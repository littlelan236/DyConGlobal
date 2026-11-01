"""Pinned WebArena Verified evaluation in a dedicated Python environment.

Only ``prepare_runtime`` installs software. Evaluation is offline and consumes
the frozen private dataset, the submitted answer and the flushed browser HAR.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from core.io import hash_json, now, owned_path, read_json, sha256, write_json

REVISION = "6473f72db5dcefc97b5725b59e734504edc28a21"
REPOSITORY = "https://github.com/ServiceNow/webarena-verified.git"
RUNTIME_REF = f"runtime/evaluator-snapshots/webarena-{REVISION}/manifest.json"


def _python(cfg):
    value = cfg.get("execution_context", {}).get("evaluator_python") or cfg.get("webarena", {}).get(
        "evaluator_python", ".venvs/webarena-eval/bin/python")
    return Path(cfg["project_root"]) / value


def runtime_reference(cfg):
    """Verify the published runtime receipt without importing the evaluator."""
    root = Path(cfg["project_root"])
    ref = cfg.get("execution_context", {}).get("evaluator_runtime")
    path = owned_path(root, ref["path"] if ref else RUNTIME_REF)
    if ref and sha256(path) != ref["sha256"]:
        raise ValueError("WebArena evaluator runtime receipt changed")
    receipt = read_json(path)
    if receipt.get("schema_version") != 1 or receipt.get("revision") != REVISION:
        raise ValueError("Unqualified WebArena evaluator runtime")
    if Path(receipt["python"]).absolute() != _python(cfg).absolute():
        raise ValueError("WebArena evaluator interpreter differs from its publication")
    return {"path": str(path.relative_to(root)), "sha256": sha256(path), "revision": REVISION}


def _run_logged(argv, log, **kwargs):
    with Path(log).open("ab") as out:
        out.write(("\n$ " + " ".join(map(str, argv)) + "\n").encode())
        out.flush()
        subprocess.run(list(map(str, argv)), stdout=out, stderr=subprocess.STDOUT, check=True, **kwargs)


def _restrict_tree(path):
    """Keep bundled evaluator answers readable only by the owning account."""
    path = Path(path)
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_symlink():
            continue
        if child.is_dir():
            child.chmod(0o700)
        else:
            child.chmod(0o700 if child.stat().st_mode & 0o111 else 0o600)
    path.chmod(0o700)


def prepare_runtime(cfg, source=None):
    """Publish a dedicated, source-verified evaluator; never called by rollout."""
    from core.locking import file_lock

    root = Path(cfg["project_root"]).resolve()
    python = _python(cfg).absolute()
    if python.name != "python" or python.parent.name != "bin":
        raise ValueError("Evaluator Python must name a dedicated POSIX venv bin/python")
    target = root / RUNTIME_REF
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "state/webarena-evaluator-prepare.lock"):
        if target.exists():
            ref = runtime_reference(cfg)
            receipt = read_json(target)
            _restrict_tree(receipt["source"])
            _restrict_tree(python.parent.parent)
            request = directory / "verify-request.json"
            result = directory / "verification.json"
            request.unlink(missing_ok=True)
            result.unlink(missing_ok=True)
            write_json(request, {"runtime": receipt})
            request.chmod(0o600)
            subprocess.run([str(python), "-m", "benchmark_evaluation.webarena", "verify", str(request),
                            str(result)], cwd=root, check=True, timeout=120)
            result.chmod(0o600)
            outcome = read_json(result)
            request.unlink()
            result.unlink()
            return {"status": "complete", "runtime": ref, "verification": outcome}
        source = Path(source) if source else root / "benchmarks" / ("webarena-evaluator-" + REVISION[:12])
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        log = directory / "prepare.log"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            _run_logged(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, source],
                        log, timeout=600)
            _run_logged(["git", "-C", source, "fetch", "--depth=1", "origin", REVISION], log, timeout=600)
            _run_logged(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"], log, timeout=600)
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if head != REVISION:
            raise ValueError("Evaluator source checkout is not the pinned commit")
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                                        text=True)
        if dirty:
            raise ValueError("Evaluator source checkout has tracked modifications")
        _restrict_tree(source)
        if not python.exists():
            _run_logged([sys.executable, "-m", "venv", python.parent.parent], log, timeout=120)
        index_url = cfg.get("skillsbench_images", {}).get("python_index") or "https://pypi.org/simple"
        pip_command = [python, "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "30",
                       "--retries", "2", "--index-url", index_url]
        proxy = (cfg.get("skillsbench_images", {}).get("host_proxy")
                 or cfg.get("webarena_images", {}).get("host_proxy"))
        if proxy:
            pip_command.extend(["--proxy", proxy])
        build_requirements = ["hatchling==1.33.0", "hatch-vcs==0.5.0", "setuptools==84.2.1"]
        _run_logged([*pip_command, *build_requirements], log, timeout=600)
        _run_logged([*pip_command, "--no-build-isolation", str(source)], log, timeout=1200)
        _restrict_tree(python.parent.parent)
        probe_file = directory / "probe.json"
        subprocess.run([str(python), "-m", "benchmark_evaluation.webarena", "probe", str(probe_file)],
                       cwd=root, check=True, timeout=120)
        probe = read_json(probe_file)
        # Match every tracked evaluator Python source against the installed wheel.
        tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "src/webarena_verified"],
                                          text=True).splitlines()
        expected = {str(Path(name).relative_to("src/webarena_verified")): sha256(source / name)
                    for name in tracked if name.endswith(".py")}
        if not expected or any(probe["files"].get(name) != digest for name, digest in expected.items()):
            raise ValueError("Installed evaluator Python sources differ from the pinned checkout")
        receipt = {"schema_version": 1, "revision": REVISION, "repository": REPOSITORY,
                   "source": str(source), "python": str(python), "index_url": index_url,
                   "proxy_used": bool(proxy), "build_requirements": build_requirements,
                   "package_version": probe["package_version"],
                   "files": probe["files"], "dependencies": probe["dependencies"],
                   "evaluator_checksum": probe["evaluator_checksum"], "created_at": now()}
        receipt["identity_sha256"] = hash_json({k: v for k, v in receipt.items() if k != "created_at"})
        write_json(target, receipt)
        return {"status": "complete", "runtime": runtime_reference(cfg), "package_version": probe["package_version"]}


setup_evaluator = prepare_runtime


def _probe_runtime():
    import webarena_verified.api.webarena_verified as api
    from webarena_verified.core.utils.checksum import compute_evaluator_checksum

    package = Path(api.__file__).parents[1]
    # Exclude the host project distribution discovered from the adapter import root.
    project_distribution = "dycon-global"
    return {"package_version": importlib.metadata.version("webarena-verified"),
            "files": {path.relative_to(package).as_posix(): sha256(path)
                      for path in sorted(package.rglob("*.py"))},
            "dependencies": sorted(str(d.metadata["Name"]) + "==" + d.version
                                   for d in importlib.metadata.distributions()
                                   if d.metadata.get("Name", "").lower() != project_distribution),
            "evaluator_checksum": compute_evaluator_checksum()}


def _verify_runtime(receipt):
    if receipt.get("revision") != REVISION:
        raise ValueError("Evaluator revision mismatch")
    actual = _probe_runtime()
    for key in ("package_version", "files", "dependencies", "evaluator_checksum"):
        if actual[key] != receipt[key]:
            raise ValueError("Evaluator runtime changed: " + key)
    return {"verified": True, "revision": REVISION, "evaluator_checksum": actual["evaluator_checksum"]}


def _worker(request):
    from webarena_verified.api import WebArenaVerified
    from webarena_verified.types.config import WebArenaVerifiedConfig

    _verify_runtime(request["runtime"])
    dataset = Path(request["dataset"]["path"])
    if not dataset.is_file() or sha256(dataset) != request["dataset"]["sha256"]:
        raise ValueError("Frozen evaluator dataset changed or is unavailable")
    config = WebArenaVerifiedConfig(test_data_file=dataset, environments=request["environments"])
    if config.test_data_file.resolve() != dataset.resolve():
        raise ValueError("Official evaluator attempted a dataset fallback")
    evaluator = WebArenaVerified(config=config)
    task = evaluator.get_task(request["task_id"])
    if (task.revision != request["task_revision"] or hash_json(read_json(dataset)[request["task_index"]])
            != request["source_task_sha256"]):
        raise ValueError("Private evaluator task differs from the frozen public task identity")
    result = evaluator.evaluate_task(task_id=request["task_id"],
                                     agent_response=Path(request["answer"]), network_trace=Path(request["har"]))
    return result.model_dump(mode="json")


def classify_result(result):
    """Official errors remain unknown; a valid negative assertion is a failure."""
    status = result.get("status")
    if status in {"success", "failure", "partial_match"}:
        return {"status": "completed", "resolved": status == "success", "score": result["score"]}
    if status == "error":
        return {"status": "infra_error", "resolved": None, "failure_type": "evaluator_error",
                "reason": result.get("error_msg") or "OfficialEvaluatorError"}
    raise ValueError("Unknown official evaluator status: " + str(status))


def _invoke_worker(command, root, log, timeout):
    from core.cancellation import CURRENT

    cancellation = CURRENT.get()
    if cancellation:
        cancellation.check()
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    remove_callback = cancellation.add_callback(process.terminate) if cancellation else lambda: None
    try:
        try:
            process.wait(timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise
        if cancellation:
            cancellation.check()
        return process
    finally:
        remove_callback()


def evaluate(cfg, task, subject_path, output_dir):
    """Evaluate one verified web subject using only persisted bytes."""
    root, output = Path(cfg["project_root"]).resolve(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)  # Official detailed results contain expected answers.
    subject_path = Path(subject_path)
    subject = read_json(subject_path)
    if (subject.get("kind") != "web_trace" or not subject.get("complete")
            or not subject.get("har_flushed")):
        raise ValueError("WebArena evaluation requires a complete flushed web_trace subject")
    outputs = {row["path"]: row for row in subject["outputs"]}
    for name in (subject["answer_path"], subject["har_path"]):
        file = owned_path(subject_path.parent, name)
        if name not in outputs or file.is_symlink() or sha256(file) != outputs[name]["sha256"]:
            raise ValueError("WebArena subject output changed: " + name)
    environments = subject.get("environments") or subject.get("runtime_identity", {}).get("environments")
    if not environments or any(not environments.get(site, {}).get("urls") for site in task["instance"]["sites"]):
        raise ValueError("WebArena subject lacks its exact site URL mapping")
    runtime_ref = runtime_reference(cfg)
    dataset_ref = cfg["execution_context"]["evaluator"]
    dataset = owned_path(root, dataset_ref["path"])
    if dataset_ref.get("revision") != REVISION or sha256(dataset) != dataset_ref["sha256"]:
        raise ValueError("WebArena frozen evaluator dataset changed")
    definitions = read_json(dataset)
    task_id = task["instance"]["source_task_id"]
    indexes = [i for i, row in enumerate(definitions) if row["task_id"] == task_id]
    if len(indexes) != 1:
        raise ValueError("Evaluator dataset has no unique task definition")
    payload = {"runtime": read_json(root / runtime_ref["path"]),
               "dataset": {**dataset_ref, "path": str(dataset)}, "environments": environments,
               "task_id": task_id, "task_index": indexes[0],
               "task_revision": task["instance"]["task_revision"],
               "source_task_sha256": task["instance"]["source_task_sha256"],
               "answer": str(owned_path(subject_path.parent, subject["answer_path"])),
               "har": str(owned_path(subject_path.parent, subject["har_path"]))}
    request_path, result_path = output / "request.json", output / "official-result.json"
    write_json(request_path, payload)
    os.chmod(request_path, 0o600)
    result_path.unlink(missing_ok=True)
    try:
        command = [str(_python(cfg)), "-m", "benchmark_evaluation.webarena", "worker",
                   str(request_path), str(result_path)]
        with (output / "worker.log").open("wb") as log:
            completed = _invoke_worker(command, root, log, cfg.get("timeouts", {}).get("evaluation_seconds", 300))
        if completed.returncode or not result_path.is_file():
            log_path = output / "worker.log"
            detail = ""
            if log_path.is_file():
                detail = log_path.read_text(errors="replace")[-4000:]
            return {"status": "infra_error", "resolved": None, "failure_type": "evaluator_process_error",
                    "reason": "EvaluatorProcessExit", "returncode": completed.returncode,
                    "error": detail}
        os.chmod(result_path, 0o600)
        official = read_json(result_path)
        return {**classify_result(official), "official_status": official["status"],
                "official_result_path": str(result_path.relative_to(root)),
                "official_result_sha256": sha256(result_path), "evaluator_runtime": runtime_ref,
                "dataset_sha256": dataset_ref["sha256"],
                "evaluator_checksum": official.get("webarena_verified_evaluator_checksum")}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"status": "infra_error", "resolved": None, "failure_type": "evaluator_process_error",
                "reason": type(exc).__name__, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Internal isolated WebArena evaluation worker")
    parser.add_argument("action", choices=("probe", "verify", "worker"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.action == "probe":
        write_json(args.paths[0], _probe_runtime())
    elif args.action == "verify":
        write_json(args.paths[1], _verify_runtime(read_json(args.paths[0])["runtime"]))
    else:
        write_json(args.paths[1], _worker(read_json(args.paths[0])))


if __name__ == "__main__":
    main()
