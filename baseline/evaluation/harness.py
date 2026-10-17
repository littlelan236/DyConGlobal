from __future__ import annotations

import concurrent.futures
import inspect
import time
from pathlib import Path

from baseline.catalog import Catalog
from core.io import hash_json, identifier, now, update_status, write_json
from core.locking import exclusive_run
from runtime.containers.factory import ContainerFactory, TaskContext


def install_harness_adapter(cfg: dict, factory: ContainerFactory, rows: list[dict], output: Path):
    import importlib.metadata

    from swebench.harness import run_evaluation as official
    from swebench.harness.test_spec import python as sb_python
    from swebench.harness.test_spec.test_spec import TestSpec

    if importlib.metadata.version("swebench") != "4.1.0":
        raise RuntimeError("Harness adapter is validated only against swebench 4.1.0")
    required = {"test_spec", "pred", "client", "run_id", "timeout"}
    if not required.issubset(inspect.signature(official.run_instance).parameters):
        raise RuntimeError("SWE harness interface changed")
    instances = {r["instance_id"]: r for r in rows}
    saved_official = {
        key: getattr(official, key)
        for key in ["build_container", "cleanup_container", "RUN_EVALUATION_LOG_DIR", "exec_run_with_timeout"]
    }
    original_script = TestSpec.eval_script
    original_get = sb_python.requests.get
    original_requirements = sb_python.get_requirements_by_commit

    # Requirements metadata is advisory for the project's locked prebuilt SWE
    # images. If all remote candidates fail, use the image environment and keep
    # the failure visible in the evaluation log instead of aborting spec build.
    def cached_requirements(repo, commit):
        # Network/metadata failures are infrastructure errors, never fabricated
        # empty dependency metadata. Real 404s still reach the official fallback.
        return original_requirements(repo, commit)

    # Keep exact downloaded metadata content in the project, then prefer it offline.
    cache = Path(
        cfg["environment"].get(
            "requirements_cache", str(Path(cfg["project_root"]) / "data/swebench/requirements-cache")
        )
    )
    cache.mkdir(parents=True, exist_ok=True)

    def cached_get(url, *args, **kwargs):
        import requests

        if not url.startswith("https://raw.githubusercontent.com/"):
            return original_get(url, *args, **kwargs)
        p = cache / (hash_json(url) + ".txt")
        if p.exists():
            response = requests.Response()
            response.status_code = 200
            response._content = p.read_bytes()
            response.url = url
            return response
        kwargs.setdefault("timeout", cfg["timeouts"].get("metadata_request_seconds", 30))
        try:
            response = original_get(url, *args, **kwargs)
            # Official requirements lookup tries several candidate paths.
            # A real missing file must reach that lookup, not abort it here.
            if response.status_code in (404, 410):
                return response
            response.raise_for_status()
        except requests.RequestException:
            parts = url.removeprefix("https://raw.githubusercontent.com/").split("/", 3)
            if len(parts) != 4:
                raise
            owner, repo, revision, path = parts
            response = original_get(
                f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{revision}/{path}", *args, **kwargs
            )
            if response.status_code in (404, 410):
                return response
            response.raise_for_status()
        p.write_bytes(response.content)
        return response

    def build_container(test_spec, client, run_id, logger, nocache, force_rebuild):
        if force_rebuild or nocache:
            raise ValueError("Project harness uses locked prebuilt images, not implicit rebuilds")
        iid = test_spec.instance_id
        ctx = TaskContext(
            cfg.get("evaluation_run_id", output.name),
            iid,
            run_id,
            "harness",
            output / "environment" / run_id / iid,
        )
        return factory.create(instances[iid], ctx)

    # This is scoped to a dedicated evaluation process, not docker SDK globally.
    official.build_container = build_container
    official.cleanup_container = lambda client, container, logger: (
        factory.close(container) if container else None
    )
    official.RUN_EVALUATION_LOG_DIR = output / "harness"
    sb_python.requests.get = cached_get
    sb_python.get_requirements_by_commit = cached_requirements
    TestSpec.eval_script = property(
        lambda self: factory.policy_for(instances[self.instance_id]).wrap_eval(original_script.fget(self))
    )

    def execute_eval(container, cmd, timeout):
        started = time.monotonic()
        out = factory.execute(container, "source /opt/miniconda3/bin/activate && " + cmd, timeout)
        timed_out = out["returncode"] in [124, 137]
        if not timed_out:
            try:
                factory.probe(container, "environment-after-eval.json")
            except Exception as exc:
                write_json(container.context.output_dir / "post-eval-probe-error.json", {"error": str(exc)})
                raise
        return out["output"], timed_out, time.monotonic() - started

    official.exec_run_with_timeout = execute_eval

    def restore():
        for key, value in saved_official.items():
            setattr(official, key, value)
        sb_python.requests.get = original_get
        sb_python.get_requirements_by_commit = original_requirements
        TestSpec.eval_script = original_script

    official.dycon_restore = restore
    return official


@exclusive_run
def evaluate(cfg, run, eval_id, gold=False, factory=None):
    from core.io import owned_path, sha256

    output = owned_path(cfg["project_root"], f"baseline/evaluations/{identifier(eval_id)}")
    # Snapshot selection is authoritative even if the live manifest has since changed.
    selected = cfg.get("selection")
    if not selected or not selected.get("rollout_ids"):
        raise ValueError("Evaluation requires a run snapshot with explicit rollout selection")
    output.mkdir(parents=True, exist_ok=False)
    cfg = {**cfg, "evaluation_run_id": run.name}
    write_json(output / "resolved-config.json", cfg)
    rows = {}
    for rid, row in selected["rollouts"].items():
        task = selected["tasks"][row["task_key"]]
        if task["instance_id"] in rows and rows[task["instance_id"]] != task["instance"]:
            raise ValueError("Same instance ID has incompatible benchmark definitions")
        rows[task["instance_id"]] = task["instance"]
        patch = owned_path(cfg["project_root"], row["patch_path"])
        if sha256(patch) != row["patch_sha256"]:
            raise ValueError("Patch changed after run selection was frozen")
    factory = factory or ContainerFactory(cfg)
    official = None
    outcomes = {}
    update_status(output / "status.json", "evaluation")
    try:
        factory.lock_images(list(rows.values()))
        official = install_harness_adapter(cfg, factory, list(rows.values()), output)

        def worker(rid, row):
            iid = selected["tasks"][row["task_key"]]["instance_id"]
            patch = (
                rows[iid].get("patch", "")
                if gold
                else owned_path(cfg["project_root"], row["patch_path"]).read_text()
            )
            result = {
                "patch_sha256": hash_json(patch) if gold else row["patch_sha256"],
                "status": "infra_error",
                "resolved": None,
            }
            if not patch:
                return rid, {**result, "reason": "MissingPatch"}
            try:
                factory.deadline.check()
                spec = official.make_test_spec(rows[iid], namespace="swebench")
                prediction = {"instance_id": iid, "model_name_or_path": rid, "model_patch": patch}
                timeout = rows[iid].get("evaluation_timeout_seconds", cfg["timeouts"]["evaluation_seconds"])
                graded = official.run_instance(
                    spec,
                    prediction,
                    False,
                    False,
                    factory.client,
                    "task-" + hash_json({"evaluation": eval_id, "rollout": rid})[:24],
                    timeout,
                )
                result.update(
                    status="completed" if graded["completed"] else "infra_error",
                    resolved=bool(graded["resolved"]) if graded["completed"] else None,
                )
            except (Exception,) as exc:
                # A worker failure is a recorded infrastructure outcome; remaining tasks continue.
                result["error"] = repr(exc)
            return rid, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["execution"]["harness_workers"]) as pool:
            futures = [pool.submit(worker, rid, row) for rid, row in selected["rollouts"].items()]
            for future in concurrent.futures.as_completed(futures):
                rid, result = future.result()
                outcomes[rid] = result
                write_json(output / "progress.json", outcomes)
        resolved = sum(row["resolved"] is True for row in outcomes.values())
        summary = {
            "evaluation_id": eval_id,
            "gold": gold,
            "requested": len(outcomes),
            "resolved": resolved,
            "resolve_rate": resolved / len(outcomes),
            "graded": sum(row["status"] == "completed" for row in outcomes.values()),
            "outcomes": outcomes,
            "finished_at": now(),
        }
        write_json(output / "summary.json", summary)
        if not gold:
            Catalog(cfg["project_root"]).record_evaluation(
                eval_id,
                output.relative_to(Path(cfg["project_root"])) / "summary.json",
                outcomes,
                {
                    "harness": "swebench==4.1.0",
                    "run_id": run.name,
                    "config_hash": cfg["config_hash"],
                    "catalog_revision": selected["catalog_revision"],
                },
            )
        update_status(
            output / "status.json", "complete" if summary["graded"] == summary["requested"] else "incomplete"
        )
        return summary
    except BaseException as exc:
        update_status(output / "status.json", "failed", error=str(exc))
        raise
    finally:
        try:
            factory.cleanup_owned(run.name)
        finally:
            if official is not None:
                official.dycon_restore()
