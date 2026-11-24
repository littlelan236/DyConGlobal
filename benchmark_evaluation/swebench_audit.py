"""Bound evidence collection and immutable offline SWE-bench audit service."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from benchmark_evaluation.swebench_settlement import official_result
from benchmark_evaluation.swebench_verifier import classify, digest, summarize
from core.io import identifier, read_json, sha256, write_json


def collect(output, evaluation_task_id, rollout_id, task, outcome, *, gold=False):
    # New summaries contain effective settlement; evidence collection must replay
    # physical execution/official scoring, never the projected null label.
    if outcome.get("scoring_policy"):
        outcome = {**outcome, **outcome.get("raw_outcome", {})}
    try:
        return _collect(output, evaluation_task_id, rollout_id, task, outcome, gold=gold)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"task": task, "identity": {"instance_id": task.get("instance_id")},
                "official_result": official_result(outcome),
                "gold": gold, "collection_errors": [f"Cannot read attempt evidence: {type(exc).__name__}: {exc}"]}


def _collect(output, evaluation_task_id, rollout_id, task, outcome, *, gold=False):
    """Read exactly one physical attempt; never fetch metadata or run tests."""
    from swebench.harness.grading import get_logs_eval
    from swebench.harness.constants import FAIL_ONLY_REPOS
    from importlib.metadata import version

    output = Path(output).resolve()
    iid = identifier(task["instance_id"])
    base = output / "harness" / identifier(evaluation_task_id) / identifier(rollout_id) / iid
    environment = output / "environment" / identifier(evaluation_task_id) / iid
    document = {
        "identity": {"instance_id": iid, "evaluation_task_id": evaluation_task_id, "rollout_id": rollout_id,
                     "task_sha256": digest(task), "patch_sha256": outcome.get("patch_sha256"),
                     "test_patch_sha256": digest(task.get("test_patch", "")),
                     "base_commit": task.get("base_commit"), "image": task.get("nas_image", task.get("image")),
                     "harness_version": "swebench==" + version("swebench")},
        "task": task, "gold": gold,
        "eval_type": "fail_only" if task["repo"] in FAIL_ONLY_REPOS else "pass_and_fail",
        "official_result": official_result(outcome),
        "report": {}, "statuses": {}, "parser_found": False, "test_output": "",
        "sources": [], "collection_errors": [], "environment": {},
    }
    captured = []
    raw_sources = {}
    if version("swebench") != "4.1.0":
        document["collection_errors"].append("Unsupported official harness/parser version")
    def read(path, required=False):
        if not path.exists():
            if required:
                document["collection_errors"].append(f"Missing evidence: {path}")
            return None
        resolved = path.resolve()
        if not resolved.is_relative_to(output):
            document["collection_errors"].append(f"Evidence escapes harness root: {path}")
            return None
        before = sha256(path)
        raw_sources[path] = path.read_bytes()
        value = raw_sources[path].decode("utf-8", errors="replace")
        if sha256(path) != before:
            document["collection_errors"].append(f"Source changed while reading: {path}")
        document["sources"].append({"path": str(path), "sha256": before})
        captured.append((path, before))
        return value
    completed = document["official_result"].get("completed") is True
    text = read(base / "test_output.txt", completed)
    document["test_output"] = text or ""
    report = read(base / "report.json", completed)
    if report:
        try:
            value = json.loads(report)
            document["report"] = value[iid]
            if set(value) != {iid}:
                document["collection_errors"].append("Unexpected report instance IDs")
        except (ValueError, KeyError, TypeError):
            document["collection_errors"].append("Malformed official report")
    eval_script = read(base / "eval.sh")
    instance_log = read(base / "run_instance.log")
    # A flag or a run name is insufficient to apply gold-only policy to model
    # output. Bind the harness input to the frozen reference patch, and compare
    # the actual official patch file whenever execution reached that stage.
    patch_text = read(base / "patch.diff")
    reference = task.get("patch")
    reference_ok = isinstance(reference, str) and bool(reference.strip())
    input_matches = reference_ok and outcome.get("patch_sha256") == digest(reference)
    artifact_matches = patch_text == reference if patch_text is not None else None
    document["gold_provenance"] = {
        "verified": bool(gold and input_matches and artifact_matches is not False),
        "input_matches_reference": bool(input_matches),
        "patch_artifact_matches_reference": artifact_matches,
        "reference_sha256": digest(reference) if reference_ok else None,
    }
    document["attempt_result"] = {k: outcome[k] for k in
        ("status", "resolved", "failure_type", "failure_detail", "error") if k in outcome}
    if gold and (not input_matches or artifact_matches is False):
        document["collection_errors"].append("Gold input/reference patch identity not established")
    if not gold and patch_text is not None:
        import hashlib
        if hashlib.sha256(patch_text.encode()).hexdigest() != outcome.get("patch_sha256"):
            document["collection_errors"].append("Submitted patch digest mismatch")
        else:
            # Only malformed diff syntax proves a submission defect here. A
            # generic hunk failure might instead reflect a wrong base/image.
            import re
            terminal = (instance_log or "").split(">>>>> Patch Apply Failed:")
            if (len(terminal) == 2 and ">>>>> Applied Patch" not in (instance_log or "")
                    and not eval_script and re.search(
                        r"(?im)^error: (?:No valid patches in input|corrupt patch at line \d+|unrecognized input)",
                        terminal[1])):
                document["submission_failure"] = {"reason_code": "invalid_patch",
                    "attribution": "subject", "patch_sha256": outcome.get("patch_sha256"),
                    "basis": "Bound submitted diff has explicit git syntax rejection before test execution"}
    for name in ("container-spec.json", "environment-policy.json", "bootstrap-receipt.json",
                 "environment.json", "environment-after-eval.json", "eval-execution.json", "dependency-failure.json"):
        content = read(environment / name)
        if content:
            try:
                value = json.loads(content)
                if name == "eval-execution.json":
                    document["execution"] = value
                elif name == "dependency-failure.json":
                    document["dependency_failure"] = value
                else:
                    document["environment"][name] = value
            except ValueError:
                document["collection_errors"].append(f"Malformed environment receipt: {name}")
    if outcome.get("dependency_failure"):
        document["dependency_failure"] = outcome["dependency_failure"]
    if document.get("execution"):
        execution = document["execution"]
        # Docker merges stdout/stderr chunks without preserving cross-stream
        # order. The exact returned output and command identity bind a receipt
        # even when stdout was appended after stderr's End Test Output marker.
        if text is not None and execution.get("output_sha256"):
            import hashlib
            # The official runner appends its own timeout message after the host
            # receipt was written. Only that exact known suffix may be removed.
            observed = raw_sources[base / "test_output.txt"]
            if execution.get("timed_out"):
                import re
                observed = re.sub(rb"\n\nTimeout error: \d+ seconds exceeded\.$", b"", observed)
            if hashlib.sha256(observed).hexdigest() != execution["output_sha256"]:
                document["collection_errors"].append("Execution output digest mismatch")
            else:
                execution["output_verified"] = True
        if execution.get("instance_id") != iid or execution.get("evaluation_task_id") != evaluation_task_id:
            document["collection_errors"].append("Execution receipt identity mismatch")
        if eval_script and execution.get("test_command_sha256"):
            from benchmark_evaluation.swebench_receipts import get_test_command, original_script, get_pytest_binding
            details = get_test_command(original_script(eval_script))
            if not details or details["command_sha256"] != execution["test_command_sha256"]:
                document["collection_errors"].append("Execution/test script command mismatch")
            if get_pytest_binding(eval_script) != execution.get("pytest_binding"):
                document["collection_errors"].append("Execution/pytest plugin binding mismatch")
    if document.get("execution", {}).get("receipt_protocol") == 2:
        execution = document["execution"]
        sessions = []
        if execution.get("receipt_error"):
            document["collection_errors"].append("Sidecar collection failed: " + execution["receipt_error"])
        for item in execution.get("receipt_files", []):
            name = item.get("name", "")
            if not name or Path(name).name != name or not name.endswith((".json", ".txt")):
                document["collection_errors"].append("Invalid receipt filename")
                continue
            path = environment / "pytest-receipts" / name
            content = read(path, True)
            if content is None:
                continue
            if sha256(path) != item.get("sha256"):
                document["collection_errors"].append("Sidecar receipt digest mismatch")
                continue
            if name.endswith(".txt"):
                continue
            try:
                value = json.loads(content)
                if name == "command.json":
                    execution["sidecar_test_result"] = value
                else:
                    if value.get("kind") != "phase":
                        sessions.append(value)
            except ValueError:
                document["collection_errors"].append("Malformed sidecar receipt")
        execution["pytest_sessions"] = sessions
    if text:
        try:
            from swebench.harness import grading
            from benchmark_evaluation.swebench_test_identity import parser_for, validate_collection, contract
            if contract(task):
                original = grading.MAP_REPO_TO_PARSER[task['repo']]
                try:
                    grading.MAP_REPO_TO_PARSER[task['repo']] = parser_for(task, getattr(original, '_dycon_identity_base', original))
                    statuses, found = get_logs_eval(SimpleNamespace(repo=task["repo"], version=task["version"], instance_id=iid),
                                                  str(base / "test_output.txt"))
                finally:
                    grading.MAP_REPO_TO_PARSER[task['repo']] = original
                validate_collection(task, document.get('execution', {}))
            else:
                statuses, found = get_logs_eval(SimpleNamespace(repo=task["repo"], version=task["version"], instance_id=iid),
                                              str(base / "test_output.txt"))
            document.update(statuses=statuses, parser_found=found)
        except Exception as exc:
            document["collection_errors"].append(f"Official parser failed: {type(exc).__name__}: {exc}")
    for path, before in captured:
        if not path.exists() or sha256(path) != before:
            document["collection_errors"].append(f"Source changed during collection: {path}")
    return document


def audit_run(root, run_id, output, *, rollout_ids=(), scope="all", labels=None):
    """Formal CLI service: independent output, no writes within historical runs."""
    root = Path(root).resolve()
    run = root / "runs" / identifier(run_id)
    harness = run / "harness"
    output = Path(output).resolve()
    if output.is_relative_to(root / "runs"):
        raise ValueError("Audit output must be outside historical runs")
    if output.exists():
        raise FileExistsError(output)
    summary_path, selection_path = harness / "summary.json", harness / "selection.json"
    before = {str(p): sha256(p) for p in (summary_path, selection_path)}
    summary, selection = read_json(summary_path), read_json(selection_path)
    config = read_json(harness / "resolved-config.json")
    receipt_path = harness / "selection-receipt.json"
    if receipt_path.exists():
        from core.io import hash_json
        if read_json(receipt_path).get("selection_sha256") != hash_json(selection):
            raise ValueError("Harness selection receipt mismatch")
    if config.get("selection") and digest(config["selection"]) != digest(selection):
        raise ValueError("Frozen selection and evaluation config disagree")
    if config.get("benchmark", {}).get("backend", "swebench") != "swebench":
        raise ValueError("SWE-bench audit requires SWE-bench harness evidence")
    selected = set(rollout_ids) if rollout_ids else set(summary["outcomes"])
    if selected - set(summary["outcomes"]):
        raise ValueError("Unknown rollout IDs")
    label_map = read_json(labels) if labels else {}
    if labels:
        before[str(Path(labels).resolve())] = sha256(Path(labels))
    if set(label_map) - set(summary["outcomes"]):
        raise ValueError("Attribution labels reference unknown rollout IDs")
    output.mkdir(parents=True)
    audits, index = [], []
    for rid in sorted(selected):
        outcome = summary["outcomes"][rid]
        raw = outcome.get("raw_outcome", outcome)
        category = ("positive" if raw.get("resolved") is True else "negative"
                    if raw.get("status") == "completed" else "infra")
        if scope != "all" and category != scope:
            continue
        row = selection["rollouts"][rid]
        task = selection["tasks"][row["task_key"]]["instance"]
        attempts = outcome.get("attempts", [])
        result = None
        if not attempts:
            document = {"task": task, "identity": {"instance_id": task["instance_id"]},
                        "official_result": official_result(outcome),
                        "collection_errors": ["No physical evaluation attempt"], "gold": summary.get("gold", False)}
            if outcome.get("scoring_policy"):
                # Empty submissions intentionally have no container attempt.
                # Revalidate the same frozen bytes and hash-bound submission
                # audit; do not reinterpret every missing attempt as a zero.
                reference = outcome.get("verifier_audit", {})
                audit_path = Path(reference.get("path", "")).resolve()
                try:
                    if not audit_path.is_relative_to(harness.resolve() / "verifier"):
                        raise ValueError("Submission audit escapes harness root")
                    if sha256(audit_path) != reference.get("sha256"):
                        raise ValueError("Submission audit digest mismatch")
                    observed = read_json(audit_path)
                    if summary.get("gold"):
                        empty = not task.get("patch", "").strip() and outcome["patch_sha256"] == digest(task.get("patch", ""))
                        reason = "gold_reference_patch_missing"
                    else:
                        from core.io import owned_path
                        patch = owned_path(root, row["patch_path"])
                        empty = not patch.read_text().strip() and sha256(patch) == row["patch_sha256"] == outcome["patch_sha256"]
                        reason = "missing_patch"
                    if not empty or observed.get("reason_code") != reason:
                        raise ValueError("Submission audit does not bind an empty input")
                    from benchmark_evaluation.swebench_settlement import project, validate
                    replay = project(outcome.get("raw_outcome", {}), observed, gold=summary.get("gold") is True)
                    validate(replay)
                    if (replay["resolved"], replay["status"]) != (outcome["resolved"], outcome["status"]):
                        raise ValueError("Submission settlement and audit disagree")
                    result = observed
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    document["collection_errors"].append(str(exc))
        else:
            document = collect(harness, attempts[-1]["evaluation_task_id"], rid, task, outcome,
                               gold=summary.get("gold", False))
        document["findings"] = label_map.get(rid, [])
        result = result if result is not None else classify(document)
        result.setdefault("candidate_reason_codes", [])
        result.update(instance_id=task["instance_id"], rollout_id=rid, original_category=category)
        write_json(output / "tasks" / f"{identifier(rid)}.json", result)
        audits.append(result)
        index.append({k: result[k] for k in ("instance_id", "rollout_id", "original_category", "classification",
                                            "reason_code", "evidence_digest", "candidate_reason_codes")})
    changed = [p for p, h in before.items() if sha256(Path(p)) != h]
    result = {**summarize(audits), "run_id": run_id, "scope": scope, "sources": before,
              "complete": not changed and not any(x["reason_code"] == "evidence_integrity_error" for x in audits),
              "source_changes": changed, "output": str(output)}
    write_json(output / "index.json", index)
    write_json(output / "summary.json", result)
    return result
