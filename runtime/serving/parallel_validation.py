"""Optional bounded real-query validation inside the official startup gate.

Only check-only runs call this routine; queries never execute generated tools.
All modes reuse the real model wrapper, capture, finalization and NAS reader.
"""

import importlib
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

from baseline.rollout.model import make_model
from core.io import read_json, sha256, write_json
from hidden.artifacts import event_records, read_arrays, read_branch_index
from hidden.backends.vllm.online import capture_request_metadata
from runtime.serving.parallel import artifact_contract


def validate_parallel_queries(cfg, run_id, factory, runtime):
    repetitions = cfg["parallel"]["validation_repetitions"]
    out = Path(cfg["project_root"]) / "runs" / run_id / "startup/parallel-validation"
    out.mkdir(parents=True, exist_ok=False)
    controller = runtime["capture_controller"]
    inventory = runtime["capture_token_inventory"]
    contract = artifact_contract(cfg["parallel"]["estimator_ref"], cfg["parallel"]["estimator_sha256"],
                                 cfg["model"]["fingerprint"])
    model_config = read_json(Path(cfg["model"]["path"]) / "config.json")
    num_layers = model_config.get("text_config", model_config)["num_hidden_layers"]
    module, name = contract["loader"].split(":")
    independent = getattr(importlib.import_module(module), name)(
        Path(cfg["parallel"]["estimator_ref"]).parent, contract)
    modes = [("single", True, None, None),
             ("parallel_unscored", False, "first_valid_unscored", 2),
             ("parallel_scored", False, None, 2)]
    if cfg["parallel"]["branch_count"] >= 4:
        modes += [("parallel4_scored", False, None, 4)]
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cases = []
    case_lock = threading.Lock()

    def run_case(repeat, mode, single, selection, branches):
        sid = f"startup-bench-{run_id}-{repeat}-{mode}"
        md = capture_request_metadata(sid, "startup-benchmark", run_id, sid, 0,
            cfg["capture"]["anchors"], newline_token_ids=inventory["newline_token_ids"],
            close_token_ids=inventory["close_token_ids"],
            tokenizer_fingerprint=inventory["tokenizer_fingerprint"],
            capture_select=inventory["capture_select"])
        controller.start(md)
        messages = [{"role": "system", "content": cfg["agent"]["system_template"]},
                    {"role": "user", "content": "Think briefly, then call bash exactly once to run "
                     "printf 'PARALLEL_LATENCY_CHECK\\n'. Do not write files. "
                     "This tool call will be inspected without execution."}]
        model = make_model(cfg, factory.deadline, md, {
            **runtime, "diagnostic_single_query": single,
            "diagnostic_selection_mode": selection, "diagnostic_branch_count": branches})
        start = time.monotonic()
        message = model.query(messages)
        query_ms = (time.monotonic() - start) * 1000
        path = out / f"{repeat}-{mode}.json"
        write_json(path, {"messages": [*messages, message]})
        endpoint = cfg["serving"]["base_url"].removesuffix("/v1").rstrip("/")
        response = requests.post(endpoint + "/dycon/capture/engine/flush",
                                 timeout=cfg["capture"]["finalize_seconds"])
        response.raise_for_status()
        controller.stop(sid)
        controller.prepare_trajectory(sid, path)
        result = controller.finalize(sid, rollout_id=sid, trajectory_path=str(path),
            trajectory_sha256=sha256(path), model_fingerprint=cfg["model"]["fingerprint"])
        if result["capture_status"] != "complete":
            raise ValueError("Benchmark session not completely captured/published")
        source = controller.spool.nas if controller.spool else controller.export_root
        graphs = Counter()
        score_errors = []
        summary = message.get("extra", {}).get("dycon_parallel")
        if summary:
            branch = read_branch_index(source, sid)
            if branch["coverage"]["missing"] or summary["fallback_reason"]:
                raise ValueError("Parallel benchmark missing branches or used fallback")
            by_branch = {c["branch_id"]: c for c in summary["candidates"]}
            for row in branch["samples"]:
                event = next(e for e in row["close_events"] if e["sample_id"] ==
                             by_branch[row["branch_id"]]["close_sample_id"])
                arrays = read_arrays(source, event)
                graphs[event["graph_execution"]] += 1
                if selection is None:
                    features = np.stack([arrays[f"layer_{num_layers if s == num_layers - 1 else s}"]
                                         for s in contract["sites"]])[None]
                    offline = float(independent.predict(features)[0])
                    online = by_branch[row["branch_id"]]["score"]
                    np.testing.assert_allclose(online, offline, rtol=1e-10, atol=1e-10)
                    score_errors.append(abs(online - offline))
            if selection is None:
                winner = min(summary["candidates"], key=lambda c: (c["score"], c["branch_id"]))
                if winner["branch_id"] != summary["selected_branch_id"]:
                    raise ValueError("Selected branch differs from minimum real Ridge score")
            tokens = summary["total_usage"]["completion_tokens"]
        else:
            tokens = message["extra"]["response"]["usage"]["completion_tokens"]
            for _, event in event_records(source):
                if event["session_id"] == sid and event["anchor"] == "think_close":
                    graphs[event["graph_execution"]] += 1
        if not graphs.get("full"):
            raise ValueError("Benchmark has no FULL close evidence")
        case = {"mode": mode, "repeat": repeat, "query_ms": query_ms,
                "completion_tokens": tokens, "tokens_per_second": tokens / (query_ms / 1000),
                "graph_close_counts": dict(graphs), "session_id": sid,
                "score_absolute_errors": score_errors, "capture_status": result["capture_status"],
                "parallel": summary, "actions_executed": False}
        with case_lock:
            cases.append(case)
            write_json(out / "progress.json", list(cases))

    for repeat in range(repetitions):
        # Rotate modes so C is not always favored by warmup/cache order.
        for mode in modes[repeat % len(modes):] + modes[:repeat % len(modes)]:
            run_case(repeat, *mode)
    # Three B=2 groups exceed max_num_seqs=4 in total: verify whole-group
    # admission, cross-decision isolation, concurrent spool/finalization.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_case, repetitions + index, "parallel_overlap", False, None, 2)
                   for index in range(3)]
        for future in futures:
            future.result()
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["mode"]].append(case)
    metrics = {mode: {"queries": len(rows),
        "query_ms_p50": float(np.percentile([r["query_ms"] for r in rows], 50)),
        "query_ms_p95": float(np.percentile([r["query_ms"] for r in rows], 95)),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "aggregate_tokens_per_second": sum(r["completion_tokens"] for r in rows) /
                                       (sum(r["query_ms"] for r in rows) / 1000)} for mode, rows in grouped.items()}
    report = {"passed": True, "cases": cases, "metrics": metrics,
              "scope": "small natural-query diagnostic; shared GPUs, variable lengths, no success-rate claim"}
    write_json(out / "report.json", report)
    return {"passed": True, "metrics": metrics, "report": str(out / "report.json")}
