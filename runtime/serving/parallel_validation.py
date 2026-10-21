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

from core.io import read_json, sha256, write_json
from hidden.artifacts import event_records, read_arrays, read_branch_index
from hidden.backends.vllm.online import capture_request_metadata
from rollout.model import make_model
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
        query_end = time.monotonic()
        query_ms = (query_end - start) * 1000
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
                "query_started_monotonic": start, "query_finished_monotonic": query_end,
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
    loop_check = None
    if cfg["parallel"].get("reasoning_loop", {}).get("enabled"):
        loop_check = validate_reasoning_loop_abort(cfg, run_id, factory, runtime, out)
    # Exercise the configured task/branch concurrency through the same model
    # wrapper, admission controller and concurrent publication/finalization.
    branches = cfg["parallel"]["branch_count"]
    groups = min(cfg["execution"]["rollout_workers"],
                 cfg["parallel"]["max_inflight_branches"] // branches)
    with ThreadPoolExecutor(max_workers=groups) as pool:
        futures = [pool.submit(run_case, repetitions + index, "parallel_overlap", False, None, branches)
                   for index in range(groups)]
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
    overlap = grouped["parallel_overlap"]
    window = max(r["query_finished_monotonic"] for r in overlap) - min(
        r["query_started_monotonic"] for r in overlap)
    report = {"passed": True, "cases": cases, "metrics": metrics, "reasoning_loop_abort": loop_check,
              "overlap": {"groups": groups, "branches_per_group": branches,
                          "query_window_ms": window * 1000,
                          "completion_tokens_per_wall_second": sum(r["completion_tokens"] for r in overlap) / window},
              "scope": "small natural-query diagnostic; shared GPUs, variable lengths, no success-rate claim"}
    write_json(out / "report.json", report)
    return {"passed": True, "metrics": metrics, "report": str(out / "report.json")}


def validate_reasoning_loop_abort(cfg, run_id, factory, runtime, out):
    """Controlled token repetition through the same real non-streaming transport.

    This check-only case uses the existing native allowed-token diagnostic
    mechanism. It never executes tools and does not create training trajectories.
    """
    from minisweagent.exceptions import LimitsExceeded

    controller, inventory = runtime["capture_controller"], runtime["capture_token_inventory"]
    sid = f"startup-loop-{run_id}"
    md = capture_request_metadata(sid, "startup-loop", run_id, sid, 0, cfg["capture"]["anchors"],
        newline_token_ids=inventory["newline_token_ids"], close_token_ids=inventory["close_token_ids"],
        tokenizer_fingerprint=inventory["tokenizer_fingerprint"], capture_select=inventory["capture_select"])
    controller.start(md)
    model = make_model(cfg, factory.deadline, md, runtime)
    model.config.model_kwargs["extra_body"]["allowed_token_ids"] = [inventory["newline_token_ids"][0]]
    options = cfg["parallel"]["reasoning_loop"]
    max_tokens = max(options["min_generated_tokens"], options["min_repeated_tokens"],
                     options["repeat_count"]) + options["check_interval_tokens"] * 8 + 1024
    model.config.model_kwargs["max_tokens"] = max_tokens
    messages = [{"role": "user", "content": "Controlled reasoning loop abort diagnostic; do not execute tools."}]
    started = time.monotonic()
    try:
        model.query(messages)
    except LimitsExceeded as exc:
        if exc.messages[-1]["extra"].get("limit_reason") != "all_parallel_branches_reasoning_loop":
            raise
        exit_message = exc.messages[-1]
    else:
        raise ValueError("Controlled repeated-token branches did not stop with LimitsExceeded")
    decisions = [row for row in controller.parallel_decisions.values() if row["session_id"] == sid]
    if len(decisions) != 1 or decisions[0]["selected_branch_id"] is not None:
        raise ValueError("Loop abort must preserve exactly one decision with no winner")
    branches = []
    for candidate in decisions[0]["candidates"]:
        receipt = controller.request_outputs[candidate["native_request_id"]]
        evidence = receipt.get("reasoning_loop") or {}
        if (candidate["reason"] != "reasoning_loop" or receipt.get("finish_reason") != "abort"
                or not evidence.get("abort_completed") or len(receipt["output_token_ids"]) >= max_tokens):
            raise ValueError("Native loop branch did not abort before its output limit")
        branches.append({"branch_id": candidate["branch_id"], "native_request_id": receipt["request_id"],
                         "output_tokens": len(receipt["output_token_ids"]), "limit": max_tokens,
                         "reasoning_loop": evidence})
    path = out / "reasoning-loop-abort.json"
    write_json(path, {"messages": [*messages, exit_message], "branches": branches, "diagnostic": True})
    endpoint = cfg["serving"]["base_url"].removesuffix("/v1").rstrip("/")
    response = requests.post(endpoint + "/dycon/capture/engine/flush", timeout=cfg["capture"]["finalize_seconds"])
    response.raise_for_status()
    controller.stop(sid)
    controller.prepare_trajectory(sid, path)
    finalized = controller.finalize(sid, rollout_id=sid, trajectory_path=str(path),
        trajectory_sha256=sha256(path), model_fingerprint=cfg["model"]["fingerprint"])
    # An all-aborted diagnostic has no accepted assistant and no close hidden.
    # Its partial session is retained explicitly; normal cases require complete.
    coverage = read_json(controller.export_root / f"sessions/{sid}/parallel/branch-coverage.json")
    if coverage["missing"] or len(coverage["not_required"]) != len(branches):
        raise ValueError("Loop abort close requirements were not classified")
    return {"passed": True, "query_ms": (time.monotonic() - started) * 1000,
            "session_id": sid, "branches": branches, "actions_executed": False,
            "capture_status": finalized["capture_status"],
            "scope": "controlled native allowed-token diagnostic; no accepted assistant or close tensor"}
