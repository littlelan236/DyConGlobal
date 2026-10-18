"""GPU regression for managed decode capture, exact receipts and finite latents."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from core.config import resolve
from core.io import read_json, write_json
from hidden.backends.vllm.online import capture_request_metadata
from runtime.serving.manager import ServingManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).absolute()
    out.mkdir(parents=True, exist_ok=False)
    cfg = resolve(args.config, args.set + [
        "capture.enabled=true", "capture.anchors=[reasoning_newline,think_close]",
        "capture.prefill_token_chunk_size=0", "serving.dtype=float16",
        f"capture.export.root={out / 'capture'}",
    ])
    base = cfg["serving"]["base_url"].removesuffix("/v1")
    report = {"passed": False, "gpus": cfg["serving"]["gpus"], "dtype": "float16"}

    def post(route, body):
        response = requests.post(base + route, json=body, timeout=600)
        response.raise_for_status()
        return response.json()

    try:
        with ServingManager(cfg, out / "service") as service:
            controller = service.capture_controller
            # Keep diagnostic manifests alongside this run's artifacts.
            controller.root = out / "manifests"
            inventory = service.capture_token_inventory
            report["phase_execution"] = []
            report["request_seconds"] = []

            def observe(phase):
                value = requests.get(base + "/dycon/capture/engine/status", timeout=60).json()
                report["phase_execution"].append({
                    "phase": phase,
                    "graph_replays": [worker["graph_replays"] for worker in value["workers"]],
                    "eager_capture_forwards": [worker["eager_capture_forwards"] for worker in value["workers"]],
                })
                write_json(out / "phase-execution.json", report["phase_execution"])

            def metadata(sid):
                value = capture_request_metadata(sid, "diagnostic", sid, sid, 0,
                    ["reasoning_newline", "think_close"],
                    newline_token_ids=inventory["newline_token_ids"],
                    close_token_ids=inventory["close_token_ids"],
                    tokenizer_fingerprint=inventory["tokenizer_fingerprint"])
                controller.start(value)
                return value

            def request(md=None, *, long=False, allowed=198, natural=False):
                body = {
                    "model": cfg["model"]["served_name"],
                    "messages": [
                        {"role": "system", "content": "Answer carefully. " +
                         ("Repository note: inspect tests before editing. " * 180 if long else "")},
                        {"role": "user", "content": "What is 1+1?"},
                    ],
                    "temperature": 0.0, "seed": 42, "max_tokens": 48 if natural else 16,
                    "return_token_ids": True,
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
                }
                if not natural:
                    body["allowed_token_ids"] = [allowed]
                if md is not None:
                    body["dycon_capture"] = md
                started = time.perf_counter()
                response = post("/v1/chat/completions", body)
                report["request_seconds"].append({
                    "session_id": md["session_id"] if md else "capture-disabled",
                    "long": long, "natural": natural, "seconds": time.perf_counter() - started,
                })
                return response

            observe("after_engine_warmup")
            post("/dycon/capture/engine/mode", {"enabled": False})
            plain = request()
            plain_natural = request(natural=True)
            observe("after_capture_disabled")
            post("/dycon/capture/engine/mode", {"enabled": True})
            short = request(metadata("short"))
            observe("after_short_decode")
            natural = request(metadata("natural"), natural=True)
            observe("after_natural_decode")
            close_id, = inventory["close_token_ids"]
            closed = request(metadata("close"), allowed=close_id)
            observe("after_close_decode")
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(request, metadata("long-cold"), long=True),
                           pool.submit(request, metadata("mixed-short"))]
                mixed = [future.result() for future in futures]
            observe("after_concurrent_long_prefill_and_short_decode")
            warm = request(metadata("long-warm"), long=True)
            observe("after_warm_long_prompt")
            flushed = post("/dycon/capture/engine/flush", {})
            controller.queue.queue.join()
            status = requests.get(base + "/dycon/capture/engine/status", timeout=60).json()
            report["responses"] = {"plain": plain, "short": short, "close": closed,
                                   "plain_natural": plain_natural, "natural": natural,
                                   "mixed": mixed, "warm": warm}
            report["engine"] = status
            report["flush"] = flushed

            def tokens(response):
                return response["choices"][0]["token_ids"]

            assert tokens(plain) == tokens(short) == [198] * 16
            assert tokens(closed) == [close_id] * 16
            assert all(tokens(response) == [198] * 16 for response in mixed + [warm])
            report["natural_equal"] = tokens(plain_natural) == tokens(natural)
            assert report["natural_equal"], "Natural generation changed under capture"
            report["behavior_passed"] = True
            assert not status["failures"], status["failures"]
            assert all(worker["graph_replays"] > 0 for worker in status["workers"])
            report["eager_capture_forwards"] = [worker["eager_capture_forwards"] for worker in status["workers"]]
            report["fullgraph_only_passed"] = not any(report["eager_capture_forwards"])
            rows = [read_json(path) for path in (controller.export_root / "index").glob("*.json")]
            report["captured_rows"] = len(rows)
            report["anchors"] = sorted({row["anchor"] for row in rows})
            assert set(report["anchors"]) == {"reasoning_newline", "think_close"}
            for row in rows:
                receipt = controller.request_outputs[row["request_id"]]
                ids = receipt["prompt_token_ids"] + receipt["output_token_ids"]
                assert ids[row["absolute_position"]] == row["token_id"]
                with np.load(controller.export_root / row["tensor"]) as arrays:
                    assert all(np.isfinite(arrays[key]).all() for key in arrays.files), row
            report["finite_hidden_and_alignment_passed"] = True
            report["cached_tokens"] = [receipt["num_cached_tokens"]
                                        for receipt in controller.request_outputs.values()]
            assert max(report["cached_tokens"]) > 0, "No APC hit on repeated long prompt"
            assert not controller.queue.status()["failed"]
            assert report["fullgraph_only_passed"], report["eager_capture_forwards"]
            report["passed"] = True
    finally:
        write_json(out / "report.json", report)
        print({key: value for key, value in report.items() if key not in {"responses", "engine"}}, flush=True)


if __name__ == "__main__":
    main()
