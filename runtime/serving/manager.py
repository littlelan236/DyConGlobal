"""Managed pinned EasySteer serving or explicitly configured external service."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from core.io import hash_json, now, sha256, write_json
from runtime.serving.capture import CaptureControlServer, OnlineCaptureController
from runtime.serving.resources import reserve_serving_devices


class ServingManager:
    def __init__(self, cfg, output):
        self.cfg, self.output = cfg, Path(output)
        self.process = None
        self.log = None
        self.capture_controller = None
        self.capture_server = None
        self.capture_token_inventory = {}
        self.runtime = {}
        self.device_reservation = None
        self.device_inventory = None

    def build_args(self, serving_python, host, port):
        cfg, s = self.cfg, self.cfg["serving"]
        if cfg.get("parallel", {}).get("enabled") and cfg["parallel"]["branch_count"] > 1:
            from runtime.serving.parallel import artifact_contract

            opts = cfg["parallel"]
            artifact_contract(opts["estimator_ref"], opts["estimator_sha256"], cfg["model"]["fingerprint"])
        cap = cfg.get("capture", {})
        args = [
            serving_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            cfg["model"]["path"],
            "--served-model-name",
            cfg["model"]["served_name"],
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(s["tensor_parallel_size"]),
            "--gpu-memory-utilization",
            str(s["gpu_memory_utilization"]),
            "--max-model-len",
            str(s["max_model_len"]),
            "--dtype",
            s["dtype"],
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            s["tool_parser"],
            "--reasoning-parser",
            s["reasoning_parser"],
            "--generation-config",
            "vllm",
            "--no-async-scheduling",
            "--max-num-seqs",
            str(s["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(s["max_num_batched_tokens"]),
        ]
        if s.get("kv_cache_memory_bytes") is not None:
            args += ["--kv-cache-memory-bytes", str(s["kv_cache_memory_bytes"])]
        algorithms = set()
        if cfg["steering"]["enabled"]:
            from runtime.steering.easysteer.payload import steering_payload

            algorithms.update(v["algorithm"] for v in steering_payload(cfg)["vectors"])
        if cap.get("enabled"):
            algorithms.add("direct")
            sizes = sorted(
                {n for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] if n <= s["max_num_seqs"]}
                | {s["max_num_seqs"]}
            )
            args += [
                "--compilation-config",
                json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": sizes}),
                "--worker-extension-cls",
                "runtime.serving.capture_worker.CaptureWorkerExtension",
            ]
        if algorithms:
            args += [
                "--enable-steer-vector",
                "--steer-algorithms",
                ",".join(sorted(algorithms)),
                "--steer-graph-mode",
                "in_graph",
                "--steer-vector-dtype",
                "float32",
            ]
        if s["language_model_only"]:
            args.append("--language-model-only")
        if cfg["model"]["trust_remote_code"]:
            args.append("--trust-remote-code")
        return args

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _enter(self):
        cfg, s = self.cfg, self.cfg["serving"]
        if cfg.get("parallel", {}).get("enabled") and cfg["parallel"]["branch_count"] > 1:
            from runtime.serving.parallel import artifact_contract

            options = cfg["parallel"]
            artifact_contract(options["estimator_ref"], options["estimator_sha256"], cfg["model"]["fingerprint"])
        # ``resolve`` canonicalizes ordinary paths, which would turn a venv's
        # python symlink into the shared interpreter.  Managed serving must use
        # the pinned EasySteer venv, so retain the authoritative raw path.
        from hidden.backends.vllm.engine import serving_interpreter

        serving_python = str(serving_interpreter(cfg))
        self.output.mkdir(parents=True, exist_ok=True)
        capture_cfg = cfg.get("capture", {})
        if capture_cfg.get("enabled") and s["management"] != "managed":
            raise ValueError("Online capture requires managed serving")
        if s["management"] == "managed":
            if not s["tool_parser"] or not s["reasoning_parser"]:
                raise ValueError("Managed serving requires explicit tool_parser and reasoning_parser")
            endpoint = urlparse(s["base_url"])
            if endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.scheme != "http":
                raise ValueError("Managed service must bind a local HTTP endpoint")
            port = endpoint.port or 80
            with socket.socket() as check:
                check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                check.bind((endpoint.hostname, port))
            root = Path(cfg["project_root"])
            if capture_cfg.get("enabled"):
                model_config = json.loads((Path(cfg["model"]["path"]) / "config.json").read_text())
                architectures = model_config.get("architectures", [])
                is_qwen_gdn = model_config.get("model_type") in {
                    "qwen3_5",
                    "qwen3_5_moe",
                    "qwen3_next",
                } or any("Qwen3_5" in str(name) for name in architectures)
                if is_qwen_gdn and capture_cfg.get("prefill_token_chunk_size", 0) == 1:
                    raise ValueError(
                        "Qwen GDN cannot start with capture.prefill_token_chunk_size=1: "
                        "a fresh prompt can enter the decode path without state initialization, "
                        "producing NaN logits and token 0. Use 0 for native prefill scheduling."
                    )
                if (
                    is_qwen_gdn
                    and "think_open" in capture_cfg.get("anchors", [])
                    and not capture_cfg.get("allow_unsupported_gdn", False)
                ):
                    raise ValueError(
                        "Qwen GDN FULL graphs support decode capture only. Remove think_open "
                        "from capture.anchors; allow_unsupported_gdn is only for prefill diagnostics."
                    )
            model_type = json.loads((Path(cfg["model"]["path"]) / "config.json").read_text()).get(
                "model_type", ""
            )
            if model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_next"} and (
                s["tool_parser"] not in {"qwen3_xml", "qwen3_coder"} or s["reasoning_parser"] != "qwen3"
            ):
                raise ValueError("Current Qwen requires explicit qwen3_xml/qwen3_coder and qwen3 parsers")
            verify = root / "runtime/scripts/verify_easysteer_runtime.py"
            subprocess.run([serving_python, str(verify)], check=True, capture_output=True, text=True)
            if capture_cfg.get("enabled"):
                if not capture_cfg.get("layers"):
                    raise ValueError("Online capture requires capture.layers to be configured")
                inventory_code = (
                    "import json,sys; from transformers import AutoTokenizer; "
                    "from hidden.backends.vllm.online import tokenizer_newline_inventory; "
                    "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=True); "
                    "ids,texts=tokenizer_newline_inventory(t,policy=sys.argv[2]); "
                    "print(json.dumps(dict(newline_token_ids=ids,newline_token_texts=texts,"
                    "close_token_ids=t.encode('</think>',add_special_tokens=False),open_token_ids=t.encode('<think>',add_special_tokens=False),source='active_tokenizer')))"
                )
                raw_inventory = subprocess.check_output(
                    [
                        serving_python,
                        "-c",
                        inventory_code,
                        cfg["model"]["path"],
                        capture_cfg["newline_policy"],
                    ],
                    text=True,
                    cwd=root,
                )
                self.capture_token_inventory = json.loads(raw_inventory.strip().splitlines()[-1])
                self.capture_token_inventory["tokenizer_fingerprint"] = hash_json(
                    self.capture_token_inventory
                )
                parallel = cfg.get("parallel", {})
                if parallel.get("enabled") and parallel["branch_count"] > 1:
                    contract = artifact_contract(parallel["estimator_ref"], parallel["estimator_sha256"],
                                                 cfg["model"]["fingerprint"])
                    expected = contract.get("tokenizer_fingerprint")
                    if expected is not None and expected != self.capture_token_inventory["tokenizer_fingerprint"]:
                        raise ValueError("Parallel estimator tokenizer fingerprint mismatch")
                self.capture_controller = OnlineCaptureController(
                    cfg["project_root"],
                    {
                        **capture_cfg,
                        "run_id": self.output.parent.name,
                        "nas_root": cfg["paths"]["nas_root"],
                        "reserve_bytes": cfg["images"]["reserve_bytes"],
                        "open_token_ids": self.capture_token_inventory["open_token_ids"],
                        "model_fingerprint": cfg["model"]["fingerprint"],
                        "serving_dtype": s.get("dtype", "float16"),
                        "tensor_parallel_size": s["tensor_parallel_size"],
                    },
                )
                self.capture_server = CaptureControlServer(
                    self.capture_controller,
                    capture_cfg.get("control_host", "127.0.0.1"),
                    capture_cfg.get("control_port", 0),
                ).start()
            env = dict(os.environ)
            env.pop("DYCON_PARALLEL_OPTIONS", None)
            env.pop("DYCON_PARALLEL_MODEL_FINGERPRINT", None)
            if cfg.get("parallel", {}).get("enabled") and cfg["parallel"]["branch_count"] > 1:
                from runtime.serving.parallel import artifact_contract

                opts = cfg["parallel"]
                artifact_contract(opts["estimator_ref"], opts["estimator_sha256"], cfg["model"]["fingerprint"])
                env["DYCON_PARALLEL_OPTIONS"] = json.dumps(opts)
                env["DYCON_PARALLEL_MODEL_FINGERPRINT"] = cfg["model"]["fingerprint"]
            # Both native FULL graphs and online capture compile CUDA graphs
            # through ninja. The managed venv may be pip-less while uv has
            # already materialized a compatible binary in its cache.
            import shutil

            if shutil.which("ninja", path=env.get("PATH")) is None:
                candidates = sorted(
                    Path.home().glob(".cache/uv/archive-v0/*/ninja-*.data/scripts/ninja"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True,
                )
                if candidates:
                    env["PATH"] = str(candidates[0].parent) + os.pathsep + env.get("PATH", "")
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, s["gpus"]))
            env["VLLM_PLUGINS"] = "dycon_capture" if capture_cfg.get("enabled") else ""
            env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            env["VLLM_CACHE_ROOT"] = str(root / "runtime/cache/vllm")
            if self.capture_server is not None:
                env["DYCON_CAPTURE_CONTROL_URL"] = self.capture_server.base_url
                env["DYCON_CAPTURE_GRAPH_MODE"] = "FULL"
                env["DYCON_CAPTURE_CACHE_POLICY"] = "safe_boundary"
                env["DYCON_CAPTURE_STAGING_ROOT"] = str(self.capture_controller.export_root / "staging")
                env["DYCON_CAPTURE_LAYERS"] = json.dumps(capture_cfg["layers"])
                from hidden.backends.vllm.online import build_capture_select

                selector = build_capture_select(
                    self.capture_token_inventory["newline_token_ids"]
                    if "reasoning_newline" in capture_cfg["anchors"]
                    else [],
                    self.capture_token_inventory["close_token_ids"]
                    if "think_close" in capture_cfg["anchors"]
                    else [],
                    include_open="think_open" in capture_cfg["anchors"],
                )
                self.capture_token_inventory["capture_select"] = selector
                env["DYCON_CAPTURE_SELECT"] = json.dumps(selector)
                env["DYCON_CAPTURE_FAILURE_POLICY"] = capture_cfg["failure_policy"]
                env["DYCON_CAPTURE_FLUSH_SECONDS"] = str(capture_cfg["export"]["flush_interval_ms"] / 1000)
                env["DYCON_CAPTURE_CONTROL_SECONDS"] = str(capture_cfg["control_seconds"])
                env["DYCON_CAPTURE_PROJECT"] = str(root)
                env["DYCON_CAPTURE_EXPORT_OPTIONS"] = json.dumps(capture_cfg["export"])
                env["DYCON_CAPTURE_RESERVE_BYTES"] = str(cfg["images"]["reserve_bytes"])
                self.capture_controller._write("inventory.json", self.capture_token_inventory)
                self.capture_controller.write_configuration()
                self.runtime = {
                    "capture_control_url": self.capture_server.base_url,
                    "capture_token_inventory": self.capture_token_inventory,
                    "capture_controller": self.capture_controller,
                }
                env["DYCON_CAPTURE_NEWLINE_IDS"] = json.dumps(
                    self.capture_token_inventory.get("newline_token_ids", [])
                )
                env["DYCON_CAPTURE_CLOSE_IDS"] = json.dumps(
                    self.capture_token_inventory.get("close_token_ids", [])
                )
                env["DYCON_CAPTURE_OPEN_IDS"] = json.dumps(self.capture_token_inventory["open_token_ids"])
            self.device_reservation = reserve_serving_devices(
                root, serving_python, s["gpus"]
            )
            self.device_inventory = self.device_reservation.__enter__()
            if self.device_inventory["device_kind"] == "mig":
                # The pinned NVML platform fingerprints the parent GPU memory.
                env["VLLM_ENABLE_STARTUP_PLAN"] = "0"
            args = self.build_args(serving_python, endpoint.hostname, port)
            write_json(
                self.output / "launch.json",
                {
                    "args": args,
                    "config_hash": cfg["config_hash"],
                    "config_source": cfg["config_source"],
                    "config_source_sha256": cfg.get("config_source_sha256"),
                    "serving_dtype": s.get("dtype", "float16"),
                    "serving_devices": self.device_inventory,
                    "runtime_sha256": sha256(root / "vendor/easysteer-runtime.json"),
                    "capture": {
                        "enabled": bool(capture_cfg.get("enabled")),
                        "control_url": self.capture_server.base_url if self.capture_server else None,
                        "graph_mode": capture_cfg.get("graph_mode", "FULL"),
                        "cache_policy": capture_cfg.get("prefix_cache_policy", "safe_boundary"),
                        "token_inventory": self.capture_token_inventory,
                    },
                },
            )
            self.log = (self.output / "server.log").open("ab")
            attempt_id = os.environ.get("DYCON_EXECUTION_ATTEMPT")
            execution_run = os.environ.get("DYCON_EXECUTION_RUN")
            if attempt_id and execution_run:
                from uuid import uuid4
                resource_id = "serving-" + uuid4().hex
                attempt = Path(execution_run) / "execution/attempts" / attempt_id
                write_json(attempt / "spawn-intents" / (resource_id + ".json"),
                           {"resource_id": resource_id, "role": "serving", "created_at": now()})
                args = [serving_python, "-m", "application.execution.process_gate",
                        execution_run, attempt_id, resource_id, *args]
                env["DYCON_CANCEL_OWNER_RUN"] = Path(execution_run).name
            self.process = subprocess.Popen(
                args, cwd=root, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
            )
            from core.processes import identity
            write_json(self.output / "process.json", {**identity(self.process.pid), "managed": True,
                       "run_id": self.output.parent.name, "execution_attempt_id": attempt_id})
        elif cfg["steering"]["enabled"]:
            raise ValueError("Steering requires managed pinned EasySteer provenance")
        session = requests.Session()
        session.trust_env = False
        headers = {"Authorization": "Bearer " + os.environ.get(s["api_key_env"], "EMPTY")}
        deadline = time.monotonic() + s["startup_seconds"]
        try:
            while True:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("EasySteer server exited; inspect server.log")
                try:
                    r = session.get(s["base_url"].rstrip("/") + "/models", headers=headers, timeout=5)
                    r.raise_for_status()
                    models = r.json()
                    if cfg["model"]["served_name"] not in {x["id"] for x in models["data"]}:
                        raise ValueError("Requested served model is absent")
                    write_json(
                        self.output / "service.json",
                        {
                            "checked_at": now(),
                            "models": models,
                            "management": s["management"],
                            "checkpoint_attestation": "managed_local"
                            if self.process
                            else "user_configured_external",
                        },
                    )
                    return self
                except requests.RequestException:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Model service did not become ready") from None
                    time.sleep(0.2)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        finally:
            session.close()

    def __exit__(self, *exc):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=self.cfg["serving"]["shutdown_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self.capture_server is not None:
            self.capture_server.close()
            self.capture_server = None
            self.capture_controller = None
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.device_reservation is not None:
            self.device_reservation.__exit__(*exc)
            self.device_reservation = None
            self.device_inventory = None
