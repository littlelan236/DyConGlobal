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


class ServingManager:
    def __init__(self, cfg, output):
        self.cfg, self.output = cfg, Path(output)
        self.process = None
        self.log = None
        self.capture_controller = None
        self.capture_server = None
        self.capture_token_inventory = {
            "newline_token_ids": [198, 271, 1358],
            "close_token_ids": [248069],
            "source": "qwen_verified_fallback",
        }

    def __enter__(self):
        cfg, s = self.cfg, self.cfg["serving"]
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
                check.bind((endpoint.hostname, port))
            root = Path(cfg["project_root"])
            verify = root / "runtime/scripts/verify_easysteer_runtime.py"
            subprocess.run([s["python"], str(verify)], check=True, capture_output=True, text=True)
            if capture_cfg.get("enabled"):
                inventory_code = (
                    "import json,sys; from transformers import AutoTokenizer; "
                    "t=AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=True); "
                    "ids=[]; texts={}; "
                    "[ (ids.append(i), texts.__setitem__(str(i), t.decode([i], clean_up_tokenization_spaces=False))) for i in range(len(t)) if \"\\n\" in t.decode([i], clean_up_tokenization_spaces=False) ]; "
                    "print(json.dumps({'newline_token_ids':ids,'newline_token_texts':texts,'close_token_ids':t.encode('</think>', add_special_tokens=False),'source':'active_tokenizer'}))"
                )
                try:
                    raw_inventory = subprocess.check_output(
                        [s["python"], "-c", inventory_code, cfg["model"]["path"]], text=True, cwd=root
                    )
                    self.capture_token_inventory = json.loads(raw_inventory.strip().splitlines()[-1])
                    self.capture_token_inventory["tokenizer_fingerprint"] = hash_json(
                        self.capture_token_inventory
                    )
                except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
                    # Preserve the verified current-Qwen fallback and record it
                    # in launch.json; capture admission can reject unknown IDs.
                    self.capture_token_inventory["source"] = "qwen_verified_fallback"
                    self.capture_token_inventory["tokenizer_fingerprint"] = hash_json(
                        self.capture_token_inventory
                    )
                self.capture_controller = OnlineCaptureController(cfg["project_root"], capture_cfg)
                self.capture_server = CaptureControlServer(
                    self.capture_controller,
                    capture_cfg.get("control_host", "127.0.0.1"),
                    capture_cfg.get("control_port", 0),
                ).start()
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, s["gpus"]))
            env["VLLM_PLUGINS"] = ""
            env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            env["VLLM_CACHE_ROOT"] = str(root / "runtime/cache/vllm")
            if self.capture_server is not None:
                env["DYCON_CAPTURE_CONTROL_URL"] = self.capture_server.base_url
                env["DYCON_CAPTURE_GRAPH_MODE"] = "FULL"
                env["DYCON_CAPTURE_CACHE_POLICY"] = "safe_boundary"
            args = [
                s["python"],
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                cfg["model"]["path"],
                "--served-model-name",
                cfg["model"]["served_name"],
                "--host",
                endpoint.hostname,
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(s["tensor_parallel_size"]),
                "--gpu-memory-utilization",
                str(s["gpu_memory_utilization"]),
                "--max-model-len",
                str(s["max_model_len"]),
                "--dtype",
                s.get("dtype", "float16"),
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                s["tool_parser"],
                "--reasoning-parser",
                s["reasoning_parser"],
                "--generation-config",
                "vllm",
                "--no-async-scheduling",
            ]
            if capture_cfg.get("enabled"):
                # vLLM's normal path is unchanged; capture runs opt into the
                # native FULL graph dispatcher for both prefill and decode.
                args += ["--compilation-config", '{"cudagraph_mode":"FULL"}']
            if s["language_model_only"]:
                args.append("--language-model-only")
            if cfg["model"]["trust_remote_code"]:
                args.append("--trust-remote-code")
            if cfg["steering"]["enabled"]:
                from runtime.steering.easysteer.payload import steering_payload

                payload = steering_payload(cfg)
                algorithms = sorted({v["algorithm"] for v in payload["vectors"]})
                args += [
                    "--enable-steer-vector",
                    "--steer-algorithms",
                    ",".join(algorithms),
                    "--steer-graph-mode",
                    "in_graph",
                    "--steer-vector-dtype",
                    "float32",
                ]
            write_json(
                self.output / "launch.json",
                {
                    "args": args,
                    "config_hash": cfg["config_hash"],
                    "serving_dtype": s.get("dtype", "float16"),
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
            self.process = subprocess.Popen(
                args, cwd=root, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
            )
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
