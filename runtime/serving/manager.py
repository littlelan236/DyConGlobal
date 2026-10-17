"""Managed pinned EasySteer serving or explicitly configured external service."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from core.io import now, sha256, write_json


class ServingManager:
    def __init__(self, cfg, output):
        self.cfg, self.output = cfg, Path(output)
        self.process = None
        self.log = None

    def __enter__(self):
        cfg, s = self.cfg, self.cfg["serving"]
        self.output.mkdir(parents=True, exist_ok=True)
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
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, s["gpus"]))
            env["VLLM_PLUGINS"] = ""
            env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            env["VLLM_CACHE_ROOT"] = str(root / "runtime/cache/vllm")
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
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                s["tool_parser"],
                "--reasoning-parser",
                s["reasoning_parser"],
                "--generation-config",
                "vllm",
                "--no-async-scheduling",
            ]
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
                    "runtime_sha256": sha256(root / "vendor/easysteer-runtime.json"),
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
        if self.log is not None:
            self.log.close()
