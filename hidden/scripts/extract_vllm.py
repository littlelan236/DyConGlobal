"""Isolated eager vLLM entry point; accepts the same dataset/capture contract."""

import argparse
import json

from core.config import resolve
from hidden.backends.vllm.engine import configure_environment, gpu_reservation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--run-id")
    p.add_argument("dataset_id")
    p.add_argument("capture_id")
    a = p.parse_args()
    if a.run_id:
        if a.set:
            raise ValueError("Run replay uses frozen config")
        from application.execution.run_plan import load_plan
        from pathlib import Path
        cfg = load_plan(Path(a.config).resolve().parent, a.run_id)["config"]
    else:
        cfg = resolve(a.config, a.set)
    if cfg["hidden"]["adapter"] != "vllm_eager":
        raise ValueError("Set hidden.adapter=vllm_eager explicitly")
    configure_environment(cfg)
    with gpu_reservation(cfg["project_root"], cfg["hidden"]["vllm"]["gpus"]):
        from hidden.adapters.vllm import load_adapter
        from hidden.extract import extract_hidden

        adapter = load_adapter(cfg)
        try:
            out = extract_hidden(cfg, a.dataset_id, a.capture_id, adapter)
            print(json.dumps({"capture": str(out), "backend": "vllm_eager"}))
        finally:
            adapter.close()


if __name__ == "__main__":
    main()
