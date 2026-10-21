"""Run one explicit rollout shard through the isolated vLLM replay backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog.store import Catalog
from core.config import resolve
from core.io import read_json
from core.model_identity import verify_model
from enrichment.trajectory import enrich
from hidden.extract import extract_hidden


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fingerprint", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--anchors", required=True)
    a = p.parse_args()
    root = a.root.resolve()
    ids = set(read_json(a.ids))
    overrides = [
        f"model.path={a.model}",
        f"model.fingerprint={a.fingerprint}",
        f"model.served_name={Path(a.model).name}",
        "hidden.device=cpu",
        f"hidden.layers={list(range(32)) if '9B' in a.model else list(range(64))}",
        "hidden.adapter=vllm_eager",
        f"hidden.vllm.gpus=[{a.gpu}]",
        "hidden.vllm.tensor_parallel_size=1",
        f"enrichment.anchors={a.anchors}",
        "benchmark.partition=train",
        "benchmark.purposes=[training]",
    ]
    cfg = resolve(root / "config.yaml", overrides)
    catalog = Catalog(root)
    original = catalog.select

    def select(self, *, partition=None, purposes=(), profile=None):
        x = original(partition=partition, purposes=purposes, profile=profile)
        x["rollout_ids"] = [r for r in x["rollout_ids"] if r in ids]
        x["rollouts"] = {r: x["rollouts"][r] for r in x["rollout_ids"]}
        return x

    Catalog.select = select
    verify_model(cfg)
    from hidden.backends.vllm.engine import configure_environment, gpu_reservation
    configure_environment(cfg)

    from transformers import AutoTokenizer

    from hidden.adapters.vllm import load_adapter

    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True,
                                        trust_remote_code=cfg["model"]["trust_remote_code"])
    enrich(cfg, a.dataset, tok)
    with gpu_reservation(root, [int(a.gpu)]):
        adapter = load_adapter(cfg)
        try:
            extract_hidden(cfg, a.dataset, a.capture, adapter)
        finally:
            adapter.close()
    print(json.dumps({"dataset": a.dataset, "capture": a.capture, "ids": len(ids), "gpu": a.gpu}, indent=2))


if __name__ == "__main__":
    main()
