"""Build/audit historical traces by checkpoint, retaining weak evidence explicitly."""

import argparse
from collections import Counter
from pathlib import Path
import traceback

from baseline.catalog import Catalog
from core.config import resolve
from core.io import artifact_dir, read_json, sha256, write_json
from core.locking import file_lock
from enrichment.token_trace import audit_token_traces, trace_rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model-inventory", type=Path, required=True)
    p.add_argument("--dataset-id", required=True)
    args = p.parse_args()
    cfg = resolve(args.config)
    root = Path(cfg["project_root"])
    selection = Catalog(root).select(partition="train", purposes=["training"])
    models = read_json(args.model_inventory)["models"]
    by_fingerprint = {value["fingerprint"]: {**value, "name": name} for name, value in models.items()}
    out = artifact_dir(cfg, "enrichment", args.dataset_id)
    out.mkdir(parents=True, exist_ok=True)
    report = {"rollouts": len(selection["rollout_ids"]), "completed": 0, "failed": {},
              "catalog_sha256": selection["catalog_sha256"], "models": {}, "passed": False}
    with file_lock(root / "state" / (args.dataset_id + ".lock"), nonblocking=True):
        from transformers import AutoTokenizer

        tokenizer = None
        token_text_cache = {}
        previous = None
        ids = sorted(selection["rollout_ids"], key=lambda rid: (selection["rollouts"][rid]["model_fingerprint"], rid))
        for rid in ids:
            row = selection["rollouts"][rid]
            try:
                model = by_fingerprint[row["model_fingerprint"]]
                if previous != row["model_fingerprint"]:
                    model_path = str(Path(model["path"]).resolve())
                    cfg = resolve(args.config, ["model.path=" + model_path,
                        "model.fingerprint=" + row["model_fingerprint"], "model.served_name=" + model["name"]])
                    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                    token_text_cache.clear()
                    previous = row["model_fingerprint"]
                    report["models"][previous] = {"name": model["name"], "path": model_path}
                receipt = out / "token-traces" / rid / "token_trace.receipt.json"
                valid = False
                if receipt.exists():
                    prior = read_json(receipt)
                    valid = prior["trajectory_sha256"] == row["trajectory_sha256"] and all(
                        sha256(receipt.parent / name) == digest for name, digest in prior["files"].items())
                if not valid:
                    trace_rollout(cfg, rid, tokenizer, dataset_id=args.dataset_id, selection=selection,
                                  token_text_cache=token_text_cache)
                report["completed"] += 1
            except Exception:
                report["failed"][rid] = traceback.format_exc()
            if (report["completed"] + len(report["failed"])) % 25 == 0:
                write_json(out / "progress.json", report)
                print({key: report[key] for key in ["rollouts", "completed"]}, flush=True)
        report["audit"] = audit_token_traces(cfg, args.dataset_id)
        report["model_counts"] = dict(Counter(row["model_fingerprint"] for row in selection["rollouts"].values()))
        report["passed"] = not report["failed"] and report["completed"] == report["rollouts"]
        write_json(out / "audit.json", report)
        print(report["audit"], flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
