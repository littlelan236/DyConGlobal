"""Sequential functional acceptance through core.cli; one editable config.

Each child is awaited. This is not a throughput launcher and never launches
production work. Interrupted validation restores the production config bytes.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from core.io import now, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--modes", nargs="+", default=["plain", "capture", "steering", "combined"])
    args = parser.parse_args()
    root = Path.cwd()
    config_path = root / "config.yaml"
    original = config_path.read_bytes()
    raw = yaml.safe_load(original)
    out = root / "runs" / args.run_prefix
    out.mkdir(parents=True, exist_ok=True)
    payload = out / "steering.json"
    create = """
import base64,json,sys,numpy as np
from vllm.model_hooks.steering.payloads import DirectionVector
vector=np.random.default_rng(42).normal(0,0.1,4096).astype(np.float32)
np.save(sys.argv[1]+'/steering-vector.npy',vector)
wire=DirectionVector({15:vector}).to_wire()
for tensor in wire['tensors'].values(): tensor['data']=base64.b64encode(tensor['data']).decode()
json.dump({'model_fingerprint':sys.argv[2],'layer':15,'scale':1.0,'vector':'steering-vector.npy',
 'steering':{'vectors':[{'data':wire,'layers':[15],'scale':1.0,'algorithm':'direct','apply':{'generation':'all'}}]}},
 open(sys.argv[1]+'/steering.json','w'))
"""
    serving_python = root / ".venvs/easysteer/bin/python"
    subprocess.run([str(serving_python), "-c", create, str(out), raw["model"]["fingerprint"]], check=True)
    report = {"passed": False, "started_at": now(), "modes": {}}
    try:
        for mode in args.modes:
            run_id = args.run_prefix + "-" + mode
            raw["capture"]["enabled"] = mode in {"capture", "combined"}
            raw["steering"].update(
                enabled=mode in {"steering", "combined"},
                payload=str(payload) if mode in {"steering", "combined"} else None,
            )
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            print("START", mode, now(), flush=True)
            with (out / (mode + ".log")).open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "core.cli",
                        "--config",
                        str(config_path),
                        "rollout",
                        run_id,
                        "--check-only",
                        "--task-id",
                        args.task_id,
                    ],
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            report["modes"][mode] = {"exit_code": result.returncode, "run_id": run_id}
            if result.returncode:
                raise RuntimeError(f"{mode} failed; see {out / (mode + '.log')}")
            report["modes"][mode]["receipt"] = read_json(root / "runs" / run_id / "startup/receipt.json")
            print("PASS", mode, now(), flush=True)
        report["passed"] = True
    finally:
        config_path.write_bytes(original)
        report["finished_at"] = now()
        write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
