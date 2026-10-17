"""Copy the verified serving runtime into an independent extraction environment.

Never installs into or mutates the source environment. cp --reflink=auto uses
copy-on-write where supported and ordinary copies otherwise, never hardlinks.
"""

import argparse
import subprocess
from pathlib import Path

from core.config import resolve
from hidden.backends.vllm.engine import serving_interpreter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args()
    cfg = resolve(a.config, ["hidden.adapter=vllm_eager"])
    root = Path(cfg["project_root"])
    source = serving_interpreter(cfg).parent.parent
    target = Path(cfg["hidden"]["vllm"]["python"]).absolute().parent.parent
    if target.exists() or target == source:
        raise FileExistsError("Extraction runtime must be a new independent directory")
    subprocess.run(
        [str(source / "bin/python"), str(root / "runtime/scripts/verify_easysteer_runtime.py")],
        cwd=root,
        check=True,
    )
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(target)], check=True)
    subprocess.run(
        [
            str(target / "bin/python"),
            "-c",
            "from hidden.backends.vllm.engine import verify_runtime; print(verify_runtime('.'))",
        ],
        cwd=root,
        check=True,
    )
    print(f"Prepared {target}; run it using bin/python (copied console-script shebangs are not used)")


if __name__ == "__main__":
    main()
