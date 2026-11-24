from __future__ import annotations

import json
import shlex
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from core.io import hash_json, sha256

CONSTRAINT_FILE = "/tmp/dycon-constraints.txt"
WHEELS_PATH = "/opt/dycon-wheels"


class EnvironmentPolicy:
    def __init__(self, cfg: dict, *, wheel_hashes=None):
        self.cfg = cfg
        self.environment = cfg["environment"]
        wheelhouse = self.environment["wheelhouse"]
        self.wheel_hashes = dict(wheel_hashes) if wheel_hashes is not None else (
            {p.name: sha256(p) for p in sorted(Path(wheelhouse).glob("*.whl"))} if wheelhouse else {}
        )
        if (self.environment["constraints"] or any(self.environment["repo_pins"].values())) and not self.wheel_hashes:
            raise ValueError(f"Offline wheelhouse is empty for configured requirements: {wheelhouse}")
        if wheelhouse and not self.wheel_hashes:
            raise ValueError(f"Configured wheelhouse contains no wheels: {wheelhouse}")

    def specs(self, repo: str) -> list[str]:
        normalized = {}
        pins = {}
        for text in self.environment["constraints"] + self.environment["repo_pins"].get(repo, []):
            requirement = Requirement(text)
            name = canonicalize_name(requirement.name)
            requirement.name = name
            key = str(requirement)
            exact = [s.version for s in requirement.specifier if s.operator == "==" and "*" not in s.version]
            if exact and not requirement.marker:
                if name in pins and pins[name] != exact[0]:
                    raise ValueError(f"Conflicting offline requirement pins for {name}: {pins[name]} vs {exact[0]}")
                pins[name] = exact[0]
            normalized.setdefault(key, key)
        return list(normalized.values())

    def variables(self) -> dict[str, str]:
        from runtime.containers.historical_runtime import ENVS, KEY
        from runtime.containers.tools import variables as tool_variables

        env = dict(self.environment["variables"])
        env.update(tool_variables(env))
        if KEY in env:
            # The pre-bootstrap activation uses the image's testbed until the historical one exists.
            env["CONDA_ENVS_DIRS"] = ENVS + ":/opt/miniconda3/envs"
        env.update(
            {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}
        )
        env.update(
            {
                "LANG": self.environment["locale"],
                "LC_ALL": self.environment["locale"],
                "PYTHONIOENCODING": "utf-8",
                "PIP_CONSTRAINT": CONSTRAINT_FILE,
                "PIP_FIND_LINKS": WHEELS_PATH,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            }
        )
        if self.environment.get("wheelhouse") and not self.environment["network_fallback"]:
            # Also governs official eval scripts and their isolated build subprocesses.
            env["PIP_NO_INDEX"] = "1"
        return env

    def bootstrap(self, repo: str) -> str:
        specs = self.specs(repo)
        from runtime.containers.historical_runtime import prepare, verify
        from runtime.containers.tools import prepare as prepare_tools
        from runtime.containers.repairs import legacy_iterable_bootstrap
        from runtime.containers.resources import preparation
        compatibility = legacy_iterable_bootstrap(self.environment["variables"])
        text = shlex.quote("\n".join(specs) + "\n")
        install = "python -m pip install --no-input --no-index --find-links " + WHEELS_PATH
        install += " " + " ".join(map(shlex.quote, specs))
        if self.environment["network_fallback"]:
            install = "(" + install + ") || (echo DYCON_NETWORK_FALLBACK; python -m pip install "
            install += "--no-input " + " ".join(map(shlex.quote, specs)) + ")"
        return (
            "\n".join(
                [
                    "set -euo pipefail",
                    *prepare(self.environment["variables"]),
                    "source /opt/miniconda3/bin/activate",
                    "conda activate testbed",
                    *verify(self.environment["variables"]),
                    f"printf '%s' {text} > {CONSTRAINT_FILE}",
                    *(f"export {k}={shlex.quote(v)}" for k, v in self.variables().items()),
                    *prepare_tools(self.environment["variables"]),
                    "echo DYCON_BOOTSTRAP_BEGIN",
                    install if specs else "true",
                    *([compatibility] if compatibility else []),
                    preparation(self.cfg) or "true",
                    "echo DYCON_BOOTSTRAP_OK",
                ]
            )
            + "\n"
        )

    def identity(self, repo: str) -> dict:
        value = {
            "environment": self.environment,
            "timeouts": self.cfg["timeouts"],
            "repo": repo,
            "specs": self.specs(repo),
            "requirements": self.specs(repo),
            "constraints": self.specs(repo),
            "wheel_hashes": self.wheel_hashes,
            "bootstrap": self.bootstrap(repo),
        }
        return {"policy_hash": hash_json(value), **value}

    def wrap_eval(self, script: str) -> str:
        from runtime.containers.dependencies import INSTALL_ERROR_GUARD

        # Export before the official conda activation and every subsequent pip install.
        exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in self.variables().items())
        head, separator, rest = script.partition("\n")
        prefix = exports + "\n" + INSTALL_ERROR_GUARD + "\n"
        wrapped = head + separator + prefix + rest if head.startswith("#!") else prefix + script
        from benchmark_evaluation.swebench_receipts import instrument

        result = instrument(wrapped, stack_seconds=max(1, self.cfg["timeouts"]["evaluation_seconds"] - 20))[0]
        if self.environment["variables"].get("DYCON_SWEBENCH_IMAGE_DIAGNOSTICS") == "1":
            source = Path("benchmark_evaluation/swebench_image_diagnostics.py").read_text()
            result += "\npython -c " + shlex.quote(source) + "\n"
        return result


PROBE_CODE = """import json,sys,subprocess,importlib,os,re
try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
constraints=open('/tmp/dycon-constraints.txt').read().splitlines()
packages=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json']).decode('utf-8'))
versions={canonicalize_name(p['name']):p['version'] for p in packages}
checks=[]
for text in constraints:
    if not text.strip():continue
    requirement=Requirement(text)
    name=requirement.name
    actual=versions.get(canonicalize_name(name))
    checks.append({'spec':text,'package':name,'actual':actual,'satisfied':actual is not None and actual in requirement.specifier})
data={'python':sys.version,'executable':sys.executable,'conda':os.environ.get('CONDA_DEFAULT_ENV'),
      'constraints':checks,'packages':packages,
      'variables':{k:os.environ.get(k) for k in ['LANG','LC_ALL','PIP_CONSTRAINT','PIP_FIND_LINKS']}}
print(json.dumps(data))
sys.exit(0 if all(c['satisfied'] for c in checks) else 3)
"""


def probe_command() -> str:
    return "source /opt/miniconda3/bin/activate && conda activate testbed && python -c " + shlex.quote(
        PROBE_CODE
    )


def parse_probe(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "constraints" in value:
            return value
    raise ValueError("Environment probe did not return JSON")
