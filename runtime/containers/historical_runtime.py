"""Pinned historical interpreter installations for specific SWE-bench tasks."""
import shlex
from pathlib import Path

from core.io import sha256


KEY = "DYCON_SWEBENCH_HISTORICAL_RUNTIME"
ENVS = "/opt/dycon-historical-envs"
INSTALLER = "/opt/dycon-historical-installer.sh"
RUNTIMES = {
    "psf__requests-2674": (
        "Miniconda2-py27_4.8.3-Linux-x86_64.sh",
        "b820dde1a0ba868c4c948fe6ace7300a252b33b5befd078a15d4a017476b8979",
        (2, 7),
    ),
    "pyvista__pyvista-432": (
        "Miniconda3-4.7.12-Linux-x86_64.sh",
        "a23fcffe97690d3bbcd34cda798c3a3318e0f35d863c5d4aca3fc983fe8450b7",
        (3, 7),
    ),
    "sympy__sympy-13001": ("Miniconda2-py27_4.8.3-Linux-x86_64.sh", "b820dde1a0ba868c4c948fe6ace7300a252b33b5befd078a15d4a017476b8979", (2, 7)),
    "sympy__sympy-13146": ("Miniconda2-py27_4.8.3-Linux-x86_64.sh", "b820dde1a0ba868c4c948fe6ace7300a252b33b5befd078a15d4a017476b8979", (2, 7)),
    "sympy__sympy-13865": ("Miniconda2-py27_4.8.3-Linux-x86_64.sh", "b820dde1a0ba868c4c948fe6ace7300a252b33b5befd078a15d4a017476b8979", (2, 7)),
}


def validate(instance, overrides):
    variables = overrides.get("variables", {})
    selector = variables.get(KEY)
    if selector is None:
        return
    if (selector not in RUNTIMES or selector != instance["instance_id"]
            or (instance.get("environment_overrides") or {}).get("variables", {}).get(KEY) != selector):
        raise ValueError("Historical runtime must be explicitly enabled for its matching task")


def mounts(cfg):
    selector = cfg["environment"]["variables"].get(KEY)
    if selector is None:
        return {}
    name, digest, _ = RUNTIMES[selector]
    root = (Path(cfg["project_root"]) / "runtime/resources/swebench-historical-python").resolve()
    source = (root / name).resolve()
    if source.parent != root or sha256(source) != digest:
        raise ValueError("Historical interpreter installer digest mismatch")
    return {str(source): {"bind": INSTALLER, "mode": "ro"}}


def prepare(variables):
    selector = variables.get(KEY)
    if selector is None:
        return []
    _, digest, _ = RUNTIMES[selector]
    return [
        f"printf '%s  %s\\n' {digest} {INSTALLER} | sha256sum -c -",
        f"bash {INSTALLER} -b -p {ENVS}/testbed",
        f"export CONDA_ENVS_DIRS={shlex.quote(ENVS + ':/opt/miniconda3/envs')}",
        "source /opt/miniconda3/bin/activate",
    ]


def verify(variables):
    selector = variables.get(KEY)
    if selector is None:
        return []
    _, _, version = RUNTIMES[selector]
    program = (
        f"import sys; assert sys.version_info[:2] == {version!r}; "
        f"assert sys.prefix == {ENVS + '/testbed'!r}; print(sys.version)"
    )
    return [
        "python -c " + shlex.quote(program),
        "PIP_CONSTRAINT=/dev/null python -m pip install --no-input --no-index --find-links /opt/dycon-wheels pip==20.3.4",
    ]
