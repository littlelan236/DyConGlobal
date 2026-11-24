from pathlib import Path

import pytest

from runtime.containers.historical_runtime import KEY as RUNTIME_KEY, validate as validate_runtime, verify
from runtime.containers.tools import KEY as TOOL_KEY, validate as validate_tool, variables, prepare
from runtime.wheelhouse import build


def test_explicit_missing_wheel_source_is_rejected_before_publication(tmp_path):
    with pytest.raises(FileNotFoundError, match="Explicit wheel source"):
        build({"project_root": str(tmp_path)}, "candidate", [tmp_path / "not-ready"])
    assert not (tmp_path / "temp/wheelhouse/candidate").exists()
    assert not (tmp_path / "runtime/wheelhouse/candidate").exists()


@pytest.mark.parametrize("iid", ["psf__requests-2674", "pyvista__pyvista-432"])
def test_historical_runtime_requires_matching_explicit_task(iid):
    env = {"variables": {RUNTIME_KEY: iid}}
    instance = {"instance_id": iid, "environment_overrides": env}
    validate_runtime(instance, env)
    with pytest.raises(ValueError):
        validate_runtime({"instance_id": "other", "environment_overrides": env}, env)
    with pytest.raises(ValueError):
        validate_runtime({"instance_id": iid}, env)


def test_renderer_preserves_conda_path_and_requires_explicit_task():
    env = {TOOL_KEY: "9.53.3", "PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin"}
    assert "PATH" not in variables(env)
    assert any("ln -s" in command for command in prepare(env))
    overrides = {"variables": env}
    task = {"instance_id": "matplotlib__matplotlib-20488", "repo": "matplotlib/matplotlib",
            "environment_overrides": overrides}
    validate_tool(task, overrides)
    with pytest.raises(ValueError):
        validate_tool({**task, "instance_id": "other"}, overrides)
    with pytest.raises(ValueError):
        validate_tool({**task, "environment_overrides": {}}, overrides)


def test_pip_upgrade_does_not_read_uncreated_task_constraints():
    commands = verify({RUNTIME_KEY: "pyvista__pyvista-432"})
    assert commands[-1].startswith("PIP_CONSTRAINT=/dev/null ")
    assert "--no-index" in commands[-1]
