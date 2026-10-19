"""Task repairs applied at the shared creation boundary for every caller."""

from __future__ import annotations

from pathlib import Path


def environment_overrides(cfg: dict, instance: dict) -> dict:
    overrides = dict(instance.get("environment_overrides") or {})
    iid = instance["instance_id"]
    # These upstream test suites fetch real HTTP resources. Dependency bootstrap
    # remains offline; the isolated task's runtime needs the Docker bridge.
    if iid in {
        "psf__requests-1921",
        "pydicom__pydicom-1194",
        "pydicom__pydicom-1236",
        "pydicom__pydicom-1241",
        "pydicom__pydicom-1365",
        "pydicom__pydicom-1416",
        "pydicom__pydicom-1539",
        "pydicom__pydicom-1555",
        "pyvista__pyvista-3675",
        "pyvista__pyvista-3747",
        "pyvista__pyvista-4417",
        "pyvista__pyvista-4648",
        "pyvista__pyvista-4808",
        "sphinx-doc__sphinx-10435",
        "sphinx-doc__sphinx-7985",
        "sphinx-doc__sphinx-8475",
    }:
        overrides["network_mode"] = "bridge"
    fixes = {
        "sympy__sympy-23262": ["scipy==1.13.1", "numpy==1.26.4"],
        "pylint-dev__pylint-4661": ["appdirs==1.4.4"],
    }
    if iid not in fixes:
        return overrides
    # These wheels are maintained inside this project and mounted read-only.
    overrides["wheelhouse"] = str(Path(cfg["project_root"]) / "runtime/cache/gold-wheels")
    repo = instance["repo"]
    pins = {**cfg["environment"]["repo_pins"], **overrides.get("repo_pins", {})}
    names = {text.split("==")[0] for text in fixes[iid]}
    retained = [p for p in pins.get(repo, []) if p.split("==")[0] not in names]
    pins[repo] = retained + fixes[iid]
    overrides["repo_pins"] = pins
    return overrides
