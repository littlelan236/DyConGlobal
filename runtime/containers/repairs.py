"""Task repairs applied at the shared creation boundary for every caller."""

from __future__ import annotations

from pathlib import Path


def _legacy_environment_overrides(cfg: dict, instance: dict) -> dict:
    overrides = dict(instance.get("environment_overrides") or {})
    iid = instance["instance_id"]
    # These upstream test suites fetch real HTTP resources. Dependency bootstrap
    # remains offline; the isolated task's runtime needs the Docker bridge.
    network_tasks = {
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
        # Non-Verified 600: original suites require external test resources.
        "psf__requests-1635",
        "psf__requests-1689",
        "psf__requests-1713",
        "psf__requests-1733",
        "psf__requests-1768",
        "psf__requests-1776",
        "psf__requests-1944",
        "psf__requests-1962",
        "psf__requests-1963",
        "psf__requests-2148",
        "psf__requests-2153",
        "psf__requests-2193",
        "psf__requests-2413",
        "psf__requests-2617",
        "psf__requests-2674",
        "psf__requests-2821",
        "psf__requests-2873",
        "psf__requests-3362",
        "psf__requests-3738",
        "mwaskom__seaborn-2457",
        # The original mandatory HTTPS imread smoke test fetches a real image.
        "matplotlib__matplotlib-23111",
        "scikit-learn__scikit-learn-13620",
        "sphinx-doc__sphinx-11266",
        "sphinx-doc__sphinx-8509",
        "sphinx-doc__sphinx-9309",
        "sphinx-doc__sphinx-9467",
    }
    if iid in network_tasks:
        overrides["network_mode"] = "bridge"
        # Image-provided host proxies are unreachable or present the wrong TLS
        # identity from rootless task containers. These suites need direct egress.
        overrides["variables"] = {
            **cfg["environment"].get("variables", {}),
            **overrides.get("variables", {}),
            **{
                key: "" for key in [
                    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                ]
            },
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
        if iid in {"psf__requests-3362", "psf__requests-3738"}:
            # These suites test proxy discovery itself. A wildcard bypass hides
            # proxy variables set by the original tests and changes their result.
            overrides["variables"].update(NO_PROXY="", no_proxy="")
    if iid in {
        "scikit-learn__scikit-learn-13933",
        "scikit-learn__scikit-learn-14309",
        "scikit-learn__scikit-learn-14869",
    }:
        # Old OpenMP suites see all host CPUs; bound nested parallelism to
        # avoid oversubscription under the concurrent harness.
        overrides["variables"] = {
            **cfg["environment"].get("variables", {}),
            **overrides.get("variables", {}),
            **{key: "1" for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        }
    fixes = {
        "sympy__sympy-23262": ["scipy==1.13.1", "numpy==1.26.4"],
        "pylint-dev__pylint-4661": ["appdirs==1.4.4"],
        "mwaskom__seaborn-2457": ["matplotlib==3.5.3"],
        **{
            key: [
                "pytest-httpbin==1.0.2", "httpbin==0.7.0", "Flask==1.1.4", "Werkzeug==0.16.1",
                "Jinja2==2.11.3", "MarkupSafe==2.0.1", "itsdangerous==1.1.0", "click==7.1.2",
                "decorator==5.1.1", "cffi==1.15.1", "pycparser==2.21", "blinker==1.6.3",
            ]
            for key in ["psf__requests-2873", "psf__requests-3362", "psf__requests-3738"]
        },
    }
    if iid not in fixes:
        return overrides
    # These wheels are maintained inside this project and mounted read-only.
    wheel_dir = "gold600-wheels" if iid in {
        "mwaskom__seaborn-2457", "psf__requests-2873", "psf__requests-3362", "psf__requests-3738",
    } else "gold-wheels"
    overrides["wheelhouse"] = str(Path(cfg["project_root"]) / "runtime/cache" / wheel_dir)
    repo = instance["repo"]
    pins = {**cfg["environment"]["repo_pins"], **overrides.get("repo_pins", {})}
    names = {text.split("==")[0] for text in fixes[iid]}
    retained = [p for p in pins.get(repo, []) if p.split("==")[0] not in names]
    pins[repo] = retained + fixes[iid]
    overrides["repo_pins"] = pins
    return overrides


def _profile_environment_overrides(cfg: dict, instance: dict) -> dict:
    """Apply an explicitly frozen, content-bound environment control profile.

    Legacy runs have no profile variables and retain their existing policy.
    The profile only changes dependencies/output; task patches and required IDs
    are never read or rewritten here.
    """
    overrides = _legacy_environment_overrides(cfg, instance)
    variables = cfg['environment'].get('variables', {})
    profile_id = variables.get('DYCON_SWEBENCH_REPAIR_POLICY')
    expected = variables.get('DYCON_SWEBENCH_REPAIR_POLICY_SHA256')
    if not profile_id and not expected:
        return overrides
    from core.io import identifier, read_json, sha256
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if not profile_id or not expected:
        raise ValueError('SWE-bench repair profile requires both ID and SHA256')
    profile_path = Path(cfg['project_root']) / 'runtime/containers/repair_profiles' / (identifier(profile_id) + '.json')
    if sha256(profile_path) != expected:
        raise ValueError('SWE-bench repair profile checksum mismatch')
    profile = read_json(profile_path)
    if profile.get('schema_version') != 1 or profile.get('policy_id') != profile_id:
        raise ValueError('Unsupported SWE-bench repair profile identity')
    row = profile['tasks'].get(instance['instance_id'])
    if row is None:
        return overrides
    requirements = row['requirements']
    names = {canonicalize_name(Requirement(spec).name) for spec in requirements}
    repo = instance['repo']
    pins = {**cfg['environment']['repo_pins'], **overrides.get('repo_pins', {})}
    retained = [spec for spec in pins.get(repo, [])
                if canonicalize_name(Requirement(spec).name) not in names]
    # Let EnvironmentPolicy report contradictory global constraints instead of
    # silently replacing a separately declared user pin.
    pins[repo] = retained + requirements
    overrides['repo_pins'] = pins
    if row.get('variables') or row.get('resources'):
        overrides['variables'] = {
            **variables, **overrides.get('variables', {}), **row.get('variables', {})}
        if row.get('resources'):
            import json
            overrides['variables']['DYCON_SWEBENCH_RESOURCES'] = json.dumps(row['resources'], sort_keys=True)
    return overrides


def environment_overrides(cfg: dict, instance: dict) -> dict:
    """Frozen task configuration takes precedence over compatibility defaults."""
    overrides = _profile_environment_overrides(cfg, instance)
    explicit = instance.get("environment_overrides") or {}
    for key, value in explicit.items():
        if key == "variables":
            overrides[key] = {**cfg["environment"].get(key, {}),
                              **overrides.get(key, {}), **value}
        elif key == "repo_pins":
            from packaging.requirements import Requirement
            from packaging.utils import canonicalize_name
            pins = {**cfg["environment"].get(key, {}), **overrides.get(key, {})}
            for repo, specs in value.items():
                names = {canonicalize_name(Requirement(spec).name) for spec in specs}
                pins[repo] = [spec for spec in pins.get(repo, [])
                              if canonicalize_name(Requirement(spec).name) not in names] + specs
            overrides[key] = pins
        else:
            overrides[key] = value
    legacy_iterable = overrides.get("variables", {}).get("DYCON_SWEBENCH_LEGACY_ITERABLE")
    if legacy_iterable is not None and (
        legacy_iterable != "1"
        or instance["instance_id"] != "sympy__sympy-13865"
        or instance["repo"] != "sympy/sympy"
        or explicit.get("variables", {}).get("DYCON_SWEBENCH_LEGACY_ITERABLE") != "1"
    ):
        raise ValueError("Legacy Iterable compatibility requires explicit sympy-13865 configuration")
    from runtime.containers.historical_runtime import validate
    from runtime.containers.tools import validate as validate_tools

    validate(instance, {**overrides, "variables": {**cfg["environment"].get("variables", {}), **overrides.get("variables", {})}})
    validate_tools(instance, {**overrides, "variables": {**cfg["environment"].get("variables", {}), **overrides.get("variables", {})}})
    return overrides


def legacy_iterable_bootstrap(variables: dict) -> str:
    """Restore the Python 3.7 spelling of Iterable for a frozen legacy task."""
    if variables.get("DYCON_SWEBENCH_LEGACY_ITERABLE") != "1":
        return ""
    import shlex

    code = """import sys, sysconfig
from pathlib import Path
if sys.version_info[:2] != (3, 9):
    raise RuntimeError('Expected Python 3.9 for Iterable compatibility')
path = Path(sysconfig.get_path('purelib')) / 'dycon_legacy_iterable.pth'
line = ("import collections, collections.abc, os; "
        "os.environ.get('DYCON_SWEBENCH_LEGACY_ITERABLE') == '1' and "
        "collections.__dict__.setdefault('Iterable', collections.abc.Iterable)\\n")
try:
    with path.open('x', encoding='ascii') as stream:
        stream.write(line)
except FileExistsError:
    if path.is_symlink() or path.read_text(encoding='ascii') != line:
        raise RuntimeError('Conflicting Iterable compatibility startup file')
"""
    verify = "import collections, collections.abc; assert collections.Iterable is collections.abc.Iterable"
    return ("python -c " + shlex.quote(code)
            + " && python -W error::DeprecationWarning -c " + shlex.quote(verify)
            + " && echo DYCON_LEGACY_ITERABLE_READY")
