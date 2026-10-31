"""Publish immutable site-set declarations from verified NAS image references."""
from __future__ import annotations

from pathlib import Path

from core.io import hash_json, read_json, sha256, write_json
from runtime.containers.webarena_adapter import SOURCE_REVISION, validate_pool_manifest
from runtime.images.nas import NasImageStore

DEFAULTS = {
    "shopping": {"port": 80, "health_path": "/customer/account/login", "base_url": "http://shopping",
        "login": {"method": "header", "headers": {"X-M2-Customer-Auto-Login": "emma.lopez@gmail.com:Password.123"},
                  "path": "/customer/account/", "verify_selector": "a[href*='customer/account/logout']"}},
    "shopping_admin": {"port": 80, "health_path": "/", "base_url": "http://shopping-admin",
        "login": {"method": "header", "headers": {"X-M2-Admin-Auto-Login": "admin:admin1234"},
                  "path": "/admin", "verify_selector": ".admin-user-account-text"}},
    "reddit": {"port": 80, "health_path": "/login", "base_url": "http://reddit",
        "login": {"method": "header", "headers": {"X-Postmill-Auto-Login": "MarvelsGrantMan136:test1234"},
                  "path": "/", "verify_selector": "a[href*='/user/MarvelsGrantMan136']"}},
    "gitlab": {"port": 8023, "health_path": "/users/sign_in", "base_url": "http://gitlab:8023",
        "login": {"method": "ui", "path": "/users/sign_in",
            "steps": [{"action": "fill", "selector": "[data-testid='username-field']", "value": "byteblaze"},
                      {"action": "fill", "selector": "[data-testid='password-field']", "value": "hello1234"},
                      {"action": "click", "selector": "[data-testid='sign-in-button']"}],
            "verify_selector": ".header-user-dropdown-toggle"}},
}


def prepare_environments(cfg, sites=()):
    """Publish available supported sites without claiming absent baseline data."""
    root = Path(cfg["project_root"])
    options = cfg["webarena_images"]
    store = Path(cfg["paths"]["nas_root"]) / options["store"]
    requested = set(sites) if sites else set(DEFAULTS)
    unknown = requested - set(DEFAULTS) - {"map", "wikipedia"}
    if unknown:
        raise ValueError("These sites require an explicit immutable baseline publication: " + ", ".join(sorted(unknown)))
    included, excluded = {}, []
    for site in sorted(requested):
        if site in {"map", "wikipedia"}:
            excluded.append({"site": site, "reason": "Official immutable dataset baseline has not been published"})
            continue
        path = store / "images" / (site + ".json")
        if not path.exists():
            excluded.append({"site": site, "reason": "NAS image publication not complete"})
            continue
        record = read_json(path)
        reference = {"store": store.name, "manifest": "images/" + site + ".json",
                     "manifest_sha256": sha256(path), "image_id": record["manifest_digest"]}
        NasImageStore(cfg, reference).verify(deep=True)
        included[site] = {**DEFAULTS[site], "image_ref": reference}
    if not included:
        return {"status": "blocked", "passed": False, "excluded": excluded,
                "blockers": ["No verified NAS site image is available"]}
    value = {"schema_version": 1, "benchmark": "webarena_verified", "revision": SOURCE_REVISION,
             "sites": included, "reset_policy": "per_attempt",
             "scope": "Declared site subset; actual reset/login/browser validation requires preflight"}
    validate_pool_manifest(value)
    path = root / "runtime/webarena-environments" / hash_json(value) / "manifest.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Environment pool publication changed")
    write_json(path, value)
    # These are public benchmark credentials, not SSH/API secrets. Keep the
    # publication owner-readable and never include them in agent observations.
    path.parent.chmod(0o700)
    path.chmod(0o600)
    return {"schema_version": 1, "status": "complete", "passed": True,
        "environment_pool_ref": path.relative_to(root).as_posix(),
        "environment_pool_sha256": sha256(path), "sites": sorted(included), "excluded": excluded,
        "unprepared_baselines": ["map", "wikipedia"],
        "scope": value["scope"]}
