from core.io import artifact_roots, cache_roots, hash_json, run_path
from core.config import snapshot, load_snapshot

def test_configured_roots_and_run_snapshot(tmp_path):
    project = tmp_path / "project"; project.mkdir()
    physical = tmp_path / "data1"
    cfg = {
        "project_root": str(project),
        "paths": {"artifact_roots": {"run": str(physical), "difficulty_cache": str(physical)},
                  "artifact_scan_roots": [str(project), str(physical)]},
        "config_source": str(project / "config.yaml"),
    }
    cfg["config_hash"] = hash_json(cfg)
    run = snapshot(cfg, "flexible", {"tasks": {}})
    assert run == physical / "runs" / "flexible"
    assert run_path({**cfg, "run_storage_root": str(physical)}, "flexible") == run
    assert (project / "runs" / "flexible").is_symlink()
    loaded = load_snapshot(project, "flexible")
    assert loaded["run_storage_root"] == str(physical.resolve())

def test_cache_roots_include_all_scan_locations(tmp_path):
    cfg = {"project_root": str(tmp_path / "project"), "paths": {"artifact_scan_roots": [str(tmp_path / "data1"), str(tmp_path / "data2")]}}
    assert cache_roots(cfg) == [tmp_path / "data1" / "difficulty" / "cache", tmp_path / "data2" / "difficulty" / "cache"]
    assert artifact_roots(cfg, "run")[0] == (tmp_path / "data1").resolve()
