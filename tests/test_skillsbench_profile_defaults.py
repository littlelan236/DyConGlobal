from pathlib import Path

from core.io import read_json


def test_profile_defaults_match_public_oracle_proxy_and_allow_override(cfg, monkeypatch):
    from core import cli

    monkeypatch.setattr(cli, "resolve", lambda *args: cfg)
    base = ["skillsbench", "profile", "default-proxy", "--bridge-advertise", "10.0.0.1",
            "--output", "test-task:/out.txt"]
    cli.main(base)
    root = Path(cfg["project_root"])
    path = root / "runtime/agent-profiles/default-proxy/manifest.json"
    assert read_json(path)["use_host_proxy"] is True

    cli.main([*base[:2], "explicit-off", *base[3:], "--no-use-host-proxy"])
    path = root / "runtime/agent-profiles/explicit-off/manifest.json"
    assert read_json(path)["use_host_proxy"] is False

    cli.main([*base[:2], "offline-network", *base[3:], "--network", "none"])
    path = root / "runtime/agent-profiles/offline-network/manifest.json"
    assert read_json(path)["use_host_proxy"] is False
