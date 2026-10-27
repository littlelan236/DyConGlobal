"""Keep network cleanup from changing Requests proxy-discovery semantics."""
import pytest
import requests
from runtime.containers.repairs import environment_overrides


@pytest.mark.parametrize("iid", ["psf__requests-3362", "psf__requests-3738"])
@pytest.mark.parametrize("name,url,scheme", [
    ("http_proxy", "http://example.com", "http"),
    ("https_proxy", "https://example.com", "https"),
    ("all_proxy", "http://example.com", "all"),
    ("all_proxy", "https://example.com", "all"),
])
def test_task_proxy_override_remains_visible(tmp_path, monkeypatch, iid, name, url, scheme):
    cfg = {"project_root": str(tmp_path), "environment": {"variables": {}, "repo_pins": {}}}
    instance = {"instance_id": iid, "repo": "psf/requests", "environment_overrides": {}}
    policy = environment_overrides(cfg, instance)
    for key, value in policy["variables"].items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(name, "socks5://proxy.com:9876")
    actual = requests.utils.get_environ_proxies(url)
    assert actual[scheme] == "socks5://proxy.com:9876"
    assert policy["variables"]["NO_PROXY"] == policy["variables"]["no_proxy"] == ""


def test_other_network_tasks_keep_existing_cleanup(tmp_path):
    cfg = {"project_root": str(tmp_path), "environment": {"variables": {}, "repo_pins": {}}}
    policy = environment_overrides(cfg, {"instance_id": "psf__requests-1944", "repo": "psf/requests"})
    assert policy["variables"]["NO_PROXY"] == "*"


def test_matplotlib_original_https_test_has_task_network(tmp_path):
    cfg = {"project_root": str(tmp_path), "environment": {"variables": {}, "repo_pins": {}}}
    policy = environment_overrides(cfg, {'instance_id':'matplotlib__matplotlib-23111',
        'repo':'matplotlib/matplotlib'})
    assert policy['network_mode'] == 'bridge'
    assert policy['variables']['HTTPS_PROXY'] == ''
