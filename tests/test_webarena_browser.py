import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.containers.webarena_adapter import (
    SOURCE_REVISION,
    WebArenaEnvironment,
    parse_command,
    render_urls,
    validate_answer,
    validate_pool_manifest,
)


@pytest.mark.parametrize("command", [
    "ls /", "browser click [1]; browser click [2]", "browser click [1] && echo hi",
    "browser click [1]\nbrowser click [2]", "browser new_tab click [2]",
    "browser goto [file:///verifier/answers.json]", "browser stop [plain answer]",
    "browser stop [{\"task_type\":\"MUTATE\",\"status\":\"SUCCESS\"}]",
])
def test_unsafe_or_multi_action_never_reaches_worker(command):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_command(command)


def test_stop_preserves_actual_structured_answer():
    answer = {"task_type": "RETRIEVE", "status": "NOT_FOUND_ERROR", "retrieved_data": [],
              "error_details": "No matching item was visible"}
    kind, result = parse_command("browser stop [" + json.dumps(answer) + "]")
    assert kind == "stop" and result == answer
    with pytest.raises(ValueError):
        validate_answer({**answer, "retrieved_data": [float("nan")]})

@pytest.mark.parametrize("command, expected", [
    ("browser type [8410] mouth guard", "type [8410] [mouth guard]"),
    ("browser type [8410] mouth guard 1", "type [8410] [mouth guard] [1]"),
    ('browser type [8410] "mouth guard" 1', "type [8410] [mouth guard] [1]"),
    ("browser type [8410] mouth guard [0]", "type [8410] [mouth guard] [0]"),
    ("browser tab_focus 0", "tab_focus [0]"),
    ("browser goto http://shopping/search?q=night+guard", "goto [http://shopping/search?q=night+guard]"),
    ("browser press Tab", "press [Tab]"),
    ("browser press Control+a", "press [Control+a]"),
    ("browser scroll down", "scroll [down]"),
    ("browser scroll up", "scroll [up]"),
])
def test_model_concise_browser_forms_are_canonicalized(command, expected):
    assert parse_command(command) == ("action", expected)
    assert parse_command("browser " + expected) == ("action", expected)


@pytest.mark.parametrize("action", [
    "type [8410] [mouth guard] [1]", "type [8410] [mouth guard] [0]",
    "type [8410] [mouth guard]", "type [8410] [version 1]",
    "type [8410] [a [nested] label] [0]", "type [8410] [a; b && c] [0]",
    "click [12]", "hover [12]", "press [Control+a]", "goto [http://shopping/]",
    "tab_focus [0]", "scroll [down]", "new_tab", "close_tab", "go_back", "go_forward",
])
def test_original_browser_grammar_is_preserved(action):
    assert parse_command("browser " + action) == ("action", action)


@pytest.mark.parametrize("command", [
    "browser type [1] text; browser click [2]",
    "browser type [1] text && browser click [2]",
    "browser type [1] text | cat", "browser type [1] [text] browser click [2]",
    "browser type [1] [text] [1]; browser click [2]",
    "browser type [1] [unfinished",
    "browser scroll down]", "browser scroll [down", "browser tab_focus 0 extra",
    "browser goto file:///verifier/answers.json", "browser goto http://shopping/ ; browser new_tab",
    "browser press Tab && browser new_tab",
])
def test_concise_forms_do_not_admit_combined_or_malformed_commands(command):
    with pytest.raises(ValueError):
        parse_command(command)


@pytest.mark.parametrize("wrapper", ["{}", "[{}]"])
def test_stop_accepts_public_json_with_or_without_wrapper(wrapper):
    answer = {"task_type": "NAVIGATE", "status": "SUCCESS",
              "retrieved_data": None, "error_details": None}
    assert parse_command("browser stop " + wrapper.format(json.dumps(answer))) == ("stop", answer)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_command("browser stop " + wrapper.format(json.dumps(answer)) + "; browser new_tab")


def test_selected_urls_cannot_access_unleased_site():
    sites = {"reddit": {"base_url": "http://reddit"}}
    assert render_urls({"start_urls": ["__REDDIT__/f/news"]}, sites) == ["http://reddit/f/news"]
    with pytest.raises(ValueError):
        render_urls({"start_urls": ["__GITLAB__/projects"]}, sites)
    with pytest.raises(ValueError):
        render_urls({"start_urls": ["http://reddit.evil/x"]}, sites)


def test_runtime_pool_rejects_mutable_or_external_image_identity():
    ref = {"store": "wav", "manifest": "images/reddit.json", "manifest_sha256": "a" * 64,
           "image_id": "sha256:" + "b" * 64}
    value = {"schema_version": 1, "benchmark": "webarena_verified", "revision": SOURCE_REVISION,
             "sites": {"reddit": {"image_ref": ref, "port": 80, "base_url": "http://reddit"}}}
    assert validate_pool_manifest(value) == value
    value["sites"]["reddit"]["base_url"] = "http://127.0.0.1"
    with pytest.raises(ValueError):
        validate_pool_manifest(value)
    value["sites"]["reddit"]["base_url"] = "http://reddit"
    value["sites"]["reddit"]["image_ref"]["image_id"] = "latest"
    with pytest.raises(ValueError):
        validate_pool_manifest(value)


def test_worker_rejects_stale_request_without_executing():
    from runtime.containers.webarena_worker import BrowserSession
    worker = BrowserSession.__new__(BrowserSession)
    worker.env = SimpleNamespace(step=lambda _: pytest.fail("stale request executed"))
    worker.finished = False
    worker.step_index = 1
    worker.observation_sha256 = "current"
    with pytest.raises(ValueError, match="stale"):
        worker.action({"step": 0, "observation_sha256": "old", "action": "click [1]"})


def test_finish_requires_real_har_flush_and_is_idempotent(tmp_path):
    env = WebArenaEnvironment.__new__(WebArenaEnvironment)
    env._finished = None
    env.browser = object()
    env.subject_dir = tmp_path
    env.actions_path = tmp_path / "actions.jsonl"
    env.network = SimpleNamespace(id="network-1")
    env._containers = []
    env.name = "lease-1"
    env.step_index = 2
    env.sites = {"reddit": {"base_url": "http://reddit"}}
    env.factory = SimpleNamespace(_images={})
    calls = []
    def rpc(*args, **kwargs):
        calls.append(args)
        return {"har_flushed": True}
    env._rpc = rpc
    with pytest.raises(ValueError, match="flushed"):
        env.finish()
    (tmp_path / "network.har").write_text('{"log":{"entries":[]}}')
    result = env.finish()
    assert result["runtime_identity"]["environments"] == {"reddit": {"urls": ["http://reddit"]}}
    assert result["har_flushed"] is True
    assert env.finish() is result and len(calls) == 2


def test_close_flushes_before_deleting_mutable_state(tmp_path):
    env = WebArenaEnvironment.__new__(WebArenaEnvironment)
    events = []
    env._closed = False
    env._acquired = False
    env.browser = object()
    env.subject_dir = tmp_path
    env._containers = [SimpleNamespace(id="container-1", remove=lambda **_: events.append("container"))]
    env._volumes = [SimpleNamespace(name="volume-1", remove=lambda: events.append("volume"))]
    env.network = SimpleNamespace(remove=lambda: events.append("network"))
    env.factory = SimpleNamespace(_active={env})
    env.finish = lambda: events.append("flush")
    env.close()
    env.close()
    assert events == ["flush", "container", "volume", "network"]
    assert json.loads((Path(tmp_path) / "cleanup.json").read_text())["cleanup_complete"]


@pytest.mark.parametrize("site,port", [("reddit", 80), ("gitlab", 8023), ("shopping", 8080)])
def test_site_health_probe_uses_declared_port_and_host(site, port):
    observed = []
    container = SimpleNamespace(start=lambda: None, reload=lambda: None,
        attrs={"NetworkSettings": {"Networks": {"private": {"IPAddress": "172.18.0.2"}}}})
    env = WebArenaEnvironment.__new__(WebArenaEnvironment)
    env.factory = SimpleNamespace(cfg={"timeouts": {"container_start_seconds": 90}},
        client=SimpleNamespace(containers=SimpleNamespace(create=lambda *a, **kw: container)),
        _images={site: SimpleNamespace(image_id="sha256:site")})
    env._check = lambda: None
    env._containers, env._volumes = [], []
    env.name, env.labels = "test-lease", {"dycon.run": "test"}
    env.network = SimpleNamespace(name="private", disconnect=lambda c: None,
        connect=lambda c, **kw: None)
    host = site.replace("_", "-") + ("" if port == 80 else ":" + str(port))
    env.sites = {site: {"base_url": "http://" + host, "port": port,
        "health_path": "/health", "login": {"headers": {"X-Test-Login": "public"}}}}
    env._wait_http = lambda url, **kwargs: observed.append((url, kwargs))
    env._start_sites()
    assert observed == [(f"http://172.18.0.2:{port}/health", {
        "timeout": 90, "headers": {"Host": host, "X-Test-Login": "public"}})]
    assert env._containers == [container]



def test_shopping_admin_placeholder_and_login_target_the_admin_application():
    from runtime.images.webarena_environments import DEFAULTS
    spec = DEFAULTS["shopping_admin"]
    urls = render_urls({"start_urls": ["__SHOPPING_ADMIN__",
        "__SHOPPING_ADMIN__/catalog/product/edit/id/1481/"]}, {"shopping_admin": spec})
    assert urls == ["http://shopping-admin/admin",
                    "http://shopping-admin/admin/catalog/product/edit/id/1481/"]
    assert spec["base_url"].rstrip("/") + spec["login"]["path"] == "http://shopping-admin/admin/"
    assert spec["health_path"] == "/admin"



@pytest.mark.parametrize("command", ["browser scroll down", "browser scroll [down]",
                                     "  browser scroll [down]  "])
def test_startup_gate_accepts_parser_normalized_scroll(command):
    from runtime.serving.startup import _is_webarena_startup_scroll
    assert _is_webarena_startup_scroll([{"command": command}])


@pytest.mark.parametrize("actions", [[], None, [{"command": None}], ["browser scroll down"],
    [{"command": "browser scroll up"}], [{"command": "pwd"}],
    [{"command": "browser scroll down; browser new_tab"}],
    [{"command": "browser scroll down"}, {"command": "browser scroll down"}],
    [{"command": "browser stop {}"}], [{"command": "browser new_tab"}]])
def test_startup_gate_rejects_anything_except_one_down_scroll(actions):
    from runtime.serving.startup import _is_webarena_startup_scroll
    assert not _is_webarena_startup_scroll(actions)



@pytest.mark.parametrize("site,state", [("gitlab", "attached"), ("reddit", "attached"),
                                        ("shopping_admin", "attached")])
def test_login_verifies_authenticated_page_without_downgrading_ui(site, state):
    from runtime.containers.webarena_worker import BrowserSession
    from runtime.images.webarena_environments import DEFAULTS
    spec = DEFAULTS[site]
    calls = []

    class Locator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def fill(self, value):
            calls.append(("fill", self.selector))

        def click(self):
            calls.append(("click", self.selector))

        def press(self, value):
            calls.append(("press", self.selector))

        def wait_for(self, **kwargs):
            calls.append(("verify", self.selector, kwargs["state"]))

    page = SimpleNamespace(goto=lambda url, **kw: calls.append(("goto", url)),
                           locator=Locator, close=lambda: calls.append(("close",)))
    worker = BrowserSession.__new__(BrowserSession)
    worker._login(SimpleNamespace(new_page=lambda: page), spec)
    assert calls[-2] == ("verify", spec["login"]["verify_selector"], state)
    assert calls[-1] == ("close",)
    if site == "gitlab":
        assert spec["login"]["verify_selector"] == "a[href='/byteblaze']"
        assert spec["login"]["verify_state"] == "attached"
        assert any(call[0] == "click" for call in calls)
        assert sum(call[0] == "fill" for call in calls) == 2


def test_ui_login_defaults_to_visible_without_explicit_override():
    from runtime.containers.webarena_worker import BrowserSession
    calls = []
    locator = SimpleNamespace(wait_for=lambda **kw: calls.append(kw["state"]))
    locator.first = locator
    page = SimpleNamespace(goto=lambda *a, **kw: None,
                           locator=lambda _: locator, close=lambda: calls.append("close"))
    worker = BrowserSession.__new__(BrowserSession)
    worker._login(SimpleNamespace(new_page=lambda: page), {
        "base_url": "http://ui-site", "login": {"method": "ui",
        "verify_selector": "#authenticated", "steps": []}})
    assert calls == ["visible", "close"]


@pytest.mark.parametrize("state", ["hidden", "detached", "invalid"])
def test_login_rejects_non_authenticated_verification_states(state):
    from runtime.containers.webarena_worker import BrowserSession
    calls = []
    page = SimpleNamespace(goto=lambda *a, **kw: None,
                           close=lambda: calls.append("close"))
    worker = BrowserSession.__new__(BrowserSession)
    with pytest.raises(ValueError, match="verification state"):
        worker._login(SimpleNamespace(new_page=lambda: page), {
            "base_url": "http://ui-site", "login": {"method": "ui",
            "verify_selector": "#authenticated", "verify_state": state, "steps": []}})
    assert calls == []


def test_login_receipt_records_hidden_account_without_sensitive_page_data(tmp_path):
    from runtime.containers.webarena_worker import BrowserSession
    locator = SimpleNamespace(count=lambda: 1, is_visible=lambda: False)
    locator.first = locator
    page = SimpleNamespace(url="http://gitlab:8023/?private_token=SECRET",
                           locator=lambda _: locator)
    worker = BrowserSession(tmp_path)
    worker._record_login_verification(page, {"base_url": "http://gitlab:8023",
        "login": {"method": "ui", "verify_selector": "a[href='/byteblaze']",
                  "password": "SECRET"}}, "attached", True)
    raw = (tmp_path / "login-verification.jsonl").read_text()
    row = json.loads(raw)
    assert row["verified"] and row["match_count"] == 1 and row["first_visible"] is False
    assert row["page_path"] == "/" and row["page_origin"] == ["http", "gitlab", 8023]
    assert "SECRET" not in raw and "private_token" not in raw and "password" not in raw
