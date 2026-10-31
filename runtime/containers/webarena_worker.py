"""Single-thread HTTP RPC owner of the pinned upstream synchronous browser.

This module runs inside the browser image, not the Python 3.12 runner. It owns
exactly one Playwright session, keeps official AX IDs/actions, and flushes a full
embedded HAR before reporting completion. It never calls a model or evaluator.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def origin(url):
    value = urlsplit(url)
    return value.scheme, value.hostname, value.port or (443 if value.scheme == "https" else 80)


class BrowserSession:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = None
        self.finished = False
        self.step_index = 0
        self.observation_sha256 = None

    def initialize(self, task):
        if self.env is not None:
            raise ValueError("Browser session already initialized")
        # The legacy package requires every URL variable during import. Unused
        # sites remain unreachable; request routing below allows leased origins.
        for site in ("shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map", "homepage"):
            os.environ[site.upper()] = task["sites"].get(site, {}).get("base_url", "http://unused.invalid")
        from browser_env import ScriptBrowserEnv
        from playwright.sync_api import sync_playwright
        owner = self

        class RecordingEnvironment(ScriptBrowserEnv):
            def setup(self, config_file=None):
                self.context_manager = sync_playwright()
                self.playwright = self.context_manager.__enter__()
                self.browser = self.playwright.chromium.launch(executable_path="/usr/bin/chromium",
                    headless=True, args=["--no-sandbox"])
                self.context = self.browser.new_context(viewport=self.viewport_size,
                    record_har_path=str(owner.output / "network.har"),
                    record_har_mode="full", record_har_content="embed", service_workers="block")
                self.context.set_default_timeout(30000)
                allowed = {origin(spec["base_url"]): spec.get("login", {})
                           for spec in task["sites"].values()}

                def route_request(route, request):
                    target = origin(request.url)
                    if target not in allowed:
                        route.abort("blockedbyclient")
                        return
                    login = allowed[target]
                    headers = dict(request.headers)
                    if login.get("method") == "header":
                        headers.update(login["headers"])
                    route.continue_(headers=headers)

                self.context.route("**/*", route_request)
                for spec in task["sites"].values():
                    owner._login(self.context, spec)
                for url in task["start_urls"]:
                    page = self.context.new_page()
                    page.goto(url, wait_until="domcontentloaded")
                    client = self.context.new_cdp_session(page)
                    client.send("Accessibility.enable")
                    page.client = client
                if not self.context.pages:
                    raise ValueError("A browser task requires at least one initial URL")
                self.page = self.context.pages[0]
                self.page.bring_to_front()

        self.max_actions = task["max_browser_actions"]
        self.env = RecordingEnvironment(observation_type="accessibility_tree", headless=True,
                                        current_viewport_only=False)
        observation, info = self.env.reset()
        text = self._observation(observation, info)
        self.observation_sha256 = digest(text)
        return {"observation": text, "observation_sha256": self.observation_sha256,
                "protocol": "bash_browser_id_v1", "browser_actions": 0}

    def _login(self, context, spec):
        login = spec.get("login", {})
        method = login.get("method", "none")
        if method == "none":
            return
        if method not in {"header", "ui"}:
            raise ValueError("Unsupported login method")
        if not login.get("verify_selector"):
            raise ValueError("Authenticated sites require an explicit browser login verification selector")
        page = context.new_page()
        try:
            page.goto(spec["base_url"].rstrip("/") + login.get("path", "/"),
                      wait_until="domcontentloaded")
            if method == "ui":
                for step in login.get("steps", []):
                    locator = page.locator(step["selector"])
                    if step["action"] == "fill":
                        locator.fill(step["value"])
                    elif step["action"] == "click":
                        locator.click()
                    elif step["action"] == "press":
                        locator.press(step["value"])
                    else:
                        raise ValueError("Unsupported login UI step")
            page.locator(login["verify_selector"]).first.wait_for(state="visible")
        finally:
            page.close()

    def _observation(self, observation, info):
        tabs = [{"index": i, "url": page.url, "active": page == self.env.page}
                for i, page in enumerate(self.env.context.pages)]
        return ("URL: " + self.env.page.url + "\nTabs: " + json.dumps(tabs, ensure_ascii=False)
                + "\nAction error: " + str(info.get("fail_error", ""))
                + "\nAccessibility tree:\n" + str(observation["text"]))

    def action(self, value):
        if self.env is None or self.finished:
            raise ValueError("No active browser session")
        if value["step"] != self.step_index or value["observation_sha256"] != self.observation_sha256:
            raise ValueError("Browser action refers to a stale observation/version")
        if self.step_index >= self.max_actions:
            raise ValueError("Browser action budget exhausted")
        from browser_env import create_id_based_action
        # The host applies the stricter one-action grammar; this invocation is
        # deliberately the pinned official ID action parser and executor.
        action = create_id_based_action(value["action"])
        observation, reward, terminated, truncated, info = self.env.step(action)
        text = self._observation(observation, info)
        self.step_index += 1
        self.observation_sha256 = digest(text)
        return {"observation": text, "observation_sha256": self.observation_sha256,
                "fail_error": info.get("fail_error", ""), "browser_actions": self.step_index,
                "action_executed": bool(reward), "terminated": terminated, "truncated": truncated,
                "note": "action_executed is an execution receipt, not benchmark success"}

    def finish(self, value=None):
        if self.env is None:
            raise ValueError("Browser was not initialized")
        if not self.finished:
            # Playwright writes HAR during context.close(). Do not report success
            # until the JSON document is complete and includes response bodies.
            self.env.context.close()
            self.env.browser.close()
            self.env.context_manager.__exit__(None, None, None)
            self.finished = True
        path = self.output / "network.har"
        value = json.loads(path.read_text())
        if not isinstance(value.get("log", {}).get("entries"), list):
            raise ValueError("Playwright did not flush a valid HAR")
        return {"har_flushed": True, "har_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "entry_count": len(value["log"]["entries"]), "browser_actions": self.step_index,
                "chromium_version": self.env.browser.version, "playwright_version": __import__("playwright").__version__}


def main():
    token = os.environ["DYCON_BROWSER_RPC_TOKEN"]
    owner = BrowserSession(os.environ.get("DYCON_BROWSER_OUTPUT", "/output"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Request bodies include task login settings. Log only path/status.
            sys.stderr.write("browser-rpc " + self.path + "\n")

        def _reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(200 if self.path == "/health" else 404, {"ready": self.path == "/health"})

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self._reply(403, {"error": "Unauthorized browser control request"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 4 * 1024 * 1024:
                    raise ValueError("Invalid RPC payload size")
                payload = json.loads(self.rfile.read(size))
                methods = {"/initialize": owner.initialize, "/action": owner.action, "/finish": owner.finish}
                if self.path not in methods:
                    self._reply(404, {"error": "Unknown browser operation"})
                    return
                self._reply(200, methods[self.path](payload))
            except Exception as exc:
                self._reply(200, {"error": type(exc).__name__ + ": " + str(exc)})

    HTTPServer(("0.0.0.0", 8765), Handler).serve_forever()


if __name__ == "__main__":
    main()
