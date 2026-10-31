"""Real Chromium/official AX/ID parser/HAR contract test against a local fixture.

The fixture is deliberately not a WebArena site and yields no benchmark score.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import tempfile
import threading

from worker import BrowserSession


def main():
    received = []

    class PageHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'''<html><body><h1>Browser contract fixture</h1><form action="/save" method="post">
                <label for="name">Item</label><input id="name" name="name"><button>Save</button>
                </form></body></html>'''
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            received.append(self.rfile.read(int(self.headers["Content-Length"])).decode())
            body = b"<html><body><h1>Saved fixture item</h1></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), PageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="webarena-browser-contract-") as output:
            browser = BrowserSession(output)
            base = "http://127.0.0.1:" + str(server.server_port)
            first = browser.initialize({"start_urls": [base], "sites": {"fixture": {"base_url": base}},
                                        "max_browser_actions": 3})
            assert "Browser contract fixture" in first["observation"], first
            textbox = re.search(r"\[(\d+)\] textbox 'Item'", first["observation"])
            button = re.search(r"\[(\d+)\] button 'Save'", first["observation"])
            assert textbox and button, first["observation"]
            typed = browser.action({"action": "type [" + textbox.group(1) + "] [Smoke] [0]",
                "step": 0, "observation_sha256": first["observation_sha256"]})
            assert not typed["fail_error"], typed
            button = re.search(r"\[(\d+)\] button 'Save'", typed["observation"])
            clicked = browser.action({"action": "click [" + button.group(1) + "]", "step": 1,
                                       "observation_sha256": typed["observation_sha256"]})
            assert not clicked["fail_error"], clicked
            assert "Saved fixture item" in clicked["observation"], clicked
            flushed = browser.finish()
            again = browser.finish()
            assert flushed == again
            har = json.loads((Path(output) / "network.har").read_text())
            posts = [entry for entry in har["log"]["entries"] if entry["request"]["method"] == "POST"]
            assert len(posts) == 1 and received == ["name=Smoke"], posts
            assert "name=Smoke" in posts[0]["request"]["postData"]["text"], posts
            assert "Saved fixture item" in posts[0]["response"]["content"]["text"], posts
            print(json.dumps({"passed": True, "browser_source": "official pinned ScriptBrowserEnv",
                "initial_ax": True, "id_type": True, "id_click": True, "real_post_count": 1,
                "har_request_body": True, "har_embedded_response_body": True,
                "finalize_idempotent": True, "har_sha256": flushed["har_sha256"],
                "chromium_version": flushed["chromium_version"],
                "playwright_version": flushed["playwright_version"],
                "scope": "local browser fixture; WebArena site validation is separate"}))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
