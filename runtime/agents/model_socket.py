"""A single model destination over a Unix socket, including offline sandboxes.

The host endpoint forwards only to an attempt's authenticated OpenHandsBridge.
There is no destination selection or HTTP CONNECT proxy. The agent-side helper
offers that socket on its loopback and launches the unchanged pinned ACP CLI.
"""
from __future__ import annotations

import contextlib
import json
import os
import select
import socket
import socketserver
import threading
from pathlib import Path


def relay(left, right):
    with left, right:
        while True:
            ready, _, _ = select.select([left, right], [], [], 1)
            for source in ready:
                data = source.recv(65536)
                if not data:
                    return
                (right if source is left else left).sendall(data)


class UnixModelChannel:
    def __init__(self, path, host, port):
        self.path, self.destination = Path(path), (host, port)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, mode=0o711)
        destination = self.destination

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                with contextlib.suppress(ConnectionError, OSError):
                    relay(self.request, socket.create_connection(destination, timeout=10))

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.server = Server(str(self.path), Handler)
        self.path.chmod(0o666)  # Per-attempt mount; authentication stays at the model bridge.
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.path.unlink(missing_ok=True)


def launch_acp():
    """Runs inside the sandbox; API tokens arrive only through process env."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    child = os.fork()
    if child == 0:
        try:
            while True:
                incoming, _ = listener.accept()
                upstream = socket.socket(socket.AF_UNIX)
                try:
                    upstream.connect("/run/dycon-model/model.sock")
                except OSError:
                    incoming.close()
                    upstream.close()
                    continue
                threading.Thread(target=relay, args=(incoming, upstream), daemon=True).start()
        finally:
            os._exit(0)
    listener.close()
    base_url = f"http://127.0.0.1:{port}/v1"
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["BENCHFLOW_PROVIDER_BASE_URL"] = base_url
    if os.environ.get("DYCON_PROBE_NETWORK") == "1":
        import urllib.error
        import urllib.request
        checks = {}
        for label, authorization, expected in (("authenticated_channel", "Bearer " + os.environ["LLM_API_KEY"], 200),
                                                ("unauthenticated_channel", "Bearer invalid", 401)):
            try:
                with urllib.request.urlopen(urllib.request.Request(base_url + "/dycon-health",
                    headers={"Authorization": authorization}), timeout=5) as response:
                    checks[label] = response.status == expected
            except urllib.error.HTTPError as exc:
                checks[label] = exc.code == expected
        for label, host, port in (("public_internet", "1.1.1.1", 443),
                                  ("host_ssh", os.environ["DYCON_PROBE_HOST"], 22),
                                  ("host_proxy", os.environ["DYCON_PROBE_HOST"], 17890),
                                  ("external_dns_tcp", "8.8.8.8", 53)):
            try:
                with socket.create_connection((host, port), timeout=1):
                    checks[label + "_denied"] = False
            except OSError:
                checks[label + "_denied"] = True
        try:
            socket.getaddrinfo("example.com", 443)
            checks["external_dns_resolution_denied"] = False
        except OSError:
            checks["external_dns_resolution_denied"] = True
        Path.home().joinpath("dycon-network-probe.json").write_text(json.dumps(checks))
        if not all(checks.values()) or len(checks) != 7:
            raise RuntimeError("Offline model-channel/network probe failed")
    settings = Path.home() / ".openhands/agent_settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"llm": {"model": os.environ["LLM_MODEL"],
        "api_key": os.environ["LLM_API_KEY"], "base_url": base_url}}))
    settings.chmod(0o600)
    os.execv("/opt/dycon-openhands/bin/openhands", ["openhands", "acp", "--always-approve", "--override-with-envs"])


if __name__ == "__main__":
    launch_acp()
