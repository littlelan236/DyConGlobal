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
        self.path.parent.mkdir(parents=True, mode=0o711, exist_ok=True)
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



def process_identity(pid):
    """Linux PID plus start ticks, excluding exited/zombie processes."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def signal_owned(identity, sig):
    """Pin the kernel process before checking identity; never kill by UID/group."""
    import signal
    if not identity:
        return False
    try:
        fd = os.pidfd_open(identity["pid"])
    except ProcessLookupError:
        return False
    try:
        if process_identity(identity["pid"]) != identity:
            return False
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(fd)


def save_processes(path, body):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body))
    temporary.chmod(0o600)
    temporary.replace(path)


def supervise_acp(argv, relay_pid, receipt):
    """Own only ACP and relay; task services outlive these runtime processes."""
    import signal
    import subprocess
    record = {"schema_version": 1, "supervisor": process_identity(os.getpid()),
              "relay": process_identity(relay_pid), "acp": None}
    save_processes(receipt, record)
    process = None

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            process = subprocess.Popen(argv, restore_signals=True,
                preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, mask))
            record["acp"] = process_identity(process.pid)
            save_processes(receipt, record)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        return process.wait()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            signal_owned(record["acp"], signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                signal_owned(record["acp"], signal.SIGKILL)
                process.wait(timeout=3)
        import time
        signal_owned(record["relay"], signal.SIGTERM)
        deadline = time.monotonic() + 3
        with contextlib.suppress(ChildProcessError):
            while not os.waitpid(relay_pid, os.WNOHANG)[0]:
                if time.monotonic() >= deadline:
                    signal_owned(record["relay"], signal.SIGKILL)
                    os.waitpid(relay_pid, 0)
                    break
                time.sleep(.05)
        record["children_stopped"] = all(process_identity(record[k]["pid"]) != record[k]
                                         for k in ("acp", "relay") if record[k])
        save_processes(receipt, record)
        signal.signal(signal.SIGTERM, previous)


def stop_owned(receipt):
    """External docker-exec cleanup, also covering detached stdio transports."""
    import signal
    import time
    record = json.loads(Path(receipt).read_text())
    signal_owned(record["supervisor"], signal.SIGTERM)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if all(not record.get(k) or process_identity(record[k]["pid"]) != record[k]
               for k in ("supervisor", "acp", "relay")):
            break
        time.sleep(.05)
    updated = json.loads(Path(receipt).read_text())
    if updated.get("supervisor") != record.get("supervisor"):
        raise ValueError("ACP process receipt changed ownership")
    record = updated
    for key in ("acp", "relay", "supervisor"):
        signal_owned(record.get(key), signal.SIGKILL)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if all(not record.get(k) or process_identity(record[k]["pid"]) != record[k]
               for k in ("supervisor", "acp", "relay")):
            return {"verified": True, "processes": record, "task_services_preserved": True}
        time.sleep(.05)
    raise RuntimeError("Owned ACP/relay processes are still alive")


def launch_acp():
    """Runs inside the sandbox; API tokens arrive only through process env."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    child = os.fork()
    if child == 0:
        if os.environ.get("DYCON_PROCESS_RECEIPT"):
            for fd in (0, 1, 2):
                with contextlib.suppress(OSError):
                    os.close(fd)
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
    identity = process_identity(child)
    try:
        if os.environ.get("DYCON_PROCESS_RECEIPT"):
            save_processes(os.environ["DYCON_PROCESS_RECEIPT"], {"schema_version": 1,
                "supervisor": process_identity(os.getpid()), "relay": identity, "acp": None})
        return _configure_and_launch(port, child)
    finally:
        if os.environ.get("DYCON_PROCESS_RECEIPT"):
            import signal
            signal_owned(identity, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child, 0)


def _configure_and_launch(port, child):
    base_url = f"http://127.0.0.1:{port}/v1"
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["BENCHFLOW_PROVIDER_BASE_URL"] = base_url
    if os.environ.get("DYCON_PROBE_CHANNEL") == "1":
        import urllib.error
        import urllib.request
        checks = {}
        for label, token, expected in (("authenticated", os.environ["LLM_API_KEY"], 200),
                                       ("unauthenticated", "invalid", 401)):
            request = urllib.request.Request(base_url + "/dycon-health",
                headers={"Authorization": "Bearer " + token})
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    checks[label] = response.status == expected
            except urllib.error.HTTPError as exc:
                checks[label] = exc.code == expected
        checks["verified"] = all(checks.values())
        Path.home().joinpath("dycon-model-channel-probe.json").write_text(json.dumps(checks))
        if not checks["verified"]:
            raise RuntimeError("Terminal model channel authentication probe failed")
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
    argv = ["/opt/dycon-openhands/bin/openhands", "acp", "--always-approve", "--override-with-envs"]
    if os.environ.get("DYCON_PROCESS_RECEIPT"):
        raise SystemExit(supervise_acp(argv, child, os.environ["DYCON_PROCESS_RECEIPT"]))
    os.execv(argv[0], argv)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--stop":
        print(json.dumps(stop_owned(sys.argv[2])))
    else:
        launch_acp()
