"""Portable ACP SDK session wrapper for the pinned OpenHands CLI.

Uses the same ACP SDK version as BenchFlow. The environment supplies only the
stdio subprocess transport; the OpenHands CLI owns the single agent/tool loop.
"""
from __future__ import annotations

import asyncio
from pathlib import Path


class ACPEvents:
    def __init__(self):
        self.events = []

    def on_connect(self, connection):
        self.connection = connection

    async def session_update(self, session_id, update, **kwargs):
        self.events.append({"session_id": session_id,
                            "update": update.model_dump(mode="json", by_alias=True, exclude_none=True)})

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        from acp.schema import AllowedOutcome, RequestPermissionResponse
        option = next((o for o in options if o.kind in {"allow_once", "allow_always"}), None)
        if option is None:
            raise ValueError("No supported ACP permission option")
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=option.option_id))


async def run_session(command, args, env, cwd, instruction, *, seconds, probe=False, record=None):
    import json
    client = ACPEvents()
    result = record if record is not None else {}
    result["events"] = client.events
    process_cwd = cwd if cwd is None or Path(cwd).is_dir() else None
    process = await asyncio.create_subprocess_exec(command, *args, env=env, cwd=process_cwd,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, limit=1024 * 1024)
    stderr_chunks = []
    async def collect_stderr():
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            stderr_chunks.append(line.decode(errors="replace"))
    stderr_task = asyncio.create_task(collect_stderr())
    request_id = 1000

    async def request(method, params, timeout=60):
        nonlocal request_id
        request_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
            if not line:
                with __import__("contextlib").suppress(Exception):
                    await process.wait()
                await asyncio.sleep(0.1)
                detail = "".join(stderr_chunks)[-4000:]
                raise RuntimeError("ACP process exited before " + method + f" (rc={process.returncode}): " + (detail or "<no stderr>"))
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                result.setdefault("protocol_noise", []).append(text[:1000])
                continue
            if not isinstance(value, dict):
                result.setdefault("protocol_noise", []).append(repr(value)[:1000])
                continue
            if value.get("method") == "session/update":
                client.events.append({"session_id": value.get("params", {}).get("sessionId"),
                                      "update": value.get("params", {}).get("update")})
                continue
            if value.get("id") == request_id:
                if "error" in value:
                    raise RuntimeError("ACP " + method + " failed: " + str(value["error"]))
                return value.get("result", {})

    try:
        initialize = await request("initialize", {"protocolVersion": 1,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": {"name": "dycon-harbor", "version": "1"}})
        session = await request("session/new", {"cwd": cwd, "mcpServers": []})
        result.update({"initialize": initialize, "session": session, "events": client.events,
                       "model_requests": 0 if probe else None})
        if not probe:
            result["response"] = await request("session/prompt", {"sessionId": session["sessionId"],
                "prompt": [{"type": "text", "text": instruction}]}, timeout=seconds)
        try:
            await request("session/cancel", {"sessionId": session.get("sessionId")}, timeout=3)
        except Exception:
            pass
        return result
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
        if not stderr_task.done():
            stderr_task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await stderr_task
