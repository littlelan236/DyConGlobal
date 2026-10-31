"""Attempt-owned WebArena site sets and a persistent, isolated browser worker.

Only the browser worker imports the pinned upstream browser environment. The
runner uses the usual mini agent/tool protocol and never installs its old SDKs.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from core.io import hash_json, now, owned_path, read_json, sha256, write_json

SITES = ("shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map")
SOURCE_REVISION = "6473f72db5dcefc97b5725b59e734504edc28a21"
PROTOCOL = "bash_browser_id_v1"
WORKER_PORT = 8765


def validate_reference(value):
    if not isinstance(value, dict) or not all(k in value for k in ("store", "manifest", "manifest_sha256", "image_id")):
        raise ValueError("A content-addressed NAS image reference is required")
    if not re.fullmatch(r"[a-f0-9]{64}", value["manifest_sha256"]):
        raise ValueError("Invalid NAS reference checksum")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", value["image_id"]):
        raise ValueError("Invalid NAS image digest")
    for key in ("store", "manifest"):
        path = PurePosixPath(value[key])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("NAS reference must remain inside its publication")
    return copy.deepcopy(value)


def validate_pool_manifest(value):
    """Validate a frozen runtime declaration without importing Docker or SDKs."""
    if (value.get("schema_version") != 1 or value.get("benchmark") != "webarena_verified"
            or value.get("revision") != SOURCE_REVISION or not isinstance(value.get("sites"), dict)):
        raise ValueError("Invalid pinned WebArena environment pool")
    if not value["sites"] or set(value["sites"]) - set(SITES):
        raise ValueError("Unsupported/empty WebArena site set")
    for site, spec in value["sites"].items():
        validate_reference(spec["image_ref"])
        if type(spec.get("port")) is not int or not 0 < spec["port"] < 65536:
            raise ValueError("Invalid site service port")
        url = urlsplit(spec["base_url"])
        if url.scheme != "http" or url.hostname != site.replace("_", "-") or url.port != (
                None if spec["port"] == 80 else spec["port"]):
            raise ValueError("Site base URL must use its isolated network alias and service port")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Site base URL must not contain credentials/query/fragment")
        if not spec.get("health_path", "/").startswith("/"):
            raise ValueError("Health probe must be an absolute path")
        for volume in spec.get("volumes", []):
            mount = PurePosixPath(volume["mount_path"])
            if not mount.is_absolute() or str(mount) == "/" or ".." in mount.parts:
                raise ValueError("Invalid baseline volume destination")
            if volume.get("baseline"):
                baseline = volume["baseline"]
                if not re.fullmatch(r"[a-f0-9]{64}", baseline.get("manifest_sha256", "")):
                    raise ValueError("Volume baseline must have a frozen manifest checksum")
    return copy.deepcopy(value)


def parse_command(command):
    """Freeze the single-command grammar before the permissive upstream parser."""
    if not isinstance(command, str) or "\n" in command or "\r" in command or "\x00" in command:
        raise ValueError("Issue exactly one browser command on one line")
    if not command.startswith("browser "):
        raise ValueError("Only browser actions are available; use browser ACTION")
    action = command[len("browser "):].strip()
    if action.startswith("stop "):
        payload = action[len("stop "):].strip()
        if not payload.startswith("[") or not payload.endswith("]"):
            raise ValueError("Use browser stop [JSON_OBJECT]")
        answer = json.loads(payload[1:-1])
        validate_answer(answer)
        return "stop", answer
    patterns = (
        r"(?:click|hover)\s*\[\d+\]",
        r"type\s*\[\d+\]\s*\[.+\](?:\s*\[[01]\])?",
        r"press\s*\[[^\[\]]+\]",
        r"scroll\s*\[?(?:up|down)\]?",
        r"goto\s*\[https?://[^\[\]\s]+\]",
        r"tab_focus\s*\[\d+\]",
        r"(?:new_tab|go_back|go_forward|close_tab)",
    )
    if not any(re.fullmatch(pattern, action) for pattern in patterns):
        raise ValueError("Unsupported browser action or more than one action")
    # Upstream type's greedy text parser could swallow a second command. Text is
    # data, but commands outside its brackets must never be silently ignored.
    if action.startswith("type") and re.search(r"\]\s*(?:browser\s+|&&|;)", action):
        raise ValueError("Multiple actions are not supported")
    return "action", action


def validate_answer(answer):
    """The public FinalAgentResponse shape, independent of expected task values."""
    if not isinstance(answer, dict) or set(answer) != {"task_type", "status", "retrieved_data", "error_details"}:
        raise ValueError("Final answer requires task_type, status, retrieved_data and error_details")
    if answer["task_type"] not in {"RETRIEVE", "NAVIGATE", "MUTATE"}:
        raise ValueError("Invalid public task_type")
    if answer["status"] not in {"SUCCESS", "ACTION_NOT_ALLOWED_ERROR", "PERMISSION_DENIED_ERROR",
                                 "NOT_FOUND_ERROR", "DATA_VALIDATION_ERROR", "UNKNOWN_ERROR"}:
        raise ValueError("Invalid public response status")
    data = answer["retrieved_data"]
    if data is not None and (not isinstance(data, list) or any(isinstance(v, list) for v in data)):
        raise ValueError("retrieved_data must be null or an array of public result items")
    if answer["error_details"] is not None and not isinstance(answer["error_details"], str):
        raise ValueError("error_details must be null or a string")
    json.dumps(answer, allow_nan=False)
    return answer


def render_urls(instance, sites):
    urls = []
    for template in instance["start_urls"]:
        url = template
        for site, spec in sites.items():
            url = url.replace("__" + site.upper() + "__", spec["base_url"].rstrip("/"))
        if "__" in url or urlsplit(url).scheme not in {"http", "https"}:
            raise ValueError("Unrendered/unsupported task URL")
        if not any(_same_origin(url, spec["base_url"]) for spec in sites.values()):
            raise ValueError("Task URL does not belong to its leased sites")
        urls.append(url)
    return urls


def _same_origin(left, right):
    a, b = urlsplit(left), urlsplit(right)
    return (a.scheme, a.hostname, a.port or 80) == (b.scheme, b.hostname, b.port or 80)


class WebArenaFactory:
    def __init__(self, cfg, run_id, deadline=None, client=None):
        import docker

        from core.deadline import Deadline
        self.cfg, self.run_id = cfg, run_id
        self.deadline = deadline or Deadline(cfg["execution"]["pipeline_wall_seconds"])
        self.client = client or docker.DockerClient(base_url=cfg["environment"]["docker_host"],
            timeout=cfg["timeouts"]["container_start_seconds"])
        runtime = cfg.get("execution_context", {}).get("webarena_runtime")
        if not runtime:
            raise ValueError("WebArena runtime references must be frozen by run create")
        self.pool = validate_pool_manifest(runtime["environment_pool"])
        self.browser_reference = validate_reference(runtime["browser_image"])
        self._images, self._active = {}, set()
        self._stack, self._lock = contextlib.ExitStack(), threading.RLock()
        self._slots = threading.BoundedSemaphore(cfg["webarena"]["environment_slots"])
        self.started = False

    def lock_images(self, tasks):
        from runtime.images.nas import ImageLeaseManager, NasImageStore
        with self._lock:
            if self.started:
                return
            required = {site for task in tasks for site in task.get("instance", task)["sites"]}
            missing = required - set(self.pool["sites"])
            if missing:
                raise ValueError("Pool is missing required sites: " + ", ".join(sorted(missing)))
            refs = {site: self.pool["sites"][site]["image_ref"] for site in sorted(required)}
            refs["browser"] = self.browser_reference
            manager = ImageLeaseManager(self.cfg, self.client)
            try:
                for role, reference in refs.items():
                    image = NasImageStore(self.cfg, reference)
                    image.verify(deep=True)
                    self._stack.enter_context(manager.lease(image))
                    self._images[role] = image
            except BaseException:
                self._stack.close()
                raise
            self.started = True

    def acquire(self):
        from core.cancellation import CURRENT
        while not self._slots.acquire(timeout=0.2):
            self.deadline.check()
            if CURRENT.get():
                CURRENT.get().check()

    def cleanup_owned(self, run_id):
        if run_id != self.run_id:
            raise ValueError("Cannot clean another run's WebArena resources")
        errors = []
        for env in list(self._active):
            try:
                env.close()
            except Exception as exc:
                errors.append(str(exc))
        # Also recover resources whose owning runner died before registering env.
        filters = {"label": ["dycon.project=" + str(Path(self.cfg["project_root"]).resolve()),
                             "dycon.run=" + run_id, "dycon.backend=webarena_verified"]}
        for container in self.client.containers.list(all=True, filters=filters):
            try:
                container.remove(force=True, v=True)
            except Exception as exc:
                errors.append(str(exc))
        for kind in (self.client.volumes, self.client.networks):
            for resource in kind.list(filters=filters):
                try:
                    resource.remove()
                except Exception as exc:
                    errors.append(str(exc))
        self._stack.close()
        self.started = False
        if errors:
            raise RuntimeError("WebArena cleanup incomplete: " + "; ".join(errors))


class WebArenaEnvironment:
    def __init__(self, factory, instance, context):
        from core.cancellation import CURRENT
        self.factory, self.context = factory, context
        self.instance = instance.get("instance", instance)
        self.config = factory.cfg["webarena"]
        self.subject_dir = context.output_dir / "web-subject"
        self.subject_dir.mkdir(parents=True, exist_ok=True)
        self.actions_path = self.subject_dir / "actions.jsonl"
        self.actions_path.touch(exist_ok=True)
        self._containers, self._volumes = [], []
        self.network = self.browser = None
        self._finished = None
        self._closed, self._acquired = False, False
        self._token = uuid.uuid4().hex
        self.step_index = 0
        self.answer = None
        self._cancellation = CURRENT.get()
        self.labels = {"dycon.project": str(Path(factory.cfg["project_root"]).resolve()),
            "dycon.run": context.run_id, "dycon.task": context.instance_id,
            "dycon.attempt": context.attempt_id, "dycon.backend": "webarena_verified",
            "dycon.execution": os.environ.get("DYCON_EXECUTION_ATTEMPT", "direct")}
        self.name = "dycon-wav-" + hash_json(self.labels)[:12] + "-" + uuid.uuid4().hex[:8]
        try:
            factory.acquire()
            self._acquired = True
            factory._active.add(self)
            self.sites = {site: factory.pool["sites"][site] for site in sorted(self.instance["sites"])}
            self.network = factory.client.networks.create(self.name, internal=True, labels=self.labels)
            self._start_sites()
            self._start_browser()
            task = {"start_urls": render_urls(self.instance, self.sites),
                    "sites": {site: {"base_url": spec["base_url"], "login": spec.get("login", {})}
                              for site, spec in self.sites.items()},
                    "max_browser_actions": self.config["max_browser_actions"]}
            response = self._rpc("initialize", task)
            self.initial_observation = response["observation"]
            self.observation_sha256 = hash_json(self.initial_observation)
            write_json(self.subject_dir / "initial-observation.json", response)
        except BaseException:
            self.close(flush=False)
            raise

    def _check(self):
        self.factory.deadline.check()
        if self._cancellation:
            self._cancellation.check()

    def _start_sites(self):
        cfg, client = self.factory.cfg, self.factory.client
        for site, spec in self.sites.items():
            self._check()
            mounts = {}
            for index, volume in enumerate(spec.get("volumes", [])):
                if volume.get("read_only", False):
                    baseline = self._baseline(volume["baseline"])
                    mounts[str(baseline)] = {"bind": volume["mount_path"], "mode": "ro"}
                else:
                    name = self.name + "-" + site.replace("_", "-") + "-" + str(index)
                    owned = client.volumes.create(name=name, labels=self.labels)
                    self._volumes.append(owned)
                    mounts[owned.name] = {"bind": volume["mount_path"], "mode": "rw"}
                    if volume.get("baseline"):
                        baseline = self._baseline(volume["baseline"])
                        seed = client.containers.create(self.factory._images["browser"].image_id,
                            command=["cp", "-a", "/baseline/.", "/volume/"], entrypoint=[],
                            network_disabled=True, labels={**self.labels, "dycon.role": "baseline-copy"},
                            volumes={str(baseline): {"bind": "/baseline", "mode": "ro"},
                                     owned.name: {"bind": "/volume", "mode": "rw"}})
                        self._containers.append(seed)
                        seed.start()
                        result = seed.wait(timeout=cfg["timeouts"]["container_start_seconds"])
                        if result["StatusCode"] != 0:
                            raise RuntimeError("Could not clone immutable " + site + " baseline")
                        seed.remove(v=True)
                        self._containers.remove(seed)
            if site in {"map", "wikipedia"} and not mounts:
                raise ValueError(site + " requires its immutable baseline data declaration")
            # Keep the image ENTRYPOINT/CMD; each instance gets fresh writable
            # image layers and task-owned volumes. No SWE conda/bootstrap logic.
            container = client.containers.create(self.factory._images[site].image_id,
                name=self.name + "-" + site.replace("_", "-"),
                environment={"WA_ENV_CTRL_EXTERNAL_SITE_URL": spec["base_url"].rstrip("/")},
                labels={**self.labels, "dycon.role": "site", "dycon.site": site},
                network=self.network.name,
                ports={str(spec["port"]) + "/tcp": ("127.0.0.1", None)},
                volumes=mounts, shm_size="2g" if site == "gitlab" else "256m")
            self._containers.append(container)
            # Reconnect once with the stable alias used by browser URLs.
            self.network.disconnect(container)
            self.network.connect(container, aliases=[site.replace("_", "-")])
            container.start()
            container.reload()
            binding = container.attrs["NetworkSettings"]["Ports"][str(spec["port"]) + "/tcp"][0]
            self._wait_http("http://127.0.0.1:" + binding["HostPort"] + spec.get("health_path", "/"),
                            timeout=cfg["timeouts"]["container_start_seconds"])

    def _baseline(self, reference):
        nas = Path(self.factory.cfg["paths"]["nas_root"]).resolve()
        manifest = owned_path(nas, reference["manifest"])
        if sha256(manifest) != reference["manifest_sha256"]:
            raise ValueError("Changed immutable site baseline manifest")
        value = read_json(manifest)
        root = owned_path(manifest.parent, value.get("root", "."))
        for row in value["files"]:
            path = owned_path(root, row["path"])
            if path.is_symlink() or not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError("Changed immutable site baseline")
        return root

    def _start_browser(self):
        cfg = self.factory.cfg
        self.browser = self.factory.client.containers.create(self.factory._images["browser"].image_id,
            name=self.name + "-browser", labels={**self.labels, "dycon.role": "browser"},
            environment={"DYCON_BROWSER_RPC_TOKEN": self._token},
            network=self.network.name,
            ports={str(WORKER_PORT) + "/tcp": ("127.0.0.1", None)},
            volumes={str(self.subject_dir.resolve()): {"bind": "/output", "mode": "rw"}},
            shm_size="2g", cap_drop=["ALL"], security_opt=["no-new-privileges"])
        self._containers.append(self.browser)
        self.browser.start()
        self.browser.reload()
        binding = self.browser.attrs["NetworkSettings"]["Ports"][str(WORKER_PORT) + "/tcp"][0]
        self.endpoint = "http://127.0.0.1:" + binding["HostPort"]
        self._wait_http(self.endpoint + "/health", cfg["timeouts"]["container_start_seconds"])

    def _wait_http(self, url, timeout):
        import requests
        end = time.monotonic() + timeout
        with requests.Session() as session:
            session.trust_env = False
            while True:
                self._check()
                try:
                    response = session.get(url, timeout=min(5, max(0.1, end-time.monotonic())))
                    if response.status_code < 400:
                        return
                except requests.RequestException:
                    pass
                if time.monotonic() >= end:
                    raise TimeoutError("WebArena service did not become healthy: " + url)
                if self._cancellation:
                    self._cancellation.event.wait(0.3)
                else:
                    time.sleep(0.3)

    def _rpc(self, operation, payload, *, finishing=False):
        import requests
        if not finishing:
            self._check()
        timeout = self.factory.cfg["timeouts"]["tool_command_seconds"]
        with requests.Session() as session:
            session.trust_env = False
            # A POST may mutate a website. Never transparently retry a timeout.
            response = session.post(self.endpoint + "/" + operation, json=payload,
                headers={"Authorization": "Bearer " + self._token}, timeout=timeout)
            response.raise_for_status()
            value = response.json()
        if value.get("error"):
            raise RuntimeError("Browser worker: " + value["error"])
        return value

    def execute(self, action, cwd="", *, timeout=None):
        from minisweagent.exceptions import LimitsExceeded, Submitted
        try:
            kind, payload = parse_command(action.get("command"))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            result = {"returncode": 2, "output": str(exc)}
            self._receipt(action, result, action_executed=False, protocol_error=True)
            return result
        self._check()
        if self._finished is not None:
            raise RuntimeError("Browser worker was already finalized")
        if kind == "stop":
            self.answer = payload
            write_json(self.subject_dir / "agent_response.json", payload)
            self._receipt(action, {"returncode": 0, "output": "Browser task submitted"}, terminal=True)
            self.finish()
            submission = json.dumps(payload, ensure_ascii=False)
            raise Submitted({"role": "exit", "content": submission,
                             "extra": {"exit_status": "Submitted", "submission": submission}})
        if self.step_index >= self.config["max_browser_actions"]:
            raise LimitsExceeded({"role": "exit", "content": "Browser action budget exhausted",
                "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        result = self._rpc("action", {"action": payload, "step": self.step_index,
                                      "observation_sha256": self.observation_sha256})
        output = {"returncode": 0 if not result.get("fail_error") else 1,
                  "output": result["observation"]}
        self._receipt(action, output, worker=result)
        self.step_index += 1
        self.observation_sha256 = hash_json(output["output"])
        return output

    def _receipt(self, action, result, **extra):
        row = {"schema_version": 1, "at": now(), "step": self.step_index,
            "run_id": self.context.run_id, "task_id": self.context.instance_id,
            "attempt_id": self.context.attempt_id, "action": action,
            "action_sha256": hash_json(action), "input_observation_sha256": self.observation_sha256,
            "output_observation_sha256": hash_json(result["output"]),
            "returncode": result["returncode"], **extra}
        with self.actions_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self):
        if self._finished is not None:
            return self._finished
        if self.browser is None:
            raise RuntimeError("Browser was not started")
        result = self._rpc("finish", {}, finishing=True)
        har = self.subject_dir / "network.har"
        if not result.get("har_flushed") or not har.is_file() or not isinstance(read_json(har).get("log", {}).get("entries"), list):
            raise ValueError("Browser HAR was not completely flushed")
        identity = {"schema_version": 1, "protocol": PROTOCOL, "source_revision": SOURCE_REVISION,
            "network": self.network.id, "lease_id": self.name,
            "environments": {site: {"urls": [spec["base_url"].rstrip("/")]}
                             for site, spec in self.sites.items()},
            "images": {role: image.identity() for role, image in self.factory._images.items()},
            "containers": [c.id for c in self._containers], "reset_policy": "per_attempt",
            "browser_actions": self.step_index}
        self._finished = {"har_path": str(har), "actions_path": str(self.actions_path),
            "answer_path": str(self.subject_dir / "agent_response.json"),
            "runtime_identity": identity, "har_flushed": True, "complete": True}
        write_json(self.subject_dir / "browser-finalization.json", self._finished)
        return self._finished

    def get_template_vars(self, **kwargs):
        return {"cwd": "browser", "timeout": self.factory.cfg["timeouts"]["tool_command_seconds"],
                "max_browser_actions": self.config["max_browser_actions"],
                "initial_observation": self.initial_observation, **kwargs}

    def serialize(self):
        return {"info": {"config": {"environment_type": "runtime.containers.WebArenaEnvironment",
            "environment": copy.deepcopy(self.config)}}}

    def close(self, *, flush=True):
        if self._closed:
            return
        errors = []
        if flush and self.browser is not None:
            try:
                self.finish()
            except Exception as exc:
                errors.append("HAR finalization: " + str(exc))
        for container in reversed(self._containers):
            try:
                container.remove(force=True, v=True)
            except Exception as exc:
                errors.append("container " + container.id + ": " + str(exc))
        for volume in reversed(self._volumes):
            try:
                volume.remove()
            except Exception as exc:
                errors.append("volume " + volume.name + ": " + str(exc))
        if self.network is not None:
            try:
                self.network.remove()
            except Exception as exc:
                errors.append("network: " + str(exc))
        self._closed = not errors
        self.factory._active.discard(self)
        if self._acquired:
            self.factory._slots.release()
            self._acquired = False
        write_json(self.subject_dir / "cleanup.json", {"schema_version": 1,
                   "cleanup_complete": not errors, "errors": errors, "at": now()})
        if errors:
            raise RuntimeError("WebArena cleanup incomplete: " + "; ".join(errors))


def preflight(root, run_id, *, factory=None, tasks=None):
    """Exercise the same owned site/browser/reset path, without any LLM request."""
    from application.execution.run_plan import load_plan
    from runtime.containers.factory import TaskContext
    loaded = load_plan(root, run_id)
    cfg = loaded["config"]
    if cfg.get("benchmark", {}).get("backend") != "webarena_verified":
        raise ValueError("WebArena preflight requires a frozen Verified run")
    from core.io import run_path
    directory = run_path(cfg, run_id) / "validation/environments" / ("webarena-" + uuid.uuid4().hex[:12])
    own_factory = factory is None
    receipts, blockers = [], []
    try:
        factory = factory or WebArenaFactory(cfg, run_id)
        tasks = tasks or cfg["selection"]["tasks"]
        if not isinstance(tasks, dict):
            tasks = {task["instance_id"]: task for task in tasks}
        factory.lock_images(tasks.values())
        # One selected task per distinct site combination verifies the actual
        # combinations including multi-site requests, without sampling a model.
        seen = set()
        for task in tasks.values():
            combination = tuple(sorted(task["instance"]["sites"]))
            if combination in seen:
                continue
            seen.add(combination)
            previous = None
            for cycle in range(2):
                context = TaskContext(run_id, task["instance_id"], "preflight-" + uuid.uuid4().hex[:12],
                    "environment-preflight", directory / task["instance_id"] / str(cycle))
                env = None
                try:
                    env = WebArenaEnvironment(factory, task, context)
                    if not env.initial_observation.strip():
                        raise ValueError("Initial AX observation is empty")
                    action = env.execute({"command": "browser scroll [down]"})
                    if action["returncode"] != 0:
                        raise ValueError("Real browser scroll failed: " + action["output"])
                    result = env.finish()
                    current = result["runtime_identity"]
                    if previous is not None and (previous["network"] == current["network"]
                            or set(previous["containers"]) & set(current["containers"])):
                        raise ValueError("Reset reused an earlier attempt's mutable resources")
                    previous = current
                    receipts.append({"task_id": task["instance_id"], "sites": list(combination),
                        "reset_cycle": cycle, "read_action": "scroll [down]", "read_action_passed": True,
                        "initial_observation_sha256": hash_json(env.initial_observation),
                        "runtime_identity": current, "har_flushed": True,
                        "reset_scope": "Fresh containers/volumes and browser; destructive site mutation not exercised"})
                finally:
                    if env is not None:
                        env.close()
    except Exception as exc:
        blockers.append({"error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if factory is not None and own_factory:
            try:
                factory.cleanup_owned(run_id)
            except Exception as exc:
                blockers.append({"error_type": "CleanupError", "error": str(exc)})
    result = {"schema_version": 1, "run_id": run_id, "passed": not blockers,
        "status": "passed" if not blockers else "blocked", "blockers": blockers,
        "receipts": receipts, "model_requests": 0, "at": now()}
    write_json(directory / "receipt.json", result)
    return result
