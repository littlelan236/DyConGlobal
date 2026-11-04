"""SkillsBench benchmark adapter, delegating ACP/container lifecycle to BenchFlow."""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import uuid
from pathlib import Path

import docker
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.usage_tracking import UsageTrackingConfig

from core.cancellation import Cancelled
from core.deadline import Deadline
from core.io import hash_json, now, read_json, run_path, sha256, write_json
from rollout.retry import record_agent_outcome
from runtime.agents.openhands import OPENHANDS_AGENT_ENVIRONMENT, resource_override
from runtime.images.nas import ImageLeaseManager, NasImageStore

SKILLSBENCH_AGENT_ENVIRONMENT = OPENHANDS_AGENT_ENVIRONMENT


async def resolve_submission_outputs(env, task_id, declared, workspace, output, ownership):
    """Resolve the organize task's location-independent five-directory contract.

    Its prompt names directories, and the pinned verifier accepts /root/papers,
    /root, the workspace, then a common parent below /root. Discover existing
    outputs only; never move files or use oracle/verifier contents.
    """
    import shlex

    names = ['LLM', 'trapped_ion_and_qc', 'black_hole', 'DNA', 'music_history']
    if task_id != 'organize-messy-files' or declared != ['/root/' + n for n in names]:
        return declared, None
    script = '''import json
from pathlib import Path
names = %r
workspace = %r
roots = [Path('/root/papers'), Path('/root')]
if workspace:
    roots.append(Path(workspace))
observed = []
seen = set()
def inspect(root):
    text = str(root)
    if text in seen or not root.is_absolute() or root.resolve() != root:
        return False
    seen.add(text)
    present = [n for n in names if (root / n).is_dir() and not (root / n).is_symlink()]
    if present:
        observed.append({'root': text, 'present': present})
    return len(present) == len(names)
selected = next((root for root in roots if inspect(root)), None)
if selected is None:
    for path in Path('/root').rglob(names[0]):
        if inspect(path.parent):
            selected = path.parent
            break
print(json.dumps({'root': str(selected) if selected is not None else None,
                  'observed': observed[:32]}))
''' % (names, workspace)
    receipt = {'schema_version': 1, 'ownership': ownership, 'task_id': task_id,
               'policy': 'organize-messy-files-common-parent-v1', 'declared_paths': declared,
               'at': now()}
    effective = declared
    try:
        probe = await env.exec('python3 -c ' + shlex.quote(script), user='root', timeout_sec=30)
        receipt['return_code'] = probe.return_code
        if probe.return_code == 0:
            result = json.loads(probe.stdout)
            root = result.get('root')
            if root is not None:
                from application.execution.subject import safe_output_path
                root = safe_output_path(root)
                if root not in {'/root', workspace} and not root.startswith('/root/'):
                    raise ValueError('Discovered submission parent is outside the task workspace')
                effective = [root.rstrip('/') + '/' + n for n in names]
            receipt.update(status='resolved' if root is not None else 'no_complete_group',
                           discovered_root=root, observed=result.get('observed', []))
        else:
            receipt.update(status='discovery_failed', stderr=_safe_verifier_text(
                getattr(probe, 'stderr', ''))[-2000:])
    except Exception as exc:
        # Optional discovery must not prevent official verification. The normal
        # exporter still records missing/rejected declared paths on failure.
        receipt.update(status='discovery_failed', error_type=type(exc).__name__)
    receipt['effective_paths'] = effective
    path = output / 'diagnostics/submission-output-resolution.json'
    write_json(path, receipt)
    return effective, sha256(path)


def image_reference(cfg, unit, *, deep=True):
    options = cfg["skillsbench_images"]
    store = Path(cfg["paths"]["nas_root"]) / options["store"]
    path = store / "images" / (unit["key"] + ".json")
    row = read_json(path)
    source_unit = {k: v for k, v in unit.items() if k not in {"nas_reference", "image_identity"}}
    if row.get("provenance", {}).get("unit") != source_unit:
        raise ValueError("NAS image does not match its imported task package")
    ref = {"store": options["store"], "manifest": str(path.relative_to(store)),
           "manifest_sha256": sha256(path), "image_id": row["manifest_digest"]}
    if unit.get("nas_reference") is not None and unit["nas_reference"] != ref:
        raise ValueError("NAS image reference differs from the frozen task")
    image = NasImageStore(cfg, ref)
    if unit.get("image_identity") is not None and unit["image_identity"] != image.identity():
        raise ValueError("NAS image identity differs from the frozen task")
    graph = image.verify(deep=deep)
    if not graph:
        raise ValueError("NAS image graph is empty")
    return ref, image


def attach_image_references(cfg, task):
    updated = json.loads(json.dumps(task))
    for unit in updated["instance"]["images"]:
        ref, image = image_reference(cfg, unit)
        unit["nas_reference"] = ref
        unit["image_identity"] = image.identity()
    return updated


class SkillsBenchFactory:
    """Admit images through the existing rootless NAS lease manager."""
    def __init__(self, cfg, run_id, deadline=None):
        self.cfg, self.run_id, self.deadline = cfg, run_id, deadline or Deadline()
        self.client = docker.DockerClient(base_url=cfg["environment"]["docker_host"])
        self.daemon_info = self.client.info()
        self.cpu_quota_supported = not any("No cpu cfs quota support" in str(w)
                                            for w in self.daemon_info.get("Warnings", []))
        self.leases = ImageLeaseManager(cfg, self.client)
        self.stack, self.images, self.started = contextlib.ExitStack(), {}, False

    def lock_images(self, tasks, *, deep=True):
        if self.started:
            return
        profile = self.cfg.get("execution_context", {}).get("profile", {})
        if profile.get("use_host_proxy") and not self.cfg.get("skillsbench_network_proxy"):
            import socket

            from runtime.images.skillsbench import build_proxy
            if profile.get("network") != "public":
                raise ValueError("Host proxy requires the public-network profile")
            host = profile["bridge_listen"]
            with socket.socket() as listener:
                listener.bind((host, 0))
                port = listener.getsockname()[1]
            proxy = f"http://{host}:{port}"
            proxy_cfg = copy.deepcopy(self.cfg)
            proxy_cfg["skillsbench_images"].update(build_proxy=proxy, manage_build_proxy=True)
            target = run_path(self.cfg, self.run_id) / "agent/network"
            self.stack.enter_context(build_proxy(proxy_cfg, target, managed_role='network-proxy'))
            self.cfg["skillsbench_network_proxy"] = proxy
        for task in tasks:
            package = task.get("instance", task)
            task_id = task["instance_id"]
            for unit in package["images"]:
                ref, image = image_reference(self.cfg, unit, deep=deep)
                self.stack.enter_context(self.leases.lease(image) if deep else self.leases.lease(image, deep=False))
                tag = unit["tag"]
                self.client.api.tag(image.image_id, tag)
                self.images[(task_id, unit["service"])] = {
                    "tag": tag, "reference": ref, "identity": image.identity(), "unit": unit}
        self.started = True

    def cleanup_owned(self, run_id):
        try:
            self.stack.close()
        finally:
            self.started = False
            self.client.close()

    cleanup_cancelled = cleanup_owned


class OwnedPlanes:
    """Inject run/sample/attempt ownership into BenchFlow's Docker compose."""
    def __init__(self, inner, cfg, run_id, sample, attempt_id, image, output):
        self.inner, self.cfg = inner, cfg
        self.run_id, self.sample, self.attempt_id = run_id, sample, attempt_id
        self.image, self.output = image, Path(output)
        self.task_source = None
        self.ownership = {"project": str(Path(cfg["project_root"]).resolve()), "run": run_id,
            "task": sample.instance_id, "sample": sample.sample_key, "attempt": attempt_id,
            "execution": os.environ.get("DYCON_EXECUTION_ATTEMPT", "direct"),
            "role": "skillsbench-environment", "epoch": uuid.uuid4().hex}
        for name in dir(inner):
            if (name not in {"create_environment", "connect_acp", "harden_before_verify", "verifier", "deploy_skills"}
                    and not name.startswith("_") and callable(getattr(inner, name))):
                setattr(self, name, getattr(inner, name))

    async def connect_acp(self, *args, **kwargs):
        # Use BenchFlow's own client and transport. Observe its extensible
        # receive seam before ACPClient discards JSON-RPC error.data.
        result = await self.inner.connect_acp(*args, **kwargs)
        client = result[0]
        transport = client._transport
        receive = transport.receive
        async def observed_receive():
            try:
                value = await receive()
            except Exception as exc:
                diag = getattr(exc, "diagnostic", None)
                diagnostic = diag.to_dict() if hasattr(diag, "to_dict") else None
                write_json(self.output / 'diagnostics/process-transport.json', {
                    'schema_version': 1, 'ownership': self.ownership,
                    'observed_at': now(), 'error_type': type(exc).__name__,
                    'process_exit_code': (diagnostic or {}).get('process_exit_code'),
                    'process_pid': (diagnostic or {}).get('process_pid'),
                    'diagnosis': (diagnostic or {}).get('transport_diagnosis'),
                    'source': 'benchflow_live_transport_before_cleanup',
                    'cause': 'unknown'})
                await asyncio.to_thread(self._snapshot_process_evidence, 'transport_error')
                raise
            if isinstance(value, dict) and isinstance(value.get('error'), dict):
                error = value['error']
                write_json(self.output / 'diagnostics/acp-error.json', {
                    'schema_version': 1, 'ownership': self.ownership,
                    'source': 'actual_benchflow_transport_receive', 'observed_at': now(),
                    'rpc_request_id': value.get('id'), 'code': error.get('code'),
                    'message': _safe_acp_message(error.get('message')),
                    **_acp_details(error.get('data'))})
            return value
        transport.receive = observed_receive
        return result

    async def _collect_sdk_transport(self, env):
        import shlex

        from runtime.agents.openhands_transport import SDK_TRANSPORT_LOG, safe_event
        path = self.output / 'diagnostics/sdk-local-transport.json'
        row = {'schema_version': 1, 'ownership': self.ownership, 'observed_at': now(),
               'source': 'actual_sdk_local_log_before_cleanup', 'available': False, 'events': []}
        try:
            command = 'test -f ' + shlex.quote(SDK_TRANSPORT_LOG) + ' && head -c 8388608 ' + shlex.quote(SDK_TRANSPORT_LOG)
            result = await env.exec(command, user=self.cfg['execution_context']['profile']['sandbox_user'],
                timeout_sec=min(5, self.cfg['execution_context']['profile'].get('cleanup_seconds', 60)))
            row['return_code'] = result.return_code
            for line in (result.stdout or '').splitlines()[:8192]:
                try: row['events'].append(safe_event(json.loads(line)))
                except (ValueError, TypeError, AttributeError): row['invalid_lines'] = row.get('invalid_lines', 0) + 1
            row['available'] = result.return_code == 0
        except Exception as exc:
            row['collection_error_type'] = type(exc).__name__
        write_json(path, row)

    def _snapshot_process_evidence(self, phase):
        """Observe only epoch-owned containers before cleanup; never infer OOM from 137."""
        target = self.output / ('diagnostics/process-' + phase + '.json')
        if target.exists():
            return
        rows, errors, client = [], [], None
        try:
            client = docker.DockerClient(base_url=self.cfg['environment']['docker_host'], timeout=5)
            containers = client.containers.list(all=True, filters={
                'label': ['dycon.epoch=' + self.ownership['epoch']]})
            for container in containers:
                container.reload()
                attrs = container.attrs
                state, host = attrs.get('State', {}), attrs.get('HostConfig', {})
                oom = state.get('OOMKilled')
                memory_peak = None
                memory_stats_error = None
                cgroup = {'available': False, 'reason': 'container_not_running'}
                if state.get('Running'):
                    try:
                        stats = container.stats(stream=False)
                        memory_peak = stats.get('memory_stats', {}).get('max_usage')
                    except Exception as exc:
                        memory_stats_error = type(exc).__name__
                    try:
                        result = container.exec_run([
                            'sh', '-lc',
                            'for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak '
                            '/sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.events '
                            '/sys/fs/cgroup/memory/memory.usage_in_bytes '
                            '/sys/fs/cgroup/memory/memory.max_usage_in_bytes '
                            '/sys/fs/cgroup/memory/memory.oom_control; do '
                            'if test -r "$f"; then printf "FILE %s\\n" "$f"; cat "$f"; fi; done'])
                        raw = result.output.decode('utf-8', 'replace') if isinstance(result.output, bytes) else str(result.output or '')
                        cgroup = {'available': result.exit_code == 0 and bool(raw.strip()),
                                  'return_code': result.exit_code, 'raw': raw[-8192:]}
                    except Exception as exc:
                        cgroup = {'available': False, 'error_type': type(exc).__name__}
                rows.append({'container_id': container.id,
                    'state': {k: state.get(k) for k in ('Status','Running','ExitCode','OOMKilled',
                                                       'StartedAt','FinishedAt','Pid')},
                    'memory_limit_bytes': host.get('Memory'),
                    'memory_peak_bytes': memory_peak,
                    'memory_stats_error_type': memory_stats_error,
                    'cgroup': cgroup,
                    'host_kernel_log': {'available': False,
                        'reason': 'docker_host_kernel_log_not_exposed_by_container_api'},
                    'cause': 'container_oom_confirmed' if oom is True else 'unknown',
                    'source': 'docker_inspect_before_container_cleanup'})
        except Exception as exc:
            errors.append({'error_type': type(exc).__name__, 'source': 'docker_inspect'})
        finally:
            if client is not None:
                client.close()
        write_json(target, {'schema_version': 1, 'ownership': self.ownership,
            'observed_at': now(), 'phase': phase, 'containers': rows, 'errors': errors,
            'available': bool(rows), 'cause': 'unknown' if not rows else None})

    def create_environment(self, *args, **kwargs):
        env = self.inner.create_environment(*args, **kwargs)
        self.env = env
        labels = {"dycon.project": self.ownership["project"], "dycon.run": self.run_id,
            "dycon.task": self.sample.instance_id, "dycon.sample": self.sample.sample_key,
            "dycon.attempt": self.attempt_id, "dycon.role": self.ownership["role"],
            "dycon.epoch": self.ownership["epoch"], "dycon.execution": self.ownership["execution"]}
        env._mounts_compose_path = self.output / "ownership-compose.json"
        env._mounts_compose_path.parent.mkdir(parents=True, exist_ok=True)
        service = {"labels": labels}
        profile = self.cfg["execution_context"]["profile"]
        if profile.get("network") == "none":
            # Use BenchFlow's native no-network overlay for every task.
            env.task_env_config = env.task_env_config.model_copy(update={"allow_internet": False})
        proxy = self.cfg.get("skillsbench_network_proxy")
        if proxy:
            service["environment"] = {name: proxy for name in
                ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")}
            service["environment"].update(NO_PROXY="localhost,127.0.0.1," + profile["bridge_advertise"],
                                         no_proxy="localhost,127.0.0.1," + profile["bridge_advertise"])
        wheelhouse = profile.get("dependency_snapshot")
        if wheelhouse:
            from runtime.agents.dependencies import verify_snapshot
            verify_snapshot(wheelhouse)
            expected_dependency = profile.get("dependency_snapshot_sha256")
            if expected_dependency and sha256(Path(wheelhouse) / "manifest.json") != expected_dependency:
                raise ValueError("OpenHands dependency snapshot differs from frozen execution context")
            wheelhouse = str(Path(wheelhouse).resolve())
            if not Path(wheelhouse).is_dir():
                raise ValueError("OpenHands dependency snapshot is not a host directory: " + wheelhouse)
            service.setdefault("volumes", []).append({"type": "bind", "source": wheelhouse,
                "target": "/opt/dycon-wheels", "read_only": True})
            # Keep the official verifier scripts unchanged while supplying the
            # pinned uv tools from the host-side publication.
            tools = Path(self.cfg["project_root"]) / "runtime/skillsbench-tools"
            if tools.is_dir():
                service.setdefault("volumes", []).append({"type": "bind", "source": str(tools.resolve()),
                    "target": "/opt/dycon-skillsbench-tools", "read_only": True})
                service.setdefault("environment", {})["PATH"] = "/root/.local/bin:/opt/dycon-skillsbench-tools:/opt/dycon-wheels:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        cfs_supported = self.cfg.get("skillsbench_daemon_capabilities", {}).get("cpu_quota_supported", True)
        if not cfs_supported:
            task_key = hash_json({"benchmark": "skillsbench",
                "revision": self.cfg["execution_context"]["revision"],
                "instance_id": self.sample.instance_id})
            task_payload = self.cfg["selection"]["tasks"][task_key]
            sandbox = task_payload["instance"]["metadata"].get("sandbox", {})
            memory_mb = profile.get("memory_overrides", {}).get(self.sample.instance_id,
                sandbox.get("memory_mb"))
            adaptation = resource_override(sandbox.get("cpus", 0), memory_mb,
                                           cfs_supported=False, policy=profile["cpu_quota"])
            service.setdefault("environment", {}).update(adaptation["compose_override"].get("environment", {}))
            requested_memory = sandbox.get("memory_mb")
            if memory_mb and memory_mb != requested_memory:
                env._env_vars.memory = f"{memory_mb}M"
                adaptation["receipt"].update(requested_memory_mb=requested_memory,
                    applied_memory_mb=memory_mb, memory_override=True)
            if adaptation["compose_override"].get("environment", {}).get("CPUS") == "0":
                env._env_vars.cpus = 0
            if adaptation["receipt"]["reason"]:
                write_json(self.output / "resource-adaptation.json", {
                    "schema_version": 1, "task_id": self.sample.instance_id,
                    **adaptation["receipt"], "scope": "this BenchFlow compose service only",
                    "ownership": self.ownership})
        services = {"main": service}
        networks = {"default": {"labels": labels}}
        service_images = self.cfg.get("skillsbench_service_images", {})
        for name, tag in service_images.items():
            services.setdefault(name, {"labels": labels})["image"] = tag
        if self.task_source:
            import yaml
            compose = Path(self.task_source) / "environment/docker-compose.yaml"
            if compose.is_file():
                declared = yaml.safe_load(compose.read_text()) or {}
                for name in declared.get("networks", {}):
                    networks[name] = {"labels": labels}
        write_json(env._mounts_compose_path, {"services": services, "networks": networks})
        if service_images:
            original_start = env.start
            async def start(*args, **kwargs):
                result = await original_start(*args, **kwargs)
                def inspect_images():
                    client = docker.DockerClient(base_url=self.cfg['environment']['docker_host'])
                    try:
                        containers = client.containers.list(all=True, filters={'label': ['dycon.epoch=' + self.ownership['epoch']]})
                        rows = []
                        for container in containers:
                            container.reload()
                            name = container.labels.get('com.docker.compose.service')
                            expected = service_images.get(name)
                            actual = container.attrs['Image']
                            rows.append({'service': name, 'container_id': container.id, 'image_id': actual,
                                         'expected_tag': expected,
                                         'expected_image_id': client.images.get(expected).id if expected else None})
                        valid = ({r['service'] for r in rows} == set(service_images)
                                 and all(r['image_id'] == r['expected_image_id'] for r in rows))
                        write_json(self.output / 'compose-image-verification.json', {
                            'schema_version': 1, 'ownership': self.ownership, 'services': rows,
                            'verified': valid, 'at': now()})
                        if not valid:
                            raise ValueError('Running compose image IDs differ from this attempt NAS leases')
                    finally:
                        client.close()
                await asyncio.to_thread(inspect_images)
                return result
            env.start = start
        original_stop = env.stop
        async def stop(delete=True):
            # BenchFlow delete=True uses `compose down --rmi all`, bypassing
            # the shared NAS image lease and removing another run's tag.
            # It still owns container/network teardown; only the existing
            # ImageLeaseManager owns image reclamation.
            await self._collect_sdk_transport(env)
            await asyncio.to_thread(self._snapshot_process_evidence, "before_cleanup")
            await original_stop(delete=False)
            write_json(self.output / "sandbox-cleanup.json", {"status": "stopped", "at": now(),
                "ownership": self.ownership, "image_cleanup_owner": "ImageLeaseManager"})
        env.stop = stop
        return env

    async def deploy_skills(self, env, task_path, skills_dir, agent_cfg, sandbox_user, agent_cwd,
                            skills_sandbox_dir=None):
        """Keep official deployment; attest what the actual sandbox user can read."""
        import shlex
        await self.inner.deploy_skills(env, task_path, skills_dir, agent_cfg, sandbox_user, agent_cwd,
                                       skills_sandbox_dir=skills_sandbox_dir)
        from benchflow.rollout_planes import DefaultRolloutPlanes
        deploy = DefaultRolloutPlanes.deploy_skills.__globals__['deploy_skills']
        discovery = list(getattr(agent_cfg, 'skill_paths', ()) or deploy.__globals__['_ORACLE_SKILL_PATHS'])
        home = '/home/' + sandbox_user if sandbox_user else '/root'
        paths = {skills_sandbox_dir or '/skills'}
        for relative in discovery:
            relative = str(relative)
            if relative.startswith('$HOME/') or relative.startswith('~/'):
                paths.add(str(Path(home) / relative.split('/', 1)[1]))
                continue
            if relative.startswith('$WORKSPACE/'):
                paths.add(str(Path(agent_cwd) / relative.split('/', 1)[1]))
                continue
            for base in (home, agent_cwd):
                paths.add(str(Path(base) / relative.removeprefix('~/')))
        command = 'for directory in ' + ' '.join(shlex.quote(p) for p in sorted(paths)) + '; do '
        command += ('if [ -d "$directory" ]; then find -L "$directory" -type f '
                    '-name SKILL.md -readable -exec sha256sum {} + || '
                    'printf "optional discovery path unreadable: %s\\n" "$directory" >&2; fi; done')
        # Docker exec inherits the image cwd, often /root; find cannot walk
        # any path from an inaccessible cwd even when the target is readable.
        command = 'cd ' + shlex.quote(home) + ' && ' + command
        result = await env.exec(command, user=sandbox_user or 'root', timeout_sec=60)
        rows = sorted(set(line.strip() for line in (result.stdout or '').splitlines() if line.strip()))
        expected = {sha256(p) for p in Path(skills_dir).rglob('SKILL.md')} if skills_dir else set()
        observed = {line.split()[0] for line in rows}
        mode = self.cfg['execution_context']['profile']['skill_mode']
        valid = result.return_code == 0 and (not rows if mode == 'no-skill' else expected <= observed)
        record = {'schema_version': 1, 'ownership': self.ownership, 'skill_mode': mode,
                  'sandbox_user': sandbox_user or 'root', 'paths': sorted(paths),
                  'readable_skill_manifests': rows, 'expected_skill_hashes': sorted(expected),
                  'return_code': result.return_code, 'verified': valid, 'at': now()}
        record.update(cwd=home, workspace=agent_cwd, stderr=getattr(result, 'stderr', ''))
        record['receipt_sha256'] = hash_json(record)
        write_json(self.output / 'skills-visibility.json', record)
        if not valid:
            raise ValueError('Sandbox skill visibility differs from the frozen skill mode')
        dependency = self.cfg['execution_context']['profile'].get('dependency_snapshot')
        if dependency and sandbox_user:
            from runtime.agents.dependencies import probe_runtime
            publication = read_json(Path(dependency) / 'manifest.json')
            if any(a['policy'] == 'dycon_bounded_request_subprocess_terminal_v1'
                   for a in publication.get('source_adaptations', [])):
                await probe_runtime(env, self.cfg, self.output, self.ownership, agent_cwd)

    def verifier(self, *args, **kwargs):
        from benchflow.task import Verifier
        original = Verifier(*args, **kwargs)
        owner = self
        class PersistedVerifier:
            async def verify(self):
                try:
                    try:
                        result = await _run_verifier_with_evidence(original, owner)
                    except RuntimeError as exc:
                        message = str(exc)
                        if "clearing verifier output directory" not in message:
                            raise
                        await owner.env.exec(
                            "mkdir -p /logs/verifier && "
                            "if command -v find >/dev/null 2>&1; then "
                            "find /logs/verifier -mindepth 1 -exec rm -rf -- {} +; "
                            "else rm -rf /logs/verifier/* /logs/verifier/.[!.]* /logs/verifier/..?*; fi && "
                            "chmod 777 /logs/verifier",
                            user="root", timeout_sec=60)
                        await owner._export_workspace()
                        result = await _run_verifier_with_evidence(original, owner)
                    write_json(owner.output / "evaluation-raw.json", {
                        "schema_version": 1, "completed": True, "rewards": result.rewards,
                        "verifier": getattr(result, "__dict__", {}),
                        "verifier_evidence": "diagnostics/verifier-execution.json",
                        "subject_sha256": owner.subject_sha256, "finished_at": now()})
                    return result
                except BaseException as exc:
                    import importlib.metadata
                    import traceback
                    candidates = await owner.env.exec(
                        "find /root -maxdepth 4 -type f \\( -iname '*.csv' -o -iname '*.txt' -o "
                        "-iname '*.json' -o -iname '*.docx' -o -iname '*.pptx' \\) "
                        "-printf '%p\\t%s\\n' 2>/dev/null | sort",
                        user="root", timeout_sec=30)
                    try:
                        classifier_version = importlib.metadata.version("benchflow")
                    except importlib.metadata.PackageNotFoundError:
                        classifier_version = "unknown"
                    write_json(owner.output / "evaluation-raw.json", {
                        "schema_version": 2, "completed": False,
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
                        "inputs": {"subject_path": str(owner.subject_path),
                                   "subject_sha256": owner.subject_sha256,
                                   "declared_output_paths": owner.cfg["execution_context"]["profile"]
                                       .get("output_paths", {}).get(owner.sample.instance_id, []),
                                   "candidate_outputs": candidates.stdout,
                                   "candidate_scan_return_code": candidates.return_code},
                        "classifier": {"benchflow_version": classifier_version},
                        "verifier_evidence": "diagnostics/verifier-execution.json",
                        "finished_at": now()})
                    raise
        return PersistedVerifier()

    async def _export_workspace(self):
        from application.execution.subject import export_workspace
        source = self.cfg["selection"]["tasks"][self.sample.task_key]
        binding = {**self.ownership, "workspace": self.workspace, "agent": "openhands",
                   "image": self.image["identity"]}
        declared = self.cfg["execution_context"]["profile"]["output_paths"].get(self.sample.instance_id, [])
        outputs, resolution_sha256 = await resolve_submission_outputs(
            self.env, self.sample.instance_id, declared, self.workspace, self.output, self.ownership)
        if resolution_sha256 is not None:
            binding['output_resolution_sha256'] = resolution_sha256
        self.subject_path = await export_workspace(self.env, outputs,
            self.output / "subject", binding, source["instance"]["package_sha256"], self.image["identity"])
        self.subject_sha256 = sha256(self.subject_path)

    async def harden_before_verify(self, env, task, sandbox_user=None, *, workspace=None):
        if sandbox_user == "root":
            raise ValueError("Root agent hardening would terminate the sandbox; use the isolated agent user")
        proxy = self.cfg.get("skillsbench_network_proxy")
        profile = self.cfg.get("execution_context", {}).get("profile", {})
        if proxy or profile.get("network") == "none":
            variables = {name: proxy or "" for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")}
            variables.update(NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1")
            # Upstream re-pins trusted PATH during hardening. Tool preparation
            # below uses /usr/local/bin without overriding that invariant.
            if self.cfg["execution_context"]["profile"].get("dependency_snapshot"):
                # Public tasks may still need task-specific packages. The
                # snapshot supplies the pinned base closure; only the none-
                # network profile is forbidden from resolving extras online.
                variables.update(UV_FIND_LINKS="/opt/dycon-wheels/wheels",
                                 PIP_FIND_LINKS="/opt/dycon-wheels/wheels")
            if profile.get("network") == "none":
                variables.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_PYTHON_DOWNLOADS="never",
                                 PIP_NO_INDEX="1")
            task.config.verifier.env = {**(task.config.verifier.env or {}), **variables}
            timeout_override = profile.get("verifier_timeout_overrides", {}).get(self.sample.instance_id)
            if timeout_override:
                task.config.verifier.timeout_sec = timeout_override
            write_json(self.output / "verifier-environment.json", {"variables": variables,
                "verifier_timeout_override_sec": timeout_override,
                "policy": ("official-verifier-network-none-offline-python-wheels-v1"
                           if profile.get("network") == "none" else "official-verifier-public-proxy-offline-python-wheels-v1")})
        await self.inner.harden_before_verify(env, task, sandbox_user, workspace=workspace)
        from runtime.agents.verifier_tools import prepare_verifier_tools
        tools = await prepare_verifier_tools(env)
        write_json(self.output / "diagnostics/verifier-tools.json", {
            "schema_version": 2, "ownership": self.ownership,
            "source_paths": ["/opt/dycon-wheels", "/opt/dycon-skillsbench-tools"],
            "destination": "/root/.local/bin", "trusted_curl_path": "/usr/local/bin/curl",
            "policy": "published_uv_bootstrap_after_hardening_v2", "return_code": tools.return_code,
            "stdout": tools.stdout, "stderr": getattr(tools, "stderr", ""), "at": now()})
        # The official Druid verifier assumes the patched single server is
        # listening after oracle build; start it in the owned sandbox and
        # record readiness before verifier execution.
        if self.sample.instance_id == "fix-druid-loophole-cve":
            start = await env.exec(
                "cd /opt/druid && nohup ./bin/start-single-server-small > /logs/agent/druid-server.log 2>&1 &",
                user="root", timeout_sec=30)
            ready = await env.exec(
                "i=0; until curl -fsS http://127.0.0.1:8888/status/health >/dev/null 2>&1; do "
                "i=$((i+1)); test $i -lt 600 || exit 1; sleep 1; done",
                user="root", timeout_sec=610)
            write_json(self.output / "druid-readiness.json", {
                "start_return_code": start.return_code,
                "ready_return_code": ready.return_code,
                "status": "ready" if ready.return_code == 0 else "failed"})
            if ready.return_code:
                raise RuntimeError("Druid patched server did not become ready before verifier")
        self.env, self.workspace = env, workspace
        # Export the immutable submission after BenchFlow hardens the workspace,
        # while the task container is live and before the verifier modifies it.
        await self._export_workspace()


def _safe_acp_message(value):
    import re
    text = str(value or '')[:2048]
    text = re.sub(r'https?://[^\s]+', '[url-redacted]', text)
    text = re.sub(r'(?i)(bearer\s+|api[_-]?key[=: ]+|password[=: ]+|secret[=: ]+)\S+',
                  '[credential-redacted]', text)
    text = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[credential-redacted]', text)
    return text[:256]


def _acp_details(data):
    """Keep typed cause fields and bounded structure, never raw data messages/prompts."""
    import re
    text = json.dumps(data, ensure_ascii=False, default=str)[:65536]
    types = sorted(set(re.findall(r'\b(?:litellm\.(?:Timeout|APIError)|APITimeoutError|'
                                 r'ConnectTimeout|ReadTimeout|WriteTimeout|PoolTimeout|LLMTimeoutError)\b', text)))
    allowed = {'code','status','status_code','http_status','error_type','exception_type',
               'type','phase','timeout','timeout_seconds','num_retries','cause','details','error','data',
               'request_id','correlation_id'}
    def keep(value, depth=0):
        if depth > 4:
            return '[depth-limited]'
        if isinstance(value, dict):
            return {str(k):keep(v,depth+1) for k,v in list(value.items())[:24] if k in allowed}
        if isinstance(value, list):
            return [keep(v,depth+1) for v in value[:8]]
        if type(value) in (int,float,bool) or value is None:
            return value
        if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,96}',value) and not value.startswith('sk-'):
            return value
        return '[text-redacted]'
    cause = 'sdk_api_timeout' if any(t in types for t in (
        'litellm.Timeout','APITimeoutError','LLMTimeoutError','ConnectTimeout',
        'ReadTimeout','WriteTimeout','PoolTimeout')) else 'unknown'
    phases = {t.removesuffix('Timeout').lower() for t in types
              if t in {'ConnectTimeout','ReadTimeout','WriteTimeout','PoolTimeout'}}
    phase = next(iter(phases)) if len(phases) == 1 else 'unknown'
    return {'data': keep(data), 'data_available': data is not None,
            'exception_types': types, 'cause': cause, 'phase': phase}


def _json_events(events):
    return [{"index": i, "event": e if isinstance(e, dict) else {"unparsed": repr(e)}}
            for i, e in enumerate(events or [])]


def _result_json(result):
    return {key: getattr(result, key, None) for key in (
        "task_name", "rollout_name", "rewards", "agent", "agent_name", "model", "n_tool_calls",
        "n_skill_invocations", "n_prompts", "n_input_tokens", "n_output_tokens", "total_tokens",
        "cost_usd", "usage_source", "price_source", "usage_details", "error", "error_category",
        "verifier_error", "verifier_error_category", "export_error", "partial_trajectory",
        "trajectory_source", "started_at", "finished_at")}


def _map_evaluation(result, report, capture_status):
    if report.get("completed") is not True or result.get("verifier_error"):
        if result.get('error') and not result.get('verifier_error'):
            category=str(result.get('error_category') or '').lower()
            return ('AgentTimeout' if 'timeout' in category else 'OpenHandsError'), None, 'agent_failed_before_verification'
        return "VerifierError", None, "verifier_incomplete"
    rewards = report.get("rewards")
    if (not isinstance(rewards, dict) or not rewards or
            any(type(v) not in (int, float) or not 0 <= v <= 1 for v in rewards.values())):
        return "UnsupportedRewardPolicy", None, "unsupported_reward_range"
    # Verifier truth is durable independently of capture finalization. A partial
    # capture excludes training points, but must not erase an observed reward.
    success = all(v == 1 or v == 1.0 for v in rewards.values())
    return ("Submitted" if success else "AgentCompleted"), success, "normalized_all_dimensions_equal_one_v1"


def agent_rollout_config(cfg, task_source, image, planes, output, env, *, attempt_id, session_id, seconds):
    """One agent environment/install/skill contract for preflight and rollout."""
    from runtime.agents.dependencies import install_runtime
    profile = cfg['execution_context']['profile']
    async def install(environment):
        await install_runtime(environment, cfg, output, planes.ownership)
    mode = profile['skill_mode']
    skills = task_source / 'environment/skills'
    return RolloutConfig(task_path=task_source, agent='openhands', model=cfg['model']['served_name'],
        agent_env=env, skill_mode=mode, skills_dir=skills if mode == 'with-skill' and skills.is_dir() else None,
        sandbox_user=profile['sandbox_user'], timeout=seconds,
        agent_idle_timeout=max(profile['agent_idle_seconds'], seconds), jobs_dir=output / 'benchflow',
        job_name=attempt_id, rollout_name=session_id,
        environment_manifest=EnvironmentManifest(name=task_source.name, image=image['tag'],
            keep_alive=True, isolation='per_task', owns_lifecycle=True),
        skip_agent_install=profile['skip_agent_install'],
        pre_agent_hooks=[install] if profile['skip_agent_install'] else [],
        usage_tracking=UsageTrackingConfig(mode='off'), planes=planes)


def environment_contract(cfg, task):
    from runtime.agents import dependencies, openhands, sandbox_probe
    modules = [Path(__file__), Path(dependencies.__file__), Path(sandbox_probe.__file__), Path(openhands.__file__)]
    return {'schema_version': 1, 'policy': 'skillsbench-owned-environment-v1',
        'run_id': cfg['run_id'], 'configuration_sha256': cfg['config_hash'],
        'execution_context_sha256': cfg['execution_context']['context_sha256'],
        'model_fingerprint': cfg['model']['fingerprint'], 'task_key': task['key'] if 'key' in task else hash_json(task),
        'task_package_sha256': task['instance']['package_sha256'],
        'images': task['instance']['images'],
        'dependency_snapshot_sha256': cfg['execution_context']['profile'].get('dependency_snapshot_sha256'),
        'code': {p.name: sha256(p) for p in modules}}


def audit_task_environments(cfg, run_id, *, workers=1):
    """Public BenchFlow lifecycle methods; no LLM, oracle or verifier runs."""
    if not 1 <= workers <= 8:
        raise ValueError('Environment audit workers must be in [1,8]')
    async def execute():
        semaphore = asyncio.Semaphore(workers)
        async def one(key, task):
            async with semaphore:
                return await _audit_task_environment(cfg, run_id, key, task)
        rows = await asyncio.gather(*(one(key, task) for key, task in cfg['selection']['tasks'].items()))
        return {'schema_version': 1, 'run_id': run_id, 'task_count': len(rows),
                'verified': all(r['verified'] for r in rows), 'model_requests': 0,
                'failed_tasks': [r['task_id'] for r in rows if not r['verified']], 'tasks': rows, 'at': now()}
    result = asyncio.run(execute())
    write_json(run_path(cfg, run_id) / 'validation/environments/summary.json', result)
    return result


def _terminal_evidence(output, *, agent_status, capture_status, evaluation_status, error=None):
    output = Path(output)
    names = [
        'benchflow-diagnostics.json', 'benchflow/result.json', 'openhands/acp-events.json',
        'diagnostics/acp-error.json', 'diagnostics/process-transport.json',
        'diagnostics/process-transport_error.json', 'diagnostics/process-before_cleanup.json',
        'diagnostics/process-supervisor.json', 'capture-drain.json',
        'capture-finalization.json', 'capture-stop-error.json', 'evaluation-raw.json',
        'sample-token-budget.json', 'diagnostics/verifier-zero-audit.json',
    ]
    files = []
    primary_cause, cause_source = None, None
    process = []
    for name in names:
        path = output / name
        if not path.is_file():
            continue
        row = None
        try:
            row = read_json(path)
        except (OSError, ValueError):
            pass
        files.append({'path': name, 'sha256': sha256(path)})
        if name == 'benchflow-diagnostics.json' and isinstance(row, dict):
            candidate = row.get('cause')
            if candidate and candidate != 'unknown':
                primary_cause, cause_source = candidate, name
        if name.startswith('diagnostics/process-') and isinstance(row, dict):
            for container in row.get('containers', []):
                process.append({key: container.get(key) for key in
                    ('container_id', 'state', 'memory_limit_bytes', 'memory_peak_bytes', 'cgroup', 'cause')})
            if name == 'diagnostics/process-transport.json':
                process.append({key: row.get(key) for key in
                    ('process_pid', 'process_exit_code', 'diagnosis', 'source', 'cause')})
    supervisor_path = output / 'diagnostics/process-supervisor.json'
    supervisor = read_json(supervisor_path) if supervisor_path.is_file() else None
    if not primary_cause:
        primary_cause = agent_status or (type(error).__name__ if error else 'unknown')
        cause_source = 'agent_status' if agent_status else 'backend_exception' if error else 'unavailable'
    return {'schema_version': 1, 'primary_cause': primary_cause, 'cause_source': cause_source,
        'agent_exit_status': agent_status, 'capture_status': capture_status,
        'evaluation_status': evaluation_status, 'process': process,
        'supervisor_termination': supervisor, 'evidence_files': files,
        'supervisor_log': {'available': False, 'reason': 'no attempt-bound supervisor log was provided'}}


def _safe_verifier_text(value):
    import re

    text = value.decode('utf-8', 'replace') if isinstance(value, bytes) else str(value or '')
    text = re.sub(r'https?://[^\s/@]+:[^\s/@]+@', 'https://[credential-redacted]@', text)
    text = re.sub(r'(?i)(bearer\s+|api[_-]?key[=: ]+|password[=: ]+|secret[=: ]+|token[=: ]+)\S+',
                  r'\1[credential-redacted]', text)
    return text


def _verifier_stdout_tail(verifier, limit=4096):
    path = getattr(getattr(verifier, '_rollout_paths', None), 'test_stdout_path', None)
    if path is None:
        return None
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return _safe_verifier_text(stream.read())[-limit:]
    except OSError:
        return None


async def _run_verifier_with_evidence(verifier, owner):
    import re
    import time
    from urllib.parse import urlsplit

    path = owner.output / 'diagnostics/verifier-execution.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = read_json(path) if path.exists() else {
        'schema_version': 1, 'ownership': owner.ownership, 'commands': [], 'attempts': []}
    started_at, started = now(), time.monotonic()
    attempt = {'started_at': started_at, 'phase': 'verifier_running'}
    receipt['attempts'].append(attempt)
    receipt['attempts'] = receipt['attempts'][-8:]
    receipt['phase'] = 'verifier_running'

    def persist():
        attempt['elapsed_seconds'] = time.monotonic() - started
        receipt['updated_at'] = now()
        write_json(path, receipt)

    def classify(command):
        value = command if isinstance(command, str) else ' '.join(map(str, command or []))
        lowered = value.lower()
        if re.search(r'\b(apt-get|apt)\b', lowered):
            phase = 'apt'
        elif re.search(r'\b(curl|wget)\b', lowered):
            phase = 'download'
        elif re.search(r'\b(uv|pip|pip3)\b', lowered):
            phase = 'python_dependency'
        elif re.search(r'(test\.sh|pytest|unittest|/tests?/|\btest\b)', lowered):
            phase = 'task_test'
        else:
            phase = 'verifier_setup'
        hosts = set()
        for match in re.findall(r"https?://[^\s'\"<>]+", value):
            try:
                if urlsplit(match).hostname:
                    hosts.add(urlsplit(match).hostname)
            except ValueError:
                continue
        hosts = sorted(hosts)
        summary = re.sub(r"https?://[^\s'\"<>]+", '[url]', value)
        return phase, value, _safe_verifier_text(summary)[:256], hosts

    sandbox = getattr(verifier, '_sandbox', None)
    original_exec = getattr(sandbox, 'exec', None) if sandbox is not None else None
    patched = False
    heartbeat_stop = asyncio.Event()

    async def heartbeat():
        while not heartbeat_stop.is_set():
            try:
                await asyncio.wait_for(heartbeat_stop.wait(), timeout=15)
            except asyncio.TimeoutError:
                attempt['heartbeat_at'] = now()
                attempt['phase'] = receipt.get('active_phase', 'verifier_running')
                persist()

    async def tracked_exec(*args, **kwargs):
        command = kwargs.get('command', args[0] if args else '')
        phase, raw, summary, hosts = classify(command)
        row = {'phase': phase, 'command_summary': summary, 'command_sha256': hash_json(raw),
               'target_hosts': hosts, 'timeout_seconds': kwargs.get('timeout_sec'),
               'started_at': now(), 'status': 'running'}
        receipt['commands'].append(row)
        receipt['commands'] = receipt['commands'][-512:]
        receipt['active_phase'] = phase
        persist()
        command_started = time.monotonic()
        try:
            result = await original_exec(*args, **kwargs)
            row.update(status='completed', return_code=getattr(result, 'return_code', None),
                       stdout_tail=_safe_verifier_text(getattr(result, 'stdout', ''))[-2000:],
                       stderr_tail=_safe_verifier_text(getattr(result, 'stderr', ''))[-2000:])
            return result
        except BaseException as exc:
            row.update(status='timeout' if isinstance(exc, TimeoutError) else 'failed',
                       error_type=type(exc).__name__, error=_safe_verifier_text(str(exc))[:512])
            raise
        finally:
            row['duration_seconds'] = time.monotonic() - command_started
            row['finished_at'] = now()
            receipt.pop('active_phase', None)
            persist()

    try:
        if original_exec is not None:
            try:
                sandbox.exec = tracked_exec
                patched = True
            except (AttributeError, TypeError):
                attempt['sandbox_exec_instrumentation'] = 'unavailable'
        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            result = await verifier.verify()
            stdout_tail = _verifier_stdout_tail(verifier)
            rewards = getattr(result, 'rewards', None)
            if (isinstance(rewards, dict) and type(rewards.get('reward')) in (int, float)
                    and rewards['reward'] == 0 and stdout_tail
                    and re.search(r'/verifier/test\.sh: line \d+: uvx: command not found', stdout_tail)
                    and re.search(r'/root/\.local/bin/env: No such file or directory', stdout_tail)):
                from runtime.agents.verifier_audit import classify_verifier_logs, inspect_ctrf
                logs, unreadable = [], False
                paths = getattr(verifier, '_rollout_paths', None)
                for name in ('test_stdout_path', 'test_stderr_path'):
                    log_path = getattr(paths, name, None)
                    if log_path is not None and Path(log_path).exists():
                        try:
                            logs.append(Path(log_path).read_text(encoding='utf-8', errors='replace'))
                        except OSError:
                            unreadable = True
                classification, _ = classify_verifier_logs(logs,
                    ctrf_executed=inspect_ctrf(owner.output)[0], unreadable=unreadable)
                # Keep the existing bootstrap guard under the shared policy;
                # recovered installation/test execution must not become an error.
                if classification == 'VerifierError':
                    attempt['failure_reason'] = 'verifier_bootstrap_tools_missing'
                    raise RuntimeError('SkillsBench verifier bootstrap failed before test execution')
            attempt['phase'] = 'completed'
            attempt['return_code'] = 0
            attempt['test_stdout_tail'] = stdout_tail
            return result
        except BaseException as exc:
            attempt['phase'] = 'timed_out' if isinstance(exc, TimeoutError) else 'failed'
            attempt['error_type'] = type(exc).__name__
            attempt['error'] = _safe_verifier_text(str(exc))[:512]
            attempt['test_stdout_tail'] = _verifier_stdout_tail(verifier)
            raise
        finally:
            heartbeat_stop.set()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            if patched:
                sandbox.exec = original_exec
    finally:
        receipt['phase'] = attempt['phase']
        receipt['finished_at'] = now()
        persist()


async def _audit_task_environment(cfg, run_id, key, task):
    from benchflow.rollout_planes import DefaultRolloutPlanes
    cfg = copy.deepcopy(cfg)
    contract = environment_contract(cfg, task)
    base = run_path(cfg, run_id) / 'validation/environments' / task['instance_id'] / hash_json(contract)
    output = base
    latest = base / 'latest.json'
    if latest.exists():
        pointer = read_json(latest)
        candidate = (base / pointer['path']).resolve()
        if not candidate.is_relative_to(base.resolve()) or sha256(candidate) != pointer['sha256']:
            raise ValueError('Environment audit receipt pointer changed')
        output = candidate.parent
    target = output / 'result.json'
    if target.exists():
        previous = read_json(target)
        if (previous.get('contract') == contract and previous.get('verified')
                and all((output / name).is_file() and sha256(output / name) == digest
                        for name, digest in previous.get('evidence_sha256', {}).items())):
            return previous
    attempt_id = 'environment-' + uuid.uuid4().hex
    output = base / 'attempts' / attempt_id
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'result.json'
    row = {'schema_version': 1, 'run_id': run_id, 'task_id': task['instance_id'],
           'contract': contract, 'verified': False, 'attempt_id': attempt_id, 'started_at': now()}
    factory, rollout, planes = None, None, None
    try:
        factory = SkillsBenchFactory(cfg, run_id)
        await asyncio.to_thread(factory.lock_images, [task])
        image = factory.images[(task['instance_id'], 'main')]
        cfg['skillsbench_service_images'] = {s: r['tag'] for (iid, s), r in factory.images.items() if iid == task['instance_id']}
        cfg['skillsbench_daemon_capabilities'] = {'cpu_quota_supported': factory.cpu_quota_supported}
        sample = type('EnvironmentSample', (), {'instance_id': task['instance_id'], 'task_key': key, 'sample_key': attempt_id})()
        planes = OwnedPlanes(DefaultRolloutPlanes(), cfg, run_id, sample, attempt_id, image, output)
        source = Path(task['instance']['task_path'])
        planes.task_source = source
        model = 'openai/' + cfg['model']['served_name']
        env = {**SKILLSBENCH_AGENT_ENVIRONMENT, 'BENCHFLOW_PROVIDER_BASE_URL': 'http://127.0.0.1:9/v1',
               'BENCHFLOW_PROVIDER_API_KEY': 'environment-probe', 'BENCHFLOW_PROVIDER_MODEL': model,
               'LLM_MODEL': model, 'MODEL': model, 'LITELLM_LOCAL_MODEL_COST_MAP': 'true',
               'OPENHANDS_SUPPRESS_BANNER': '1', 'DYCON_OPENHANDS_MODEL_SECONDS': str(cfg['timeouts']['model_seconds'])}
        config = agent_rollout_config(cfg, source, image, planes, output, env,
                                      attempt_id=attempt_id, session_id=attempt_id, seconds=1200)
        rollout = await Rollout.create(config)
        await rollout.setup()
        await rollout.start()
        await rollout.install_agent()
        for name in ('dependency-install.json', 'skills-visibility.json', 'agent-environment.json', 'compose-image-verification.json'):
            value = read_json(output / name)
            if value.get('verified', value.get('return_code') == 0) is not True:
                raise ValueError('Environment evidence failed: ' + name)
        row['verified'] = True
    except Exception as exc:
        import traceback
        row.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        if rollout is not None:
            try:
                await rollout.cleanup()
            except Exception as exc:
                row.update(cleanup_error=repr(exc), verified=False)
        if factory is not None:
            try:
                remaining = factory.client.containers.list(all=True, filters={'label': ['dycon.epoch=' + planes.ownership['epoch']]}) if planes else []
                row['remaining_containers'] = [c.id for c in remaining]
                if remaining or not (output / 'sandbox-cleanup.json').exists():
                    row['verified'] = False
                await asyncio.to_thread(factory.cleanup_owned, run_id)
            except Exception as exc:
                row.update(cleanup_error=repr(exc), verified=False)
    row['evidence_sha256'] = {p.name: sha256(p) for p in output.glob('*.json') if p != target}
    row['finished_at'] = now()
    row['receipt_sha256'] = hash_json(row)
    write_json(target, row)
    write_json(latest, {'path': target.relative_to(base).as_posix(), 'sha256': sha256(target),
                        'verified': row['verified'], 'at': now()})
    return row


def preflight(root, run_id, *, audit_environments=True, workers=1):
    import importlib.metadata

    from application.execution.run_plan import load_plan
    from catalog.skillsbench import verify_package
    from core.backends import capabilities, identity
    from core.config import require_model
    from core.model_identity import model_identity_verification_enabled, verify_model
    from runtime.agents.openhands import check_version, validate_profile

    loaded = load_plan(root, run_id)
    cfg, tasks = loaded["config"], loaded["config"]["selection"]["tasks"]
    from core.config import validate_generation_contract
    validate_generation_contract(cfg)
    if identity(cfg) != ("skillsbench", "openhands"):
        raise ValueError("Run does not freeze the SkillsBench/OpenHands profile")
    capabilities(cfg)
    profile = validate_profile(cfg["execution_context"]["profile"])
    from core.preflight import SKIP_REASON, long_checks_enabled, timed_check
    long_checks = long_checks_enabled(cfg)
    checks = {name: {"status": "not_run", "seconds": 0.0, "calls": 0}
              for name in ("model_sha", "task_package_sha", "oci_deep_verify",
                           "dependency_snapshot_sha", "environment_audit")}
    if not long_checks:
        for row in checks.values():
            row.update(status="skipped", reason=SKIP_REASON)
    blockers, limitations = [], []
    if cfg["execution_context"].get("validation_only"):
        blockers.append("Validation-only task snapshot cannot execute an agent rollout")
    try:
        check_version()
        versions = {"benchflow": importlib.metadata.version("benchflow")}
        if versions["benchflow"] != "0.6.3":
            blockers.append("isolated BenchFlow version differs from frozen profile")
    except Exception as exc:
        versions = {}
        blockers.append("agent runtime dependency unavailable: " + str(exc))
    try:
        if model_identity_verification_enabled(cfg):
            require_model(cfg)
            with timed_check(checks, "model_sha", enabled=long_checks):
                verify_model(cfg) if long_checks else verify_model(cfg, deep=False)
        else:
            checks["model_sha"].update(status="skipped", reason="model.verify_identity=false")
    except Exception as exc:
        blockers.append("configured model identity unavailable: " + str(exc))
    try:
        if cfg["serving"]["management"] == "managed":
            from hidden.backends.vllm.engine import serving_interpreter
            from runtime.serving.resources import inspect_devices
            devices = cfg["serving"]["gpus"]
            if cfg["serving"].get("trajectory_sharding") == "per_gpu_tp1":
                serving_devices = {"device_kind": "mig" if str(devices[0]).startswith("MIG-") else "physical",
                                   "workers": [inspect_devices(serving_interpreter(cfg), [device], probe=True)
                                               for device in devices]}
            else:
                serving_devices = inspect_devices(serving_interpreter(cfg), devices, probe=True)
            mig_rows = serving_devices.get("device_memory", [])
            if "workers" in serving_devices:
                mig_rows = [row for worker in serving_devices["workers"]
                            for row in worker.get("device_memory", [])]
            if (serving_devices["device_kind"] == "mig"
                    and any(row["process_count"] for row in mig_rows)
                    and not cfg["serving"].get("allow_mig_sharing", False)):
                blockers.append("Configured MIG instance is already occupied")
        else:
            serving_devices = {"management": "external", "gpus": list(cfg["serving"]["gpus"])}
    except Exception as exc:
        serving_devices = {"error": str(exc)}
        blockers.append("Serving GPU identity cannot be probed: " + str(exc))
    task_capabilities = {}
    images = []
    client = docker.DockerClient(base_url=cfg["environment"]["docker_host"])
    try:
        daemon_info = client.info()
    finally:
        client.close()
    cpu_quota_supported = not any("No cpu cfs quota support" in str(w)
                                  for w in daemon_info.get("Warnings", []))
    cfg["skillsbench_daemon_capabilities"] = {"cpu_quota_supported": cpu_quota_supported}
    if not cpu_quota_supported and profile["cpu_quota"] == "strict" and any(
            (task["instance"]["metadata"].get("sandbox", {}).get("cpus") or 0) > 0
            for task in tasks.values()):
        blockers.append("Docker daemon cannot enforce task CPU quotas; select an explicit resource adaptation profile")
    for key, task in tasks.items():
        try:
            official_root = (Path(cfg["project_root"]) / cfg["skillsbench_images"]["source_root"] / "tasks").resolve()
            if Path(task["instance"]["task_path"]).resolve().parent != official_root:
                raise ValueError("Task package escaped the pinned official task root")
            with timed_check(checks, "task_package_sha", enabled=long_checks):
                capability = verify_package(task, deep=long_checks)
            task_capabilities[task["instance_id"]] = capability
            if not capability["supported"]:
                blockers.extend(task["instance_id"] + ": " + x for x in capability["reasons"])
            if not profile["output_paths"].get(task["instance_id"]):
                blockers.append(task["instance_id"] + ": profile has no declared submission output paths")
            for unit in task["instance"]["images"]:
                with timed_check(checks, "oci_deep_verify", enabled=long_checks):
                    ref, image = image_reference(cfg, unit, deep=long_checks)
                images.append({"task_id": task["instance_id"], "unit": unit["key"],
                               "image": image.identity(), "nas_reference": ref})
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(task["instance_id"] + ": startup task/image validation failed: " + str(exc))
    if cfg.get("capture", {}).get("enabled") and profile["request_roles"] != "openhands-1.22.1":
        blockers.append("OpenHands request-role qualification is incomplete; exact capture is blocked")
    if cfg.get("parallel", {}).get("enabled") and profile["request_roles"] != "openhands-1.22.1":
        blockers.append("OpenHands request-role qualification is incomplete; parallel is blocked")
    if profile["network"] == "public":
        limitations.append("Task network is public; dependency installation is not offline-qualified")
    if not profile["dependency_snapshot"]:
        limitations.append("No platform-specific locked wheel/uv/apt dependency snapshot is attached")
    else:
        try:
            from runtime.agents.dependencies import verify_snapshot
            with timed_check(checks, "dependency_snapshot_sha", enabled=long_checks):
                dependency = verify_snapshot(profile["dependency_snapshot"], deep=long_checks)
                if (profile.get("dependency_snapshot_sha256")
                        and sha256(Path(profile["dependency_snapshot"]) / "manifest.json")
                            != profile["dependency_snapshot_sha256"]):
                    raise ValueError("Dependency snapshot differs from the frozen profile")
            versions["dependency_receipt_sha256"] = dependency["receipt_sha256"]
            adaptations = {row['policy'] for row in dependency.get('source_adaptations', [])}
            required = {'dycon_bounded_request_subprocess_terminal_v1', 'preserve_explicit_terminal_backend_on_reset_v1'}
            if not required <= adaptations:
                blockers.append('Dependency publication does not implement shared bounded SDK/subprocess terminal contract')
            if not dependency.get('agent_runtime'):
                blockers.append('Dependency publication has no hash-bound portable CPython 3.12 agent runtime')
        except (OSError, ValueError, KeyError) as exc:
            blockers.append("Offline dependency closure is invalid: " + str(exc))
    environment_audit = None
    if not long_checks:
        environment_audit = {"status": "skipped", "verified": False, "reason": SKIP_REASON}
    elif not audit_environments:
        checks["environment_audit"].update(status="skipped", reason="audit_environments=false")
    if long_checks and audit_environments and not blockers:
        with timed_check(checks, "environment_audit"):
            environment_audit = audit_task_environments(cfg, run_id, workers=workers)
        if not environment_audit['verified']:
            checks['environment_audit']['status'] = 'failed'
            blockers.append('Task environment audit failed: ' + ', '.join(environment_audit['failed_tasks']))
    return {"run_id": run_id, "configuration_sha256": cfg["config_hash"],
        "execution_context_sha256": cfg["execution_context"]["context_sha256"],
        "benchmark": "skillsbench", "revision": cfg["execution_context"]["revision"],
        "agent": "openhands", "profile_sha256": cfg["execution_context"]["profile_sha256"],
        "dependency_versions": versions, "tasks": task_capabilities, "images": images,
        "expected_openhands_cli_revision": profile["cli_revision"],
        "expected_openhands_sdk_version": profile["sdk_version"],
        "expected_openhands_tools_version": profile["tools_version"],
        "serving_devices": serving_devices,
        "openhands_agent_install_probe": 'verified' if environment_audit and environment_audit['verified'] else 'pending_in_sandbox',
        "environment_audit": environment_audit,
        "long_checks": long_checks, "checks": checks,
        'startup_policy': {
            'version': 'skillsbench-shared-startup-v3',
            'entry_paths': ['run create/start', 'frozen clone', 'run resume', 'run retry-failed/resume'],
            'generation': cfg['generation'],
            'native_loop_guard': {
                'options': cfg.get('parallel', {}).get('reasoning_loop', {}),
                'scope': 'agent_decision native generated tokens; baseline and all parallel candidates',
                'capture_required': False, 'steering_capability_changed': False},
            'cross_step_guard': {
                'loop_repeat_limit': cfg['agent'].get('loop_repeat_limit'),
                'loop_max_period': cfg['agent'].get('loop_max_period', 7),
                'scope': 'accepted canonical decisions within one physical attempt',
                'match_policy': 'exact_reasoning_content_and_canonical_actions_v1',
                'stop_before_tool_execution': True},
            'dependency_publication_sha256': profile.get('dependency_snapshot_sha256'),
            'configuration_source': 'immutable run snapshot; mutable root is not reread on resume'},
        "request_roles_qualified": profile["request_roles"] == "openhands-1.22.1",
        "cpu_quota_policy": profile["cpu_quota"],
        "docker_daemon": {"server_version": daemon_info.get("ServerVersion"),
            "cgroup_version": daemon_info.get("CgroupVersion"),
            "cpu_quota_supported": cpu_quota_supported,
            "warnings": daemon_info.get("Warnings", [])},
        "limitations": limitations, "blockers": blockers, "ready": not blockers}


def capture_gpu_smoke(root, run_id):
    """Record a managed model readiness probe scoped to one frozen MIG UUID."""
    from application.execution.run_plan import load_plan
    from core.config import assert_frozen
    from hidden.backends.vllm.engine import serving_interpreter
    from runtime.serving.resources import inspect_devices

    loaded = load_plan(root, run_id)
    cfg = loaded["config"]
    assert_frozen(cfg)
    if cfg["serving"]["management"] != "managed":
        raise ValueError("GPU smoke requires a run-frozen managed model endpoint")
    devices = cfg["serving"]["gpus"]
    if len(devices) != 1 or not isinstance(devices[0], str) or not devices[0].startswith("MIG-"):
        raise ValueError("GPU smoke only accepts exactly one MIG UUID")
    identity = inspect_devices(serving_interpreter(cfg), devices, probe=True)
    if identity["device_count"] != 1 or identity["device_kind"] != "mig":
        raise ValueError("Configured serving device is not one MIG instance")
    shared = any(row["process_count"] for row in identity["device_memory"])
    if shared and not cfg["serving"].get("allow_mig_sharing", False):
        raise ValueError("Configured MIG instance is not exclusively available")
    import os

    import requests
    token = os.environ.get(cfg["serving"]["api_key_env"], "EMPTY")
    response = requests.get(cfg["serving"]["base_url"].rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + token}, timeout=12)
    response.raise_for_status()
    models = sorted(row.get("id") for row in response.json().get("data", []))
    if cfg["model"]["served_name"] not in models:
        raise ValueError("Managed MIG endpoint does not serve the frozen model identity")
    record = {"schema_version": 1, "run_id": run_id, "model_fingerprint": cfg["model"]["fingerprint"],
        "serving_devices": identity, "served_models": models,
        "probe": "GET /v1/models; no generation requested", "agent_oracle_verifier_exercised": False,
        "mig_shared": shared,
        "at": now(), "status": "passed"}
    target = loaded["run"] / "validation/mig-serving-smoke.json"
    if target.exists():
        existing = read_json(target)
        if any(existing.get(k) != record[k] for k in (
                "run_id", "model_fingerprint", "serving_devices", "served_models", "probe", "status")):
            raise FileExistsError("MIG smoke receipt is immutable")
        return existing
    write_json(target, record)
    return record


def run_oracles(root_cfg, run_id, *, offline=False, workers=1, task_ids=None, retry_failed=False):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("Oracle workers must be between 1 and 16")
    return asyncio.run(_run_oracles(root_cfg, run_id, offline=offline, workers=workers,
                                    task_ids=task_ids, retry_failed=retry_failed))


def _oracle_valid(rewards, agent_error, verifier_error, cleanup_confirmed):
    return (bool(rewards) and not agent_error and not verifier_error and cleanup_confirmed
            and all(not isinstance(v, bool) and isinstance(v, (int, float)) and v == 1
                    for v in rewards.values()))


def _oracle_attempt(output, key, package_sha256, context_sha256, retry_failed):
    """Failed results are immutable; retries get an independently owned sandbox."""
    previous = [output / "result.json"] + sorted(output.glob("attempts/attempt-*/result.json"))
    previous = [p for p in previous if p.is_file()]
    if previous:
        row = read_json(previous[-1])
        if (row.get("task_key") != key or row.get("task_package_sha256") != package_sha256
                or row.get("execution_context_sha256", context_sha256) != context_sha256):
            raise ValueError("Existing oracle receipt belongs to another snapshot")
        if row.get("valid") or not retry_failed:
            return previous[-1].parent, row
    if (output / "result.json").exists() or any(output.iterdir()):
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        number = len(list(attempts.glob("attempt-*"))) + 1
        target = attempts / f"attempt-{number:03d}"
        target.mkdir(exist_ok=False)
        return target, None
    return output, None


async def _run_oracles(root_cfg, run_id, *, offline=False, workers=1, task_ids=None, retry_failed=False):
    from application.execution.run_plan import load_plan
    from core.backends import identity
    from core.locking import file_lock
    from runtime.agents.openhands import check_version

    loaded = load_plan(root_cfg["project_root"], run_id)
    cfg, run = loaded["config"], loaded["run"]
    if offline and (cfg["execution_context"]["profile"].get("network") != "none"
                    or not cfg["execution_context"]["profile"].get("dependency_snapshot")):
        raise ValueError("Offline acceptance requires network=none and a locked dependency snapshot")
    if identity(cfg) != ("skillsbench", "openhands"):
        raise ValueError("Oracle validation requires a frozen SkillsBench/OpenHands run")
    check_version()
    tasks = cfg["selection"]["tasks"]
    selected = set(task_ids or [t["instance_id"] for t in tasks.values()])
    unknown = selected - {t["instance_id"] for t in tasks.values()}
    if unknown:
        raise ValueError("Oracle tasks absent from frozen selection: " + str(sorted(unknown)))
    base = run / "validation" / ("offline-oracle" if offline else "oracle")
    base.mkdir(parents=True, exist_ok=True)
    with file_lock(base / ".sweep.lock", nonblocking=True):
        results = []
        semaphore = asyncio.Semaphore(workers)

        async def execute(key, task):
            async with semaphore:
                row = await _run_oracle_task(copy.deepcopy(cfg), run_id, key, task, base,
                                             offline=offline, retry_failed=retry_failed)
                results.append(row)
                summary = {"schema_version": 1, "run_id": run_id, "selected_count": len(selected),
                    "completed_count": len(results), "valid_oracles": sum(bool(r["valid"]) for r in results),
                    "results": sorted(results, key=lambda r: r["task_id"]), "updated_at": now()}
                write_json(base / "progress.json", summary)
                print(json.dumps({"oracle_task": row["task_id"], "valid": row["valid"],
                    "completed": len(results), "selected": len(selected)}, ensure_ascii=False), flush=True)
                return row

        network = None
        try:
            if cfg["execution_context"]["profile"].get("use_host_proxy"):
                network = SkillsBenchFactory(copy.deepcopy(cfg), run_id)
                await asyncio.to_thread(network.lock_images, [])
                cfg["skillsbench_network_proxy"] = network.cfg["skillsbench_network_proxy"]
            await asyncio.gather(*(execute(k, t) for k, t in sorted(tasks.items(),
                key=lambda item: item[1]["instance_id"]) if t["instance_id"] in selected))
        finally:
            if network is not None:
                await asyncio.to_thread(network.cleanup_owned, run_id)
        summary = {"run_id": run_id, "oracle_count": len(results),
            "valid_oracles": sum(bool(r["valid"]) for r in results),
            "results": sorted(results, key=lambda r: r["task_id"]),
            "capability_scope": "official oracle validation only; agent capabilities are unchanged"}
        write_json(base / "summary.json", summary)
        return summary


async def _run_oracle_task(cfg, run_id, key, task, base, *, offline=False, retry_failed=False):
    cfg = copy.deepcopy(cfg)
    from benchflow.rollout_planes import DefaultRolloutPlanes

    from catalog.skillsbench import verify_package

    output = base / task["instance_id"]
    output.mkdir(parents=True, exist_ok=True)
    context_sha256 = cfg["execution_context"]["context_sha256"]
    output, previous = _oracle_attempt(output, key, task["instance"]["package_sha256"],
                                       context_sha256, retry_failed)
    if previous is not None:
        return previous
    oracle_id = "oracle-" + hash_json({"run": run_id, "task_key": key,
                                       "attempt": str(output.relative_to(base))})[:20]
    row = {"schema_version": 1, "run_id": run_id, "task_key": key,
        "task_id": task["instance_id"], "benchmark_revision": task["revision"],
        "task_package_sha256": task["instance"]["package_sha256"],
        "execution_context_sha256": context_sha256, "agent": "oracle", "verifier": "BenchFlow-0.6.3",
        "rewards": {}, "verifier_error": None, "agent_error": None, "valid": False,
        "re_evaluation": False, "created_at": now(), "cleanup_confirmed": False,
        "attempt_id": oracle_id, "evidence_path": str(output)}
    factory, rollout, planes = None, None, None
    try:
        capability = verify_package(task)
        validation_only = cfg["execution_context"].get("validation_only", False)
        if not capability["supported"] and not validation_only:
            raise ValueError("Oracle task capability refused: " + task["instance_id"])
        if capability["sandbox"].get("gpus", 0) or task["instance"]["metadata"].get("user"):
            raise ValueError("GPU and interactive-user oracle lifecycles require separate qualification")
        factory = SkillsBenchFactory(cfg, run_id)
        await asyncio.to_thread(factory.lock_images, [task])
        image = factory.images[(task["instance_id"], "main")]
        row["image"] = image["identity"]
        row["service_images"] = {service: value["identity"] for (iid, service), value in factory.images.items()}
        cfg["skillsbench_service_images"] = {service: value["tag"]
            for (iid, service), value in factory.images.items()}
        cfg["skillsbench_daemon_capabilities"] = {"cpu_quota_supported": factory.cpu_quota_supported}
        task_source = Path(task["instance"]["task_path"])
        manifest = EnvironmentManifest(name=task["instance_id"], image=image["tag"],
            keep_alive=True, isolation="per_task", owns_lifecycle=True)
        sample = type("OracleSample", (), {"instance_id": task["instance_id"],
            "sample_key": oracle_id, "task_key": key})()
        planes = OwnedPlanes(DefaultRolloutPlanes(), cfg, run_id, sample, oracle_id, image, output)
        planes.task_source = task_source
        mode = cfg["execution_context"]["profile"]["skill_mode"]
        skills_dir = task_source / "environment/skills" if mode == "with-skill" else None
        hooks = []
        if offline:
            from runtime.agents.offline import install_probe
            async def check_offline(env):
                await install_probe(env, factory.client, planes, cfg, output)
            hooks.append(check_offline)
        config = RolloutConfig(task_path=task_source, agent="oracle", model=None,
            skill_mode=mode, skills_dir=skills_dir,
            sandbox_user=cfg['execution_context']['profile']['sandbox_user'] if offline else None,
            timeout=None,  # Keep the official task's oracle budget; verifier retains its own timeout.
            jobs_dir=output / "benchflow", job_name=oracle_id, rollout_name=oracle_id,
            environment_manifest=manifest, pre_agent_hooks=hooks,
            usage_tracking=UsageTrackingConfig(mode="off"), planes=planes)
        rollout = await Rollout.create(config)
        result = await rollout.run()
        row.update(rewards=result.rewards, verifier_error=result.verifier_error, agent_error=result.error)
    except Exception as exc:
        import traceback
        row["agent_error"] = repr(exc)
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if rollout is not None and not (output / "sandbox-cleanup.json").exists():
            try:
                await rollout.cleanup()
            except Exception as exc:
                row["cleanup_error"] = repr(exc)
        if factory is not None:
            try:
                if planes is not None:
                    remaining = factory.client.containers.list(all=True, filters={
                        "label": ["dycon.run=" + run_id, "dycon.epoch=" + planes.ownership["epoch"]]})
                    row["remaining_containers"] = [c.id for c in remaining]
                    row["cleanup_confirmed"] = (output / "sandbox-cleanup.json").exists() and not remaining
            except Exception as exc:
                row["cleanup_error"] = repr(exc)
            try:
                await asyncio.to_thread(factory.cleanup_owned, run_id)
            except Exception as exc:
                row["cleanup_error"] = repr(exc)
                row["cleanup_confirmed"] = False
    row["valid"] = _oracle_valid(row["rewards"], row["agent_error"], row["verifier_error"],
                                  row["cleanup_confirmed"])
    if offline:
        probe_path = output / "offline-install.json"
        probe = read_json(probe_path) if probe_path.exists() else {}
        row.update(offline=True, dependency_probe_sha256=sha256(probe_path) if probe_path.exists() else None,
                   offline_install_verified=probe.get("verified", False))
        row["valid"] = row["valid"] and row["offline_install_verified"]
    row["finished_at"] = now()
    row["receipt_sha256"] = hash_json(row)
    write_json(output / "result.json", row)
    return row


def execute_attempt(cfg, run_id, task, runtime, sample, context, output, factory):
    return asyncio.run(_execute_attempt(cfg, run_id, task, runtime, sample, context, output, factory))


def drain_capture(cfg, output, capture_client, session_id):
    """One producer barrier and native/hidden drain for every terminal path."""
    import time

    import requests

    endpoint = cfg['serving']['base_url'].removesuffix('/v1').rstrip('/')
    started_at, started = now(), time.monotonic()
    receipt = {'schema_version': 1, 'session_id': session_id, 'started_at': started_at,
               'phase': 'native_flush'}
    try:
        flush_started = time.monotonic()
        response = requests.post(endpoint + '/dycon/capture/engine/flush', json={'session_id': session_id},
            timeout=cfg['capture'].get('flush_timeout_seconds') or cfg['capture']['finalize_seconds'])
        response.raise_for_status()
        value = response.json()
        receipt['native_flush_seconds'] = time.monotonic() - flush_started
        if value.get('failures'):
            raise RuntimeError('Native capture drain failed: ' + str(value['failures']))
        receipt['native_flush'] = value
        receipt['phase'] = 'session_stop'
        stop_started = time.monotonic()
        # Stop can wait for native output and export queues sequentially. The
        # normal control-request timeout must not expire before these bounded
        # drains can finish and persist their diagnostic. No producers remain.
        stop_timeout = max(cfg['capture'].get('control_seconds', 10),
                           cfg['capture'].get('finalize_request_seconds') or 0,
                           2 * cfg['capture']['finalize_seconds'] + 30)
        receipt['session_stop_timeout_seconds'] = stop_timeout
        previous_timeout = getattr(capture_client, 'timeout', None)
        try:
            capture_client.timeout = stop_timeout
            stopped = capture_client.stop(session_id)
        finally:
            capture_client.timeout = previous_timeout
        receipt.update(session_stop=stopped, session_stop_seconds=time.monotonic() - stop_started,
            order=['bridge_closed', 'inflight_completed_or_cancelled', 'native_and_hidden_drained', 'session_stopped'],
            elapsed_seconds=time.monotonic() - started, phase='complete', finished_at=now())
        write_json(Path(output) / 'capture-drain.json', receipt)
    except Exception as exc:
        receipt.update(failed=True, error_type=type(exc).__name__, error=_safe_verifier_text(str(exc))[:512],
            elapsed_seconds=time.monotonic() - started, finished_at=now())
        write_json(Path(output) / 'capture-drain.json', receipt)
        raise


def backend_failure_result(sample, context, output, started, error, *, bridge=None,
                           capture_session=None, agent_status=None, persist=True):
    """Retain independent agent/verifier/usage facts if finalization fails."""
    output = Path(output)
    if getattr(bridge, 'limit_reason', None) == 'sample_completion_token_budget':
        agent_status = 'SampleTokenBudgetExceeded'
    elif getattr(bridge, 'limit_reason', None) == 'sample_token_usage_unknown':
        agent_status = 'SampleTokenBudgetUsageUnknown'
    calls = [call for record in getattr(bridge, 'records', []) for call in record.get('model_calls', [])]
    known = [call.get('usage') or {} for call in calls if call.get('usage_known')]
    dependency = read_json(output / 'dependency-failure.json') if (output / 'dependency-failure.json').exists() else None
    official = read_json(output / 'evaluation-raw.json') if (output / 'evaluation-raw.json').exists() else {}
    resolved = None
    rewards = official.get('rewards')
    if (official.get('completed') is True and isinstance(rewards, dict) and rewards
            and all(type(v) in (int, float) and 0 <= v <= 1 for v in rewards.values())):
        resolved = all(v == 1 for v in rewards.values())
    if dependency:
        status = 'EnvironmentSetupFailed'
    elif agent_status in {'Submitted', 'AgentCompleted', 'AgentTimeout', 'OpenHandsError',
                          'LimitsExceeded', 'reasoning_loop',
                          'EnvironmentSetupFailed', 'TimeoutError', 'VerifierError'}:
        status = agent_status
    elif agent_status in {'SampleTokenBudgetExceeded', 'SampleTokenBudgetUsageUnknown'}:
        status = agent_status
    elif isinstance(error, TimeoutError):
        status = 'TimeoutError'
    elif 'verifier' in type(error).__name__.lower():
        status = 'VerifierError'
    elif 'acp' in (type(error).__name__ + ' ' + str(error)).lower():
        status = 'OpenHandsError'
    else:
        status = type(error).__name__
    capture_errors = []
    capture_error_paths = [output / 'capture-finalization-error.json',
                           output / 'capture-stop-error.json']
    for capture_error_path in capture_error_paths:
        if capture_error_path.exists():
            capture_errors.append(read_json(capture_error_path).get('error', 'capture finalization failed'))
    missing_receipts = [
        {'request_id': record.get('request_id'),
         'api_attempt_index': call.get('api_attempt_index'),
         'status': call.get('status')}
        for record in getattr(bridge, 'records', [])
        for call in record.get('model_calls', []) if not call.get('usage_known')]
    result = {**sample.as_dict(), **context, 'exit_status': status, 'agent_exit_status': agent_status,
        'capture_status': 'partial' if capture_session else 'disabled', 'capture_session_id': capture_session,
        'capture_errors': capture_errors, 'disposition': 'failed', 'error': repr(error),
        'started_at': started, 'finished_at': now(),
        'raw_reward': rewards, 'resolved': resolved,
        'evaluation_status': 'raw_verifier_persisted' if official.get('completed') else 'not_run',
        'usage': {'prompt_tokens': sum(r.get('prompt_tokens', 0) for r in known),
                  'completion_tokens': sum(r.get('completion_tokens', 0) for r in known),
                  'cost': None, 'complete': len(known) == len(calls),
                  'known_model_calls': len(known), 'model_calls': len(calls),
                  'missing_model_call_receipts': missing_receipts},
        'failure_type': 'dependency_error' if dependency else
            'capture_finalization_error' if capture_errors else 'backend_error'}
    result['termination_evidence'] = _terminal_evidence(output, agent_status=agent_status,
        capture_status=result['capture_status'], evaluation_status=result['evaluation_status'], error=error)
    if dependency:
        result['dependency_failure'] = dependency
    if official:
        result['raw_evaluation_sha256'] = sha256(output / 'evaluation-raw.json')
    if persist:
        write_json(output / 'attempt.json', result)
    return result


async def _execute_attempt(cfg, run_id, task, runtime, sample, context, output, factory):
    # Image/service bindings and daemon capabilities belong to this attempt.
    cfg = copy.deepcopy(cfg)
    from benchflow.rollout_planes import DefaultRolloutPlanes

    from application.execution.subject import journal, verify_subject
    from catalog.skillsbench import verify_package
    from core.model_identity import model_identity_verification_enabled, verify_model
    from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
    from rollout.adapters.openhands import canonical_trajectory
    from rollout.adapters.openhands_bridge import OpenHandsBridge
    from rollout.progress import generation_phase

    output = Path(output)
    run = run_path(cfg, run_id)
    output.mkdir(parents=True, exist_ok=True)
    from rollout.retry import publish_once
    publish_once(output / 'attempt-started.json', {
        'attempt_id': context['attempt_id'], 'started_at': now(),
        'retry_kind': runtime.get('retry_kind', 'initial'),
        'execution_attempt_id': os.environ.get('DYCON_EXECUTION_ATTEMPT')})
    if cfg["execution_context"].get("validation_only"):
        raise ValueError("Validation-only task snapshot cannot execute an agent rollout")
    source_task = task["instance"]
    capability = verify_package(task)
    if not capability["supported"]:
        raise ValueError("Unsupported SkillsBench task: " + "; ".join(capability["reasons"]))
    official_root = (Path(cfg["project_root"]) / cfg["skillsbench_images"]["source_root"]
                     / "tasks").resolve()
    task_source = Path(source_task["task_path"]).resolve()
    if not task_source.is_relative_to(official_root) or task_source.parent != official_root:
        raise ValueError("Task package escaped the pinned official task root")
    unit = next(u for u in source_task["images"] if u["service"] == "main")
    image = factory.images[(task["instance_id"], "main")]
    # Resolve the leased NAS tag explicitly in the prebuilt compose overlay.
    cfg["skillsbench_service_images"] = {"main": image["tag"]}
    if model_identity_verification_enabled(cfg):
        verify_model(cfg)
    max_seconds = cfg["timeouts"].get("agent_seconds") or cfg["timeouts"]["container_lifetime_seconds"]
    deadline = Deadline(min(max_seconds, cfg["timeouts"]["container_lifetime_seconds"]))
    cfg["skillsbench_daemon_capabilities"] = {"cpu_quota_supported": factory.cpu_quota_supported}
    binding = {**context, "benchmark": "skillsbench", "agent": "openhands",
               "session_id": "oh-" + uuid.uuid4().hex,
               "profile_sha256": cfg["execution_context"]["profile_sha256"]}
    import importlib.metadata
    import sys

    from rollout import sample_token_budget as budget_module
    from rollout.adapters import openhands_bridge as bridge_module
    from runtime.agents import dependencies as dependency_module
    from runtime.agents import openhands as driver_module
    from runtime.agents import verifier_audit as verifier_audit_module
    from runtime.agents import verifier_tools as verifier_tools_module
    source_modules = [Path(__file__), Path(driver_module.__file__),
                      Path(dependency_module.__file__), Path(bridge_module.__file__),
                      Path(budget_module.__file__), Path(verifier_tools_module.__file__),
                      Path(verifier_audit_module.__file__)]
    dependency_ref = cfg['execution_context']['profile'].get('dependency_snapshot')
    dependency_path = Path(dependency_ref) if dependency_ref else None
    write_json(output / 'implementation-identity.json', {
        'schema_version': 1, 'binding': binding, 'observed_at': now(),
        'python': sys.executable, 'benchflow_version': importlib.metadata.version('benchflow'),
        'source_files': {str(path): sha256(path) for path in source_modules},
        'dependency_snapshot': str(dependency_path) if dependency_path else None,
        'dependency_manifest_sha256': sha256(dependency_path / 'manifest.json') if dependency_path else None,
        'execution_attempt_id': runtime.get('execution_attempt_id', os.environ.get('DYCON_EXECUTION_ATTEMPT')),
        'execution_binding': cfg.get('execution_binding'),
        'sdk_effective_timeout': None, 'sdk_request_phase': 'unknown',
        'timeout_evidence_policy': 'configured_driver_budget_is_not_actual_sdk_request_evidence'})
    capture, capture_client, capture_session, capture_status = None, None, None, "disabled"
    capture_errors = []
    started = now()
    try:
        if cfg.get("capture", {}).get("enabled"):
            inventory = runtime["capture_token_inventory"]
            capture = capture_request_metadata("cap-" + sample.sample_key + "-" + context["attempt_id"],
                task["instance_id"], run_id, "pending-" + context["attempt_id"], 0,
                cfg["capture"]["anchors"], newline_token_ids=inventory["newline_token_ids"],
                close_token_ids=inventory["close_token_ids"], tokenizer_fingerprint=inventory["tokenizer_fingerprint"],
                newline_policy=cfg["capture"]["newline_policy"], capture_select=inventory["capture_select"])
            capture_session = capture["session_id"]
            capture_client = CaptureClient(runtime["capture_control_url"], timeout=cfg["capture"]["control_seconds"])
            capture_client.start(capture, graph_mode="FULL", prefix_cache_policy="safe_boundary")
            capture_status = "collecting"
            write_json(output / "capture-session.json", {"session_id": capture_session, "metadata": capture})

        from rollout.adapters.openhands_bridge import OpenHandsBridge
        bridge_cfg = copy.deepcopy(cfg)
        bridge_cfg["generation"]["seed"] = context["effective_seed"]
        with OpenHandsBridge(bridge_cfg, binding, cfg["execution_context"]["profile"], output, deadline, runtime, capture) as bridge:
            # LiteLLM requires an explicit provider prefix for OpenAI-compatible
            # endpoints. Keep the vLLM served name unchanged in request bodies.
            from runtime.agents.openhands import driver_environment
            env = driver_environment(cfg, bridge.base_url, bridge.secret, deadline)
            from runtime.agents.openhands import check_version
            check_version()
            skill_mode = cfg["execution_context"]["profile"]["skill_mode"]
            bundled_skills = task_source / "environment/skills"
            skills = bundled_skills if skill_mode == "with-skill" and bundled_skills.is_dir() else None
            manifest = EnvironmentManifest(name=task["instance_id"], image=image["tag"],
                                            keep_alive=True, isolation="per_task", owns_lifecycle=True)
            planes = OwnedPlanes(DefaultRolloutPlanes(), cfg, run_id, sample, context["attempt_id"],
                                 image, output)
            planes.task_source = task_source
            write_json(output / "image-resolution.json", {
                "schema_version": 1, "task_id": task["instance_id"],
                "service": "main", "nas_identity": image["identity"],
                "resolved_compose_image": image["tag"], "build": None,
                "source": "nas-lease-tag", "verified": True, "at": now()})
            rollout_cfg = agent_rollout_config(cfg, task_source, image, planes, output, env,
                attempt_id=context['attempt_id'], session_id=binding['session_id'],
                seconds=int(deadline.remaining(max_seconds)))
            rollout = await Rollout.create(rollout_cfg)
            with generation_phase(runtime):
                from runtime.agents.lifecycle import run_cancellable
                result = await run_cancellable(rollout, runtime.get("cancellation"), deadline,
                    cfg["execution_context"]["profile"]["cleanup_seconds"],
                    evidence_path=output / 'diagnostics/process-supervisor.json', binding=binding)

        result_row = _result_json(result)
        # This diagnostic is separate from immutable result/terminal receipts.
        # Public exit status deliberately retains the frozen retry semantics.
        cause = {"schema_version": 1, "binding": binding,
                 "category": result_row.get("error_category"),
                 "public_status_policy": "unchanged_v1", "cause": "unknown"}
        observed = output / "diagnostics/acp-error.json"
        if observed.exists():
            cause.update(read_json(observed))
        else:
            for event in _json_events(result.trajectory):
                text = json.dumps(event, ensure_ascii=False)
                if "litellm.Timeout" in text or "APITimeoutError" in text:
                    cause.update(cause="sdk_api_timeout", exception_type="APITimeoutError",
                                 source="actual_acp_trajectory", phase="unknown")
                    break
        sdk_files = sorted((output / "diagnostics").glob("sdk-transport-sdk-*.json"))
        local_sdk = output / "diagnostics/sdk-local-transport.json"
        if local_sdk.exists(): sdk_files.append(local_sdk)
        cause["sdk_transport_receipts"] = [{"path": str(p.relative_to(output)), "sha256": sha256(p)}
                                           for p in sdk_files]
        for path in sdk_files:
            sdk_row = read_json(path)
            for event in sdk_row.get("events", []):
                if event.get("event") == "sdk_call_failed":
                    types = event.get("exception_types", [])
                    if any(t.rsplit(".", 1)[-1] in {"Timeout", "APITimeoutError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "LLMTimeoutError"} for t in types):
                        cause.update(cause="sdk_api_timeout", phase=event.get("phase", "unknown"),
                                     sdk_request_id=event.get("sdk_request_id", sdk_row.get("sdk_request_id")),
                                     source="actual_sdk_http_transport", exception_types=types)
        write_json(output / "benchflow-diagnostics.json", cause)
        write_json(output/'requests/receipt.json',{'schema_version':1,'binding':binding,
            'files':{p.name:sha256(p) for p in sorted((output/'requests').glob('req-*.json'))},
            'scope':'authenticated original/effective requests and all generated candidate usage'})
        result_row = json.loads(json.dumps(result_row, default=lambda value: (
            value.isoformat() if hasattr(value, "isoformat") else repr(value))))
        write_json(output / "benchflow/result.json", result_row)
        write_json(output / "openhands/acp-events.json", {"schema_version": 1,
            "events": _json_events(result.trajectory), "trajectory_source": result.trajectory_source})
        termination = ("AgentCompleted" if not result.error else
            "AgentTimeout" if "timeout" in str(result.error_category).lower() else "OpenHandsError")
        if result.error and bridge.limit_reason:
            if bridge.limit_reason == 'sample_completion_token_budget':
                termination = 'SampleTokenBudgetExceeded'
            elif bridge.limit_reason == 'sample_token_usage_unknown':
                termination = 'SampleTokenBudgetUsageUnknown'
            else:
                termination = 'reasoning_loop' if bridge.limit_reason == 'reasoning_loop' else 'LimitsExceeded'
        dependency_failure = read_json(output / 'dependency-failure.json') if (output / 'dependency-failure.json').exists() else None
        if dependency_failure:
            termination = 'EnvironmentSetupFailed'
        record_agent_outcome(run, output, context, termination)
        report_path = output / "evaluation-raw.json"
        report = read_json(report_path) if report_path.exists() else {
            "schema_version": 1, "completed": False,
            "error": result.verifier_error or "Verifier not run", "rewards": None, "finished_at": now()}
        canonical = canonical_trajectory(task["instance_id"], binding, bridge.records,
                                         _json_events(result.trajectory), termination)
        canonical["info"].update({"rewards": report.get("rewards"), "verifier_error": result.verifier_error,
            "agent_error": result.error, "agent_usage": result_row,
            "skills": {"mode": skill_mode, "bundle_sha256": hash_json([
                {"path": p.relative_to(skills).as_posix(), "sha256": sha256(p)}
                for p in sorted(skills.rglob("*")) if p.is_file()]) if skills else None}})
        raw_path = output / "openhands/canonical.traj.json"
        write_json(raw_path, canonical)
        if capture_client is not None and capture_session:
            drain_capture(cfg, output, capture_client, capture_session)
            prepared = capture_client.prepare(capture_session, raw_path)
            write_json(output / "capture-prepared.json", prepared)
        trajectory_sha = sha256(raw_path)
        subject_path = getattr(planes, "subject_path", None)
        if subject_path is None:
            subject_path = output / "subject/subject.json"
            write_json(subject_path, {"schema_version": 1, "kind": "workspace", "binding": binding,
                "task_package_sha256": source_task["package_sha256"], "image_identity": image["identity"],
                "outputs": [], "missing_outputs": cfg["execution_context"]["profile"]["output_paths"].get(task["instance_id"], []),
                "complete": False, "re_evaluation": False,
                "re_evaluation_reason": "Live sandbox verification was not completed"})
        subject_sha = sha256(subject_path)
        journal(output, "agent_terminal", binding, terminal=termination,
                trajectory_sha256=trajectory_sha, request_count=len(bridge.records))
        journal(output, "subject_bound", binding, subject_sha256=subject_sha,
                subject_path=str(subject_path))
        result_status, resolved, policy = _map_evaluation(result_row, report, capture_status)
        if result.error and termination in {'LimitsExceeded', 'reasoning_loop', 'EnvironmentSetupFailed'}:
            result_status = termination
        if termination in {'SampleTokenBudgetExceeded', 'SampleTokenBudgetUsageUnknown'}:
            result_status = termination
        result_status, resolved, policy, verifier_audit = verifier_audit_module.audit_zero_evaluation(
            output, task, result_status, resolved, policy, report)
        if verifier_audit is not None:
            write_json(output / 'diagnostics/verifier-zero-audit.json', verifier_audit)
        evaluation_id = "sb-" + hash_json({"attempt": context["attempt_id"],
            "report": sha256(report_path) if report_path.exists() else None})[:24]
        evaluation_report = output / "evaluation-report.json"
        outcome = {"subject_sha256": subject_sha,
                   "status": "completed" if resolved is not None else "completed_unknown",
                   "resolved": resolved, "rewards": report.get("rewards"), "success_policy": policy}
        if report_path.exists():
            write_json(evaluation_report, {"schema_version": 1, "evaluation_id": evaluation_id,
                "attempt_id": context["attempt_id"], "subject_sha256": subject_sha,
                "raw_rewards": report.get("rewards"), "evaluation_status": outcome["status"],
                "resolved": resolved, "success_policy": policy, "report_sha256": sha256(report_path),
                "verifier_zero_audit": verifier_audit,
                "evaluated_at": now()})
        from rollout.artifacts import import_rollout
        rid = import_rollout(cfg["project_root"], {"source": str(raw_path), "task": task,
            "run_id": run_id, "attempt_id": context["attempt_id"], "purposes": ["baseline"],
            "agent": "openhands",
            "model_fingerprint": cfg["model"]["fingerprint"], "sample": sample.as_dict(),
            "attempt_context": context, "run_local": True, "subject": str(subject_path),
            "subject_sha256": subject_sha, "evaluation": outcome if report_path.exists() else None,
            "evaluation_id": evaluation_id if report_path.exists() else None,
            "evaluation_report": str(evaluation_report) if report_path.exists() else None})
        if capture_session and capture_client:
            try:
                finalized = capture_client.finalize(capture_session, rollout_id=rid,
                    trajectory_path=str(raw_path), trajectory_sha256=trajectory_sha,
                    model_fingerprint=cfg["model"]["fingerprint"])
            except Exception as exc:
                write_json(output / "capture-finalization-error.json",
                           {"error": _safe_verifier_text(repr(exc))[:512], "error_type": type(exc).__name__, "at": now()})
                raise
            capture_status = finalized["capture_status"]
            write_json(output / "capture-finalization.json", finalized)
            if capture_status not in {"complete", "not_exported"}:
                capture_errors = finalized.get('errors') or finalized.get('failures') or [
                    'capture finalization returned ' + str(capture_status)]
        journal(output, "artifacts_committed", binding, rollout_id=rid,
                trajectory_sha256=trajectory_sha, subject_sha256=subject_sha,
                report_sha256=sha256(report_path) if report_path.exists() else None,
                capture_status=capture_status)
        subject_manifest = read_json(subject_path)
        row = {"evaluation_subject": {"kind": "workspace"}, "trajectory_path": str(raw_path),
            "trajectory_sha256": trajectory_sha, "subject_path": str(subject_path), "subject_sha256": subject_sha}
        verify_subject(Path(cfg["project_root"]), row)
        cleanup = output / "sandbox-cleanup.json"
        cleanup_ok = cleanup.is_file() and read_json(cleanup).get("status") == "stopped"
        if resolved is True and not subject_manifest.get("complete"):
            result_status = "SubmissionExportError"
        journal(output, "cleanup_confirmed", binding, confirmed=cleanup_ok,
                cleanup_receipt_sha256=sha256(cleanup) if cleanup_ok else None)
        journal(output, "settled", binding, rollout_id=rid, raw_reward=report.get("rewards"),
                resolved=resolved, capture_status=capture_status, cleanup_confirmed=cleanup_ok)
        if not cleanup_ok:
            result_status = "CleanupError"
        result = {**sample.as_dict(), **context, "instance_id": sample.instance_id,
            "attempt_id": context["attempt_id"], "rollout_id": rid, "exit_status": result_status,
            "agent_exit_status": termination,
            "disposition": "accepted" if result_status in {"Submitted", "AgentCompleted", "AgentTimeout"} else "failed",
            "raw_reward": report.get("rewards"), "resolved": resolved,
            "evaluation_status": outcome["status"] if report_path.exists() else "not_run",
            "evaluation_id": evaluation_id if report_path.exists() else None,
            "evaluation_policy": policy, "evaluation_report_sha256": sha256(evaluation_report) if report_path.exists() else None,
            "verifier_zero_audit": verifier_audit,
            "subject_sha256": subject_sha, "capture_status": capture_status,
            "capture_errors": capture_errors,
            "capture_session_id": capture_session, "cost": result_row["cost_usd"],
            "input_tokens": result_row["n_input_tokens"], "output_tokens": result_row["n_output_tokens"],
            "timings": {"duration_seconds": 0}, "started_at": started, "finished_at": now(),
            "failure_type": "unsupported_reward_policy" if result_status == "UnsupportedRewardPolicy" else
                "verifier_incomplete" if result_status == "VerifierError" else
                "capture_finalization_error" if capture_errors else None,
            "can_reevaluate": False, "re_evaluation_reason": subject_manifest["re_evaluation_reason"]}
        calls = [call for record in bridge.records for call in record.get('model_calls', [])]
        known = [call.get('usage') or {} for call in calls if call.get('usage_known')]
        result['usage'] = {'prompt_tokens': sum(u.get('prompt_tokens', 0) for u in known),
                           'completion_tokens': sum(u.get('completion_tokens', 0) for u in known),
                           'cost': result_row.get('cost_usd'),
                           'complete': len(known) == len(calls), 'known_model_calls': len(known),
                           'model_calls': len(calls),
                           'missing_model_call_receipts': [
                               {'request_id': record.get('request_id'),
                                'api_attempt_index': call.get('api_attempt_index'),
                                'status': call.get('status')}
                               for record in bridge.records for call in record.get('model_calls', [])
                               if not call.get('usage_known')],
                           'source': 'all_physical_model_call_receipts'}
        result['input_tokens'], result['output_tokens'] = result['usage']['prompt_tokens'], result['usage']['completion_tokens']
        result['termination_evidence'] = _terminal_evidence(output, agent_status=termination,
            capture_status=capture_status,
            evaluation_status=outcome['status'] if report_path.exists() else 'not_run')
        result['sample_token_budget'] = bridge.token_budget.snapshot()
        if dependency_failure:
            result.update(dependency_failure=dependency_failure, failure_type='dependency_error',
                          exit_status='EnvironmentSetupFailed', disposition='failed')
        if bridge.limit_reason:
            result['limit_reason'] = bridge.limit_reason
            result['loop_detection'] = bridge.loop_hit
        from rollout.retry import disposition
        result['disposition'] = ('failed' if dependency_failure else
            'blocked' if capture_status not in {'complete', 'not_exported', 'disabled'}
            and cfg['capture']['failure_policy'] == 'block_rollout' else
            disposition(cfg, result['exit_status']))
        write_json(output / "attempt.json", result)
        return result
    except Cancelled as exc:
        if capture_client is not None and capture_session:
            try:
                drain_capture(cfg, output, capture_client, capture_session)
            except Exception as cleanup_error:
                write_json(output / "capture-stop-error.json", {"error": repr(cleanup_error)})
        partial = backend_failure_result(sample, context, output, started, exc,
            bridge=locals().get('bridge'), capture_session=capture_session,
            agent_status=locals().get('termination'), persist=False)
        interrupted = {**partial, "exit_status": "Cancelled",
            "disposition": "interrupted", "capture_status": "partial" if capture_session else "disabled",
            "capture_session_id": capture_session, "error": str(exc), "finished_at": now()}
        interrupted['failure_type'] = 'cancelled'
        write_json(output / "attempt.json", interrupted)
        return interrupted
    except BaseException as exc:
        if capture_client is not None and capture_session:
            try:
                drain_capture(cfg, output, capture_client, capture_session)
            except Exception as cleanup_error:
                write_json(output / "capture-stop-error.json", {"error": repr(cleanup_error)})
        write_json(output / "backend-error.json", {"error_type": type(exc).__name__,
            "error": str(exc), "binding": binding, "at": now()})
        if not isinstance(exc, Exception):
            raise
        return backend_failure_result(sample, context, output, started, exc,
            bridge=locals().get('bridge'), capture_session=capture_session,
            agent_status=locals().get('termination'))
