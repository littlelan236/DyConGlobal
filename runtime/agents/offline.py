"""Offline acceptance through BenchFlow's normal container and verifier lifecycle."""
import asyncio
import json
import shlex
from pathlib import Path

from core.io import hash_json, now, sha256, write_json
from runtime.agents.dependencies import install_runtime, verify_snapshot
from runtime.agents.verifier_tools import prepare_verifier_tools


async def install_probe(env, client, planes, cfg, output):
    profile = cfg["execution_context"]["profile"]
    snapshot = await asyncio.to_thread(verify_snapshot, profile["dependency_snapshot"])
    expected = profile.get('dependency_snapshot_sha256')
    if expected and sha256(Path(profile['dependency_snapshot']) / 'manifest.json') != expected:
        raise ValueError('OpenHands dependency snapshot differs from frozen execution context')
    containers = client.containers.list(filters={"label": [
        "dycon.run=" + planes.run_id, "dycon.epoch=" + planes.ownership["epoch"]]})
    if len(containers) != 1:
        raise ValueError("Offline sandbox ownership is ambiguous")
    container = containers[0]
    container.reload()
    mode = container.attrs["HostConfig"]["NetworkMode"]
    if mode != "none":
        raise ValueError("Offline install requires Docker NetworkMode=none")
    probe = (
        "import json; from pathlib import Path; "
        "interfaces=sorted(p.name for p in Path('/sys/class/net').iterdir()); "
        "assert interfaces==['lo'],interfaces; "
        "print(json.dumps({'interfaces':interfaces,'routes':Path('/proc/net/route').read_text()}))")
    executable = '/opt/dycon-wheels/python/bin/python3.12' if snapshot.get('agent_runtime') else 'python3'
    network = await env.exec(executable + " -c " + shlex.quote(probe), user="root", timeout_sec=30)
    if network.return_code:
        raise RuntimeError("Offline sandbox has unexpected network interfaces")
    await install_runtime(env, cfg, output, planes.ownership)
    # Some official verifier scripts run a uv installer before invoking uvx.
    # Preinstall the snapshot's hash-bound binary and normal shell environment;
    # keep the original verifier script and record this environment variant.
    if profile.get('verifier_resource_policy'):
        from runtime.agents.verifier_dependencies import task_environment
        variables=task_environment(profile,planes.sample.instance_id)
        tools=await prepare_verifier_tools(env,planes.sample.instance_id,variables)
    else:
        tools = await prepare_verifier_tools(env)
    inventory = await env.exec("cat /etc/os-release && "+executable+" --version && dpkg -l",
                               user="root", timeout_sec=60)
    if inventory.return_code:
        raise RuntimeError("Offline image package inventory failed")
    container.reload()
    if container.attrs["HostConfig"]["NetworkMode"] != "none":
        raise ValueError("Offline sandbox network changed during installation")
    proof = {"schema_version": 1, "run_id": planes.run_id, "ownership": planes.ownership,
             "container_id": container.id, "network_mode": mode, "network_probe": json.loads(network.stdout),
             "image_identity": planes.image["identity"], "system_packages": inventory.stdout,
             "system_packages_sha256": hash_json(inventory.stdout),
             "dependency_receipt_sha256": snapshot["receipt_sha256"],
             "dependency_install_sha256": sha256(output / "dependency-install.json"),
             "verifier_tool_preinstallation": tools.stdout,
             "environment_variant": "snapshot_uv_preinstalled_in_root_home_v1",
             "python": snapshot["python"], "platform": snapshot["platform"],
             "cache_disabled": True, "verified": True, "at": now()}
    proof["receipt_sha256"] = hash_json(proof)
    write_json(output / "offline-install.json", proof)
