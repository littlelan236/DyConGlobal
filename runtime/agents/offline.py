"""Offline acceptance through BenchFlow's normal container and verifier lifecycle."""
import json
import shlex

from core.io import hash_json, now, sha256, write_json
from runtime.agents.dependencies import install_runtime, verify_snapshot


async def install_probe(env, client, planes, cfg, output):
    profile = cfg["execution_context"]["profile"]
    snapshot = verify_snapshot(profile["dependency_snapshot"])
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
    tools = await env.exec(
        "mkdir -p /root/.local/bin && "
        "{ test -x /root/.local/bin/uv || cp /opt/dycon-wheels/uv /root/.local/bin/uv; } && "
        "{ test -x /root/.local/bin/uvx || "
        "printf '%s\\n' '#!/bin/sh' 'exec /opt/dycon-wheels/uv tool run \"$@\"' > /root/.local/bin/uvx; } && "
        "{ test -r /root/.local/bin/env || "
        "printf '%s\\n' 'export PATH=/root/.local/bin:$PATH' > /root/.local/bin/env; } && "
        "sha256sum /root/.local/bin/uv && /root/.local/bin/uv --version",
        user="root", timeout_sec=30)
    if tools.return_code:
        raise RuntimeError("Offline verifier uv preinstallation failed")
    inventory = await env.exec("cat /etc/os-release && python3 --version && dpkg -l",
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
