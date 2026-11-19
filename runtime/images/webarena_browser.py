"""Build, smoke-test and publish the isolated pinned WebArena browser image.

This is an image preparation API used by ``dycon webarena browser-prepare``.
It does not choose benchmark tasks, call an LLM or manufacture rollout evidence.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from core.io import hash_json, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.oci import descriptor_graph, platform_manifest
from runtime.images.ownership import json_blob, publish_blob

BROWSER_SOURCE = "dce04686a56253aefba7b18a4fa0937cf1dc987b"
BASE_IMAGE = "docker.io/library/python:3.10-slim-bookworm"


def _worker_update_compatible(previous, build_inputs):
    """Reuse dependencies only when every frozen dependency input still matches."""
    return (previous.get("selftest", {}).get("passed") is True
            and bool(previous.get("worker_sha256"))
            and all(previous.get(key) == value for key, value in build_inputs.items()
                    if key != "worker_sha256"))


def _prepare_base(cfg):
    """Resolve and transfer the base through the shared NAS registry transport."""
    from runtime.images.nas import NasImageStore
    from runtime.images.registry import image_metadata, transfer_blob
    namespace = Path(cfg["paths"]["nas_root"]) / "webarena-browser-python-base-3.10"
    blobs = namespace / "blobs/sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    lock_path = namespace / "source-lock.json"
    proxy = cfg.get("webarena_images", {}).get("host_proxy")
    with file_lock(namespace / ".prepare.lock", nonblocking=True):
        if lock_path.exists():
            lock = read_json(lock_path)
        else:
            root, descriptors = image_metadata("library/python", blobs,
                reference="3.10-slim-bookworm", proxy=proxy)
            selected, manifest = platform_manifest(blobs, root, cfg["environment"]["platform"])
            by_digest = {row["digest"]: row for row in descriptors}
            required = [by_digest[row["digest"]] for row in (selected, manifest["config"], *manifest["layers"])]
            lock = {"requested": BASE_IMAGE, "root": selected, "descriptors": required,
                    "platform": cfg["environment"]["platform"]}
            write_json(lock_path, lock)
        if lock["platform"] != cfg["environment"]["platform"]:
            raise ValueError("Browser base platform differs from frozen NAS source lock")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda row: transfer_blob(row, blobs, proxy=proxy), lock["descriptors"]))
        graph = descriptor_graph(blobs, lock["root"], deep=True)
        manifest = read_json(blobs / lock["root"]["digest"][7:])
        record = {"schema_version": 2, "requested": BASE_IMAGE,
            "manifest_digest": lock["root"]["digest"], "manifest_size": lock["root"]["size"],
            "root_media_type": lock["root"]["mediaType"], "runtime_manifest_digest": lock["root"]["digest"],
            "config": manifest["config"], "layers": manifest["layers"],
            "compressed_bytes": sum(row["size"] for row in manifest["layers"]),
            "repo_digests": ["docker.io/library/python@" + lock["root"]["digest"]],
            "platform": lock["platform"], "descriptor_count": len(graph), "deep_verified": True,
            "provenance": {"source_lock_sha256": sha256(lock_path)}}
        path = namespace / "images/base.json"
        if path.exists() and read_json(path) != record:
            raise ValueError("Browser base immutable publication changed")
        write_json(path, record)
        reference = {"store": namespace.name, "manifest": "images/base.json",
            "manifest_sha256": sha256(path), "image_id": lock["root"]["digest"]}
    return NasImageStore(cfg, reference)


def export_image(client, image_id, store, provenance):
    from runtime.images.export import export_image as shared_export
    return shared_export(client, image_id, store, provenance,
                         requested="dycon/webarena-browser:" + BROWSER_SOURCE[:12],
                         manifest_name="images/browser.json")


def prepare_browser(cfg):
    """Publish a reproducible browser image and an explicit local AX/HAR smoke."""
    import docker
    root = Path(cfg["project_root"]).resolve()
    directory = root / "state/webarena-browser" / BROWSER_SOURCE
    directory.mkdir(parents=True, exist_ok=True)
    source = root / "benchmarks/webarena-browser-source"
    proxy = cfg.get("webarena_images", {}).get("host_proxy")
    environment = os.environ.copy()
    if proxy:
        environment.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
    with file_lock(directory / ".prepare.lock", nonblocking=True), contextlib.ExitStack() as assets:
        print("Preparing pinned browser source", flush=True)
        if not (source / ".git").exists():
            source.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(source)], check=True, env=environment, capture_output=True)
            subprocess.run(["git", "-C", str(source), "remote", "add", "origin",
                            "https://github.com/web-arena-x/webarena.git"], check=True, env=environment)
        if not (source / "browser_env/envs.py").exists():
            subprocess.run(["git", "-C", str(source), "config", "core.sparseCheckout", "true"], check=True)
            (source / ".git/info/sparse-checkout").write_text("/browser_env/\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "--filter=blob:none", "origin", BROWSER_SOURCE],
                           check=True, env=environment, capture_output=True, timeout=180)
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"],
                           check=True, env=environment, capture_output=True)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != BROWSER_SOURCE:
            raise ValueError("Browser environment source is not the pinned commit")
        tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "browser_env"], text=True).splitlines()
        files = {}
        for name in tracked:
            expected = subprocess.check_output(["git", "-C", str(source), "show", BROWSER_SOURCE + ":" + name])
            if (source / name).read_bytes() != expected:
                raise ValueError("Changed pinned browser source: " + name)
            files[name] = sha256(source / name)
        context = directory / "build-context"
        (context / "upstream").mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "browser_env", context / "upstream/browser_env", dirs_exist_ok=True)
        for name in ("Dockerfile", "browser-requirements.txt", "browser-selftest.py"):
            shutil.copyfile(root / "runtime/webarena" / name, context / name)
        shutil.copyfile(root / "runtime/containers/webarena_worker.py", context / "worker.py")
        client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
        print("Resolving pinned Python base through the existing registry transport", flush=True)
        from runtime.images.nas import ImageLeaseManager
        base_store = _prepare_base(cfg)
        assets.enter_context(ImageLeaseManager(cfg, client).lease(base_store))
        base = client.images.get(base_store.image_id)
        # FROM uses the digest of the resolved base, retained as build provenance.
        base_reference = "docker.io/library/python@" + base_store.image_id
        # BuildKit resolves FROM references through the daemon registry path; tag the leased local image so the build never falls back to Docker Hub.
        local_base = "dycon/webarena-python-base:" + base_store.image_id.split(":", 1)[1][:12]
        base.tag(local_base.split(":", 1)[0], tag=local_base.split(":", 1)[1])
        build_args = {"PYTHON_IMAGE": local_base}
        forwarding = contextlib.nullcontext()
        if proxy:
            # Rootless build containers cannot reach a host loopback proxy.
            # Reuse the shared managed forwarder, with its own ephemeral port.
            from runtime.images.skillsbench import build_proxy
            settings = copy.deepcopy(cfg)
            hostname = urlsplit(cfg.get("skillsbench_images", {}).get("build_proxy", "")).hostname
            if not hostname or hostname in {"127.0.0.1", "localhost"}:
                raise ValueError("Configure a host-reachable skillsbench_images.build_proxy address for browser build")
            with socket.socket() as probe:
                probe.bind((hostname, 0))
                port = probe.getsockname()[1]
            endpoint = "http://" + hostname + ":" + str(port)
            settings["skillsbench_images"] = {**settings.get("skillsbench_images", {}),
                "host_proxy": proxy, "build_proxy": endpoint, "manage_build_proxy": True}
            forwarding = build_proxy(settings, directory / "network-proxy")
            build_args.update(HTTP_PROXY=endpoint, HTTPS_PROXY=endpoint)
        build_inputs = {"browser_source_revision": BROWSER_SOURCE,
            "browser_source_sha256": files, "base_image": base_reference, "base_image_id": base.id,
            "worker_sha256": sha256(context / "worker.py"),
            "requirements_sha256": sha256(context / "browser-requirements.txt"),
            "dockerfile_sha256": sha256(context / "Dockerfile")}
        image = None
        previous_store = None
        previous = {}
        derivation = {}
        previous_receipt = directory / "receipt.json"
        if previous_receipt.exists():
            from runtime.images.nas import NasImageStore
            previous_store = NasImageStore(cfg, read_json(previous_receipt)["reference"])
            previous = previous_store.record["provenance"]
            if all(previous.get(key) == value for key, value in build_inputs.items()):
                try:
                    image = client.images.get(previous["build_image_id"])
                except docker.errors.ImageNotFound:
                    pass
        if image is None:
            dockerfile = "Dockerfile"
            if previous_store is not None and _worker_update_compatible(previous, build_inputs):
                assets.enter_context(ImageLeaseManager(cfg, client).lease(previous_store))
                parent = client.images.get(previous_store.image_id)
                parent_tag = "dycon/webarena-browser-worker-base:" + parent.id.split(":", 1)[1][:12]
                parent.tag(parent_tag.split(":", 1)[0], tag=parent_tag.split(":", 1)[1])
                dockerfile = "Dockerfile.worker-update"
                (context / dockerfile).write_text("ARG BROWSER_IMAGE\nFROM $" + "{BROWSER_IMAGE}\n"
                    "COPY worker.py /opt/webarena/worker.py\n"
                    "COPY browser-selftest.py /opt/webarena/browser-selftest.py\n", encoding="utf-8")
                build_args = {"BROWSER_IMAGE": parent_tag}
                forwarding = contextlib.nullcontext()
                derivation = {"worker_update_parent": read_json(previous_receipt)["reference"],
                              "worker_update_parent_image_id": parent.id,
                              "worker_update_dockerfile_sha256": sha256(context / dockerfile)}
                print("Updating worker on provenance-matched immutable browser dependencies", flush=True)
            print("Building isolated browser image", flush=True)
            tag = "dycon/webarena-browser:" + BROWSER_SOURCE[:12]
            with forwarding, (directory / "build.log").open("w", encoding="utf-8") as log:
                for row in client.api.build(path=str(context), rm=True, forcerm=True,
                        tag=tag, dockerfile=dockerfile, buildargs=build_args, decode=True,
                        platform=cfg["environment"]["platform"]):
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if row.get("stream"):
                        print(row["stream"], end="", flush=True)
                    if row.get("error"):
                        raise docker.errors.BuildError(row["error"], [row])
            image = client.images.get(tag)
        else:
            print("Reusing provenance-matched local browser image " + image.id, flush=True)
        provenance = {**build_inputs, **derivation, "build_image_id": image.id}
        smoke = client.containers.create(image.id, entrypoint=["python", "/opt/webarena/browser-selftest.py"],
            network_disabled=True, shm_size="2g", cap_drop=["ALL"], security_opt=["no-new-privileges"],
            labels={"dycon.project": str(root), "dycon.backend": "webarena_verified",
                    "dycon.role": "browser-contract-selftest"})
        try:
            print("Testing real Chromium AX/actions and flushed embedded HAR", flush=True)
            smoke.start()
            status = smoke.wait(timeout=180)
            output = smoke.logs().decode(errors="replace")
            (directory / "selftest.log").write_text(output, encoding="utf-8")
            if status["StatusCode"] != 0:
                raise RuntimeError("Browser AX/action/HAR selftest failed; see " + str(directory / "selftest.log"))
            checks = [json.loads(line) for line in output.splitlines() if line.startswith('{"passed"')]
            if len(checks) != 1 or not checks[0]["passed"]:
                raise ValueError("Missing browser selftest receipt")
            write_json(directory / "selftest.json", checks[0])
        finally:
            smoke.remove(force=True, v=True)
        provenance["selftest"] = checks[0]
        store = Path(cfg["paths"]["nas_root"]) / ("webarena-browser-" + hash_json(provenance)[:20])
        print("Publishing verified browser image to NAS", flush=True)
        reference = export_image(client, image.id, store, provenance)
        publication = root / "runtime/webarena-browser" / store.name / "reference.json"
        write_json(publication, reference)
        result = {"schema_version": 1, "status": "complete", "reference": reference,
                  "browser_image_ref": publication.relative_to(root).as_posix(),
                  "build_image_id": image.id, "browser_source_revision": BROWSER_SOURCE,
                  "browser_contract": checks[0],
                  "scope": "browser image publication; site rollout is validated separately"}
        write_json(directory / "receipt.json", result)
        return result
