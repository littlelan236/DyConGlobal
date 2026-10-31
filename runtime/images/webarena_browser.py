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


class _Chunks(io.RawIOBase):
    def __init__(self, source):
        self.source, self.buffer = iter(source), bytearray()

    def readable(self):
        return True

    def readinto(self, target):
        while not self.buffer:
            try:
                self.buffer.extend(next(self.source))
            except StopIteration:
                return 0
        count = min(len(target), len(self.buffer))
        target[:count] = self.buffer[:count]
        del self.buffer[:count]
        return count


def export_image(client, image_id, store, provenance):
    """Convert a daemon save stream to the project's shared NAS OCI contract."""
    blobs = store / "blobs/sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    entries, manifest = {}, None
    stream = io.BufferedReader(_Chunks(client.api.get_image(image_id)))
    with stream, tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            source = tar.extractfile(member)
            if member.name == "manifest.json":
                manifest = json.load(source)
                continue
            if (member.name.startswith("blobs/sha256/") or member.name.endswith(".json") or member.name.endswith("/layer.tar")):
                fd, name = tempfile.mkstemp(prefix=".browser-export-", dir=blobs)
                try:
                    with os.fdopen(fd, "wb") as output:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    with open(name, "rb") as layer_file:
                        magic = layer_file.read(4)
                    media = "application/vnd.oci.image.layer.v1.tar"
                    if magic.startswith(b"\x1f\x8b"):
                        media += "+gzip"
                    elif magic == b"\x28\xb5\x2f\xfd":
                        media += "+zstd"
                    entries[member.name] = {**publish_blob(Path(name), blobs), "mediaType": media}
                finally:
                    Path(name).unlink(missing_ok=True)
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise ValueError("Browser export must contain exactly one Docker image")
    raw = manifest[0]
    config = {**entries[raw["Config"]], "mediaType": "application/vnd.oci.image.config.v1+json"}
    layers = [{**entries[name], "mediaType": entries[name]["mediaType"]}
              for name in raw["Layers"]]
    value = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
             "config": config, "layers": layers}
    root = {**json_blob(value, blobs), "mediaType": value["mediaType"]}
    graph = descriptor_graph(blobs, root, deep=True)
    record = {"schema_version": 2, "requested": "dycon/webarena-browser:" + BROWSER_SOURCE[:12],
        "manifest_digest": root["digest"], "manifest_size": root["size"],
        "root_media_type": root["mediaType"], "runtime_manifest_digest": root["digest"],
        "config": config, "layers": layers, "compressed_bytes": sum(d["size"] for d in layers),
        "repo_digests": [], "platform": "linux/amd64", "descriptor_count": len(graph),
        "deep_verified": True, "provenance": provenance}
    path = store / "images/browser.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("Browser image publication already exists with different bytes")
    write_json(path, record)
    return {"store": store.name, "manifest": "images/browser.json", "manifest_sha256": sha256(path),
            "image_id": root["digest"]}


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
        previous_receipt = directory / "receipt.json"
        if previous_receipt.exists():
            from runtime.images.nas import NasImageStore
            previous = NasImageStore(cfg, read_json(previous_receipt)["reference"]).record["provenance"]
            if all(previous.get(key) == value for key, value in build_inputs.items()):
                try:
                    image = client.images.get(previous["build_image_id"])
                except docker.errors.ImageNotFound:
                    pass
        if image is None:
            print("Building isolated browser image", flush=True)
            tag = "dycon/webarena-browser:" + BROWSER_SOURCE[:12]
            with forwarding, (directory / "build.log").open("w", encoding="utf-8") as log:
                for row in client.api.build(path=str(context), rm=True, forcerm=True,
                        tag=tag, buildargs=build_args, decode=True,
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
        provenance = {**build_inputs, "build_image_id": image.id}
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
