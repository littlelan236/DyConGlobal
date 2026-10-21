"""Digest-verified Docker Hub to NAS transfer; no Docker daemon or local tar."""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from pathlib import Path

import requests

from core.io import sha256
from runtime.images.oci import INDEX_TYPES, MANIFEST_TYPES, blob_path

ACCEPT = ", ".join(sorted(INDEX_TYPES | MANIFEST_TYPES))


class Registry:
    def __init__(self, proxy=None):
        self.session = requests.Session()
        self.session.trust_env = False
        if proxy:
            self.session.proxies = {"https": proxy, "http": proxy}
        self.tokens = {}

    def get(self, repository, endpoint, *, stream=False, offset=0):
        if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", repository):
            raise ValueError("Invalid Docker Hub repository")
        for attempt in range(5):
            try:
                if repository not in self.tokens:
                    with self.session.get(
                        "https://auth.docker.io/token",
                        params={"service": "registry.docker.io", "scope": f"repository:{repository}:pull"},
                        timeout=(20, 60),
                    ) as response:
                        response.raise_for_status()
                        self.tokens[repository] = response.json()["token"]
                headers = {"Accept": ACCEPT, "Authorization": "Bearer " + self.tokens[repository]}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                response = self.session.get(
                    f"https://registry-1.docker.io/v2/{repository}/{endpoint}",
                    headers=headers, stream=stream, timeout=(20, 120),
                )
                if response.status_code == 429:
                    response.close()
                    headers.pop("Authorization", None)
                    response = self.session.get(
                        f"https://mirror.gcr.io/v2/{repository}/{endpoint}",
                        headers=headers, stream=stream, timeout=(20, 120),
                    )
                if response.status_code == 401:
                    self.tokens.pop(repository, None)
                if response.status_code in {403, 404}:
                    response.raise_for_status()
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in {403, 404}:
                    raise
                if attempt == 4:
                    raise
            except requests.RequestException:
                if attempt == 4:
                    raise
            time.sleep(min(20, 2**attempt))
        raise RuntimeError("Registry retries exhausted")

    def close(self):
        self.session.close()


def persist_bytes(directory, raw, expected=None):
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("Registry descriptor SHA mismatch")
    target = blob_path(directory, digest)
    if target.exists():
        if target.stat().st_size != len(raw) or sha256(target) != digest[7:]:
            raise ValueError(f"Existing immutable blob is corrupt: {digest}")
        return digest
    temporary = directory / (".metadata-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def image_metadata(repository, directory, *, reference="latest", proxy=None):
    """Retain the original index, every platform/attestation manifest and config."""
    import json

    client = Registry(proxy)
    descriptors = {}
    active = set()

    def visit(reference, expected=None):
        if len(active) >= 16 or reference in active:
            raise ValueError("Cyclic/deep registry descriptor graph")
        active.add(reference)
        try:
            with client.get(repository, "manifests/" + reference) as response:
                raw = response.content
            digest = persist_bytes(directory, raw, expected["digest"] if expected else None)
            obj = json.loads(raw)
            if expected and len(raw) != expected["size"]:
                raise ValueError("Registry manifest size mismatch")
            descriptor = {"digest": digest, "size": len(raw), "mediaType": obj["mediaType"]}
            descriptors[digest] = {**descriptor, "repository": repository}
            if obj["mediaType"] in INDEX_TYPES:
                for child in obj["manifests"]:
                    visit(child["digest"], child)
            elif obj["mediaType"] in MANIFEST_TYPES:
                config = obj["config"]
                with client.get(repository, "blobs/" + config["digest"]) as response:
                    data = response.content
                if len(data) != config["size"]:
                    raise ValueError("Registry config size mismatch")
                persist_bytes(directory, data, config["digest"])
                for child in [config, *obj["layers"]]:
                    prior = descriptors.get(child["digest"])
                    if prior and prior["size"] != child["size"]:
                        raise ValueError("Conflicting descriptor size")
                    descriptors[child["digest"]] = {**child, "repository": repository}
            else:
                raise ValueError("Unsupported registry manifest")
            return descriptor
        finally:
            active.remove(reference)

    try:
        root = visit(reference)
        return root, list(descriptors.values())
    finally:
        client.close()


def transfer_blob(item, directory, seed_directories=(), proxy=None, progress=None):
    """Resume partial network bytes; validate complete SHA before atomic publish."""
    target = blob_path(directory, item["digest"])
    if target.exists():
        if target.stat().st_size != item["size"] or sha256(target) != item["digest"][7:]:
            raise ValueError(f"Existing immutable blob is corrupt: {target}")
        return {"digest": item["digest"], "source": "existing", "bytes": item["size"]}
    for source_directory in seed_directories:
        source = blob_path(Path(source_directory), item["digest"])
        if source.is_file() and source.stat().st_size == item["size"]:
            if sha256(source) != item["digest"][7:]:
                continue
            os.link(source, target)
            return {"digest": item["digest"], "source": "nas-hardlink", "bytes": item["size"]}
    part = target.with_suffix(".part")
    client = Registry(proxy)
    received = 0
    try:
        for attempt in range(6):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset > item["size"]:
                    part.unlink()
                    offset = 0
                if offset < item["size"]:
                    with client.get(item["repository"], "blobs/" + item["digest"], stream=True,
                                    offset=offset) as response:
                        if offset and response.status_code == 206:
                            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                                raise ValueError("Wrong HTTP range")
                        elif offset:
                            offset = 0
                        with part.open("ab" if offset else "wb") as output:
                            for block in response.iter_content(4 * 1024**2):
                                if output.tell() + len(block) > item["size"]:
                                    raise ValueError("Registry blob exceeds descriptor size")
                                output.write(block)
                                received += len(block)
                                if progress:
                                    progress(len(block))
                            output.flush()
                            os.fsync(output.fileno())
                if part.stat().st_size != item["size"] or sha256(part) != item["digest"][7:]:
                    part.unlink(missing_ok=True)
                    raise ValueError("Downloaded blob SHA/size mismatch")
                part.replace(target)
                return {"digest": item["digest"], "source": "network", "bytes": item["size"],
                        "received_bytes": received}
            except (requests.RequestException, OSError, ValueError):
                if attempt == 5:
                    raise
                time.sleep(min(20, 2**attempt))
    finally:
        client.close()
