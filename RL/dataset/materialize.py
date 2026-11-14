"""Bounded typed shards: no pickle, no retokenization, and one copy per prompt SHA."""
import hashlib
import os
from pathlib import Path
import sys
from collections import OrderedDict
import numpy as np

from core.io import hash_json, read_json, sha256, write_json
from .artifacts import directory, publication, require_space, seal, verify_local
from .build import iter_rows, json_line
from .schema import validate


class ShardWriter:
    def __init__(self, stage, dtype, max_bytes):
        self.stage = Path(stage); self.dtype = np.dtype(dtype); self.max_bytes = max_bytes
        self.stream = None; self.shard = -1; self.offset = 0
        (self.stage / "shards").mkdir(exist_ok=True)

    def put(self, values):
        array = np.asarray(values, dtype=self.dtype)
        if array.ndim != 1 or array.nbytes > self.max_bytes:
            raise ValueError("Single sequence exceeds shard byte budget")
        if self.stream is None or (self.offset + len(array)) * self.dtype.itemsize > self.max_bytes:
            if self.stream: self.stream.close()
            self.shard += 1; self.offset = 0
            name = f"shards/part-{self.shard:06d}.{self.dtype.str[1:]}"
            self.stream = (self.stage / name).open("wb"); self.name = name
        raw = array.tobytes()
        result = {"shard": self.name, "offset": self.offset, "count": len(array),
                  "dtype": self.dtype.str, "sha256": hashlib.sha256(raw).hexdigest()}
        self.stream.write(raw); self.offset += len(array)
        return result

    def close(self):
        if self.stream:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close(); self.stream = None


def read_array(root, ref, *, verify=True):
    path = Path(root) / ref["shard"]
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Shard reference escapes artifact")
    dtype = np.dtype(ref["dtype"])
    if dtype.hasobject or dtype not in (np.dtype("<u4"), np.dtype("<f4")):
        raise ValueError("Unsupported array dtype")
    if type(ref["offset"]) is not int or ref["offset"] < 0 or type(ref["count"]) is not int or ref["count"] < 1:
        raise ValueError("Invalid array offset/count")
    if (ref["offset"] + ref["count"]) * dtype.itemsize > path.stat().st_size:
        raise ValueError("Array extends past shard boundary")
    array = np.memmap(path, mode="r", dtype=dtype, offset=ref["offset"] * dtype.itemsize, shape=(ref["count"],))
    if verify and hashlib.sha256(array.tobytes()).hexdigest() != ref["sha256"]:
        raise ValueError("Array content checksum mismatch")
    return array


def materialize(dataset_path, cache_root, *, max_shard_bytes=None):
    from .loader import iter_groups
    dataset_path = Path(dataset_path); verify_local(dataset_path, "rl-dataset")
    config = read_json(dataset_path / "resolved-config.json")
    max_shard_bytes = max_shard_bytes or config["storage"]["max_shard_bytes"]
    max_shard_bytes = min(max_shard_bytes, int(config["storage"]["memory_budget_gib"] * 1024**3 / 8))
    identity = {"dataset_receipt": sha256(dataset_path / "receipt.json"), "dtype": "<u4", "schema": 1,
                "mask_policy": read_json(dataset_path / "dataset.json")["mask_policy"], "order": "frozen_group_branch_order_v1",
                "audit_enabled": read_json(dataset_path / "dataset.json").get("audit", {}).get("enabled", False)}
    cache_id = "tokens-" + hash_json(identity)[:48]
    target = directory(cache_root, "rl-cache", cache_id)
    if target.exists():
        verify_local(target, "rl-cache")
        if read_json(target / "cache.json")["identity"] != identity: raise ValueError("Cache identity mismatch")
        return {"status": "verified", "path": str(target), "cache_id": cache_id, "reused": True}
    summary = read_json(dataset_path / "summary.json")
    require_space(cache_root, 4 * (summary["prompt_lengths"].get("sum", 0) + summary["completion_lengths"].get("sum", 0)))
    with publication(cache_root, "rl-cache", cache_id) as (stage, destination):
        writer = ShardWriter(stage, "<u4", max_shard_bytes)
        prompts = {}; count = 0; group_count = 0
        try:
            with (stage / "groups.jsonl").open("w", encoding="utf-8") as gs, (stage / "branches.jsonl").open("w", encoding="utf-8") as bs:
                for group in iter_groups(dataset_path):
                    g = group["group"]
                    json_line(gs, {"group_id": g["group_id"]}); group_count += 1
                    for row in group["members"]:
                        p = row["prompt_token_ids"]; key = hash_json(p)
                        if key not in prompts: prompts[key] = writer.put(p)
                        pref = prompts[key]
                        cref = writer.put(row["completion_token_ids"])
                        json_line(bs, {"branch_sample_id": row["branch_sample_id"], "group_id": row["group_id"],
                                       "prompt_ref": pref, "completion_ref": cref}); count += 1
        finally:
            writer.close()
        write_json(stage / "cache.json", {"schema_version": 1, "identity": identity, "group_count": group_count, "branch_count": count})
        seal(stage, "rl-cache", cache_id)
    return {"status": "materialized", "path": str(destination), "cache_id": cache_id, "groups": group_count, "branches": count, "reused": False}


class CacheReader:
    def __init__(self, path, dataset_receipt, *, prompt_budget_bytes=64*1024**2):
        self.path = Path(path); verify_local(path, "rl-cache")
        identity = read_json(self.path / "cache.json")["identity"]
        if identity["dataset_receipt"] != dataset_receipt:
            raise ValueError("Cache belongs to another dataset receipt")
        if identity.get("dtype") != "<u4": raise ValueError("Token cache requires uint32")
        self.prompt_budget = prompt_budget_bytes; self.prompt_bytes = 0
        self.prompts = OrderedDict()
        if identity.get("dtype") != "<u4": raise ValueError("Token cache requires uint32")
        self.prompt_budget = prompt_budget_bytes; self.prompt_bytes = 0
        self.prompts = OrderedDict()
        self.audit = identity["audit_enabled"]
        self.rows = iter(iter_rows(self.path / "branches.jsonl"))

    def tokens(self, branch):
        row = next(self.rows, None)
        # A resumed group cursor may skip earlier cache rows.
        while row is not None and row["branch_sample_id"] != branch["branch_sample_id"]:
            row = next(self.rows, None)
        if row is None:
            raise ValueError("Cache branch order mismatch")
        if row["group_id"] != branch["group_id"]: raise ValueError("Cache group mismatch")
        pref, cref = row["prompt_ref"], row["completion_ref"]
        if pref["dtype"] != "<u4" or cref["dtype"] != "<u4": raise ValueError("Token cache requires uint32 arrays")
        file = self.path / pref["shard"]
        stat = file.stat()
        key = (hash_json(pref), branch["prompt_sha256"], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        saved = self.prompts.pop(key, None)
        if saved is None:
            p = read_array(self.path, pref).tolist()
            # Validate shared prompt once; completion remains checked for every branch.
            if self.audit and hash_json(p) != branch["prompt_sha256"]: raise ValueError("Prompt token checksum mismatch")
            size = sys.getsizeof(p) + sum(map(sys.getsizeof, p))
            if size <= self.prompt_budget:
                while self.prompts and self.prompt_bytes + size > self.prompt_budget:
                    _, (_, old_size) = self.prompts.popitem(last=False); self.prompt_bytes -= old_size
                self.prompts[key] = (p, size); self.prompt_bytes += size
        else:
            p, size = saved; self.prompts[key] = saved
        c = read_array(self.path, cref).tolist()
        if self.audit and hash_json(c) != branch["completion_sha256"]: raise ValueError("Completion token checksum mismatch")
        validate(p, c, branch["prompt_sha256"], branch["completion_sha256"], verify_hashes=False)
        if (len(p), len(c)) != (branch["prompt_count"], branch["completion_count"]): raise ValueError("Cache token count mismatch")
        return p, c
