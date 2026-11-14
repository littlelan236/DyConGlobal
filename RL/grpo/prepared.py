"""Immutable, indexed native-token groups and prefix-preserving epoch schedules.

SQLite indexes contain metadata/offsets only. Little-endian int64 token arrays
share one prompt per group; float64 scores/advantages preserve Python precision.
No padded tensors or native ledgers are needed by readers or exact resume.
"""
from collections import Counter
from contextlib import closing
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import time
from typing import TypedDict

from core.io import hash_json, now, owned_path, read_json, sha256, write_json
from RL.artifacts import source_identity, verify_receipt
from RL.config import schedule_contract
from RL.dataset.artifacts import directory, publication, seal, verify_local
from RL.dataset.schema import MASK_POLICY
from .advantage import CONSENSUS_ADVANTAGE_VERSION
from .data import freeze_group, source_groups, tensor_groups
from .schedule import LEGACY_SHUFFLE, order_groups, shuffle_metadata

VERSION = 1
SHUFFLE = LEGACY_SHUFFLE


class ArtifactBinding(TypedDict):
    id: str
    root: str
    manifest_sha256: str
    receipt_sha256: str


class Segment(TypedDict):
    segment_id: str
    parent_segment_id: str | None
    group_ids: list[str]
    group_order_sha256: str
    window_sha256: str
    group_batch_size: int
    shuffle_algorithm: str
    shuffle_seed: int
    sha256: str


class PreparedManifest(TypedDict):
    schema_version: int
    prepared_id: str
    producer_fingerprint: str
    parent: ArtifactBinding | None
    proof: dict
    schedule: dict
    shards: list[dict]


class ExtensionRequest(TypedDict):
    schema_version: int
    extension_id: str
    parent_checkpoint: str
    parent_receipt_sha256: str
    parent_contract_sha256: str
    prepared: ArtifactBinding
    scheduler_extension: str


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def hash_sequence(rows):
    """core.io.hash_json(list(rows)) without retaining token records."""
    digest = hashlib.sha256(b"[")
    for i, row in enumerate(rows):
        if i:
            digest.update(b",")
        digest.update(_json(row).encode())
    digest.update(b"]")
    return digest.hexdigest()


def source_record(value):
    # Native source identities include token hashes, mask spans and ledger receipt
    # locators. Synthetic inputs carry tokens directly and bind them here too.
    return {"group": value["group"], "members": [
        {k: v for k, v in row.items() if k != "audit_enabled"} for row in value["members"]]}


def order_record(value, consensus=False):
    keys = ("branch_sample_id", "prompt_token_ids", "completion_token_ids", "historical_score")
    if consensus:
        keys += ("retained",)
    return {"group_id": value["group"]["group_id"],
            "branches": [{k: row[k] for k in keys} for row in value["members"]],
            "advantages": value["frozen_advantages"]}


def segment(ids, batch, seed, ordinal, parent=None, *, algorithm=LEGACY_SHUFFLE):
    value = {"segment_id": f"segment-{ordinal:06d}", "parent_segment_id": parent,
             "group_ids": list(ids), "group_order_sha256": hash_json(ids),
             "group_batch_size": batch, **shuffle_metadata(algorithm), "shuffle_seed": seed,
             "window_sha256": hash_sequence(ids[i:i+batch] for i in range(0, len(ids), batch))}
    value["sha256"] = hash_json(value)
    return value


def validate_schedule(value):
    if value.get("schema_version") != VERSION or not value.get("segments"):
        raise ValueError("Unsupported or empty prepared schedule")
    seen = set()
    for i, item in enumerate(value["segments"]):
        ids = item["group_ids"]
        if (not ids or any(not isinstance(g, str) or not g for g in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)
                or type(item["group_batch_size"]) is not int or item["group_batch_size"] < 1
                or item != segment(ids, item["group_batch_size"], item["shuffle_seed"], i,
                                   value["segments"][i-1]["segment_id"] if i else None,
                                   algorithm=item.get("shuffle_algorithm", LEGACY_SHUFFLE))):
            raise ValueError("Prepared schedule segment identity/order/window mismatch")
        seen.update(ids)
    if value.get("sha256") != hash_json({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Prepared schedule manifest checksum mismatch")
    return value


def validate_extension_schedule(parent, child):
    validate_schedule(parent); validate_schedule(child)
    n = len(parent["segments"])
    if (len(child["segments"]) != n + 1 or child["segments"][:n] != parent["segments"]
            or child.get("parent_schedule_hash") != parent["sha256"]
            or child.get("parent_last_segment_hash") != parent["segments"][-1]["sha256"]):
        raise ValueError("Append schedule is not a strict immutable prefix extension")


def _readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def binding(path, kind="rl-prepared"):
    path = Path(path)
    return {"id": path.name, "root": str(path.parents[2]),
            "manifest_sha256": sha256(path / "manifest.json"),
            "receipt_sha256": sha256(path / "receipt.json")}


def resolve(ref, kind="rl-prepared"):
    path = directory(ref["root"], kind, ref["id"])
    if (sha256(path / "manifest.json") != ref["manifest_sha256"]
            or sha256(path / "receipt.json") != ref["receipt_sha256"]):
        raise ValueError(f"{kind} manifest/receipt binding checksum mismatch")
    return path


class PreparedReader:
    """Bounded reads by group ID or frozen ordinal, with parent body references."""
    def __init__(self, path, *, verify=True, ancestors=()):
        self.path = Path(path)
        if str(self.path.resolve()) in ancestors or len(ancestors) >= 100:
            raise ValueError("Prepared parent cycle/depth exceeded")
        started = time.perf_counter()
        receipt = verify_local(self.path, "rl-prepared", verify_files=verify)
        self.manifest = read_json(self.path / "manifest.json")
        m = self.manifest
        if m.get("schema_version") != VERSION or m.get("prepared_id") != self.path.name:
            raise ValueError("Prepared artifact schema/identity mismatch")
        self.schedule = validate_schedule(read_json(self.path / "schedule.json"))
        if self.schedule != m["schedule"]:
            raise ValueError("Prepared/schedule manifest mismatch")
        self.proof = m["proof"]
        self.execution_plan = self.proof.get("execution_plan")
        if self.execution_plan:
            from .execution import validate
            validate(self.execution_plan)
            if self.schedule.get("execution_schedule_sha256") != self.execution_plan["sha256"]:
                raise ValueError("Prepared execution schedule identity mismatch")
        self.ids = [g for s in self.schedule["segments"] for g in s["group_ids"]]
        self.parent = PreparedReader(resolve(m["parent"]), verify=verify,
            ancestors=(*ancestors, str(self.path.resolve()))) if m.get("parent") else None
        if self.parent:
            validate_extension_schedule(self.parent.schedule, self.schedule)
        with closing(_readonly(self.path / "inventory.sqlite3")) as db:
            local = {r[0]: dict(zip(("shard", "branch_count", "token_count", "content_sha256", "source_sha256"), r[1:]))
                     for r in db.execute("SELECT id,shard,branches,tokens,content_sha256,source_sha256 FROM groups WHERE eligible=1")}
            self.local_ids = set(local)
            self.records = {**(self.parent.records if self.parent else {}), **local}
            if self.parent and self.parent.all_ids.intersection(r[0] for r in db.execute("SELECT id FROM groups")):
                raise ValueError("Prepared child shadows a parent group identity")
            self.all_ids = (self.parent.all_ids if self.parent else set()) | {r[0] for r in db.execute("SELECT id FROM groups")}
            self.branch_ids = (self.parent.branch_ids if self.parent else set()) | {r[0] for r in db.execute("SELECT id FROM branches")}
        if (set(self.ids) != set(self.records) or len(self.ids) != len(self.records)
                or self.ids != self.proof["eligible_group_ids"] or len(self.ids) != self.proof["group_count"]
                or hash_json(sorted(self.records)) != self.proof["eligible_groups_sha256"]
                or hash_json({g: self.records[g]["content_sha256"] for g in sorted(self.records)}) != m["group_inventory_sha256"]):
            raise ValueError("Prepared inventory/group order mismatch")
        expected_shards = {r["shard"] for r in local.values()}
        if expected_shards != {s["name"] for s in m["shards"]}:
            raise ValueError("Prepared shard list mismatch")
        for shard in m["shards"]:
            folder = owned_path(self.path, shard["name"])
            if sha256(folder / "receipt.json") != shard["receipt_sha256"]:
                raise ValueError("Prepared shard receipt checksum mismatch")
            if verify:
                verify_receipt(folder, "rl-prepared-shard-v1")
        self.statistics = {"manifest_verification_seconds": time.perf_counter() - started,
                           "bytes_read": sum(x["bytes"] for x in receipt["files"].values()) if verify else 0}
        if self.execution_plan:
            self.verify_metadata_indexes()
            from .execution import parent_record
            parents = {p["parent_group_id"]: p for p in self.execution_plan["parents"]}
            if (set(parents) != set(self.records) or any(r["branch_count"] != 32 for r in self.records.values())
                    or any(parent_record(self.metadata(gid)) != p for gid, p in parents.items())):
                raise ValueError("Prepared 8x16 execution/native parent inventory mismatch")

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return (self.raw(g) for g in self.ids)

    def source_hash(self, group_id):
        with closing(_readonly(self.path / "inventory.sqlite3")) as db:
            row = db.execute("SELECT source_sha256 FROM groups WHERE id=?", (group_id,)).fetchone()
        return row[0] if row else self.parent.source_hash(group_id) if self.parent else None

    def branch_ids_for(self, group_id):
        if group_id not in self.local_ids and self.parent:
            return self.parent.branch_ids_for(group_id)
        with closing(_readonly(self.path / "inventory.sqlite3")) as db:
            row = db.execute("SELECT payload FROM groups WHERE id=?", (group_id,)).fetchone()
        return [r["branch_sample_id"] for r in json.loads(row[0])["members"]]

    def verify_metadata_indexes(self):
        """Verify required metadata, never opening token arrays; deep verify stays explicit."""
        receipt = read_json(self.path / "receipt.json")
        for name in ("inventory.sqlite3", "schedule.json", "manifest.json"):
            entry = receipt["files"].get(name)
            if not entry or sha256(self.path / name) != entry["sha256"]:
                raise ValueError("Prepared metadata receipt mismatch: " + name)
        if self.parent:
            self.parent.verify_metadata_indexes()

    def metadata(self, group_id, *, token_preview=0):
        """Frozen metadata and bounded optional previews, sharing shard path/hash rules."""
        if not 0 <= token_preview <= 128:
            raise ValueError("Token preview must be between 0 and 128")
        if group_id not in self.local_ids:
            if self.parent:
                return self.parent.metadata(group_id, token_preview=token_preview)
            raise KeyError(group_id)
        with closing(_readonly(self.path / "inventory.sqlite3")) as db:
            row = db.execute("SELECT payload FROM groups WHERE id=?", (group_id,)).fetchone()
        value = json.loads(row[0])
        location = value.pop("storage")
        folder = owned_path(self.path, location["shard"])
        receipt = read_json(folder / "receipt.json")
        score_file = folder / "scores-advantages.bin"
        if sha256(score_file) != receipt["files"]["scores-advantages.bin"]["sha256"]:
            raise ValueError("Prepared score/advantage shard receipt mismatch")
        def preview(name, offset, count):
            path = folder / name
            if min(offset, count) < 0 or (offset+count)*8 > path.stat().st_size:
                raise ValueError("Prepared token preview offset out of bounds")
            with path.open("rb") as stream:
                stream.seek(offset*8)
                return [x[0] for x in struct.iter_unpack("<q", stream.read(min(token_preview,count)*8))]
        advantages = []
        with score_file.open("rb") as scores:
            for row in value["members"]:
                offset, length, score_offset = row.pop("storage")
                if score_offset < 0 or (score_offset+2)*8 > score_file.stat().st_size:
                    raise ValueError("Prepared score offset out of bounds")
                scores.seek(score_offset*8)
                score, advantage = struct.unpack("<dd", scores.read(16))
                if score != row["historical_score"] or not all(math.isfinite(x) for x in (score,advantage)):
                    raise ValueError("Prepared score/advantage identity invalid")
                advantages.append(advantage)
                row.update(prompt_count=location["prompt_length"], completion_count=length)
                if token_preview:
                    row["prompt_token_ids"] = preview("prompt-tokens.bin",location["prompt_offset"],location["prompt_length"])
                    row["completion_token_ids"] = preview("completion-tokens.bin",offset,length)
        ids = [r["branch_sample_id"] for r in value["members"]]
        if len(ids) != value["group"]["expected_branch_count"] or value["group"].get("branch_ids",ids) != ids:
            raise ValueError("Prepared native branch order mismatch")
        value["frozen_advantages"] = advantages
        return value

    def raw(self, group_id, *, branch_ordinals=None):
        if group_id not in self.local_ids:
            if self.parent:
                return self.parent.raw(group_id, branch_ordinals=branch_ordinals)
            raise KeyError(group_id)
        with closing(_readonly(self.path / "inventory.sqlite3")) as db:
            row = db.execute("SELECT payload FROM groups WHERE id=?", (group_id,)).fetchone()
        value = json.loads(row[0])
        location = value.pop("storage")
        folder = owned_path(self.path, location["shard"])
        def read_values(name, offset, count, fmt):
            width = struct.calcsize(fmt)
            path = folder / name
            if (type(offset) is not int or type(count) is not int or offset < 0 or count < 1
                    or (offset + count) * width > path.stat().st_size):
                raise ValueError("Prepared shard offset out of bounds")
            with path.open("rb") as stream:
                stream.seek(offset * width)
                data = stream.read(count * width)
            if len(data) != count * width:
                raise ValueError("Truncated prepared token/advantage array")
            return [r[0] for r in struct.iter_unpack(fmt, data)]
        prompt = read_values("prompt-tokens.bin", location["prompt_offset"], location["prompt_length"], "<q")
        if branch_ordinals is not None:
            if (branch_ordinals != sorted(set(branch_ordinals))
                    or any(type(i) is not int or not 0 <= i < len(value["members"]) for i in branch_ordinals)):
                raise ValueError("Prepared native branch ordinals invalid")
            value["members"] = [value["members"][i] for i in branch_ordinals]
        advantages = []
        for row in value["members"]:
            offset, length, score_offset = row.pop("storage")
            row["prompt_token_ids"] = prompt
            row["completion_token_ids"] = read_values("completion-tokens.bin", offset, length, "<q")
            pair = read_values("scores-advantages.bin", score_offset, 2, "<d")
            if pair[0] != row["historical_score"] or not all(math.isfinite(x) for x in pair):
                raise ValueError("Prepared score/advantage identity invalid")
            advantages.append(pair[1])
            row["audit_enabled"] = True
        value["frozen_advantages"] = advantages
        if branch_ordinals is not None:
            from RL.dataset.schema import validate
            for row in value["members"]:
                validate(row["prompt_token_ids"], row["completion_token_ids"],
                         row["prompt_sha256"], row["completion_sha256"])
            return value
        if (len(value["members"]) != value["group"]["expected_branch_count"]
                or value["group"].get("branch_ids", [r["branch_sample_id"] for r in value["members"]])
                   != [r["branch_sample_id"] for r in value["members"]]
                or hash_json(order_record(value, self.proof["advantage_policy"] == CONSENSUS_ADVANTAGE_VERSION))
                   != self.records[group_id]["content_sha256"]):
            raise ValueError("Prepared branch/token/mask/advantage content mismatch")
        return value

    def cursor(self, count):
        if getattr(self, "execution_plan", None):
            from .execution import cursor
            return cursor(self.execution_plan, count)
        if type(count) is not int or not 0 <= count <= len(self.ids):
            raise ValueError("Prepared cursor is beyond the schedule")
        consumed = self.ids[:count]
        remaining = count
        segments = self.schedule["segments"]
        current = segments[-1]
        for current in segments:
            if remaining < len(current["group_ids"]):
                break
            if remaining == len(current["group_ids"]) and current is segments[-1]:
                break
            remaining -= len(current["group_ids"])
        return {"segment_id": current["segment_id"], "segment_ordinal": remaining,
                "last_consumed_group_id": consumed[-1] if consumed else None,
                "consumed_group_ids_sha256": hash_json(consumed)}

    def validate_state(self, state):
        if self.execution_plan:
            from .execution import validate_state
            return validate_state(self.execution_plan, state)
        count = state["group_cursor"]
        expected = self.cursor(count)
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError("Prepared checkpoint cursor/consumed ID hash mismatch: " + key)
        if (state["consumed_groups"] != count
                or state["consumed_branches"] != sum(self.records[g]["branch_count"] for g in self.ids[:count])
                or state["consumed_tokens"] != sum(self.records[g]["token_count"] for g in self.ids[:count])
                or state["global_step"] != sum(1 for end in self.window_ends() if end <= count)
                or count not in {0, *self.window_ends()}):
            raise ValueError("Prepared checkpoint group/branch/token/update cursor mismatch")

    def window_ends(self):
        offset = 0
        for item in self.schedule["segments"]:
            n, batch = len(item["group_ids"]), item["group_batch_size"]
            yield from (offset + min(n, i + batch) for i in range(0, n, batch))
            offset += n

    def windows(self, start=0):
        previous = 0
        for end in self.window_ends():
            if previous >= start:
                yield tuple(tensor_groups(self.raw(g) for g in self.ids[previous:end]))
            previous = end


def validate_config(reader, config, *, exact=True):
    actual = schedule_contract(config)
    expected = reader.manifest["training_schedule"]
    if not exact:
        actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
        for value in (actual, expected):
            value["training"].pop("max_updates")
            value["algorithm"].pop("dataset_root")
            value["algorithm"].pop("dataset_id")
            value["grpo"].pop("reference_cache", None)
    if actual != expected:
        raise ValueError("Prepared schedule/dataset/advantage/model identity mismatch")


def validate_dataset_binding(proof, config):
    if proof["synthetic"]:
        fixture = Path(__file__).parents[1] / "fixtures/grpo-native-smoke-v1.json"
        digest = sha256(fixture)
    else:
        path = directory(config.algorithm.dataset_root, "rl-dataset", config.algorithm.dataset_id)
        verify_local(path, "rl-dataset", verify_files=False)
        digest = sha256(path / "receipt.json")
    if (digest != proof["dataset_receipt_sha256"] or config.algorithm.dataset_id != proof["dataset_id"]):
        raise ValueError("Prepared source dataset receipt identity mismatch")


def verify_prepared(path):
    """Explicit deep artifact verification checks every index/offset and token hash."""
    reader = PreparedReader(path)
    from RL.dataset.schema import validate
    for value in reader:
        for row in value["members"]:
            validate(row["prompt_token_ids"], row["completion_token_ids"],
                     row["prompt_sha256"], row["completion_sha256"])
    if reader.parent:
        expected_order = hash_json({"parent":reader.parent.proof["group_order_sha256"],
            "appended":[reader.records[g]["content_sha256"] for g in reader.schedule["segments"][-1]["group_ids"]]})
    else:
        expected_order = hash_sequence(order_record(reader.raw(g), reader.proof["advantage_policy"] == CONSENSUS_ADVANTAGE_VERSION)
                                       for g in reader.ids)
    if expected_order != reader.proof["group_order_sha256"]:
        raise ValueError("Prepared frozen group order/content hash mismatch")
    return reader


def materialize(config, project_root, artifact_id, *, shard_groups=16, parent=None,
                append_seed=None, extension_id=None, legacy=None, source_attempt=None):
    """Freeze one group at a time, shuffle only compact IDs, then atomically seal."""
    if type(shard_groups) is not int or shard_groups < 1:
        raise ValueError("shard_groups must be positive")
    if parent:
        validate_config(parent, config, exact=False)
    from .execution import enabled, parent_record, from_config, apply_proof
    if parent and (enabled(config) or parent.proof.get("execution_plan")):
        raise ValueError("8x16 append execution schedules are not supported")
    execution_parents = []
    raw, proof = source_groups(config) if legacy is None else (iter(legacy), copy.deepcopy(legacy.proof))
    producer = source_identity(project_root)["fingerprint"]
    root = config.runtime.output_root
    with publication(root, "rl-prepared", artifact_id, signature=hash_json(schedule_contract(config))) as (stage, destination):
        db = sqlite3.connect(stage / "inventory.sqlite3")
        db.execute("CREATE TABLE groups(id TEXT PRIMARY KEY, eligible INTEGER, source_sha256 TEXT, content_sha256 TEXT, shard TEXT, branches INTEGER, tokens INTEGER, payload TEXT)")
        db.execute("CREATE TABLE branches(id TEXT PRIMARY KEY, group_id TEXT)")
        ids, exclusions, shards, advantages = [], [], [], []
        prompt_lengths = {}
        total = overlap = native_scanned = 0
        streams = None
        offsets = None
        folder = None
        seen_source = set()
        def close_shard():
            nonlocal streams
            if streams is None:
                return
            for stream in streams.values():
                stream.flush(); __import__("os").fsync(stream.fileno()); stream.close()
            from RL.artifacts import commit_receipt
            commit_receipt(folder, "rl-prepared-shard-v1")
            shards.append({"name": folder.relative_to(stage).as_posix(),
                           "receipt_sha256": sha256(folder / "receipt.json"),
                           "group_count": min(shard_groups, len(ids) - (len(shards) * shard_groups))})
            streams = None
        try:
            for value in raw:
                group_id = value["group"]["group_id"]
                if group_id in seen_source:
                    raise ValueError("Duplicate group identity in dataset")
                seen_source.add(group_id)
                source_hash = hash_json(source_record(value))
                if parent and group_id in parent.all_ids:
                    if source_hash != parent.source_hash(group_id):
                        raise ValueError("Append changes an existing group/branch/token/score/retained identity")
                    overlap += 1
                    continue
                total += 1
                if legacy is None:
                    reason = freeze_group(value, config)
                    native_scanned += 1
                else:
                    reason = None
                for row in value["members"]:
                    branch_id = row["branch_sample_id"]
                    if parent and branch_id in parent.branch_ids:
                        raise ValueError("Append branch ID conflicts with parent inventory")
                    try:
                        db.execute("INSERT INTO branches VALUES (?,?)", (branch_id, group_id))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("Duplicate branch identity") from exc
                if reason:
                    exclusions.append({"group_id": group_id, "reason": reason})
                    db.execute("INSERT INTO groups VALUES (?,?,?,?,?,?,?,?)", (group_id, 0, source_hash, None, None, 0, 0, None))
                    continue
                if enabled(config):
                    execution_parents.append(parent_record(value))
                from RL.dataset.schema import validate
                for row in value["members"]:
                    validate(row["prompt_token_ids"], row["completion_token_ids"],
                             row["prompt_sha256"], row["completion_sha256"])
                    if not proof["synthetic"] and row.get("tokenizer_fingerprint") != proof["tokenizer_fingerprint"]:
                        raise ValueError("Prepared branch tokenizer identity mismatch")
                if len(ids) % shard_groups == 0:
                    close_shard()
                    folder = stage / "shards" / f"shard-{len(shards):06d}"
                    folder.mkdir(parents=True)
                    streams = {name: (folder / name).open("wb") for name in ("prompt-tokens.bin", "completion-tokens.bin", "scores-advantages.bin")}
                    offsets = dict.fromkeys(streams, 0)
                def write_values(name, values, fmt):
                    start = offsets[name]
                    if fmt == "<q" and any(type(token) is not int or not 0 <= token < 2**63 for token in values):
                        raise ValueError("Token ID does not fit the int64 native contract")
                    from array import array
                    import sys
                    packed = array("q" if fmt == "<q" else "d", values)
                    if sys.byteorder != "little":
                        packed.byteswap()
                    streams[name].write(packed.tobytes())
                    offsets[name] += len(values)
                    return start
                payload = copy.deepcopy({k:v for k,v in value.items() if k != "members"})
                payload.pop("frozen_advantages", None)
                prompt = value["members"][0]["prompt_token_ids"]
                payload["storage"] = {"shard": folder.relative_to(stage).as_posix(),
                    "prompt_offset": write_values("prompt-tokens.bin", prompt, "<q"), "prompt_length": len(prompt)}
                payload["members"] = []
                tokens = 0
                for i, row in enumerate(value["members"]):
                    if row["prompt_token_ids"] != prompt:
                        raise ValueError("Prepared group prompt mismatch")
                    c = row["completion_token_ids"]
                    member = {k:v for k,v in row.items() if k not in {"prompt_token_ids", "completion_token_ids", "audit_enabled"}}
                    member["storage"] = [write_values("completion-tokens.bin", c, "<q"), len(c),
                        write_values("scores-advantages.bin", [row["historical_score"], value["frozen_advantages"][i]], "<d")]
                    payload["members"].append(member)
                    tokens += len(c)
                digest = hash_json(order_record(value, config.grpo.advantage_policy == CONSENSUS_ADVANTAGE_VERSION))
                db.execute("INSERT INTO groups VALUES (?,?,?,?,?,?,?,?)", (group_id, 1, source_hash, digest,
                    folder.relative_to(stage).as_posix(), len(value["members"]), tokens, _json(payload)))
                ids.append(group_id); advantages.extend(value["frozen_advantages"])
                prompt_lengths[group_id] = len(prompt)
            close_shard()
            db.commit()
        finally:
            db.close()
            if streams:
                for stream in streams.values():
                    stream.close()
        if not ids:
            raise ValueError("No eligible new GRPO groups after exclusions")
        seed = config.grpo.shuffle_seed if append_seed is None else append_seed
        algorithm = (legacy.proof.get("shuffle_algorithm", LEGACY_SHUFFLE)
                     if legacy is not None else config.grpo.shuffle_algorithm)
        if legacy is None:
            ids = order_groups(ids, seed=seed, batch_size=config.grpo.group_batch_size,
                algorithm=algorithm, prompt_length=prompt_lengths.__getitem__)
        if algorithm != LEGACY_SHUFFLE:
            proof.update(shuffle_metadata(algorithm))
        segments = copy.deepcopy(parent.schedule["segments"]) if parent else []
        segments.append(segment(ids, config.grpo.group_batch_size, seed, len(segments),
            segments[-1]["segment_id"] if segments else None, algorithm=algorithm))
        schedule = {"schema_version": VERSION, "schedule_id": artifact_id, "segments": segments,
                    "parent_schedule_hash": parent.schedule["sha256"] if parent else None,
                    "parent_last_segment_hash": parent.schedule["segments"][-1]["sha256"] if parent else None,
                    "extension_request_id": extension_id}
        if enabled(config):
            execution_plan = from_config(execution_parents, config)
            schedule["execution_schedule_sha256"] = execution_plan["sha256"]
        schedule["sha256"] = hash_json(schedule)
        if parent:
            validate_extension_schedule(parent.schedule, schedule)
        settings = config.grpo.model_dump(include={"top_k", "top_k_fraction", "score_scale_floor", "boundary_tie_sigma"})
        if config.grpo.advantage_policy == CONSENSUS_ADVANTAGE_VERSION:
            settings["advantage_policy"] = config.grpo.advantage_policy
        all_ids = (parent.ids if parent else []) + ids
        all_exclusions = (parent.proof["exclusions"] if parent else []) + exclusions
        branch_count = (parent.proof["branch_count"] if parent else 0) + len(advantages)
        all_total = (parent.proof["total_groups"] if parent else 0) + total
        proof.update(objective_version="offline_grpo_k3_v1", mask_policy=MASK_POLICY,
            preflight=config.preflight.model_dump(), advantage_policy=config.grpo.advantage_policy,
            advantage_config_sha256=hash_json(settings), group_order_seed=config.grpo.shuffle_seed,
            group_batch_size=config.grpo.group_batch_size, max_epochs=1, probability_mode="raw_t1",
            clip_epsilon=config.grpo.clip_epsilon, kl_beta=config.grpo.kl_beta,
            eligible_group_ids=all_ids, eligible_groups_sha256=hash_json(sorted(all_ids)),
            group_count=len(all_ids), branch_count=branch_count,
            planned_updates=sum(math.ceil(len(s["group_ids"])/s["group_batch_size"]) for s in segments),
            exclusions=all_exclusions, exclusion_reasons=dict(Counter(x["reason"] for x in all_exclusions)),
            total_groups=all_total, group_coverage=len(all_ids)/all_total)
        if enabled(config):
            apply_proof(proof, execution_plan)
        if config.training.max_updates != proof["planned_updates"]:
            raise ValueError(f"GRPO max_updates must equal one epoch's planned_updates={proof['planned_updates']}")
        with closing(_readonly(stage / "inventory.sqlite3")) as index:
            records = dict(parent.records) if parent else {}
            records.update({r[0]: {"content_sha256":r[1]} for r in index.execute("SELECT id,content_sha256 FROM groups WHERE eligible=1")})
        manifest = {"schema_version": VERSION, "prepared_id": artifact_id, "created_at": now(),
            "producer_fingerprint": producer, "parent": binding(parent.path) if parent else None,
            "training_schedule": schedule_contract(config), "proof": proof, "schedule": schedule,
            "shards": shards, "token_dtype": "<i8", "score_advantage_dtype": "<f8",
            "mask_encoding": "all_native_completion_targets_v1", "source_attempt": source_attempt,
            "group_inventory_sha256": hash_json({g: records[g]["content_sha256"] for g in sorted(records)}),
            "observations": {"source_scan_groups": total+overlap, "native_scan_groups": native_scanned,
                "materialized_groups": len(ids), "old_groups_compared": overlap}}
        # Temporarily read our unpublished arrays through the same bounds/hash reader.
        # No receipt is published until all consistency checks and order hashes pass.
        provisional = object.__new__(PreparedReader)
        provisional.path, provisional.parent = stage, parent
        provisional.local_ids, provisional.records, provisional.proof = set(ids), records, proof
        proof["group_order_sha256"] = (hash_json({"parent":parent.proof["group_order_sha256"],
            "appended":[records[g]["content_sha256"] for g in ids]}) if parent else
            hash_sequence(order_record(provisional.raw(g), config.grpo.advantage_policy == CONSENSUS_ADVANTAGE_VERSION) for g in ids))
        if legacy is not None and proof["group_order_sha256"] != legacy.proof["group_order_sha256"]:
            raise ValueError("Legacy frozen-group order/content hash differs; migration rejected")
        # Distribution combines compact frozen floats; token arrays remain bounded.
        if parent:
            old = parent.proof["advantage_distribution"]
            distribution = {"min":min(old["min"], min(advantages)), "max":max(old["max"], max(advantages)),
                "sum":old["sum"]+math.fsum(advantages),
                "positive":old["positive"]+sum(a>0 for a in advantages),
                "negative":old["negative"]+sum(a<0 for a in advantages)}
        else:
            distribution = {"min":min(advantages), "max":max(advantages), "sum":math.fsum(advantages),
                "positive":sum(a>0 for a in advantages), "negative":sum(a<0 for a in advantages)}
        distribution["mean"] = distribution["sum"]/branch_count
        proof["advantage_distribution"] = distribution
        write_json(stage / "manifest.json", manifest)
        write_json(stage / "schedule.json", schedule)
        write_json(stage / "dataset-binding.json", {k:proof[k] for k in ("dataset_id", "dataset_receipt_sha256", "tokenizer_fingerprint")})
        seal(stage, "rl-prepared", artifact_id, manifest_sha256=sha256(stage / "manifest.json"))
    return PreparedReader(destination)


class PreparedGroups:
    """Trainer protocol avoids list(groups) and validates counters from indexes."""
    def __init__(self, reader):
        self.reader = reader

    def __len__(self):
        return len(self.reader)

    def __iter__(self):
        return tensor_groups(iter(self.reader))

    def validate_state(self, state):
        self.reader.validate_state(state)

    def windows(self, start):
        return self.reader.windows(start)

    def cursor(self, count):
        return self.reader.cursor(count)


def summary(reader):
    return {"prepared_id": reader.manifest["prepared_id"], "group_count": len(reader),
            **({key: reader.proof[key] for key in ("planned_parent_groups", "planned_execution_chunks",
                "execution_schedule_sha256", "group_batch_semantics")} if reader.execution_plan else {}),
            "branch_count": reader.proof["branch_count"], "planned_updates": reader.proof["planned_updates"],
            "segments": len(reader.schedule["segments"]), "local_shards": len(reader.manifest["shards"]),
            "manifest_sha256": sha256(reader.path / "manifest.json"),
            "schedule_sha256": reader.schedule["sha256"], "parent": reader.manifest["parent"]}
