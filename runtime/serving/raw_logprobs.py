"""Raw T=1 request sidecars; no training roles or objective policy live here."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import threading

import numpy as np

from core.disk_budget import DiskBudget
from core.io import hash_json, owned_path, sha256

SEMANTICS = "raw_t1_sampled_token_v1"
STREAM = "raw_t1_output_tokens"
COLLECTOR = "dycon_vllm_v2_pre_grammar_target_v1"


def validate_descriptor(descriptor):
    if descriptor.get("probability_semantics") != SEMANTICS:
        raise ValueError("Raw probability semantics mismatch")
    if (type(descriptor.get("normalization_temperature")) is not float or
            descriptor.get("normalization_temperature") != 1.0 or
            descriptor.get("sampling_transforms_applied") is not False or
            descriptor.get("target_only") is not True or
            descriptor.get("values_dtype") != "float32"):
        raise ValueError("Raw T=1 pre-transform target-only contract mismatch")
    if descriptor.get("status") != "sealed":
        raise ValueError("Raw probability sidecar is not sealed")


def read_values(root, descriptor, *, output_ids=None, record=None, verify_hashes=True):
    validate_descriptor(descriptor)
    path = owned_path(root, descriptor["locator"])
    count = descriptor["count"]
    if (type(count) is not int or count < 0 or descriptor["bytes"] != count * 4 or
            (verify_hashes and sha256(path) != descriptor["sha256"])):
        raise ValueError("Raw probability shard length/checksum mismatch")
    if output_ids is not None and (len(output_ids) != count or
            (verify_hashes and hash_json(output_ids) != descriptor["output_token_ids_sha256"])):
        raise ValueError("Raw probability output token alignment mismatch")
    if record is not None:
        for key in ("request_id", "run_id", "session_id", "decision_id", "branch_id", "step_id", "api_attempt_id"):
            if descriptor["identity"].get(key) != record.get(key):
                raise ValueError("Raw probability request identity mismatch: " + key)
        if verify_hashes and descriptor["prompt_token_ids_sha256"] != hash_json(record["prompt_token_ids"]):
            raise ValueError("Raw probability prompt alignment mismatch")
        if descriptor["tokenizer_fingerprint"] != record.get("tokenizer_fingerprint"):
            raise ValueError("Raw probability tokenizer identity mismatch")
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != count * 4:
            raise ValueError("Raw probability shard length mismatch")
        values = np.fromfile(stream, dtype="<f4")
    if len(values) != count:
        raise ValueError("Raw probability shard truncated")
    if np.isnan(values).any() or np.isposinf(values).any() or (values > 0).any():
        raise ValueError("Raw probability values contain NaN or +inf")
    if int(np.isneginf(values).sum()) != descriptor["negative_infinity_count"]:
        raise ValueError("Raw probability nonfinite summary mismatch")
    if int(np.isfinite(values).sum()) != descriptor["finite_count"]:
        raise ValueError("Raw probability finite summary mismatch")
    return values


class RawLogprobWriter:
    """Bounded writer, independent of activation queues, sharing spool governance."""
    def __init__(self, project, root, options, *, spool=None, source_identity=None):
        self.root = Path(root)
        self.options = options
        self.spool = spool
        self.source = dict(source_identity or {})
        self.capacity = threading.BoundedSemaphore(options.get("queue_size", 64))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raw-t1-writer")
        # Separate quota ledger; shared physical spool admission still accounts
        # these bytes through spool.write_bytes(). A smaller raw quota must not
        # accidentally compare against all activation reservations in the project.
        self.budget = DiskBudget(self.root / "raw-logprob-budget", self.root / "raw-logprob-budget",
                                options.get("spool_max_bytes", 1073741824))
        self.lock = threading.Lock()
        self.pending = 0
        self.peak_pending = 0
        self.written_bytes = 0
        self.failures = 0

    def write(self, record, values):
        if not self.capacity.acquire(blocking=False):
            raise RuntimeError("Raw probability writer queue is full")
        with self.lock:
            self.pending += 1
            self.peak_pending = max(self.peak_pending, self.pending)
        future = self.pool.submit(self._write, record, values)
        try:
            return future.result()
        except Exception:
            with self.lock:
                self.failures += 1
            raise
        finally:
            with self.lock:
                self.pending -= 1
            self.capacity.release()

    def _write(self, record, values):
        if values is None or not isinstance(values, (list, np.ndarray)):
            raise ValueError("Raw probability vector is missing")
        array = np.asarray(values, dtype="<f4")
        count = len(record["output_token_ids"])
        if array.ndim != 1 or len(array) != count:
            raise ValueError("Raw probability vector length mismatch")
        if count > self.options.get("max_request_tokens", 131072):
            raise ValueError("Raw probability request token budget exceeded")
        if array.nbytes > self.options.get("shard_max_bytes", 4194304):
            raise ValueError("Raw probability shard byte budget exceeded")
        if np.isnan(array).any() or np.isposinf(array).any() or (array > 0).any():
            raise ValueError("Invalid raw probability values")
        relative = "requests/" + hash_json(record["request_id"]) + ".raw_t1.f32"
        payload = array.tobytes()
        reservation = "raw-t1:" + str(self.root / relative)
        self.budget.acquire(reservation, len(payload) * 2 + 4096)
        try:
            if self.spool:
                path = self.spool.write_bytes(relative, payload, immutable=True)
            else:
                path = owned_path(self.root, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if path.read_bytes() != payload:
                        raise ValueError("Immutable raw probability shard changed")
                else:
                    part = path.with_suffix(path.suffix + ".part")
                    with part.open("wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(part, path)
            self.budget.acquire(reservation, len(payload))
        except Exception:
            self.budget.release(reservation)
            raise
        with self.lock:
            self.written_bytes += len(payload)
        return {"schema_version": 1, "status": "sealed", "origin": "online_rollout",
                "probability_semantics": SEMANTICS, "stream": STREAM,
                "normalization_temperature": 1.0, "sampling_transforms_applied": False,
                "target_only": True, "values_dtype": "float32", "count": count,
                "locator": relative, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "negative_infinity_count": int(np.isneginf(array).sum()),
                "finite_count": int(np.isfinite(array).sum()),
                "output_token_ids_sha256": hash_json(record["output_token_ids"]),
                "prompt_token_ids_sha256": hash_json(record["prompt_token_ids"]),
                "tokenizer_fingerprint": record.get("tokenizer_fingerprint"),
                "identity": {k: record.get(k) for k in ("request_id", "run_id", "sample_key", "rollout_id",
                             "session_id", "decision_id", "branch_id", "step_id", "api_attempt_id")},
                "source_identity": self.source,
                "generation_identity": record.get("raw_t1_generation_identity"),
                "collector": COLLECTOR}

    def status(self):
        with self.lock:
            return {"pending": self.pending, "peak_pending": self.peak_pending,
                    "written_bytes": self.written_bytes, "failures": self.failures}

    def close(self):
        self.pool.shutdown(wait=True)
