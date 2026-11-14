"""One ordered token producer with explicit selection and bounded host ownership."""
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Condition, Event, Thread

from core.io import read_json, sha256
from ..loader import iter_groups, native_tokens
from ..materialize import CacheReader


def selected_groups(dataset, config, *, with_tokens=False):
    wanted = set(config.selection.group_ids); seen = set()
    missing = None
    if config.selection.coverage_plan:
        from ..probability_artifacts import replay_selection
        plan = replay_selection(config.selection.coverage_plan, dataset)
        missing = {(r["group_id"], r["branch_sample_id"]): r for r in plan["missing"]}
    consumed = set()
    for group in iter_groups(dataset, with_tokens=with_tokens):
        gid = group["group"]["group_id"]
        if wanted and gid not in wanted: continue
        seen.add(gid)
        if missing is not None:
            members = []
            for row in group["members"]:
                key = (gid, row["branch_sample_id"])
                if key not in missing: continue
                from ..probability import BINDINGS
                target = missing[key]
                if any(row[k] != target[k] for k in (*BINDINGS, "mask_spans")):
                    raise ValueError("Frozen missing branch coordinates changed")
                if with_tokens:
                    from core.io import hash_json
                    if (hash_json(row["prompt_token_ids"]) != row["prompt_sha256"]
                            or hash_json(row["completion_token_ids"]) != row["completion_sha256"]):
                        raise ValueError("Frozen missing branch token hash changed")
                consumed.add(key); members.append(row)
            if not members: continue
            group = {**group, "members": members}
        yield group
    if missing is not None and consumed != set(missing):
        raise ValueError("Frozen missing branches are absent or ineligible")
    if wanted - seen: raise ValueError("Selected group IDs are missing or ineligible: " + str(sorted(wanted - seen)[:10]))


class Prefetch:
    """CacheReader is created and used solely on this producer thread.

    Leases include expanded Python tokens, metadata, request serialization and
    pending result vectors; groups stay charged until all requests are finished.
    """
    END = object()

    def __init__(self, dataset, cache, config, *, vocab_size=None, group_ids=None, completed_ids=()):
        self.dataset, self.cache, self.config = dataset, cache, config
        self.group_ids = None if group_ids is None else set(group_ids)
        self.completed_ids = set(completed_ids)
        self.limit = int(config.io.prefetch_memory_budget_gib * 1024**3)
        self.prompt_budget = min(64*1024**2, self.limit // 8)
        self.limit -= self.prompt_budget
        self.vocab_size = vocab_size
        self.used = 0; self.peak = 0
        self.condition = Condition(); self.stop = Event()
        self.queue = Queue(config.io.prefetch_groups)
        self.thread = Thread(target=self._produce, name="replay-token-producer", daemon=True)
        self.thread.start()

    def _send(self, item):
        while not self.stop.is_set():
            try: self.queue.put(item, timeout=.1); return
            except Full: continue

    def _produce(self):
        try:
            audit = read_json(self.dataset / "dataset.json").get("audit", {}).get("enabled", False)
            reader = CacheReader(self.cache, sha256(self.dataset / "receipt.json"), prompt_budget_bytes=self.prompt_budget) if self.cache else None
            with ThreadPoolExecutor(max_workers=self.config.io.ledger_read_workers) as pool:
                for group in selected_groups(self.dataset, self.config):
                    if self.group_ids is not None and group['group']['group_id'] not in self.group_ids:
                        continue
                    if self.completed_ids:
                        group = {**group, 'members': [r for r in group['members'] if r['branch_sample_id'] not in self.completed_ids]}
                        if not group['members']:
                            continue
                    members = group["members"]
                    # Conservative upper bound includes two serialized copies of the
                    # full prompt for every active request plus completion/output.
                    charge = sum(128*(r["prompt_count"] + r["completion_count"]) + 16384 for r in members)
                    if charge > self.limit: raise ValueError("One full group exceeds expanded host memory budget")
                    with self.condition:
                        while self.used + charge > self.limit and not self.stop.is_set(): self.condition.wait(.1)
                        if self.stop.is_set(): return
                        self.used += charge; self.peak = max(self.peak, self.used)
                    if reader:
                        payloads = map(reader.tokens, members)
                    else:
                        payloads = pool.map(lambda row: native_tokens(row, audit=audit), members)
                    shared = {}
                    for row, (prompt, completion) in zip(members, payloads):
                        if self.vocab_size is not None and any(t >= self.vocab_size for t in prompt + completion): raise ValueError("Native token IDs exceed historical model vocabulary")
                        if self.config.selection.coverage_plan:
                            from core.io import hash_json
                            if hash_json(prompt) != row["prompt_sha256"] or hash_json(completion) != row["completion_sha256"]:
                                raise ValueError("Frozen missing branch token hash changed")
                        key = row["prompt_sha256"]
                        if key in shared and prompt != shared[key]: raise ValueError("Group prompt payload differs")
                        prompt = shared.setdefault(key, prompt)
                        row.update(prompt_token_ids=prompt, completion_token_ids=completion)
                    group["host_charge"] = charge
                    self._send(group)
                    if self.stop.is_set(): return
            self._send(self.END)
        except BaseException as error:
            self._send(error)

    def get(self, *, block=True):
        try: item = self.queue.get(block=block, timeout=.1 if block else None)
        except Empty: return None
        if isinstance(item, BaseException): raise item
        return item

    def release(self, group):
        with self.condition:
            self.used -= group["host_charge"]; self.condition.notify_all()

    def close(self):
        self.stop.set()
        with self.condition: self.condition.notify_all()
        self.thread.join(timeout=30)
        if self.thread.is_alive(): raise RuntimeError("Replay input producer did not stop")
