"""One scheduler for all methods, serving replicas, scans, and retry attempts."""

import math
import random
import time
from collections import Counter

from core.io import hash_json


class MixedQueue:
    def __init__(self, items, settings, workers, journal=None):
        self.settings, self.workers, self.journal = settings, workers, journal
        ordered = sorted(items, key=lambda x: x["sample_key"])
        random.Random(settings["schedule_seed"]).shuffle(ordered)
        self.rank = {x["sample_key"]: i for i, x in enumerate(ordered)}
        self.family = {
            x["sample_key"]: x.get("continuation", {}).get("family") or x["instance_id"].split("__")[0]
            for x in ordered
        }
        self.active = {}
        self.last_service = {}
        self.not_before = {}
        self.cursor = 0
        if journal and journal.exists():
            from rollout.key_steps.state import rows

            for row in rows(journal):
                if row["event"] == "started":
                    self.cursor += 1
                    for service in row["services"]:
                        self.last_service[service] = row["at_unix"]
                if row["event"] == "retry":
                    self.not_before[row["sample_key"]] = row["at_unix"] + settings["retry_backoff_seconds"]

    def services(self, item):
        return self.settings["task_services"].get(
            item["instance_id"],
            [self.settings["network_default"]] if self.settings["network_default"] != "local-only" else [],
        )

    def pop(self, pending, clock=None):
        clock = time.time() if clock is None else clock
        tasks = {x["instance_id"] for x in self.active.values()}
        families = Counter(self.family[k] for k in self.active)
        services = {s for x in self.active.values() for s in self.services(x)}
        nfamilies = len({self.family[x["sample_key"]] for x in pending})
        cap = max(1, math.ceil(self.workers / max(1, nfamilies)))
        eligible = [
            x
            for x in pending
            if x["instance_id"] not in tasks
            and families[self.family[x["sample_key"]]] < cap
            and self.not_before.get(x["sample_key"], 0) <= clock
            and not services.intersection(self.services(x))
            and all(
                clock - self.last_service.get(s, -float("inf")) >= self.settings["service_interval_seconds"]
                for s in self.services(x)
            )
        ]
        if not eligible:
            return None
        # Family tie-break rotates independently of branch rank.
        names = sorted(
            {self.family[x["sample_key"]] for x in pending},
            key=lambda f: hash_json([self.settings["schedule_seed"], f]),
        )
        rotation = names[self.cursor % len(names) :] + names[: self.cursor % len(names)]
        item = min(
            eligible,
            key=lambda x: (
                families[self.family[x["sample_key"]]],
                rotation.index(self.family[x["sample_key"]]),
                self.rank[x["sample_key"]],
            ),
        )
        pending.remove(item)
        self.active[item["sample_key"]] = item
        for service in self.services(item):
            self.last_service[service] = clock
        self.cursor += 1
        self.record("started", item, clock)
        return item

    def release(self, item, retry=False, clock=None):
        clock = time.time() if clock is None else clock
        self.active.pop(item["sample_key"], None)
        if retry:
            self.not_before[item["sample_key"]] = clock + self.settings["retry_backoff_seconds"]
        self.record("retry" if retry else "finished", item, clock)

    def record(self, event, item, clock):
        if self.journal:
            from rollout.key_steps.state import append

            append(
                self.journal,
                {
                    "event": event,
                    "sample_key": item["sample_key"],
                    "rank": self.rank[item["sample_key"]],
                    "family": self.family[item["sample_key"]],
                    "services": self.services(item),
                    "at_unix": clock,
                },
            )
