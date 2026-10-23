"""Descriptive action distributions by post-hoc trajectory position."""

from collections import Counter, defaultdict

from action.analyzers.taxonomy import LABELS


def legacy_summarize(rows, sources):
    counts = Counter(row["label"] for row in rows)
    phases = [Counter() for _ in range(5)]
    groups = defaultdict(lambda: [Counter() for _ in range(5)])
    cohorts = defaultdict(lambda: [Counter() for _ in range(5)])
    by_id = {s["rollout_id"]: s for s in sources}
    for row in rows:
        label, phase = row["label"], row["phase_bin"]
        phases[phase][label] += 1
        groups[row["rollout_id"]][phase][label] += 1
        cohorts[by_id[row["rollout_id"]].get("cohort") or "selected"][phase][label] += 1
    macro = []
    for phase in range(5):
        valid = [g[phase] for g in groups.values() if sum(g[phase].values())]
        macro.append(
            {k: sum(c[k] / sum(c.values()) for c in valid) / len(valid) if valid else None for k in LABELS}
        )
    return {
        "sources": len(sources),
        "steps": len(rows),
        "repos": len({s["repo"] for s in sources}),
        "models": len({s["model_fingerprint"] for s in sources}),
        "category_counts": {k: counts[k] for k in LABELS},
        "category_fractions": {k: counts[k] / len(rows) if rows else None for k in LABELS},
        "compound_actions": sum(r["compound"] for r in rows),
        "uncertain_actions": sum(bool(r["uncertainties"]) for r in rows),
        "uncertainty_counts": dict(Counter(x for r in rows for x in r["uncertainties"])),
        "rules": dict(Counter(x for r in rows for x in r["rules"])),
        "cohort_rollouts": dict(Counter(s.get("cohort") or "selected" for s in sources)),
        "phase_definition": "min(4, floor(5*s/max(1,T-1))); descriptive only, never an input feature",
        "phase_counts_micro": [{k: c[k] for k in LABELS} for c in phases],
        "phase_fractions_macro": macro,
        "phase_rollout_counts": [sum(bool(g[i]) for g in groups.values()) for i in range(5)],
        "cohort_phase_counts": {k: list(map(dict, v)) for k, v in cohorts.items()},
    }


# Versioned project-level analysis statistics.
from action.evaluation.distributions import summarize
