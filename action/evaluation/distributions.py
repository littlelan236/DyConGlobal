"""Versioned descriptive matrices; never connect separate canonical trajectories."""

from collections import Counter, defaultdict

from action.analyzers.taxonomy import LABELS
from action.datasets.trajectories import source_key

PHASE_SCHEMA = "phase5-endpoints-v1"
TRANSITION_SCHEMA = "adjacent-assistant-row-conditional-v1"
PHASE_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def phase_metadata(step, total):
    denominator = max(1, total - 1)
    phase = min(4, 5 * step // denominator)
    return {"phase_bin": phase, "phase_start_fraction": phase / 5,
            "phase_end_fraction": (phase + 1) / 5, "phase_denominator": denominator}


def matrices(rows, sources):
    labels = list(LABELS)
    by_source = {source_key(s): s for s in sources}
    trajectories = {key: [] for key in by_source}
    phases = [Counter() for _ in range(5)]
    trajectory_phases = {key: [Counter() for _ in range(5)] for key in by_source}
    for row in rows:
        key = source_key(row)
        label, phase = row["label"], row["phase_bin"]
        if key not in by_source or label not in LABELS or type(phase) is not int or not 0 <= phase < 5:
            raise ValueError("Unknown action source, label or phase")
        trajectories[key].append(row)
        phases[phase][label] += 1
        trajectory_phases[key][phase][label] += 1
    denominators = [sum(counts.values()) for counts in phases]
    rollout_denominators = [sum(bool(p[i]) for p in trajectory_phases.values()) for i in range(5)]
    counts = [[phase[label] for phase in phases] for label in labels]
    micro = [[count / den if den else None for count, den in zip(line, denominators)] for line in counts]
    macro = [[sum(p[i][label] / sum(p[i].values()) for p in trajectory_phases.values() if p[i]) / rollout_denominators[i]
              if rollout_denominators[i] else None for i in range(5)] for label in labels]
    pairs = Counter()
    for sequence in trajectories.values():
        sequence.sort(key=lambda row: row["step_id"])
        if [r["step_id"] for r in sequence] != list(range(len(sequence))):
            raise ValueError("Action step sequence has duplicates or gaps")
        for current, following in zip(sequence, sequence[1:]):
            pairs[current["label"], following["label"]] += 1
    transition_counts = [[pairs[a, b] for b in labels] for a in labels]
    transition_denominators = [sum(line) for line in transition_counts]
    total_pairs = sum(transition_denominators)
    return {
        "phase": {"schema": PHASE_SCHEMA, "axis_order": ["action", "phase"], "labels": labels,
                  "bins": PHASE_LABELS, "counts": counts, "micro": micro, "macro": macro,
                  "action_denominators": denominators, "rollout_denominators": rollout_denominators,
                  "definition": "min(4,floor(5*s/max(1,T-1))); descriptive post-hoc position",
                  "empty_bin": None},
        "transition": {"schema": TRANSITION_SCHEMA, "axis_order": ["current_action", "next_action"],
                       "labels": labels, "counts": transition_counts, "row_denominators": transition_denominators,
                       "probabilities": [[v / den if den else None for v in line]
                                         for line, den in zip(transition_counts, transition_denominators)],
                       "frequencies": [[v / total_pairs if total_pairs else None for v in line] for line in transition_counts],
                       "pairs": total_pairs, "single_step_trajectories": sum(len(v) == 1 for v in trajectories.values()),
                       "zero_step_trajectories": sum(not v for v in trajectories.values()), "empty_row": None},
    }


def summarize(rows, sources):
    data = matrices(rows, sources)
    counts = Counter(row["label"] for row in rows)
    cohorts = sorted({s.get("cohort") or "selected" for s in sources})
    run_ids = sorted({source_key(s)[0] for s in sources})
    result = {
        "sources": len(sources), "steps": len(rows), "repos": len({s["repo"] for s in sources}),
        "models": len({s["model_fingerprint"] for s in sources}),
        "category_counts": {k: counts[k] for k in LABELS},
        "category_fractions": {k: counts[k] / len(rows) if rows else None for k in LABELS},
        "compound_actions": sum(r["compound"] for r in rows),
        "uncertain_actions": sum(bool(r["uncertainties"]) for r in rows),
        "uncertainty_counts": dict(Counter(x for r in rows for x in r["uncertainties"])),
        "rules": dict(Counter(x for r in rows for x in r["rules"])),
        "cohort_rollouts": dict(Counter(s.get("cohort") or "selected" for s in sources)),
        "source_run_ids": run_ids, **data,
    }
    # Preserve the legacy field orientations for existing report consumers.
    result["phase_definition"] = data["phase"]["definition"]
    result["phase_counts_micro"] = [{k: data["phase"]["counts"][j][i] for j, k in enumerate(LABELS)} for i in range(5)]
    result["phase_fractions_micro"] = [{k: data["phase"]["micro"][j][i] for j, k in enumerate(LABELS)} for i in range(5)]
    result["phase_fractions_macro"] = [{k: data["phase"]["macro"][j][i] for j, k in enumerate(LABELS)} for i in range(5)]
    result["phase_rollout_counts"] = data["phase"]["rollout_denominators"]
    result["by_source_run"], result["by_cohort"] = {}, {}
    for field, values, destination in [("source_run_id", run_ids, "by_source_run"), ("cohort", cohorts, "by_cohort")]:
        for value in values:
            group = [s for s in sources if (source_key(s)[0] if field == "source_run_id" else s.get("cohort") or "selected") == value]
            keys = {source_key(s) for s in group}
            result[destination][value] = matrices([r for r in rows if source_key(r) in keys], group)
    result["cohort_phase_counts"] = {
        name: [{label: group["phase"]["counts"][j][i] for j, label in enumerate(LABELS)
                if group["phase"]["counts"][j][i]} for i in range(5)]
        for name, group in result["by_cohort"].items()
    }
    return result
