"""Default adaptive detector; selection sees only full scores and step IDs."""

import math

import numpy as np
from scipy.signal import find_peaks

VERSION = "full_branch_adaptive_v1"


def mad_scale(values):
    v = np.asarray(values, dtype=float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else 0.0


def detect(steps, distance=3, wlen=9):
    steps = sorted(steps, key=lambda x: x["step"])
    if len({s["step"] for s in steps}) != len(steps):
        raise ValueError("Duplicate step ID")
    valid = [s for s in steps if s["complete"]]
    if not valid:
        return [], {}
    values = np.array([s["std"] for s in valid])
    baseline = float(np.median(values))
    scale = max(mad_scale(values), 0.05)
    diffs = [b["std"] - a["std"] for a, b in zip(valid, valid[1:]) if b["step"] == a["step"] + 1]
    noise = mad_scale(diffs) / math.sqrt(2)
    threshold = max(0.10, 1.5 * noise)
    segments = []
    for point in valid:
        if not segments or point["step"] != segments[-1][-1]["step"] + 1:
            segments.append([])
        segments[-1].append(point)
    peaks = []
    for segment in segments:
        if len(segment) < 3:
            continue
        y = np.array([s["std"] for s in segment])
        ids, props = find_peaks(y, prominence=threshold, distance=distance, wlen=wlen)
        candidates = [(int(i), float(p), "interior") for i, p in zip(ids, props["prominences"])]
        for i, edge in [(0, "left"), (len(segment) - 1, "right")]:
            if segment[i]["step"] != steps[0 if i == 0 else -1]["step"]:
                continue
            neighbor = y[1 : min(5, len(y))] if i == 0 else y[max(0, i - 4) : i]
            rise = float(y[i] - max(neighbor))
            if rise >= threshold:
                candidates.append((i, rise, edge))
        accepted = []
        for i, p, edge in sorted(candidates, key=lambda c: (-y[c[0]], c[0])):
            if (y[i] - baseline) / scale < 0:
                continue
            if any(abs(i - other[0]) < distance for other in accepted):
                continue
            accepted.append((i, p, edge))
        for i, p, edge in sorted(accepted):
            peaks.append(
                dict(
                    step=segment[i]["step"],
                    std=float(y[i]),
                    prominence=p,
                    prominence_threshold=threshold,
                    robust_z=float((y[i] - baseline) / scale),
                    detector_step=segment[i]["step"],
                    edge=edge,
                )
            )
    return sorted(peaks, key=lambda x: x["step"]), dict(
        baseline=baseline, scale=scale, noise=noise, threshold=threshold
    )


def signal(step, branches, expected=8):
    ids = [b.get("branch_id") for b in branches]
    complete = (
        len(ids) == expected
        and all(type(i) is int for i in ids)
        and set(ids) == set(range(expected))
        and all(type(b.get("score")) in (float, int) and math.isfinite(b["score"]) for b in branches)
    )
    return {
        "step": step,
        "complete": complete,
        "std": float(np.std([b["score"] for b in branches], ddof=0)) if complete else None,
    }
