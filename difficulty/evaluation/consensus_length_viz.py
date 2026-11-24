"""Receipt-bound rank heatmaps and token-excess distributions for consensus evaluations."""

import csv
import hashlib
import html
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import rankdata

from core.io import artifact_dir, identifier, now, sha256, write_json
from core.locking import exclusive_artifact

WEIGHTING = "equal tasks -> rollouts -> complete multi-member steps -> branches (heatmap only)"
CODE_FILES = ("difficulty/evaluation/consensus_length_viz.py", "core/cli.py")
FILES = ("summary.json", "steps.csv", "branches.csv")


def rank_normalized(values):
    """Average tied zero-based ranks divided by n-1; constants sit at 0.5."""
    if len(values) < 2:
        return [0.0] * len(values)
    return ((rankdata(values, method="average") - 1) / (len(values) - 1)).tolist()


def _key(row):
    return tuple(row[k] for k in ("rollout_id", "session_id", "decision_id"))


def _finite(row, field):
    try:
        value = float(row[field])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Missing numeric {field} in complete group") from exc
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {field} in complete group")
    return value


def build_plot_data(steps, branches):
    """Use complete non-singleton groups with the evaluation's task-balanced weights."""
    if len({_key(s) for s in steps}) != len(steps):
        raise ValueError("Duplicate decision in steps CSV")
    selected = {_key(s): s for s in steps if s["status"] == "comparable"}
    if not selected:
        raise ValueError("No complete multi-member groups available")
    members = defaultdict(list)
    for row in branches:
        if _key(row) in selected:
            members[_key(row)].append(row)
    tasks = defaultdict(lambda: Counter())
    for step in selected.values():
        tasks[step["task_key"]][step["rollout_id"]] += 1
    points, excess = [], []
    for key, step in selected.items():
        group = members[key]
        count = int(step["winner_count"])
        if count < 2 or len(group) != count or len({b["branch_id"] for b in group}) != count:
            raise ValueError("Winner group inventory differs from steps CSV")
        if any(b["task_key"] != step["task_key"] for b in group):
            raise ValueError("Winner group task identity differs")
        lengths = [_finite(b, "completion_tokens") for b in group]
        scores = [_finite(b, "score") for b in group]
        if any(x < 0 or not x.is_integer() for x in lengths):
            raise ValueError("Invalid completion token count")
        best = min(range(count), key=lambda i: (scores[i], int(group[i]["branch_id"])))
        token_excess = lengths[best] - min(lengths)
        if _finite(step, "token_excess") != token_excess:
            raise ValueError("Token excess differs from winner candidates")
        rollouts = tasks[step["task_key"]]
        weight = 1 / (len(tasks) * len(rollouts) * rollouts[step["rollout_id"]])
        binding = {k: step[k] for k in ("task_key", "rollout_id", "session_id", "decision_id")}
        excess.append({**binding, "token_excess": token_excess, "weight": weight})
        for row, x, y in zip(group, rank_normalized(lengths), rank_normalized(scores)):
            points.append({**binding, "branch_id": int(row["branch_id"]),
                           "length_rank": x, "score_rank": y, "weight": weight / count})
    return {"tasks": len(tasks), "rollouts": sum(len(r) for r in tasks.values()),
            "steps": len(selected), "branches": len(points),
            "excluded_step_statuses": dict(Counter(s["status"] for s in steps if s["status"] != "comparable")),
            "rank_points": points, "excess": excess}


def weighted_ecdf(excess):
    """Coalesce ties so F(x) includes the full probability mass at x, including x=0."""
    masses = defaultdict(float)
    for row in excess:
        masses[row["token_excess"]] += row["weight"]
    total, cumulative, rows = sum(masses.values()), 0.0, []
    for value, mass in sorted(masses.items()):
        cumulative += mass / total
        rows.append({"token_excess": value, "cdf": min(cumulative, 1.0)})
    rows[-1]["cdf"] = 1.0
    return rows


def read_evaluation(cfg, evaluation_id):
    directory = artifact_dir(cfg, "difficulty", identifier(evaluation_id), "evaluations")
    raw_receipt = (directory / "receipt.json").read_bytes()
    receipt = json.loads(raw_receipt)
    if receipt.get("state") != "complete":
        raise ValueError("Source evaluation receipt is incomplete")
    raw = {name: (directory / name).read_bytes() for name in FILES}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    if any(hashes[name] != receipt["files"].get(name) for name in FILES):
        raise ValueError(f"Source evaluation checksum mismatch: {evaluation_id}")
    summary = json.loads(raw["summary.json"])
    rows = {name: list(csv.DictReader(io.StringIO(raw[name].decode("utf-8"))))
            for name in ("steps.csv", "branches.csv")}
    data = build_plot_data(rows["steps.csv"], rows["branches.csv"])
    if data["steps"] != summary["coverage"]["step_status"].get("comparable", 0):
        raise ValueError("Plot group coverage differs from evaluation summary")
    run_id = summary["run_id"]
    label = {"verified80": "Verified80", "skillsbench": "SkillsBench"}.get(run_id.split("-")[0], run_id)
    source = {"evaluation_id": evaluation_id, "run_id": run_id, "label": label,
              "directory": str(directory), "input_sha256": hashes,
              "receipt_sha256": hashlib.sha256(raw_receipt).hexdigest(),
              "coverage": {k: v for k, v in data.items() if k not in {"rank_points", "excess"}}}
    return source, data


def heatmap_matrix(points):
    matrix, _, _ = np.histogram2d([p["length_rank"] for p in points], [p["score_rank"] for p in points],
                                 bins=(np.linspace(0, 1, 11), np.linspace(0, 1, 11)),
                                 weights=[p["weight"] for p in points])
    return matrix.T / matrix.sum()


def _save(fig, stage, name, plt):
    try:
        for extension in ("png", "svg"):
            fig.savefig(stage / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def render(stage, sources, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    matrices = [heatmap_matrix(d["rank_points"]) for d in data]
    max_density = max(m.max() for m in matrices)
    fig, axes = plt.subplots(1, len(sources), figsize=(5.6 * len(sources) + 1, 5.6),
                             squeeze=False, layout="constrained")
    for ax, source, d, matrix in zip(axes[0], sources, data, matrices):
        image = ax.imshow(matrix, origin="lower", extent=(0, 1, 0, 1),
                          vmin=0, vmax=max_density, cmap="YlOrRd", aspect="equal")
        ax.plot([0, 1], [0, 1], color="#252525", linewidth=1, linestyle="--", alpha=.8)
        ax.set_title(f'{source["label"]}\n{d["tasks"]} tasks | {d["steps"]:,} steps | {d["branches"]:,} branches')
        ax.set_xlabel("Completion-token rank (0 = shortest)")
        ax.set_ylabel("Estimator-score rank (0 = lowest score)")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
    bar = fig.colorbar(image, ax=list(axes[0]), shrink=.75, label="Weighted probability per bin")
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Do shorter branches also rank better by estimator?\nWithin each historical winning group; task-balanced weights", fontsize=13)
    _save(fig, stage, "rank_heatmap", plt)

    curves = [weighted_ecdf(d["excess"]) for d in data]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, layout="constrained")
    colors = ("#bd3c32", "#2468a2", "#32734b", "#7d4a9b")
    global_max = max(curve[-1]["token_excess"] for curve in curves)
    for index, (source, curve) in enumerate(zip(sources, curves)):
        xs = [r["token_excess"] for r in curve]
        ys = [r["cdf"] for r in curve]
        zero_mass = ys[0] if xs[0] == 0 else 0.0
        for ax in axes:
            ax.step([0] + xs + [max(global_max * 1.05, 100)], [0] + ys + [1],
                    where="post", color=colors[index % len(colors)], linewidth=2,
                    label=f'{source["label"]}: F(0) = {zero_mass:.1%}')
            if xs[0] == 0:
                ax.scatter([0], [zero_mass], color=colors[index % len(colors)], zorder=4, s=22)
    axes[0].set_xlim(0, 100)
    axes[0].set_title("Most differences: 0–100 tokens")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlim(0, max(global_max * 1.05, 1))
    axes[1].set_title("Full tail (symlog scale)")
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Estimator-best tokens − shortest tokens")
        ax.grid(True, alpha=.22)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Cumulative share of comparable steps (task-balanced)")
    fig.suptitle("How much longer is the estimator-best branch?", fontsize=13)
    _save(fig, stage, "token_excess_ecdf", plt)
    write_json(stage / "rank_heatmap.json", {"bins": np.linspace(0, 1, 11).tolist(),
               "matrix_axes": "rows = estimator rank; columns = length rank; both ascending",
               "weighting": WEIGHTING, "sources": sources, "matrices": [m.tolist() for m in matrices]})
    write_json(stage / "ecdf.json", {"weighting": WEIGHTING, "sources": sources, "curves": curves})


def write_html(stage, sources):
    rows = "".join(f'<li><strong>{html.escape(s["label"])}</strong>：{html.escape(s["run_id"])}'
                   f'<br>评测：{html.escape(s["evaluation_id"])}</li>' for s in sources)
    figures = []
    for name, title, explanation in (
        ("rank_heatmap", "组内排名热力图", "横轴越小，分支越短；纵轴越小，estimator 分数越优。"
         "颜色越深表示加权概率越高，两图共享色标。若排序一致，概率应沿虚线分布。并列名次取平均；"
         "常数列位于 0.5，且小组排名离散，集中点并不单独代表强相关。"),
        ("token_excess_ecdf", "Token excess 累积分布", "横轴为 estimator 最优分支比最短分支多出的 completion tokens。"
         "曲线在 x 处的高度表示差值不超过 x 的加权比例。左图看 0–100 tokens，右图看完整长尾。"
         "F(0) 是长度相同的比例，可能高于选中同一分支的比例。"),
    ):
        svg = (stage / f"{name}.svg").read_text(encoding="utf-8")
        svg = svg[svg.index("<svg"):]
        figures.append(f'<section><h2>{title}</h2><p>{explanation}</p>{svg}'
                       f'<p>下载 <a href="{name}.png">PNG</a> · <a href="{name}.svg">SVG</a></p></section>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>共识分支长度与 estimator 可视化</title><style>'
            'body{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;'
            'color:#243044;background:#f5f6f8}section{background:white;padding:24px;margin:24px 0;border-radius:12px}'
            'svg{width:100%;height:auto}li{margin:10px 0;overflow-wrap:anywhere}a{color:#2367a0}</style>'
            '<h1>最短分支与 estimator 最优分支</h1><p>仅比较历史获胜共识组内、长度和分数完整的多成员组。'
            '按任务等权，再在任务内依次对轨迹、决策步等权；热力图还对组内分支等权。'
            '这两张图描述历史候选分支的关系，不表示更换策略后的完整轨迹收益。</p><ul>'
            + rows + '</ul>' + ''.join(figures) + '</html>')
    (stage / "index.html").write_text(page, encoding="utf-8")


@exclusive_artifact("difficulty", "plot_id")
def plot_consensus_length(cfg, plot_id, *, evaluation_ids):
    if (isinstance(evaluation_ids, (str, bytes)) or not evaluation_ids
            or len(set(evaluation_ids)) != len(evaluation_ids)):
        raise ValueError("Provide one or more unique --evaluation-id values")
    out = artifact_dir(cfg, "difficulty", identifier(plot_id), "evaluations")
    if out.exists():
        raise FileExistsError(f"Use a new plot ID: {out}")
    inputs = [read_evaluation(cfg, eid) for eid in evaluation_ids]
    sources, data = map(list, zip(*inputs))
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".consensus-length-viz-", dir=out.parent) as temporary:
        stage = Path(temporary)
        render(stage, sources, data)
        write_html(stage, sources)
        write_json(stage / "sources.json", {"weighting": WEIGHTING, "sources": sources})
        implementation = Path(__file__).resolve().parents[2]
        write_json(stage / "receipt.json", {"schema_version": 1, "state": "complete", "created_at": now(),
                   "plot_id": plot_id, "sources": sources, "weighting": WEIGHTING,
                   "code": {name: sha256(implementation / name) for name in CODE_FILES},
                   "files": {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}})
        stage.rename(out)
    return {"output": str(out), "report": str(out / "index.html"),
            "rank_heatmap": str(out / "rank_heatmap.png"),
            "token_excess_ecdf": str(out / "token_excess_ecdf.png"), "sources": sources}
