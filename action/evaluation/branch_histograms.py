"""Integer-bin histograms of per-step diversity and actual-winner support."""

from pathlib import Path


def render_histograms(directory, summary, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(plot_style()):
        for key, stats in summary["metrics"].items():
            if not (key.startswith("action_types_") or key.startswith("winner_count_")):
                continue
            view = key.rsplit("_", 1)[1]
            coverage = summary["branch_coverage"][view]
            diversity = key.startswith("action_types")
            scope = "全部分支" if view == "all" else "estimator 筛选后"
            sizes = coverage["branch_count_histogram"]
            if view == "all" and len(sizes) == 1:
                scope = f"全部 {next(iter(sizes))} 分支"
            title = ("Action 种类数" if diversity else "获胜 Action 的共识分支数") + "：" + scope
            values = stats["histogram"]
            low = 0 if values.get("0", 0) else 1
            x = list(range(low, stats["bin_upper"] + 1))
            counts = [values[str(i)] for i in x]
            fig, ax = plt.subplots(figsize=(12.5, 6.1), dpi=analysis["dpi"])
            fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.23)
            colors = {"action_types_all": "#2878B5", "action_types_retained": "#2A9D8F",
                      "winner_count_all": "#8064A2", "winner_count_retained": "#D7832F"}
            bars = ax.bar(x, counts, width=0.78, color=colors[key], edgecolor="white", linewidth=0.6)
            if len(x) <= 80:
                ax.bar_label(bars, labels=[str(n) if n else "" for n in counts], padding=4, fontsize=8)
                ax.set_xticks(x)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=32))
            ax.set_xlim(low - 0.65, stats["bin_upper"] + 0.65)
            ax.set_ylim(0, max(max(counts, default=0) * 1.22, 1))
            ax.set_xlabel("严格一致的 action 种类数" if diversity else "与实际获胜 action 完全一致的分支个数", labelpad=11)
            ax.set_ylabel("Step 数", labelpad=10)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#b8bdc4")
            fig.suptitle(title, fontsize=18, fontweight="bold", x=0.075, y=0.965, ha="left", color="#182B45")
            mean = f'{stats["mean"]:.2f}' if stats["mean"] is not None else "N/A"
            median = f'{stats["median"]:g}' if stats["median"] is not None else "N/A"
            reason = analysis["termination_reason"] or "已选择"
            subtitle = (f'{summary["trajectories"]} 条 {reason} 轨迹 · {summary["steps"]:,} parallel steps'
                        f'  |  均值 {mean}  |  中位数 {median}')
            fig.text(0.075, 0.88, subtitle, fontsize=11, color="#4e5968")
            fig.text(0.075, 0.12, "逐 step 等权；整数分箱；柱顶为 step 数。命令空白、引号及工具调用顺序保留。", fontsize=9, color="#596474")
            fig.text(0.075, 0.075,
                     f'{scope}：{coverage["unavailable"]:,} / {coverage["outputs"]:,} 个输出无可比 action，'
                     f'涉及 {coverage["steps_with_unavailable"]:,} steps；不作为额外 action 种类。',
                     fontsize=9, color="#596474")
            fig.text(0.075, 0.035,
                     f'有效统计 {stats["valid_steps"]:,} steps；N/A {stats["unavailable_steps"]:,} steps。'
                     '筛选名单沿用历史 receipt；共识数始终针对实际获胜 action。', fontsize=9, color="#596474")
            for extension in analysis["formats"]:
                fig.savefig(directory / (key + "." + extension), dpi=analysis["dpi"])
            plt.close(fig)



def plot_style():
    """Put the Latin font first and use a CJK fallback for Chinese glyphs."""
    from matplotlib import font_manager
    fonts = ["DejaVu Sans"]
    for candidate in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "C:/Windows/Fonts/msyh.ttc"):
        if Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            fonts.append(font_manager.FontProperties(fname=candidate).get_name())
            break
    return {"font.family": fonts, "axes.unicode_minus": False, "svg.fonttype": "path", "font.size": 11}
