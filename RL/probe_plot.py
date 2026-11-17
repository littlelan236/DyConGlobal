"""CPU-only fixed-group charts; numeric training-step spacing, no new forwards."""
import io
import math
from pathlib import Path

from core.io import hash_json, read_json, sha256


def output_paths(root, output):
    from RL.probe import _path
    image = _path(root, output)
    if image.suffix.lower() not in {'.png', '.svg', '.pdf'}:
        raise ValueError('Probe plot output must end in .png, .svg or .pdf')
    data = image.with_suffix('.data.json')
    for path in (image, data):
        if 'RL' in path.parts or any((p / 'receipt.json').exists() or (p / 'attempt.json').exists()
                                    for p in path.parents):
            raise ValueError('Probe plots must be outside RL source and sealed artifact/attempt trees')
        if path.exists():
            raise FileExistsError('Probe plot output already exists: ' + str(path))
    return image, data


def plot_data(report, schedule, *, include_untrained=False):
    """Pure chart data, preserving every explicit group; never infer uniform spacing."""
    ids = report['group_ids']
    if (report.get('kind') != 'fixed_grpo_preference' or report.get('schema_version') != 1
            or not ids or len(set(ids)) != len(ids)
            or [g['group_id'] for g in report['groups']] != ids
            or report['probe_sha256'] != hash_json(report['probe_identity'])
            or [g['group_id'] for g in report['probe_identity']['groups']] != ids):
        raise ValueError('Invalid fixed probe report identity/group order')
    if (schedule.get('version') != 1 or schedule.get('x_coordinate') != 'scheduled_first_step'
            or schedule['checkpoint_step'] != report['checkpoint']['global_step']
            or [g['group_id'] for g in schedule['groups']] != ids):
        raise ValueError('Training coordinates do not match the probe report')
    positions = {g['group_id']: g for g in schedule['groups']}
    rows, omitted = [], []
    for group in report['groups']:
        pos = positions[group['group_id']]
        steps = pos['scheduled_steps']
        if (not steps or any(type(s) is not int or s < 1 for s in steps)
                or steps != sorted(set(steps)) or pos['scheduled_first_step'] != steps[0]
                or pos['scheduled_last_step'] != steps[-1]):
            raise ValueError('Invalid frozen group training steps')
        committed = [s for s in steps if s <= schedule['checkpoint_step']]
        expected = ('trained' if len(committed) == len(steps)
                    else 'partially_trained' if committed else 'not_yet_trained')
        if (pos['status_at_checkpoint'] != expected
                or pos['first_training_step'] != (committed[0] if committed else None)
                or pos['last_training_step'] != (committed[-1] if committed else None)):
            raise ValueError('Invalid group training status at checkpoint')
        row = {**pos, 'x': pos['scheduled_first_step'], 'metrics': group['metrics'],
               'branch_count': len(group.get('branches', [])),
               'token_count': sum(b['token_count'] for b in group.get('branches', [])),
               'token_changes': group.get('token_changes')}
        if not include_untrained and expected != 'trained':
            omitted.append({'group_id': group['group_id'], 'status_at_checkpoint': expected})
        else:
            rows.append(row)
    if not rows:
        raise ValueError('No fully trained selected groups to plot; use --include-untrained to show the fixed sample')
    rows.sort(key=lambda r: (r['x'], r['group_ordinal']))
    def finite(value):
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite probe plot value')
    finite(rows)
    return {'version': 1, 'checkpoint': report['checkpoint'], 'probe_sha256': report['probe_sha256'],
            'summary': summarize(rows, report),
            'training_schedule_sha256': schedule['schedule_sha256'],
            'x_axis': 'group positioned at scheduled first optimizer step; numeric linear scale',
            'reference': 'original_training_anchor', 'include_untrained': include_untrained,
            'omitted_groups': omitted, 'groups': rows,
            'notes': ['Each point is a selected group, not an observation of model quality at that historical step.',
                      'All points are scored by the checkpoint in this report.',
                      'Groups sharing a first step share x; no jitter or unsampled-group interpolation.',
                      'Preference metrics weight branches equally; token statistics pool target tokens.',
                      'Partial 8x16 parents are excluded by default; all scheduled steps and status are retained.']}


def summarize(rows, report):
    """Report text and tables use only displayed groups and explicit weighting."""
    import statistics
    def mean(values):
        return math.fsum(values) / len(values) if values else None
    cohorts = []
    partitions = [(status, [r for r in rows if r['status_at_checkpoint'] == status])
                  for status in ('trained', 'partially_trained', 'not_yet_trained')]
    partitions.append(('all_displayed', rows))
    for status, selected in partitions:
        if not selected:
            continue
        metrics = {}
        for key in ('positive_logprob_change', 'negative_logprob_change', 'preference_gap_change',
                    'advantage_logprob_alignment'):
            values = [r['metrics'][key] for r in selected if r['metrics'].get(key) is not None]
            metrics[key] = {'mean': mean(values), 'median': statistics.median(values) if values else None}
        gaps = [r['metrics']['preference_gap_change'] for r in selected
                if r['metrics'].get('preference_gap_change') is not None]
        positives = [r['metrics']['positive_logprob_change'] for r in selected
                     if r['metrics'].get('positive_logprob_change') is not None]
        tokens = {}
        for sign in ('all', 'positive', 'negative'):
            stats = [(r.get('token_changes') or {}).get(sign) for r in selected]
            # Never silently summarize just the rows which happen to have token statistics.
            if any(t is None for t in stats):
                tokens[sign] = None
                continue
            count = sum(t['token_count'] for t in stats)
            def weighted(path):
                values = []
                for t in stats:
                    if not t['token_count']:
                        continue
                    v = t
                    for key in path:
                        v = v[key]
                    values.append(t['token_count'] * v)
                return math.fsum(values) / count if count else None
            tokens[sign] = {'token_count': count,
                           'logprob_change_mean': weighted(['logprob_change_mean']),
                           'abs_logprob_change_mean': weighted(['abs_logprob_change_mean']),
                           'relative_probability_change': {
                               f'gt_{k}pct': {direction: weighted(['relative_probability_change', f'gt_{k}pct', direction])
                                             for direction in ('increase_fraction', 'decrease_fraction', 'either_fraction')}
                               for k in (10, 50)}}
        cohorts.append({'cohort': status, 'group_count': len(selected),
                        'branch_count': sum(r['branch_count'] for r in selected),
                        'token_count': sum(r['token_count'] for r in selected), 'metrics': metrics,
                        'improved_groups': sum(g > 0 for g in gaps), 'comparable_groups': len(gaps),
                        'positive_up_groups': sum(p > 0 for p in positives), 'positive_groups': len(positives),
                        'tokens': tokens})
    # Exact pooled quantiles/top-k shares cannot be reconstructed by averaging group summaries.
    complete = len(rows) == len(report['groups']) and {r['group_id'] for r in rows} == set(report['group_ids'])
    return {'scope': 'displayed_groups_only', 'cohorts': cohorts,
            'exact_pooled_token_changes': report.get('token_changes') if complete else None,
            'pooled_distribution_note': ('Exact pooling over all displayed target tokens.' if complete
                else 'Pooled quantiles/top-k unavailable for this subset; group quantiles/top-k are not averaged.'),
            'source_report_comparison': report.get('comparison'),
            'source_report_group_count': len(report['groups'])}


def _report_tables(axes, rows, summary, data):
    """In-figure report, including values that are hard to read from sparse points."""
    fmt = lambda value: 'N/A' if value is None else f'{value:+.6f}'
    num = lambda value: 'N/A' if value is None else f'{value:.6f}'
    pct = lambda value: 'N/A' if value is None else f'{100 * value:.2f}%'
    names = {'trained': 'Trained', 'partially_trained': 'Partial',
             'not_yet_trained': 'Untrained', 'all_displayed': 'All shown'}
    def table(ax, title, headers, values, *, size=8):
        ax.axis('off')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=10)
        tab = ax.table(cellText=values, colLabels=headers, cellLoc='center', loc='center', bbox=[0, 0, 1, .94])
        tab.auto_set_font_size(False)
        tab.set_fontsize(size)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor('#dce3eb')
            if r == 0:
                cell.set_facecolor('#e5edf5')
                cell.get_text().set_fontweight('bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f5f8fb')
    values = []
    for row in rows:
        m = row['metrics']
        t = (row.get('token_changes') or {}).get('all')
        values.append([str(row['group_ordinal']), str(row['x']), names[row['status_at_checkpoint']],
                       fmt(m.get('positive_logprob_change')), fmt(m.get('negative_logprob_change')),
                       fmt(m.get('preference_gap_change')), fmt(m.get('advantage_logprob_alignment')),
                       str(t['token_count'] if t else row['token_count']),
                       num(t['abs_logprob_change_mean']) if t else 'N/A',
                       pct(t['top_abs_change']['top_1pct']['abs_change_share']) if t else 'N/A',
                       pct(t['relative_probability_change']['gt_10pct']['either_fraction']) if t else 'N/A',
                       pct(t['relative_probability_change']['gt_50pct']['either_fraction']) if t else 'N/A'])
    table(axes[0], 'Per-group results (ordinal is zero-based; first step is one-based)',
          ['Group', 'First step', 'Status', 'Positive\ndelta logp', 'Negative\ndelta logp', 'Delta gap',
           'Advantage\nalignment', 'Tokens', 'Mean\n|delta logp|', 'Top 1%\ncontribution', 'Prob change\n>10%', 'Prob change\n>50%'], values)
    values = []
    for cohort in summary['cohorts']:
        for reduction in ('mean', 'median'):
            m = cohort['metrics']
            values.append([names[cohort['cohort']] + ' / ' + reduction, str(cohort['group_count']),
                           *[fmt(m[k][reduction]) for k in ('positive_logprob_change', 'negative_logprob_change',
                                                          'preference_gap_change', 'advantage_logprob_alignment')],
                           f"{cohort['improved_groups']}/{cohort['comparable_groups']}",
                           f"{cohort['positive_up_groups']}/{cohort['positive_groups']}"])
    table(axes[1], 'Cohort preference summary (branches equal within sign, then groups equal)',
          ['Cohort / statistic', 'Groups', 'Positive delta logp', 'Negative delta logp', 'Delta gap',
           'Advantage alignment', 'Gap improved', 'Positive logp up'], values)
    values = []
    for cohort in summary['cohorts']:
        for sign in ('all', 'positive', 'negative'):
            t = cohort['tokens'][sign]
            values.append([names[cohort['cohort']] + ' / ' + sign,
                           str(t['token_count']) if t else 'N/A',
                           fmt(t['logprob_change_mean']) if t else 'N/A',
                           num(t['abs_logprob_change_mean']) if t else 'N/A',
                           *[pct(t['relative_probability_change'][f'gt_{k}pct'][direction]) if t else 'N/A'
                             for k, direction in ((10, 'either_fraction'), (10, 'increase_fraction'),
                                                  (10, 'decrease_fraction'), (50, 'either_fraction'))]])
    table(axes[2], 'Cohort token summary (each target token equal; + / - signs refer to frozen advantage)',
          ['Cohort / sign', 'Tokens', 'Mean delta logp', 'Mean |delta logp|', 'Prob change\n>10%',
           'Prob up\n>10%', 'Prob down\n>10%', 'Prob change\n>50%'], values)
    pooled = summary['exact_pooled_token_changes']
    if pooled:
        values = []
        for sign in ('all', 'positive', 'negative'):
            t = pooled[sign]
            values.append([sign, num(t['abs_logprob_change_quantiles']['p50']),
                           num(t['abs_logprob_change_quantiles']['p95']),
                           num(t['abs_logprob_change_quantiles']['p99']),
                           *[pct(t['top_abs_change'][f'top_{k}pct']['abs_change_share']) for k in (1, 5, 10)]])
        table(axes[3], 'Exact pooled distribution over all displayed groups (not averages of group quantiles)',
              ['Sign', 'Median |delta logp|', 'P95 |delta logp|', 'P99 |delta logp|',
               'Top 1% contribution', 'Top 5% contribution', 'Top 10% contribution'], values)
    else:
        axes[3].axis('off')
        axes[3].text(0, .65, summary['pooled_distribution_note'], fontsize=10,
                     transform=axes[3].transAxes, va='top')
    all_shown = summary['cohorts'][-1]
    lines = [f"Displayed sample: {len(rows)} groups, {all_shown['branch_count']:,} branches, "
             f"{all_shown['token_count']:,} target tokens; {len(data['omitted_groups'])} source groups excluded by the display filter."]
    for cohort in summary['cohorts']:
        if cohort['cohort'] == 'all_displayed':
            continue
        m = cohort['metrics']
        lines.append(f"{names[cohort['cohort']]}: mean delta gap {fmt(m['preference_gap_change']['mean'])}; "
                     f"gap improved in {cohort['improved_groups']}/{cohort['comparable_groups']} groups; "
                     f"positive logprob increased in {cohort['positive_up_groups']}/{cohort['positive_groups']} groups.")
    if pooled and pooled['all']['token_count']:
        t = pooled['all']
        lines.append(f"Change concentration: largest 1% / 10% of tokens account for "
                     f"{pct(t['top_abs_change']['top_1pct']['abs_change_share'])} / "
                     f"{pct(t['top_abs_change']['top_10pct']['abs_change_share'])} of total |delta logprob|.")
        median = t['abs_logprob_change_quantiles']['p50']
        if median is not None and median < 1:
            lines.append(f"Median |delta logprob| = {median:.8f}: at least half the target tokens have relative probability "
                         f"change no larger than {100 * math.expm1(median):.5f}%.")
    comparison = summary.get('source_report_comparison')
    if comparison:
        deltas = [v for v in comparison.get('aggregate_delta', {}).values() if v is not None]
        if deltas:
            maximum = max(abs(v) for v in deltas)
            baseline_step = comparison.get('baseline_checkpoint', {}).get('global_step', '?')
            lines.append(f"Source-report comparison (all {summary['source_report_group_count']} source groups): "
                         f"max absolute aggregate metric difference vs baseline step {baseline_step} = {maximum:.6g}.")
    lines.extend(['All changes use the original training anchor at fixed teacher-forced contexts, raw T=1.',
                  'The x-axis locates each group in training; every point is scored by the current checkpoint, not by the historical model.',
                  'Group-weighted and token-weighted means can have different signs; long branches carry more weight in token summaries.',
                  'Large changes may concentrate on few tokens. These training-pool probes do not measure held-out or downstream success.',
                  'N/A means unavailable / unsupported; a missing pooled distribution is never estimated by averaging group quantiles.'])
    axes[4].axis('off')
    axes[4].set_title('Reading the results and analysis scope', loc='left', fontsize=11, fontweight='bold')
    axes[4].text(0, .96, '\n'.join(lines), transform=axes[4].transAxes,
                 va='top', fontsize=9, linespacing=1.65)


def render(data, image_format):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    rows = data['groups']
    checkpoint_step = data['checkpoint']['global_step']
    summary = data['summary']
    cohorts = summary['cohorts']
    heights = [2.7] * 5 + [max(1.8, .23 * (len(rows) + 2)),
                                max(1.3, .26 * (2 * len(cohorts) + 2)),
                                max(1.8, .25 * (3 * len(cohorts) + 2)), 1.2, 2.3]
    fig = plt.figure(figsize=(17, sum(heights) + 2), layout='constrained')
    grid = fig.add_gridspec(len(heights), 1, height_ratios=heights)
    axes = [fig.add_subplot(grid[0])]
    axes.extend(fig.add_subplot(grid[i], sharex=axes[0]) for i in range(1, 5))
    tables = [fig.add_subplot(grid[i]) for i in range(5, 10)]
    colors = {'positive': '#14866d', 'negative': '#d45d38', 'gap': '#4769b2',
              'all': '#4769b2', 'p95': '#ab67a0', 'p50': '#a28935'}
    markers = {'trained': 'o', 'partially_trained': 'D', 'not_yet_trained': 'x'}
    def points(ax, values, label, color, *, percent=False):
        for status, marker in markers.items():
            pairs = [(row['x'], value) for row, value in zip(rows, values)
                     if row['status_at_checkpoint'] == status and value is not None]
            if pairs:
                suffix = '' if status == 'trained' else ' (partial)' if status == 'partially_trained' else ' (untrained)'
                ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=37, marker=marker,
                           color=color, alpha=.85, label=label + suffix, zorder=3)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    def metrics(key):
        return [row['metrics'].get(key) for row in rows]
    def tokens(sign, *keys):
        values = []
        for row in rows:
            value = (row.get('token_changes') or {}).get(sign)
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value)
        return values
    for sign in ('positive', 'negative'):
        points(axes[0], metrics(sign + '_logprob_change'), sign + ' advantage', colors[sign])
    axes[0].set(title='Signed mean logprob change by advantage (branch-equal)', ylabel='Mean delta logprob')
    points(axes[1], metrics('preference_gap_change'), 'positive minus negative change', colors['gap'])
    axes[1].set(title='Preference-gap change (branch-equal)', ylabel='Delta preference gap')
    for sign in ('all', 'positive', 'negative'):
        points(axes[2], tokens(sign, 'abs_logprob_change_mean'), sign + ': mean |delta logprob|', colors[sign])
    points(axes[2], tokens('all', 'abs_logprob_change_quantiles', 'p50'), 'all: median |delta logprob|', colors['p50'])
    points(axes[2], tokens('all', 'abs_logprob_change_quantiles', 'p95'), 'all: p95 |delta logprob|', colors['p95'])
    axes[2].set(title='Token change magnitude (token-equal within each group)', ylabel='Absolute delta logprob')
    for threshold, color in ((10, colors['all']), (50, colors['negative'])):
        points(axes[3], tokens('all', 'relative_probability_change', f'gt_{threshold}pct', 'either_fraction'),
               f'|p / anchor - 1| > {threshold}%', color, percent=True)
    axes[3].set(title='Tokens with large relative probability changes (all signs)', ylabel='Fraction of target tokens')
    for k, color in ((1, colors['negative']), (5, colors['p95']), (10, colors['all'])):
        points(axes[4], tokens('all', 'top_abs_change', f'top_{k}pct', 'abs_change_share'),
               f'top {k}% tokens', color, percent=True)
    axes[4].set(title='Concentration of absolute logprob change (all signs)',
                ylabel='Share of sum |delta logprob|',
                xlabel='Group position = first scheduled training step (linear optimizer-step scale)')
    max_x = max(checkpoint_step, max(r['x'] for r in rows), 1)
    for i, ax in enumerate(axes):
        if i < 4:
            ax.tick_params(labelbottom=False)
        ax.set_xscale('linear')
        ax.set_xlim(0, max_x * 1.025)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
        ax.grid(alpha=.2)
        if i < 2:
            ax.axhline(0, color='#777777', linewidth=.8)
        ax.axvline(checkpoint_step, color='#555555', linestyle='--', linewidth=1,
                   label=f'evaluated checkpoint: step {checkpoint_step}')
        if max_x > checkpoint_step:
            ax.axvspan(checkpoint_step, max_x * 1.025, color='#999999', alpha=.08)
        if i >= 2 and not any(r.get('token_changes') for r in rows):
            ax.text(.5, .5, 'Token distributions absent in this historical JSON; rerun probe to obtain them.',
                    ha='center', transform=ax.transAxes)
        ax.legend(loc='best', fontsize=8, ncol=2)
    if len(rows) <= 24:
        by_x = {}
        for row in rows:
            by_x.setdefault(row['x'], []).append(str(row['group_ordinal']))
        for x, ordinals in by_x.items():
            axes[0].annotate('group ' + ','.join(ordinals), (x, 1), xycoords=('data', 'axes fraction'),
                             xytext=(0, 5), textcoords='offset points', rotation=70, fontsize=8,
                             ha='left', va='bottom', annotation_clip=False)
        axes[0].set_title(axes[0].get_title(), pad=72)
    fig.suptitle(f'Fixed-group probe at checkpoint step {checkpoint_step} | {len(rows)} selected groups\n'
                 'All changes vs original anchor; each point is scored at this checkpoint', fontsize=14)
    _report_tables(tables, rows, summary, data)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format=image_format, dpi=160)
        return buffer.getvalue()
    finally:
        plt.close(fig)


def publish(report, schedule, root, output, *, include_untrained=False, report_sha256=None):
    from RL.probe import _publish
    image, data_path = output_paths(root, output)
    data = plot_data(report, schedule, include_untrained=include_untrained)
    data['source_report_sha256'] = report_sha256
    content = render(data, image.suffix.lower()[1:])
    # Reuse the report's fsynced, no-overwrite publication for both artifacts.
    _publish(image, content, binary=True)
    try:
        _publish(data_path, data)
    except BaseException:
        image.unlink()
        raise
    return {'image': str(image), 'data': str(data_path), 'group_count': len(data['groups']),
            'omitted_group_count': len(data['omitted_groups']), 'checkpoint_step': schedule['checkpoint_step']}


def _matplotlib_available():
    import importlib.util
    return importlib.util.find_spec('matplotlib') is not None


def run(root_config, *, report, output, include_untrained=False):
    from RL.probe import _path, _checkpoint_context
    from RL.probe_schedule import checkpoint_positions
    root = Path(root_config['project_root']).resolve()
    output_paths(root, output)
    path = _path(root, report)
    if not _matplotlib_available():
        # Training environments need not carry plotting dependencies. Reuse the
        # same public CLI in the project's CPU runner, never an alternate scorer.
        import json
        import subprocess
        import sys
        runner = root / '.venvs/runner/bin/dycon'
        if not runner.is_file() or (root / '.venvs/runner').resolve() == Path(sys.prefix).resolve():
            raise ValueError('matplotlib is required for plotting; use the project runner environment')
        args = [str(runner), 'lora-probe-plot', '--report', str(path), '--output', str(_path(root, output))]
        if include_untrained:
            args.append('--include-untrained')
        try:
            result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('CPU plot command exceeded 180 seconds; probe JSON is preserved') from exc
        if result.returncode:
            raise RuntimeError('CPU plot command failed: ' + result.stderr[-2000:])
        return json.loads(result.stdout)
    report_hash = sha256(path)
    source = read_json(path)
    schedule = source.get('training_schedule')
    if schedule is None:
        checkpoint = _path(root, source['checkpoint']['path'])
        metadata, _, receipt_hash, _ = _checkpoint_context(checkpoint)
        if (source['checkpoint']['receipt_sha256'] != receipt_hash
                or source['checkpoint']['global_step'] != metadata['state']['global_step']):
            raise ValueError('Historical report checkpoint receipt/step mismatch')
        schedule = checkpoint_positions(checkpoint, metadata['contract'], metadata['state'], source['group_ids'])
    if sha256(path) != report_hash:
        raise ValueError('Probe report changed during plotting')
    return {'status': 'complete', 'report': str(path),
            **publish(source, schedule, root, output, include_untrained=include_untrained,
                      report_sha256=report_hash)}
