import base64
from pathlib import Path

import numpy as np
import pytest

from difficulty.evaluation.trajectories import build_html, summarize_repetitions
from difficulty.evaluation.visualize import plot_trajectory


def test_final_selection_retains_pending_work_without_fabricating_outcome(tmp_path, monkeypatch):
    import json
    import difficulty.evaluation.trajectories as module
    import rollout.retry as retry
    items = [dict(instance_id='task-a', repetition_id='rep-000'),
             dict(instance_id='task-a', repetition_id='rep-001')]
    path = tmp_path / 'runs/run/rollout/tasks/task-a/repetitions/rep-000/result.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'rollout_id': 'r-final'}))
    monkeypatch.setattr(module, 'load_plan', lambda *args: {'work_items': {'items': items}})
    verified = []
    def verify(root, run, item, result, catalog):
        verified.append((item, result))
        return result['rollout_id']
    monkeypatch.setattr(retry, 'verify_final', verify)
    class Evidence:
        def __init__(self, *args):
            pass
        def select(self, ids):
            return {'rollout_ids': ids}
    monkeypatch.setattr(module, 'RunEvidence', Evidence)
    selection, sources, expected, pending = module.final_selection(tmp_path, 'run', {})
    assert selection['rollout_ids'] == ['r-final']
    assert len(verified) == 1
    assert len(sources) == 1
    assert expected == items
    assert pending == [items[1]]


def row(rep, status='completed', resolved=True):
    return dict(task_key='a', repetition_id=rep, evaluation_status=status, resolved=resolved)


def test_counts_preserve_zero_success_and_ungraded():
    tasks = {'a': {'instance_id': 'task-a'}, 'b': {'instance_id': 'task-b'}}
    rows = {'r1': row('rep-000'), 'r2': row('rep-001', resolved=False),
            'r3': row('rep-002', 'infra_error', None)}
    counts = summarize_repetitions(tasks, rows)
    assert counts == [dict(task_id='task-a', selected_rollouts=3, resolved=1, unresolved=1, ungraded=1),
                      dict(task_id='task-b', selected_rollouts=0, resolved=0, unresolved=0, ungraded=0)]


def test_duplicate_attempts_do_not_inflate_repetition_count():
    with pytest.raises(ValueError, match='Multiple selected'):
        summarize_repetitions({'a': {'instance_id': 'task-a'}}, {'r1': row('rep-000'), 'r2': row('rep-000')})


def capture_plot(monkeypatch):
    from matplotlib.figure import Figure
    captured = []
    def save(figure, *args, **kwargs):
        captured.append(figure)
    monkeypatch.setattr(Figure, 'savefig', save)
    return captured


def test_truth_and_predictions_are_distinct_and_missing_steps_break_lines(tmp_path, monkeypatch):
    figures = capture_plot(monkeypatch)
    plot_trajectory([dict(step_id=0, target=10, prediction=7, prediction_log1p=2),
                     dict(step_id=2, target=3, prediction=-.5, prediction_log1p=-.693)],
                    tmp_path / 'x.svg', comparison=True)
    axes = figures[0].axes
    assert len(axes) == 2
    assert list(axes[0].lines[0].get_xdata()) == [0, 1, 2]
    assert np.allclose(axes[0].lines[0].get_ydata()[[0, 2]], [10, 3])
    assert np.isnan(axes[0].lines[0].get_ydata()[1])
    assert np.isnan(axes[0].lines[1].get_ydata()[1])
    assert axes[0].lines[1].get_ydata()[2] == -.5
    assert np.allclose(axes[1].lines[0].get_ydata()[[0, 2]], np.log1p([10, 3]))
    assert axes[1].lines[1].get_ydata()[0] == 2


def test_no_data_is_visible_and_not_zero_truth(tmp_path, monkeypatch):
    figures = capture_plot(monkeypatch)
    plot_trajectory([dict(step_id=0, target=None, prediction=None)], tmp_path / 'none.svg', comparison=True)
    ax = figures[0].axes[0]
    assert np.isnan(ax.lines[0].get_ydata()).all()
    assert any('No qualified' in text.get_text() for text in ax.texts)


@pytest.mark.parametrize('steps', [[1, 0], [0, 0], [-1], [.5]])
def test_invalid_step_identity_is_rejected(tmp_path, steps):
    with pytest.raises(ValueError, match='step IDs'):
        plot_trajectory([dict(step_id=step, target=1, prediction=2) for step in steps], tmp_path / 'bad.svg')


def test_legacy_evaluation_retains_one_truth_prediction_panel(tmp_path, monkeypatch):
    figures = capture_plot(monkeypatch)
    plot_trajectory([dict(step_id=0, target=10, prediction=7)], tmp_path / 'legacy.png')
    assert len(figures[0].axes) == 1
    assert [line.get_ydata()[0] for line in figures[0].axes[0].lines] == [10, 7]


def test_html_embeds_all_same_task_images_in_one_row_and_escapes_names(tmp_path):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    (tmp_path / 'one.svg').write_bytes(svg)
    (tmp_path / 'two.svg').write_bytes(svg)
    tasks = [dict(task_id='a<script>', resolved=2, unresolved=0, ungraded=0, selected_rollouts=2),
             dict(task_id='b', resolved=0, unresolved=1, ungraded=0, selected_rollouts=1)]
    trajectories = [dict(task_id=tasks[0]['task_id'], rollout_id='r'+str(i), repetition_id='rep-'+str(i),
                         plot=path, predicted_points=1, total_steps=3, coverage_status='partial')
                    for i, path in enumerate(['one.svg', 'two.svg'])]
    summary = dict(run_id='run', attempt_id='attempt', target='reasoning', selected_rollouts=3,
                   resolved_rollouts=2, plotted_with_predictions=2, without_predictions=0)
    build_html(tasks, trajectories, tmp_path, summary)
    document = (tmp_path / 'report.html').read_text()
    assert document.count('<section ') == 2
    assert document.count('<figure>') == 2
    assert document.count('data:image/svg+xml;base64,') == 2
    assert base64.b64encode(svg).decode() in document
    assert 'flex-wrap:nowrap' in document
    assert 'a&lt;script&gt;' in document
    assert 'a<script>' not in document
    first_section = document.split('<section ')[1].split('</section>')[0]
    assert first_section.count('<figure>') == 2
