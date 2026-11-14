import json
from pathlib import Path

import pytest

from application.execution.config_diff import compare_run_configs, diff_config_values
from application.execution.run_plan import create_run
from catalog.store import Catalog
from core.cli import main
from core.io import sha256
from tests.conftest import task


def make_source(cfg):
    Catalog(cfg["project_root"]).register_tasks([task("config-diff-task")])
    return create_run(cfg, "config-diff-left", operation="rollout", task_ids=["config-diff-task"])


def test_diff_config_values_reports_added_removed_changed_and_null():
    rows = diff_config_values(
        {"explicit_null": None, "removed": 1, "nested": {"value": 1}},
        {"explicit_null": None, "added": None, "nested": {"value": 2}},
    )
    by_path = {row["path"]: row for row in rows}
    assert set(by_path) == {"/added", "/nested/value", "/removed"}
    assert by_path["/added"] == {
        "path": "/added", "kind": "added", "left_present": False,
        "left_value": None, "right_present": True, "right_value": None,
    }
    assert by_path["/removed"]["kind"] == "removed"
    assert by_path["/nested/value"]["left_value"] == 1
    assert by_path["/nested/value"]["right_value"] == 2


def test_compare_run_configs_reads_frozen_snapshots(cfg):
    left = make_source(cfg)
    right = create_run(cfg, "config-diff-right", operation="rollout", from_run=left.name,
                       clone_overrides=["serving.max_num_seqs=2"])
    report = compare_run_configs(cfg["project_root"], left.name, right.name)
    rows = {row["path"]: row for row in report["differences"]}
    assert report["difference_count"] == len(report["differences"])
    assert rows["/serving/max_num_seqs"]["left_value"] != rows["/serving/max_num_seqs"]["right_value"]
    assert report["left"]["resolved_config_sha256"] == sha256(left / "resolved-config.json")
    assert report["right"]["resolved_config_sha256"] == sha256(right / "resolved-config.json")


def test_cli_diff_config_json_markdown_and_output_no_overwrite(cfg, capsys):
    left = make_source(cfg)
    right = create_run(cfg, "config-diff-cli-right", operation="rollout", from_run=left.name,
                       clone_overrides=["serving.max_num_seqs=2"])
    output = Path(cfg["project_root"]) / "config-diff.md"
    args = ["--config", cfg["config_source"], "run", "diff-config", left.name, right.name,
            "--output", str(output)]
    main(args)
    rendered = capsys.readouterr().out
    assert "| serving.max_num_seqs |" in rendered
    assert "## Serving" in rendered
    assert output.read_text(encoding="utf-8") == rendered
    digest = sha256(output)
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert sha256(output) == digest

    main(["--config", cfg["config_source"], "run", "diff-config", left.name, right.name,
          "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert report["difference_count"] == len(report["differences"])
    assert any(row["path"] == "/serving/max_num_seqs" for row in report["differences"])


def test_cli_diff_config_uses_snapshots_not_mutable_root_yaml(cfg, capsys):
    left = make_source(cfg)
    right = create_run(cfg, "config-diff-broken-root-right", operation="rollout", from_run=left.name,
                       clone_overrides=["serving.max_num_seqs=2"])
    Path(cfg["config_source"]).write_text("invalid: [", encoding="utf-8")
    main(["--config", cfg["config_source"], "run", "diff-config", left.name, right.name,
          "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert any(row["path"] == "/serving/max_num_seqs" for row in report["differences"])


def test_schema_discovery_covers_current_and_future_configuration_fields():
    from application.execution.config_diff import configurable_rows
    from core.parallel_policy import RunConfig

    schema = RunConfig.model_json_schema()
    schema['properties']['future_section'] = {'type': 'object', 'properties': {
        'new_option': {'type': 'integer'},
        'receipt': {'type': 'string', 'readOnly': True},
    }}
    left = {'future_section': {'new_option': 8, 'receipt': 'a'},
            'agent': {'future_prompt': 'old'},
            'parallel': {'max_inflight_branches': 64, 'estimator_sha256': 'receipt',
                         'entropy': {'retention_rule': 'effective_count_plus_n_v1', 'n': 6},
                         'consensus': {'retention_ratio': .5, 'rule': 'exact_action'}},
            'model': {'fingerprint': 'configured-hash'},
            'selection': {'tasks': {'x': {'problem_statement': 'TASK_BODY_SECRET'}}},
            'config_hash': 'generated', 'execution_context': {'profile': 'DERIVED_SECRET'}}
    right = {**left, 'future_section': {'new_option': 9, 'receipt': 'b'}}
    rows = {r['path']: r for r in configurable_rows(left, right, schema=schema)}
    assert rows['/future_section/new_option']['kind'] == 'changed'
    assert rows['/agent/future_prompt']['known_field'] is False
    assert rows['/model/fingerprint']['left_value'] == 'configured-hash'
    assert rows['/parallel/entropy/n']['left_value'] == 6
    assert rows['/parallel/consensus/retention_ratio']['kind'] == 'unchanged'
    assert not any('receipt' in key or 'sha256' in key or 'max_inflight' in key for key in rows)
    assert '/parallel/consensus/rule' not in rows  # A schema constant, not a choice.
    assert not any(key.startswith(('/selection', '/execution_context', '/config_hash')) for key in rows)


def test_open_mapping_and_lists_are_atomic_and_typed_values_remain_distinct():
    from application.execution.config_diff import configurable_rows
    rows = {r['path']: r for r in configurable_rows(
        {'environment': {'variables': {'a/b~key': 'x', 'many': {'child': 2}}},
         'model': {'trust_remote_code': False}, 'serving': {'gpus': [0]},
         'agent': {'profile_ref': None}},
        {'environment': {'variables': {'a/b~key': 'y', 'many': {'child': 2}}},
         'model': {'trust_remote_code': 0}, 'serving': {'gpus': [False]},
         'agent': {}})}
    assert rows['/environment/variables']['left_value']['a/b~key'] == 'x'
    assert not any(key.startswith('/environment/variables/') for key in rows)
    assert rows['/model/trust_remote_code']['kind'] == 'changed'
    assert rows['/serving/gpus']['kind'] == 'changed'
    assert rows['/agent/profile_ref']['kind'] == 'removed'
    assert rows['/agent/profile_ref']['left_present'] is True
    assert rows['/agent/profile_ref']['right_present'] is False
    assert rows['/agent/profile_ref']['left_value'] is None


def test_schema_coverage_is_not_a_manually_maintained_field_list():
    from application.execution.config_diff import configurable_rows, _schema_shape
    from core.parallel_policy import RunConfig
    schema = RunConfig.model_json_schema()
    expected = set()

    def sample(node, parts=()):
        node = _schema_shape(node, schema['$defs'])
        if node.get('readOnly') or 'const' in node:
            return None
        if node.get('properties'):
            return {key: sample(value, (*parts, key))
                    for key, value in node['properties'].items()
                    if not _schema_shape(value, schema['$defs']).get('readOnly')
                    and 'const' not in _schema_shape(value, schema['$defs'])}
        expected.add('/' + '/'.join(parts))
        return None

    frozen = sample(schema)
    rows = configurable_rows(frozen, frozen)
    assert {row['path'] for row in rows} == expected
    assert all(row['kind'] == 'unchanged' for row in rows)


def test_compare_default_and_only_changed_exclude_derived_and_task_content(cfg):
    left = make_source(cfg)
    right = create_run(cfg, 'config-diff-filtered-right', operation='rollout', from_run=left.name,
                       clone_overrides=['serving.max_num_seqs=2'])
    before = {path: sha256(path) for run in (left, right) for path in run.rglob('*') if path.is_file()}
    report = compare_run_configs(cfg['project_root'], left.name, right.name)
    assert report['schema_version'] == 2
    assert report['field_count'] > report['difference_count']
    assert report['left']['summary']['task_count'] == 1
    assert report['left']['summary']['repeat_count'] == 1
    assert not any(row['path'].startswith(('/selection', '/config_hash', '/execution_context'))
                   for row in report['rows'])
    filtered = compare_run_configs(cfg['project_root'], left.name, right.name, only_changed=True)
    assert filtered['rows'] == report['differences']
    assert filtered['field_count'] == report['field_count']
    assert filtered['difference_count'] == report['difference_count']
    assert before == {path: sha256(path) for path in before}


def test_long_prompt_middle_difference_is_visible_and_markdown_is_escaped():
    from application.execution.config_diff import configurable_rows, format_config_diff_markdown
    head = '`<script>|**[head]\\\r\n'
    left = head + 'a' * 2000 + 'LEFT_MIDDLE' + 'z' * 2000 + '|TAIL'
    right = head + 'a' * 2000 + 'RIGHT_MIDDLE' + 'z' * 2000 + '|TAIL'
    rows = configurable_rows({'agent': {'system_template': left}},
                             {'agent': {'system_template': right}})
    report = {'left': {'run_id': 'left', 'summary': dict(operation='rollout', benchmarks=['fixture'],
                task_count=1, repeat_count=15, sample_count=15)},
              'right': {'run_id': 'right', 'summary': dict(operation='rollout', benchmarks=['fixture'],
                task_count=1, repeat_count=1, sample_count=1)},
              'field_count': 1, 'difference_count': 1, 'only_changed': False, 'rows': rows}
    rendered = format_config_diff_markdown(report)
    assert len(rendered) < 2200
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
    assert '&#124;TAIL' in rendered and '<br>' in rendered
    assert 'sha256:' in rendered
    assert 'LEFT_MIDDLE' not in rendered and 'RIGHT_MIDDLE' not in rendered
    assert rows[0]['left_value'] == left and rows[0]['right_value'] == right
    assert rows[0]['kind'] == 'changed'
    line = next(line for line in rendered.splitlines() if line.startswith('| agent.'))
    assert line.count('|') == 5
    # Same visible head/tail still carries different content fingerprints.
    import re
    assert len(set(re.findall(r'sha256:([0-9a-f]+)', line))) == 2


def test_cli_only_changed_and_identical_runs(cfg, capsys):
    left = make_source(cfg)
    main(['--config', cfg['config_source'], 'run', 'diff-config', left.name, left.name,
          '--only-changed'])
    rendered = capsys.readouterr().out
    assert 'No configurable differences.' in rendered
    assert '| task_count | 1 | 1 |' in rendered
    main(['--config', cfg['config_source'], 'run', 'diff-config', left.name, left.name,
          '--format', 'json', '--only-changed'])
    report = json.loads(capsys.readouterr().out)
    assert report['rows'] == report['differences'] == []
    assert report['field_count'] > 0


def test_nullable_section_state_and_missing_are_not_collapsed():
    from application.execution.config_diff import configurable_rows
    rows = {row['path']: row for row in configurable_rows(
        {'parallel': None}, {'parallel': {'enabled': False}})}
    assert rows['/parallel']['left_present'] is True
    assert rows['/parallel']['left_value'] is None
    assert rows['/parallel']['container_state'] is True
    assert rows['/parallel']['right_value'] == {'enabled': False}
    assert rows['/parallel/enabled']['left_present'] is False
    assert rows['/parallel/enabled']['right_value'] is False


def test_prompt_character_count_and_short_value_types_are_unambiguous():
    from application.execution.config_diff import _markdown_cell
    prompt = 'head' + 'middle' * 200 + 'tail'
    shortened = _markdown_cell(True, prompt)
    assert str(len(prompt)) + ' chars' in shortened
    assert _markdown_cell(True, 'null') != _markdown_cell(True, None)
    assert _markdown_cell(True, 'false') != _markdown_cell(True, False)
    assert _markdown_cell(True, '0') != _markdown_cell(True, 0)
