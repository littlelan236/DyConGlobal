"""Read-only, schema-driven comparison of configurable frozen run values."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from application.execution.run_plan import load_plan
from core.io import read_json, sha256


def _pointer(parts):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def diff_config_values(left, right):
    """Return every leaf-level difference, preserving missing vs explicit null."""
    rows = []

    def visit(left_present, left_value, right_present, right_value, parts):
        if left_present and right_present and isinstance(left_value, dict) and isinstance(right_value, dict):
            keys = sorted(set(left_value) | set(right_value), key=str)
            if not keys and left_value == right_value:
                return
            for key in keys:
                visit(key in left_value, left_value.get(key),
                      key in right_value, right_value.get(key), [*parts, key])
            return
        if left_present != right_present:
            present_value = left_value if left_present else right_value
            if isinstance(present_value, dict) and present_value:
                for key in sorted(present_value, key=str):
                    visit(key in left_value if left_present else False,
                          left_value.get(key) if left_present else None,
                          key in right_value if right_present else False,
                          right_value.get(key) if right_present else None,
                          [*parts, key])
                return
        if left_present and right_present and left_value == right_value:
            return
        path = _pointer(parts) if parts else "/"
        kind = ("added" if right_present else "removed") if left_present != right_present else "changed"
        rows.append({"path": path, "kind": kind,
                     "left_present": left_present, "left_value": left_value if left_present else None,
                     "right_present": right_present, "right_value": right_value if right_present else None})

    visit(True, left, True, right, [])
    return sorted(rows, key=lambda row: row["path"])


def _same_value(left, right):
    """JSON types matter: false, 0 and 0.0 are distinct frozen values."""
    return json.dumps(left, ensure_ascii=False, sort_keys=True, allow_nan=False) == json.dumps(
        right, ensure_ascii=False, sort_keys=True, allow_nan=False)


# Receipts are injected after input validation, so they have no schema fields.
# Do not exclude arbitrary '*sha256' names: user-specified hashes are configurable.
_RECEIPT_PATHS = {('parallel', 'estimator_sha256'), ('steering', 'payload_sha256')}
_SECTION_TITLES = {
    'paths': 'Storage paths', 'benchmark': 'Benchmark', 'sandbox': 'Sandbox',
    'model': 'Model', 'serving': 'Serving', 'generation': 'Generation',
    'agent': 'Agent and prompts', 'environment': 'Task environment',
    'images': 'Task images', 'execution': 'Execution and retries',
    'timeouts': 'Timeouts', 'capture': 'Capture', 'parallel': 'Parallel policy',
    'steering': 'Steering', 'enrichment': 'Enrichment', 'hidden': 'Hidden replay',
    'difficulty': 'Difficulty analysis', 'action': 'Action analysis',
    'skillsbench_images': 'SkillsBench images', 'webarena': 'WebArena',
    'webarena_images': 'WebArena images', 'harbor': 'Harbor',
}


def _schema_shape(node, definitions):
    """Resolve model/nullable/union schemas without instantiating defaults."""
    merged = dict(node)
    if '$ref' in node:
        ref = node['$ref']
        if not ref.startswith('#/$defs/'):
            raise ValueError('Unsupported configuration schema reference: ' + ref)
        merged = {**_schema_shape(definitions[ref.rsplit('/', 1)[-1]], definitions), **merged}
    properties = dict(merged.get('properties', {}))
    for variant in ('anyOf', 'oneOf', 'allOf'):
        for child in node.get(variant, []):
            shape = _schema_shape(child, definitions)
            for key, value in shape.get('properties', {}).items():
                if key in properties and properties[key] != value:
                    properties[key] = {'anyOf': [properties[key], value]}
                else:
                    properties[key] = value
    if properties:
        merged['properties'] = properties
    return merged


def configurable_rows(left, right, *, schema=None):
    """One row per frozen configurable setting, including unchanged values.

    Schema discovery admits new fields/sections automatically. Open mappings
    and lists remain atomic configuration values; unknown nested legacy keys
    stay visible without treating generated snapshot roots as configuration.
    """
    if schema is None:
        from core.parallel_policy import RunConfig
        schema = RunConfig.model_json_schema()
    definitions = schema.get('$defs', {})
    rows = []

    def visit(node, lp, lv, rp, rv, parts, known=True):
        node = _schema_shape(node, definitions)
        if parts in _RECEIPT_PATHS or node.get('readOnly') or 'const' in node:
            return
        properties = node.get('properties', {})
        if properties and (isinstance(lv, dict) or isinstance(rv, dict)):
            left_map = lv if lp and isinstance(lv, dict) else {}
            right_map = rv if rp and isinstance(rv, dict) else {}
            # A nullable model itself is also a configurable value. Preserve an
            # explicit null/type change before describing its populated fields.
            if parts and ((lp and not isinstance(lv, dict)) or
                          (rp and not isinstance(rv, dict))):
                add(parts, lp, lv, rp, rv, known, container=True)
            keys = set(left_map) | set(right_map)
            if not parts:
                keys &= set(properties)  # Snapshot identities are not input settings.
            for key in sorted(keys):
                visit(properties.get(key, {}), key in left_map, left_map.get(key),
                      key in right_map, right_map.get(key), (*parts, key),
                      known and key in properties)
            return
        if lp or rp:
            add(parts, lp, lv, rp, rv, known)

    def add(parts, lp, lv, rp, rv, known, container=False):
        kind = ('added' if rp else 'removed') if lp != rp else (
            'unchanged' if _same_value(lv, rv) else 'changed')
        rows.append({'path': _pointer(parts), 'field': '.'.join(parts),
                     'section': parts[0], 'kind': kind, 'known_field': known,
                     'container_state': container,
                     'left_present': lp, 'left_value': lv if lp else None,
                     'right_present': rp, 'right_value': rv if rp else None})

    visit(schema, True, left, True, right, ())
    return sorted(rows, key=lambda row: row['path'])


def _run_summary(item, config):
    tasks = config.get('selection', {}).get('tasks', {})
    task_count = len(tasks)
    repeat_count = item['plan'].get('repeat_count', 1)
    return {
        'operation': item['plan']['operation'],
        'benchmarks': sorted({task['benchmark'] for task in tasks.values()}),
        'task_count': task_count,
        'repeat_count': repeat_count,
        'sample_count': task_count * repeat_count,
    }


def compare_run_configs(project_root, left_run_id, right_run_id, *, only_changed=False):
    """Validate frozen plans; compare schema fields without current defaults."""
    root = Path(project_root).resolve()
    loaded = [load_plan(root, run_id) for run_id in (left_run_id, right_run_id)]
    configs, metadata = [], []
    for run_id, item in zip((left_run_id, right_run_id), loaded):
        path = item['run'] / 'resolved-config.json'
        config = read_json(path)
        configs.append(config)
        metadata.append({'run_id': run_id, 'resolved_config_path': str(path),
                         'resolved_config_sha256': sha256(path),
                         'config_hash': config.get('config_hash'),
                         'summary': _run_summary(item, config)})
    rows = configurable_rows(*configs)
    differences = [row for row in rows if row['kind'] != 'unchanged']
    return {'schema_version': 2, 'left': metadata[0], 'right': metadata[1],
            'field_count': len(rows), 'difference_count': len(differences),
            'only_changed': only_changed,
            'rows': differences if only_changed else rows, 'differences': differences}


def _markdown_cell(present, value, *, limit=240, quote_strings=True):
    if not present:
        return '&lt;missing&gt;'
    raw = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(raw) > limit:
        digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]
        keep = limit // 2
        raw = (raw[:keep] + f' … [{len(raw)} chars; sha256:{digest}] … ' + raw[-keep:])
    if isinstance(value, str) and quote_strings:
        raw = '"' + raw + '"'
    # Encode user text before adding HTML breaks. Even inline code and HTML
    # inside prompts must remain literal table content.
    return (html.escape(raw).replace('|', '&#124;').replace('`', '&#96;')
            .replace('\\', '&#92;').replace('*', '&#42;')
            .replace('[', '&#91;').replace(']', '&#93;')
            .replace('\r\n', '\n').replace('\r', '\n').replace('\n', '<br>'))


def format_config_diff_markdown(report):
    left, right = report['left'], report['right']
    lines = ['# Frozen run configuration comparison', '',
             '| Run | ID |', '|---|---|',
             '| Left | ' + _markdown_cell(True, left['run_id'], quote_strings=False) + ' |',
             '| Right | ' + _markdown_cell(True, right['run_id'], quote_strings=False) + ' |', '',
             f"Configurable settings: {report['field_count']}; "
             f"different: {report['difference_count']}. Direction: Left → Right.", '',
             'Missing values remain missing; current defaults are not inserted. '
             'Schema constants, derived fields and task bodies are omitted. '
             'Long cells show head/tail, character count and SHA256 prefix; JSON retains full values.', '',
             '## Task and sample summary', '',
             '| Item | Left | Right |', '|---|---|---|']
    for key in ('operation', 'benchmarks', 'task_count', 'repeat_count', 'sample_count'):
        lines.append('| ' + key + ' | ' + _markdown_cell(True, left['summary'][key]) +
                     ' | ' + _markdown_cell(True, right['summary'][key]) + ' |')
    section = None
    kinds = {'added': 'Right only', 'removed': 'Left only',
             'changed': 'Different', 'unchanged': 'Same'}
    for row in report['rows']:
        if row['section'] != section:
            section = row['section']
            title = _SECTION_TITLES.get(section, section.replace('_', ' ').capitalize())
            lines.extend(['', '## ' + _markdown_cell(True, title, quote_strings=False), '',
                          '| Setting | Left | Right | Comparison |', '|---|---|---|---|'])
        name = _markdown_cell(True, row['field'], quote_strings=False)
        if not row['known_field']:
            name += ' (legacy/extension)'
        if row['container_state']:
            name += ' (mapping state)'
        lines.append('| ' + ' | '.join((name,
            _markdown_cell(row['left_present'], row['left_value']),
            _markdown_cell(row['right_present'], row['right_value']), kinds[row['kind']])) + ' |')
    if not report['rows']:
        lines.extend(['', 'No configurable differences.' if report['only_changed']
                      else 'No frozen configurable settings.'])
    return '\n'.join(lines)
