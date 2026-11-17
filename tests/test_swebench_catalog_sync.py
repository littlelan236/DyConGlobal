import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catalog import swebench_sync as subject
from catalog.store import Catalog, empty_manifest
from core.backends import benchmark_family
from core.io import hash_json, read_json, write_json


def task(iid, benchmark='SWE-bench_Verified'):
    return {'benchmark': benchmark, 'revision': 'old-revision', 'instance_id': iid,
            'repo': 'org/repo', 'base_commit': 'abc', 'partition': 'train', 'exclusion_reason': None,
            'instance': {'instance_id': iid, 'repo': 'org/repo', 'base_commit': 'abc',
                         'nas_image': {'store': 'preserved', 'image_id': 'pinned'},
                         'environment_overrides': {'network_fallback': False}, 'patch': 'gold'}}


class CatalogSyncTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {'project_root': str(self.root), 'paths': {'nas_root': str(self.root / 'nas')}}
        write_json(self.root / 'catalog/manifest.json', empty_manifest())
        self.catalog = Catalog(self.root)
        self.old = task('verified-old', 'legacy-tactRefractor-swebench')
        self.other = task('untouched', 'external-backend')
        self.catalog.register_tasks([self.old, self.other])
        self.sets = {'verified': {'verified-old', 'verified-new'}, 'dev': {'dev-ok', 'dev-missing'},
                     'test': {'verified-old', 'verified-new', 'test-old'}}
        self.source = {'sources': {'verified': {'revision': 'v1'}, 'dev': {'revision': 'd1'}}}
        self.rows = {i: {'instance_id': i, 'repo': 'org/repo', 'base_commit': 'abc', 'image': 'official:latest'}
                     for i in ('verified-new', 'dev-ok', 'dev-missing')}
        self.refs = {i: {'store': 'new', 'image_id': i} for i in ('verified-new', 'dev-ok')}
        (self.root / 'source.json').write_text('{}')

    def mocks(self):
        return patch.object(subject, 'load_sources', return_value=(self.source, self.sets, self.rows)), \
            patch.object(subject, '_plan_references', return_value=self.refs), \
            patch.object(subject, 'NasImageStore')

    def test_apply_rekeys_preserves_old_task_and_other_backends_skips_missing(self):
        frozen = self.root / 'runs/old/selection.json'
        write_json(frozen, {'tasks': self.catalog.read()['tasks']})
        old_frozen = frozen.read_bytes()
        before = self.catalog.read()
        a, b, c = self.mocks()
        with a, b, c:
            result = subject.sync(self.cfg, source_path=self.root/'source.json', image_plans=[],
                                  output=self.root/'receipt', apply=True)
        after = self.catalog.read()
        self.assertEqual(result['added'], 2)
        self.assertEqual(result['rekeyed'], 1)
        self.assertEqual(result['missing_counts']['dev'], 1)
        self.assertEqual(after['catalog_revision'], before['catalog_revision'] + 1)
        old = next(t for t in after['tasks'].values() if t['instance_id'] == 'verified-old')
        self.assertEqual(old, {**self.old, 'benchmark': 'swebench_verified'})
        self.assertIn(self.other, after['tasks'].values())
        self.assertNotIn('dev-missing', {t['instance_id'] for t in after['tasks'].values()})
        self.assertEqual(read_json(self.root/'receipt/catalog-before.json'), before)
        self.assertEqual(frozen.read_bytes(), old_frozen)

    def test_preview_does_not_modify_catalog_and_rerun_is_idempotent(self):
        original = self.catalog.path.read_bytes()
        a, b, c = self.mocks()
        with a, b, c:
            subject.sync(self.cfg, source_path=self.root/'source.json', image_plans=[], output=self.root/'preview')
            self.assertEqual(self.catalog.path.read_bytes(), original)
            subject.sync(self.cfg, source_path=self.root/'source.json', image_plans=[], output=self.root/'apply', apply=True)
            changed = self.catalog.path.read_bytes()
            result = subject.sync(self.cfg, source_path=self.root/'source.json', image_plans=[], output=self.root/'repeat', apply=True)
        self.assertFalse(result['changed'])
        self.assertEqual(self.catalog.path.read_bytes(), changed)

    def test_corrupt_image_causes_no_catalog_mutation(self):
        original = self.catalog.path.read_bytes()
        a, b, c = self.mocks()
        with a, b, c as image:
            image.return_value.verify.side_effect = ValueError('corrupt graph')
            with self.assertRaisesRegex(ValueError, 'corrupt graph'):
                subject.sync(self.cfg, source_path=self.root/'source.json', image_plans=[], output=self.root/'bad', apply=True)
        self.assertEqual(self.catalog.path.read_bytes(), original)

    def test_category_rejects_unknown_and_separates_dev_from_nonverified(self):
        self.assertEqual(subject.category('test-old', self.sets), 'nonverified')
        self.assertEqual(subject.category('dev-ok', self.sets), 'dev')
        with self.assertRaisesRegex(ValueError, 'outside pinned'):
            subject.category('unknown', self.sets)

    def test_ambiguous_old_revisions_are_rejected(self):
        before = self.catalog.read()
        second = {**self.old, 'revision': 'another'}
        before['tasks']['duplicate'] = second
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            subject.build_tasks(self.cfg, before, self.source, self.sets, self.rows, {})

    def test_groups_use_existing_swe_backend_and_export_formal_tasks_file(self):
        for benchmark in subject.GROUPS.values():
            self.assertEqual(benchmark_family(benchmark), 'swebench')
        a, b, c = self.mocks()
        with a, b, c:
            subject.sync(self.cfg, source_path=self.root/'source.json', image_plans=[], output=self.root/'apply', apply=True)
        path = self.root/'dev-tasks.json'
        result = subject.list_tasks(self.cfg, group='dev', output=path)
        self.assertEqual(result['selected'], 1)
        self.assertEqual(read_json(path)['tasks'], ['dev-ok'])
        self.assertEqual(set(self.catalog.select_tasks(benchmark='swebench', task_ids=['dev-ok'])['task_keys']),
                         {hash_json({k: t[k] for k in ('benchmark', 'revision', 'instance_id')})
                          for t in self.catalog.read()['tasks'].values() if t['instance_id'] == 'dev-ok'})


if __name__ == '__main__':
    unittest.main()
