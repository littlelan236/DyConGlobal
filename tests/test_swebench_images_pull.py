import concurrent.futures
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.images import swebench_pull as subject
from runtime.images.registry import transfer_blob


class ImageCompletionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store, self.seed = self.root / 'store', self.root / 'seed'
        for path in ('blobs/sha256', 'images', 'dataset/metadata'):
            (self.store / path).mkdir(parents=True)
        self.seed.mkdir()
        self.config = json.dumps({'os': 'linux', 'architecture': 'amd64'}).encode()
        self.layer = b'an immutable compressed layer fixture'

        def descriptor(raw, media):
            return {'digest': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size': len(raw), 'mediaType': media}
        self.cd = descriptor(self.config, 'application/vnd.oci.image.config.v1+json')
        self.ld = descriptor(self.layer, 'application/vnd.oci.image.layer.v1.tar+gzip')
        self.manifest = json.dumps({'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.manifest.v1+json',
                                    'config': self.cd, 'layers': [self.ld]}).encode()
        self.md = descriptor(self.manifest, 'application/vnd.oci.image.manifest.v1+json')
        self.descriptors = [{**d, 'repository': 'swebench/sweb.eval.x86_64.fixture'}
                            for d in (self.md, self.cd, self.ld)]
        self.raws = {d['digest']: raw for d, raw in ((self.md, self.manifest), (self.cd, self.config), (self.ld, self.layer))}
        for d in self.descriptors:
            (self.seed / d['digest'][7:]).write_bytes(self.raws[d['digest']])

    def metadata(self, *args, **kwargs):
        for d in (self.cd, self.md):
            (self.store / 'blobs/sha256' / d['digest'][7:]).write_bytes(self.raws[d['digest']])
        return self.md, self.descriptors

    def test_failed_layer_is_not_published_and_resume_uses_pinned_manifest(self):
        service = subject._Transfer(self.store, [self.seed], None)
        def fail_layer(item, *args):
            if item['digest'] == self.ld['digest']:
                raise OSError('interrupted transfer')
            return transfer_blob(item, *args)
        image = 'swebench/sweb.eval.x86_64.fixture:latest'
        with patch.object(subject, 'image_metadata', side_effect=self.metadata), patch.object(subject, 'transfer_blob', side_effect=fail_layer):
            with self.assertRaises(OSError):
                service.image('fixture-1', image)
        self.assertFalse((self.store / 'images/fixture-1.json').exists())
        self.assertTrue((self.store / 'dataset/metadata/fixture-1.json').exists())
        with patch.object(subject, 'image_metadata', side_effect=AssertionError('must not re-resolve latest')):
            subject._Transfer(self.store, [self.seed], None).image('fixture-1', image)
        record = json.loads((self.store / 'images/fixture-1.json').read_text())
        self.assertTrue(record['deep_verified'])
        self.assertEqual(record['manifest_digest'], self.md['digest'])

    def test_shared_layers_transfer_once_across_workers(self):
        service = subject._Transfer(self.store, [self.seed], None)
        with patch.object(subject, 'transfer_blob', wraps=transfer_blob) as spy:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(service.blob, [self.descriptors[2]] * 12))
            self.assertEqual(spy.call_count, 1)

    def test_corrupt_existing_blob_is_rejected(self):
        target = self.store / 'blobs/sha256' / self.ld['digest'][7:]
        target.write_bytes(b'x' * len(self.layer))
        with self.assertRaises(ValueError):
            subject._Transfer(self.store, [self.seed], None).blob(self.descriptors[2])

    def test_event_written_only_after_real_download_bytes(self):
        service = subject._Transfer(self.store, [self.seed], None)
        path = self.store / 'dataset/transfer-started.json'
        self.assertFalse(path.exists())
        service.progress(4194304)
        self.assertEqual(json.loads(path.read_text())['received_bytes'], 4194304)

    def test_non_verified_plan_rejected_before_any_download(self):
        with self.assertRaises(ValueError):
            subject._validate_plan({'source': {'dataset': 'wrong'}, 'official_images': {}, 'schema_version': 1}, self.root)


if __name__ == '__main__':
    unittest.main()
