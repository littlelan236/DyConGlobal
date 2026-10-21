"""Wait on exporter exit event, then verify all catalog NAS references."""
import os
import select
import sys
from pathlib import Path
from core.config import resolve
from core.io import read_json, write_json
from catalog.store import Catalog
from runtime.images.nas import NasImageStore

root = Path('/home/wangtianyi/DyConGlobal')
try:
    fd = os.pidfd_open(int(sys.argv[1]))
except ProcessLookupError:
    fd = None
if fd is not None:
    select.select([fd], [], [])
    os.close(fd)
out = root / 'state/nas-export-258'
try:
    receipt = read_json(out / 'receipt.json')
    cfg = resolve(root / 'config.yaml')
    catalog = Catalog(root)
    manifest = catalog.read()
    records = {}
    for task in manifest['tasks'].values():
        row = task['instance']
        store = NasImageStore(cfg, row['nas_image'])
        if store.reference['store'] != 'legacy-258-20261017':
            raise ValueError('Task has unexpected NAS namespace')
        descriptors = store.verify(deep=False)
        records[row['instance_id']] = {'image_id': store.image_id, 'descriptors': len(descriptors)}
    assert len(records) == 258
    write_json(out / 'verification.json', {'verified': True, 'tasks': len(records),
               'checks': 'Export hashes every streamed blob; validate catalog, manifest checksums and OCI descriptor sizes',
               'catalog': catalog.verify(), 'images': records, 'receipt': receipt})
except BaseException as exc:
    write_json(out / 'verification.json', {'verified': False, 'error': repr(exc)})
    raise
