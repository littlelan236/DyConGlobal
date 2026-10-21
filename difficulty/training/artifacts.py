"""Shared completion gate for immutable difficulty training attempts."""

from core.io import now, sha256, write_json


def finish_attempt(output, attempt, root, before_code, *, parents=()):
    if {name: sha256(root / name) for name in before_code} != before_code:
        raise ValueError('Implementation changed during experiment')
    dataset = root / 'difficulty/datasets' / attempt['dataset_id']
    if sha256(dataset / 'receipt.json') != attempt['dataset_receipt_sha256']:
        raise ValueError('Dataset changed during experiment')
    split = dataset / 'splits' / attempt['split_id']
    if (sha256(split / 'receipt.json') != attempt['split_receipt_sha256']
            or sha256(split / 'split.json') != attempt['split_sha256']):
        raise ValueError('Split changed during experiment')
    receipt = {
        'schema_version': 1, 'state': 'complete', 'mode': attempt['mode'],
        'completed_at': now(), 'dataset_receipt_sha256': attempt['input']['receipt_sha256'],
        'dataset_id': attempt['dataset_id'], 'split_id': attempt['split_id'],
        'split_receipt_sha256': attempt['split_receipt_sha256'],
        'split_sha256': attempt['split_sha256'], 'code': before_code,
        'files': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                  if p.is_file() and p.name != 'receipt.json'},
    }
    if parents:
        receipt['parents'] = list(parents)
    write_json(output / 'receipt.json', receipt)
