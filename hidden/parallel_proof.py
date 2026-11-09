"""Frozen branch decisions and exact request identities for visualization."""
from pathlib import Path

from core.io import identifier, read_json, sha256
from hidden.artifacts import read_parallel_proof


def decision_inventory(roots, sid, rid, trajectory_sha256):
    """Read receipt-bound branches, or retain unsealed decisions as audit evidence."""
    copies = {}
    for root in roots:
        base = Path(root) / 'sessions' / identifier(sid) / 'parallel'
        if not base.exists():
            continue
        if (base / 'branch-receipt.json').exists():
            view = read_parallel_proof(root, sid, rollout_id=rid, trajectory_sha256=trajectory_sha256)
            for decision in view['decisions']:
                did = decision['decision_id']
                path = base / 'decisions' / (identifier(did) + '.json')
                copies.setdefault(did, []).append((decision, path, True))
        else:
            for path in sorted((base / 'decisions').glob('*.json')):
                decision = read_json(path)
                copies.setdefault(decision['decision_id'], []).append((decision, path, False))
    decisions, sources, receipt_bound = {}, {}, set()
    for did, variants in copies.items():
        first = variants[0][0]
        if any(item[0] != first for item in variants):
            raise ValueError('Conflicting parallel decisions: ' + did)
        if first.get('session_id') != sid:
            raise ValueError('Parallel decision belongs to another session')
        candidates = first.get('candidates')
        if not isinstance(candidates, list) or len(candidates) != first.get('branch_count'):
            raise ValueError('Incomplete parallel candidate inventory')
        ids = [candidate.get('branch_id') for candidate in candidates]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError('Duplicate parallel candidate')
        if first.get('selected_branch_id') is not None and first['selected_branch_id'] not in ids:
            raise ValueError('Parallel winner absent from candidate inventory')
        decisions[did] = first
        if any(sealed for _,_,sealed in variants): receipt_bound.add(did)
        for _, path, _ in variants:
            sources[str(path)] = sha256(path)
            base = path.parent.parent
            receipt = base / 'branch-receipt.json'
            if receipt.exists():
                sources[str(receipt)] = sha256(receipt)
                from core.io import owned_path
                for relative, digest in read_json(receipt)['files'].items():
                    sources[str(owned_path(path.parents[4], relative))] = digest
                revision=base/'branch-classification-v2.json'
                revision_receipt=base/'branch-classification-v2-receipt.json'
                if revision.exists():
                    sources[str(revision)]=sha256(revision)
                    sources[str(revision_receipt)]=sha256(revision_receipt)
                    for relative,digest in read_json(revision_receipt)['files'].items():
                        sources[str(owned_path(path.parents[4],relative))]=digest
                    for relative,digest in read_json(revision)['source_files'].items():
                        sources[str(owned_path(path.parents[4],relative))]=digest
    return decisions, sources, receipt_bound


def validate_canonical_decisions(responses, decisions):
    """Bind each parallel response to its recorded winner, never choice order."""
    selected = {}
    for index, message, response in responses:
        extra = message.get('extra') or {}
        metadata = (extra.get('request') or {}).get('dycon_capture') or {}
        summary = extra.get('dycon_parallel')
        did = metadata.get('decision_id') or (summary or {}).get('decision_id')
        if did is None:
            continue
        decision = decisions.get(did)
        if decision is None or not isinstance(summary, dict):
            raise ValueError('Canonical parallel decision evidence unavailable')
        winner = decision.get('selected_branch_id')
        if winner is None or summary.get('selected_branch_id') != winner:
            raise ValueError('Canonical and recorded winner disagree')
        for key in ('decision_id', 'query_id', 'parent_prefix_sha256'):
            if summary.get(key) != decision.get(key):
                raise ValueError('Canonical decision identity disagrees: ' + key)
        if metadata.get('branch_id') != winner or metadata.get('decision_id') != did:
            raise ValueError('Canonical capture metadata disagrees with winner')
        if metadata.get('query_id') != decision.get('query_id'):
            raise ValueError('Canonical query disagrees with decision')
        if metadata.get('parent_prefix_sha256') != decision.get('parent_prefix_sha256'):
            raise ValueError('Canonical parent prefix disagrees with decision')
        candidate = next(c for c in decision['candidates'] if c['branch_id'] == winner)
        native = candidate.get('native_request_id')
        if not native or response.get('dycon_native_request_id') != native:
            raise ValueError('Canonical native request disagrees with winner')
        if candidate.get('api_attempt_id') != metadata.get('api_attempt_id'):
            raise ValueError('Canonical API attempt disagrees with winner')
        if native in selected:
            raise ValueError('Canonical winner request reused')
        selected[native] = {'message_index': index, 'decision_id': did, 'branch_id': winner}
    return selected
