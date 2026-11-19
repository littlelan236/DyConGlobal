"""Frozen advantage queries with explicit streaming legacy indexing (no Torch)."""
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock
from RL.observability import matches, process, snapshot, source, streaming_json
from .advantage import SAME_ACTION_ADVANTAGE_VERSION
from .grouping import lineage, order_record


def file_identity(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


class TokenArray:
    def __init__(self, start, length, count):
        self.start, self.length, self.count = start, length, count


def canonical_hash(digest, value, spool):
    """Hash core.io canonical JSON while tokens stay in a one-group disk spool."""
    if isinstance(value, TokenArray):
        spool.seek(value.start)
        left = value.length
        while left:
            data = spool.read(min(left, 1024*1024))
            if not data:
                raise ValueError('Truncated temporary canonical token spool')
            digest.update(data)
            left -= len(data)
    elif isinstance(value, dict):
        digest.update(b'{')
        for i, key in enumerate(sorted(value)):
            if i:
                digest.update(b',')
            canonical_hash(digest, key, spool)
            digest.update(b':')
            canonical_hash(digest, value[key], spool)
        digest.update(b'}')
    elif isinstance(value, list):
        digest.update(b'[')
        for i, item in enumerate(value):
            if i:
                digest.update(b',')
            canonical_hash(digest, item, spool)
        digest.update(b']')
    else:
        digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())


def parse_group(events, first, spool, depth=0, key=None):
    if depth > 64:
        raise ValueError('Legacy JSON depth exceeds 64')
    event, value = first
    if event == 'start_map':
        result = {}
        while True:
            event, name = next(events)
            if event == 'end_map':
                return result
            if event != 'map_key' or name in result:
                raise ValueError('Invalid/duplicate legacy object key')
            result[name] = parse_group(events, next(events), spool, depth+1, name)
    if event == 'start_array':
        if key in ('prompt_token_ids', 'completion_token_ids'):
            start = spool.tell()
            count = 0
            buffer = bytearray(b'[')
            for event, value in events:
                if event == 'end_array':
                    break
                if event != 'number' or type(value) is not int or value < 0:
                    raise ValueError('Native token array must contain nonnegative integer IDs')
                if count:
                    buffer.extend(b',')
                buffer.extend(str(value).encode())
                count += 1
                if len(buffer) >= 65536:
                    spool.write(buffer)
                    buffer.clear()
            buffer.extend(b']')
            spool.write(buffer)
            return TokenArray(start, spool.tell()-start, count)
        result = []
        while True:
            item = next(events)
            if item[0] == 'end_array':
                return result
            result.append(parse_group(events, item, spool, depth+1))
            if len(result) > 100000:
                raise ValueError('Legacy metadata array exceeds 100000 entries')
    if event not in ('number', 'string', 'null', 'boolean'):
        raise ValueError('Malformed legacy JSON')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite frozen value')
    return value


class FingerprintedStream:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def read(self, count=-1):
        data = self.stream.read(count)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def index_status(cache, directory):
    folder = cache / 'legacy-advantages'
    record = read_json(folder / 'status.json') if (folder / 'status.json').exists() else {'state': 'missing'}
    if record.get('state') == 'building' and not matches(record.get('process')):
        record = {**record, 'state': 'failed', 'error': 'Index worker exited before publishing a ready index'}
    if record.get('source_identity') and record['source_identity'] != file_identity(directory / 'frozen-groups.json'):
        record = {**record, 'state': 'invalid', 'error': 'Frozen source changed'}
    return record


def streaming_index(directory, contract):
    """The same-action stream plan contains the frozen scores and branch diagnostics."""
    if ("streaming" not in contract
            or contract["input"].get("advantage_policy") != SAME_ACTION_ADVANTAGE_VERSION):
        return None
    from .stream_resume import PLAN, checked_path
    from .streaming import validate_plan
    path = checked_path(directory, PLAN)
    if not path.is_file():
        raise ValueError("Frozen streaming advantage plan is missing")
    if sha256(path) != contract["streaming"]["plan_sha256"]:
        raise ValueError("Frozen streaming advantage plan checksum mismatch")
    plan = validate_plan(read_json(path))
    if plan["proof"] != contract["input"]:
        raise ValueError("Frozen streaming advantage proof differs from the training contract")
    return plan["groups"]


def advantage_index_status(directory, contract, cache):
    if "prepared" in contract:
        return {"state": "ready", "source": "frozen_prepared_artifact"}
    streaming = streaming_index(directory, contract)
    if streaming is not None:
        return {"state": "ready", "source": "frozen_stream_plan", "groups": len(streaming)}
    return index_status(cache, directory)


def build_index(cfg, attempt_id, storage_root=None):
    directory, attempt, identity, cache = source(cfg, attempt_id, storage_root)
    contract = attempt.get('contract', {})
    if 'prepared' in contract:
        return {'state': 'ready', 'source': 'frozen_prepared_artifact', 'message': 'Prepared artifact already has an index'}
    streaming = streaming_index(directory, contract)
    if streaming is not None:
        return {'state': 'ready', 'source': 'frozen_stream_plan', 'groups': len(streaming)}
    path = directory / 'frozen-groups.json'
    if not path.exists() or attempt.get('status') == 'preparing_data':
        raise ValueError('Frozen source is missing or still being prepared')
    if time.time()-path.stat().st_mtime < 3:
        raise ValueError('Frozen source is too recent; retry after preparation finishes')
    folder = cache / 'legacy-advantages'
    with file_lock(folder / 'start.lock', nonblocking=True):
        status = index_status(cache, directory)
        if status['state'] == 'ready' and status.get('index_sha256') and sha256(folder/'index.sqlite3') != status['index_sha256']:
            status = {**status, 'state':'invalid', 'error':'Index checksum mismatch'}
        if status['state'] == 'ready' and not status.get('index_sha256'):
            # Explicitly attest a completed schema-1 local cache, without rescanning tokens.
            status['index_sha256'] = sha256(folder/'index.sqlite3')
            write_json(folder/'source-binding.json',status)
            write_json(folder/'status.json',status)
        if status['state'] in ('building', 'ready'):
            return status
        write_json(cache / 'source.json', identity)
        python = Path(cfg['project_root']) / '.venvs/analysis/bin/python'
        with (folder / 'worker.log').open('ab') as log:
            child = subprocess.Popen([str(python), '-m', 'RL.grpo.observability', '--build-index', str(cache / 'source.json')],
                cwd=cfg['project_root'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'schema_version': 1, 'state': 'building', 'bytes_scanned': 0, 'groups': 0,
                  'total_bytes': path.stat().st_size, 'source_identity': file_identity(path),
                  'process': process(child.pid), 'started_at': now()}
        write_json(folder / 'status.json', record)
        return record


def index_worker(descriptor):
    identity = read_json(descriptor)
    directory = Path(identity['directory'])
    folder = Path(descriptor).parent / 'legacy-advantages'
    source_path = directory / 'frozen-groups.json'
    with file_lock(folder / 'writer.lock', nonblocking=True):
        with file_lock(folder / 'start.lock'):
            pass  # Parent registration must finish before worker status publication.
        original = file_identity(source_path)
        attempt = read_json(directory / 'attempt.json')
        contract = attempt['contract']
        if hash_json(contract) != identity['contract_sha256']:
            raise ValueError('Attempt contract changed before indexing')
        proof = contract['input']
        if read_json(directory / 'grpo-contract.json') != contract:
            raise ValueError('Frozen GRPO contract mismatch')
        expected_ids = proof['eligible_group_ids']
        pending = folder / ('pending-' + str(os.getpid()) + '.sqlite3')
        pending.unlink(missing_ok=True)
        db = sqlite3.connect(pending)
        db.execute('CREATE TABLE groups(ordinal INTEGER UNIQUE, id TEXT PRIMARY KEY, payload TEXT)')
        db.execute('CREATE TABLE branches(id TEXT PRIMARY KEY, group_id TEXT)')
        status = {'schema_version': 1, 'state': 'building', 'process': process(os.getpid()),
                  'source_identity': original, 'total_bytes': original[2], 'started_at': now(), 'groups': 0}
        order_digest = hashlib.sha256(b'[')
        count = 0
        last_report = 0.
        try:
            with source_path.open('rb') as raw:
                stream = FingerprintedStream(raw)
                events = iter(streaming_json().basic_parse(stream, use_float=True, buf_size=1024*1024))
                if next(events) != ('start_array', None):
                    raise ValueError('Frozen groups must be a top-level array')
                while True:
                    first = next(events)
                    if first[0] == 'end_array':
                        break
                    with tempfile.TemporaryFile(dir=folder) as spool:
                        value = parse_group(events, first, spool)
                        gid = value['group']['group_id']
                        if count >= len(expected_ids) or gid != expected_ids[count]:
                            raise ValueError('Frozen group order/count differs from training contract')
                        members = value['members']
                        ids = [m['branch_sample_id'] for m in members]
                        if (len(members) != value['group']['expected_branch_count']
                                or value['group'].get('branch_ids', ids) != ids):
                            raise ValueError('Frozen branch native order mismatch')
                        advantages = value.get('frozen_advantages')
                        if advantages is not None:
                            if len(advantages) != len(members) or any(type(a) not in (int,float) or not math.isfinite(a) for a in advantages):
                                raise ValueError('Frozen advantages are invalid')
                            record = order_record(value, proof['advantage_policy'])
                            if count:
                                order_digest.update(b',')
                            canonical_hash(order_digest, record, spool)
                        else:
                            # Lack of producer evidence cannot be repaired by today's math.
                            status['advantage_evidence'] = 'unavailable_not_proven_as_used'
                        for member in members:
                            for key in ('prompt_token_ids', 'completion_token_ids'):
                                tokens = member.pop(key, None)
                                if isinstance(tokens, TokenArray):
                                    member['prompt_count' if key.startswith('prompt') else 'completion_count'] = tokens.count
                            db.execute('INSERT INTO branches VALUES (?,?)', (member['branch_sample_id'], gid))
                        db.execute('INSERT INTO groups VALUES (?,?,?)', (count, gid, json.dumps(value, allow_nan=False)))
                    count += 1
                    if time.monotonic()-last_report > 2:
                        status.update(bytes_scanned=stream.bytes, groups=count, updated_at=now())
                        write_json(folder / 'status.json', status)
                        last_report = time.monotonic()
                if next(events, None) is not None:
                    raise ValueError('Trailing legacy JSON content')
                order_digest.update(b']')
                if count != len(expected_ids) or count != proof['group_count']:
                    raise ValueError('Frozen group count mismatch')
                if not status.get('advantage_evidence') and order_digest.hexdigest() != proof['group_order_sha256']:
                    raise ValueError('Frozen score/retained/advantage/token/order hash differs from training contract')
                if original != file_identity(source_path):
                    raise ValueError('Frozen source changed during indexing')
                digest = stream.digest.hexdigest()
                receipt_path = directory / 'receipt.json'
                integrity = 'locally_fingerprinted_unsealed'
                receipt_hash = None
                if receipt_path.exists():
                    receipt = read_json(receipt_path)
                    if receipt['files']['frozen-groups.json']['sha256'] != digest:
                        raise ValueError('Frozen JSON receipt checksum mismatch')
                    integrity, receipt_hash = 'receipt_verified', sha256(receipt_path)
                db.commit()
                db.close()
                os.replace(pending, folder / 'index.sqlite3')
                status.update(state='ready', bytes_scanned=stream.bytes, groups=count, source_sha256=digest,
                    contract_sha256=identity['contract_sha256'], source_integrity=integrity,
                    source_receipt_sha256=receipt_hash, index_sha256=sha256(folder / 'index.sqlite3'),
                    parser_version=1, completed_at=now())
                write_json(folder / 'source-binding.json', status)
                write_json(folder / 'status.json', status)
        except BaseException as exc:
            db.close()
            pending.unlink(missing_ok=True)
            status.update(state='failed', error=f'{type(exc).__name__}: {exc}', updated_at=now())
            write_json(folder / 'status.json', status)
            raise


def group_view(value, ordinal, proof, progress, *, schedule_id=None, segment_id=None, segment_ordinal=None,
               provenance='frozen_prepared_artifact', token_preview=0):
    members = value.get('members', [])
    advantages = value.get('frozen_advantages')
    committed = progress.get('committed_consumed_groups')
    live = progress.get('live') or {}
    end = live.get('window_end_group_ordinal')
    consumption = ('consumed' if ordinal is not None and committed is not None and ordinal < committed else
                   'inflight' if ordinal is not None and end is not None and committed is not None and committed <= ordinal < end else
                   'pending' if ordinal is not None and committed is not None else 'unknown')
    if advantages is None:
        provenance = 'unavailable'
    scores = [r['historical_score'] for r in members]
    diagnostics = value.get('advantage_diagnostics') or {}
    order = diagnostics.get('order', sorted(range(len(members)), key=lambda i: (scores[i], i)))
    ranks = diagnostics.get('ranks', [sum(score < current for score in scores) for current in scores])
    if sorted(order) != list(range(len(members))) or len(ranks) != len(members):
        raise ValueError('Frozen advantage ranking diagnostics are inconsistent')
    positions = {native:position for position,native in enumerate(order)}
    branches = []
    for i, row in enumerate(members):
        advantage = advantages[i] if advantages is not None else None
        item = {'native_ordinal': i, 'branch_id': row['branch_sample_id'],
                'score': row['historical_score'], 'retained': row.get('retained'),
                'advantage': advantage, 'rank': ranks[i], 'tie_break_order': positions[i],
                'rank_provenance': 'frozen_diagnostics' if 'ranks' in diagnostics else 'frozen_score_order',
                'sign': 'positive' if advantage is not None and advantage > 0 else 'negative' if advantage is not None and advantage < 0 else 'zero' if advantage == 0 else None,
                'provenance': provenance, 'consumption': consumption,
                'prompt_length': row.get('prompt_count'), 'completion_length': row.get('completion_count')}
        for key in ('source_run','run_id','task_id','agent_step','branch_id'):
            if key in row:
                item['source_' + key] = row[key]
        if token_preview:
            item['token_preview'] = {key: row.get(key, [])[:token_preview] for key in ('prompt_token_ids','completion_token_ids')}
        branches.append(item)
    values = advantages or []
    return {'group_id':value['group']['group_id'], **lineage(value['group']),
            'group_ordinal':ordinal, 'schedule_id':schedule_id,
            'segment_id':segment_id, 'segment_local_ordinal':segment_ordinal, 'consumption':consumption,
            'provenance':provenance, 'advantage_policy':proof.get('advantage_policy'),
            'advantage_config_sha256':proof.get('advantage_config_sha256'),
            'dataset_receipt_sha256':proof.get('dataset_receipt_sha256'),
            'summary':{'sum':math.fsum(values) if values else None, 'min':min(values) if values else None,
                       'max':max(values) if values else None, 'positive':sum(a>0 for a in values),
                       'negative':sum(a<0 for a in values), 'zero':sum(a==0 for a in values)},
            'branches':branches, 'total_branches':len(branches)}


def query(cfg, attempt_id, *, storage_root=None, group_id=None, group_ordinal=None, branch_id=None,
          offset=0, limit=None, token_preview=0):
    if offset < 0 or limit is not None and not 1 <= limit <= 256 or not 0 <= token_preview <= 128:
        raise ValueError('offset >= 0; limit 1..256; token-preview 0..128 required')
    directory, attempt, identity, cache = source(cfg, attempt_id, storage_root)
    contract = attempt['contract']
    proof = contract['input']
    progress = snapshot(cfg, attempt_id, storage_root, recent=0)
    exclusions = {r['group_id']:r for r in proof.get('exclusions', [])}
    stream_groups = streaming_index(directory, contract)
    ids = proof['eligible_group_ids']
    reader = None
    db = None
    index = None
    if 'prepared' in contract:
        from RL.grpo.prepared import PreparedReader, resolve
        reader = PreparedReader(resolve(contract['prepared']), verify=False)
        reader.verify_metadata_indexes()
        if (reader.proof != proof or reader.schedule['schedule_id'] != contract['prepared_schedule_id']
                or reader.schedule['sha256'] != contract['prepared_schedule_sha256']):
            raise ValueError('Prepared query does not match training contract')
        if branch_id:
            owner = reader
            while owner:
                with closing(sqlite3.connect((owner.path/'inventory.sqlite3').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)) as con:
                    row = con.execute('SELECT group_id FROM branches WHERE id=?',(branch_id,)).fetchone()
                if row:
                    if group_id and group_id != row[0]:
                        raise ValueError('Branch does not belong to requested group')
                    group_id = row[0]
                    break
                owner = owner.parent
            else:
                raise ValueError('Unknown branch ID')
    elif stream_groups is not None:
        if token_preview:
            raise ValueError('Stream plan excludes tokens; token preview requires a prepared artifact')
        if branch_id:
            owners = [entry['group_id'] for entry in stream_groups if branch_id in entry['branch_ids']]
            if len(owners) != 1 or (group_id and group_id != owners[0]):
                raise ValueError('Unknown branch ID or branch/group mismatch')
            group_id = owners[0]
    else:
        index = index_status(cache, directory)
        if index['state'] != 'ready' or index.get('contract_sha256') != identity['contract_sha256']:
            raise ValueError('Legacy advantage index is not ready; run lora-advantages ATTEMPT_ID --build-index; inspect --index-status')
        if sha256(cache/'legacy-advantages/index.sqlite3') != index.get('index_sha256'):
            raise ValueError('Legacy metadata index checksum mismatch; rebuild required')
        db = sqlite3.connect((cache/'legacy-advantages/index.sqlite3').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        if branch_id:
            row = db.execute('SELECT group_id FROM branches WHERE id=?',(branch_id,)).fetchone()
            if not row or group_id and group_id != row[0]:
                db.close()
                raise ValueError('Unknown branch ID or branch/group mismatch')
            group_id = row[0]
        if token_preview:
            db.close()
            raise ValueError('Legacy metadata index excludes tokens; token preview requires a prepared artifact')
    try:
        if group_ordinal is not None:
            if not 0 <= group_ordinal < len(ids):
                raise ValueError('group-ordinal is outside the frozen schedule')
            if group_id and group_id != ids[group_ordinal]:
                raise ValueError('Group ID/ordinal/branch filters disagree')
            group_id = ids[group_ordinal]
        if group_id in exclusions:
            return {'schema_version':1, 'attempt_id':attempt_id, 'groups':[{'group_id':group_id,
                    **lineage(exclusions[group_id]), 'exclusion_reason':exclusions[group_id]['reason'],
                    'advantage':None, 'provenance':'unavailable', 'branches':[]}], 'has_more':False}
        selected = group_id is not None
        if selected and group_id not in ids:
            raise ValueError('Unknown group ID')
        selected_ids = [group_id] if selected else ids[offset:offset+(limit or 20)]
        results = []
        provenance = ('frozen_prepared_artifact' if reader else
                      'frozen_stream_plan' if stream_groups is not None else 'frozen_groups_json')
        for gid in selected_ids:
            ordinal = ids.index(gid)
            schedule_id = segment_id = local = None
            if reader:
                value = reader.metadata(gid, token_preview=token_preview)
                schedule_id = reader.schedule['schedule_id']
                base = 0
                for segment in reader.schedule['segments']:
                    if ordinal < base+len(segment['group_ids']):
                        segment_id, local = segment['segment_id'], ordinal-base
                        break
                    base += len(segment['group_ids'])
            elif stream_groups is not None:
                entry = stream_groups[ordinal]
                value = {**entry, 'group': {'group_id': gid, **lineage(entry)}}
                segment_id, local = 'streaming', ordinal
            else:
                row = db.execute('SELECT payload FROM groups WHERE id=?',(gid,)).fetchone()
                if not row:
                    raise ValueError('Legacy index group missing')
                value = json.loads(row[0])
            result = group_view(value, ordinal, proof, progress, schedule_id=schedule_id,
                segment_id=segment_id, segment_ordinal=local,
                provenance=provenance, token_preview=token_preview)
            if selected:
                branches = result['branches']
                if branch_id:
                    branches = [b for b in branches if b['branch_id'] == branch_id]
                total = len(branches)
                result['branches'] = branches[offset:offset+(limit or 256)]
                result.update(returned=len(result['branches']), total=total,
                              truncated=offset+len(result['branches']) < total,
                              next_offset=offset+len(result['branches']) if offset+len(result['branches']) < total else None)
            else:
                result.pop('branches')
            results.append(result)
        return {'schema_version':1, 'attempt_id':attempt_id, 'source':identity,
                'source_integrity':(index.get('source_integrity') if index else
                    'stream_plan_sha256_verified' if stream_groups is not None else 'prepared_metadata_receipts_verified'),
                'groups':results, 'total_groups':len(ids), 'returned':len(results),
                'has_more':not selected and offset+len(results)<len(ids),
                'next_offset':offset+len(results) if not selected and offset+len(results)<len(ids) else None}
    finally:
        if db:
            db.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Internal CLI-managed legacy metadata index worker')
    parser.add_argument('--build-index', required=True)
    index_worker(parser.parse_args().build_index)
