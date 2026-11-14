"""Independent TP=1 replay workers, one coordinator and one final publication.

Workers retain the native producer/scheduler/writer. Complete groups are frozen
onto one GPU, preserving prefix locality; only the coordinator imports committed
child results into the final journal. It never writes a live child journal.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from core.io import hash_json, read_json, sha256, write_json
from core.locking import file_lock
from ..artifacts import directory
from ..config import ReplayConfig
from .artifacts import BINDINGS, ReplayWriter
from .inputs import selected_groups
from .validation import raw_contract, raw_policy


class CommittedReader:
    """Read only journal-committed JSONL bytes; verify the incremental hash."""
    def __init__(self, stage):
        self.stage = Path(stage); self.offset = 0; self.count = 0
        self.digest = hashlib.sha256()

    def read(self):
        journal = read_json(self.stage / 'journal.json')
        size = journal.get('committed_bytes', 0)
        if size < self.offset:
            raise ValueError('Replay committed prefix shrank')
        with (self.stage / 'progress-branches.jsonl').open('rb') as stream:
            stream.seek(self.offset)
            data = stream.read(size - self.offset)
        if len(data) != size - self.offset or (data and not data.endswith(b'\n')):
            raise ValueError('Replay committed index is truncated')
        digest = self.digest.copy(); digest.update(data)
        if digest.hexdigest() != journal.get('index_sha256'):
            raise ValueError('Replay committed index checksum mismatch')
        rows = [json.loads(line) for line in data.splitlines()]
        if self.count + len(rows) != journal.get('branch_count'):
            raise ValueError('Replay committed branch count mismatch')
        self.offset = size; self.count += len(rows); self.digest = digest
        return rows


def source_path(binding):
    target = directory(binding.root, 'rl-replay', binding.id)
    return target if target.exists() else target.with_name('.' + binding.id + '.staging')


def check_source(source, identity, replay):
    """Migration changes execution topology, never data or raw scoring policy."""
    old = read_json(Path(source) / 'job.json')
    origin = old['identity']
    for key in ('schema_version', 'probability_semantics', 'dataset_receipt_sha256',
                'cache_receipt_sha256', 'selection_sha256', 'group_ids',
                'raw_contract_sha256', 'models', 'runtime_pin_sha256',
                'coverage_plan_sha256', 'branch_keys'):
        if origin.get(key) != identity.get(key):
            raise ValueError('Resume source scoring/data identity differs: ' + key)
    # An independent worker executes the same single-GPU scoring function.
    before = dict(origin['runtime_options']); after = replay.runtime.model_dump()
    for options in (before, after):
        for key in ('gpus', 'parallel_workers', 'subprocess_seconds'):
            options.pop(key, None)
    if before != after:
        raise ValueError('Resume source engine settings differ beyond GPU topology')
    if read_json(Path(source) / 'journal.json')['signature'] != origin:
        raise ValueError('Resume source journal signature differs')
    runtime = read_json(Path(source) / 'runtime-identity.json')
    model = next(iter(identity['models'].values()))['model']
    if (runtime.get('logprobs_mode') != 'raw_logprobs'
            or runtime.get('enable_trace_replay') is not True
            or runtime.get('model_inventory_sha256') != model['fingerprint']
            or not runtime.get('replay_implementation')):
        raise ValueError('Resume source has no verified native raw runtime identity')
    return old


def source_is_stopped(source, root, binding):
    state = read_json(source / 'state.json')
    if state.get('state') not in ('interrupted', 'failed', 'complete'):
        raise ValueError('Resume source must be stopped before migration')
    cleanup = read_json(source / 'cleanup.json')
    if cleanup.get('process_group_empty') is not True:
        raise ValueError('Resume source GPU cleanup is not confirmed')
    record = Path(root) / 'state/rl-replay' / (binding.id + '.json')
    if record.exists():
        from .jobs import _start_ticks
        proc = read_json(record)
        if proc.get('pid') and _start_ticks(proc['pid']) == proc.get('start_ticks'):
            raise ValueError('Resume source dispatcher is still alive')


def expected_rows(dataset, replay):
    return {row['branch_sample_id']: row for group in selected_groups(dataset, replay)
            for row in group['members']}


def import_rows(writer, source, rows, expected, *, policy, model, origin):
    count = 0
    for row in rows:
        bid = row['branch_sample_id']
        if bid not in expected:
            raise ValueError('Imported branch is outside the frozen selection')
        count += writer.import_record(source, row, expected[bid],
            policy_sha256=policy, model_sha256=model, origin=origin)
    return count


def seed_source(job, replay, writer, expected):
    """Lock the old publication and bind its exact committed prefix locally."""
    if replay.resume_from is None:
        return
    source = source_path(replay.resume_from)
    target = directory(replay.resume_from.root, 'rl-replay', replay.resume_from.id)
    with file_lock(target.with_name('.' + replay.resume_from.id + '.lock'), nonblocking=True):
        source_is_stopped(source, job['project_root'], replay.resume_from)
        old = check_source(source, job['identity'], replay)
        journal = read_json(source / 'journal.json')
        proof = {'source_id': replay.resume_from.id, 'source_root': replay.resume_from.root,
                 'job_sha256': sha256(source / 'job.json'), 'identity': old['identity'],
                 'journal_sha256': sha256(source / 'journal.json'),
                 'index_sha256': journal['index_sha256'], 'branch_count': journal['branch_count'],
                 'runtime': read_json(source / 'runtime-identity.json'),
                 'execution': read_json(source / 'execution.json')}
        receipt = writer.stage / 'resume-source.json'
        if receipt.exists() and read_json(receipt) != proof:
            raise ValueError('Frozen resume source changed')
        write_json(receipt, proof)
        reader = CommittedReader(source)
        import_rows(writer, source, reader.read(), expected,
                    policy=next(iter(job['plan']['policies'])),
                    model=next(iter(job['plan']['models'])), origin='resume-source')
        writer.checkpoint('seeded')


def partition_groups(dataset, replay, finished):
    """Greedy balance by remaining prompt+completion tokens, whole-group only.

    Assignment lists preserve dataset order so each worker's CacheReader remains
    a streaming ordered reader. Partitioning is frozen before workers start.
    """
    loads = [0] * replay.runtime.parallel_workers
    assignments = [[] for _ in loads]
    for group in selected_groups(dataset, replay):
        remaining = [r for r in group['members'] if r['branch_sample_id'] not in finished]
        if not remaining:
            continue
        worker = min(range(len(loads)), key=lambda i: (loads[i], i))
        assignments[worker].append(group['group']['group_id'])
        loads[worker] += sum(r['prompt_count'] + r['completion_count'] for r in remaining)
    return {'group_ids': assignments, 'estimated_tokens': loads}


def run(job_path):
    def interrupted(signum, frame):
        raise InterruptedError(f'Parallel raw replay interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    job = read_json(job_path); replay = ReplayConfig.model_validate(job['replay_config'])
    stage = Path(job['stage']); dataset = Path(job['dataset'])
    if sha256(dataset / 'receipt.json') != job['identity']['dataset_receipt_sha256']:
        raise ValueError('Parallel replay dataset receipt changed')
    writer = ReplayWriter(stage, dataset, job['identity'], replay.storage.max_shard_bytes,
        recover=job['recover'], commit_every_branches=replay.storage.commit_every_branches,
        commit_interval_seconds=replay.storage.commit_interval_seconds)
    expected = expected_rows(dataset, replay)
    seed_source(job, replay, writer, expected)
    for bid in writer.finished:
        if bid not in expected:
            raise ValueError('Recovered branch is outside the frozen selection')
        writer.verify_completed(expected[bid], policy_sha256=next(iter(job['plan']['policies'])),
                                model_sha256=next(iter(job['plan']['models'])))
    assignment_file = stage / 'worker-assignment.json'
    if assignment_file.exists():
        assignment = read_json(assignment_file)
    else:
        assignment = partition_groups(dataset, replay, writer.finished)
        assignment['initial_completed_ids'] = sorted(writer.finished)
        write_json(assignment_file, assignment)
    children = []; streams = []; readers = []; child_dirs = []
    policy = next(iter(job['plan']['policies'])); model = next(iter(job['plan']['models']))
    try:
        for rank, groups in enumerate(assignment['group_ids']):
            if not groups:
                continue
            child_stage = stage / 'workers' / f'gpu-{replay.runtime.gpus[rank]}'
            child_stage.mkdir(parents=True, exist_ok=True)
            config = replay.model_dump()
            config['runtime'] = {**config['runtime'], 'gpus': [replay.runtime.gpus[rank]], 'parallel_workers': 1}
            child = {**job, 'stage': str(child_stage), 'replay_config': config,
                     'recover': (child_stage / 'progress-branches.jsonl').exists(),
                     'partition_worker': True, 'assigned_group_ids': groups,
                     'completed_branch_ids': assignment['initial_completed_ids']}
            write_json(child_stage / 'job.json', child)
            stream = (child_stage / 'worker.log').open('ab'); streams.append(stream)
            env = os.environ.copy(); env['PYTHONPATH'] = job['project_root']
            # Inherit the coordinator's process group: formal dispatch() owns
            # and confirms cleanup of the coordinator and all GPU descendants.
            proc = subprocess.Popen([replay.runtime.python, '-m', 'RL.dataset.replay.worker', str(child_stage / 'job.json')],
                cwd=job['project_root'], env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
            children.append(proc); readers.append(CommittedReader(child_stage)); child_dirs.append(child_stage)
        write_json(stage / 'parallel-workers.json', {'gpus': replay.runtime.gpus,
                   'tensor_parallel_size': 1, 'pids': [p.pid for p in children],
                   'worker_stages': [str(p.relative_to(stage)) for p in child_dirs]})
        while children:
            for proc, reader, child_stage in zip(children, readers, child_dirs):
                if (child_stage / 'progress-branches.jsonl').exists() and (child_stage / 'journal.json').exists():
                    journal = read_json(child_stage / 'journal.json')
                    if 'index_sha256' not in journal:
                        continue
                    import_rows(writer, child_stage, reader.read(), expected, policy=policy, model=model,
                                origin=child_stage.relative_to(stage).as_posix())
                code = proc.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f'Independent replay worker exited {code}; see {child_stage / "worker.log"}')
            writer.checkpoint('replaying')
            if all(p.poll() is not None for p in children):
                break
            time.sleep(2)
        executions = {p.name: read_json(p / 'execution.json') for p in child_dirs}
        if executions:
            runtime = dict(next(iter(executions.values())))
        elif replay.resume_from:
            runtime = dict(read_json(stage / 'resume-source.json')['runtime'])
        else:
            raise ValueError('Parallel replay produced no execution identity')
        runtime.update(parallel_workers=executions, execution_topology={
            'gpus': replay.runtime.gpus, 'tensor_parallel_size': 1,
            'parallel_workers': replay.runtime.parallel_workers},
            imported_execution=read_json(stage / 'resume-source.json') if (stage / 'resume-source.json').exists() else None)
        writer.finish(job['plan']['sampling'], job['plan']['policies'], job['plan']['models'], runtime)
        write_json(stage / 'execution.json', runtime)
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        for proc in children:
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=10)
        for stream in streams:
            stream.close()
        if read_json(stage / 'journal.json').get('state') != 'complete':
            writer.checkpoint('interrupted')
