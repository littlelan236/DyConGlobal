"""Public CLI contracts; terminal formatting never enters artifact directories."""
import csv
import io
import json
import time


def register(subparsers):
    memory = subparsers.add_parser('lora-memory', help='Summarize per-rank memory events by length, phase, mode and outcome')
    memory.add_argument('attempt_id')
    memory.add_argument('--storage-root')
    memory.add_argument('--rank', type=int)
    memory.add_argument('--phase')
    memory.add_argument('--limit', type=int, default=100)
    status = subparsers.add_parser('lora-status', help='Read committed updates, liveness, stages and recoverable checkpoint')
    status.add_argument('attempt_id', nargs='?')
    status.add_argument('--active', action='store_true')
    status.add_argument('--storage-root')
    status.add_argument('--watch', type=float)
    status.add_argument('--format', choices=('table','json'), default='table')
    status.add_argument('--recent-updates', type=int, default=10)
    tb = subparsers.add_parser('lora-tensorboard', help='Manage an owned TensorBoard + metrics synchronizer')
    tb.add_argument('attempt_id')
    tb.add_argument('--action', choices=('start','status','stop','rebuild','renew'), default='status')
    tb.add_argument('--storage-root')
    tb.add_argument('--host', default='127.0.0.1')
    tb.add_argument('--port', type=int, default=16006)
    tb.add_argument('--lease-seconds', type=int, default=1800,
                    help='Query lease for a non-running attempt (default: 1800)')
    advantage = subparsers.add_parser('lora-advantages', help='Query frozen advantages; explicitly build legacy metadata index')
    advantage.add_argument('attempt_id')
    advantage.add_argument('--storage-root')
    group = advantage.add_mutually_exclusive_group()
    group.add_argument('--group-id')
    group.add_argument('--group-ordinal', type=int)
    advantage.add_argument('--branch-id')
    advantage.add_argument('--offset', type=int, default=0)
    advantage.add_argument('--limit', type=int)
    advantage.add_argument('--token-preview', type=int, default=0)
    advantage.add_argument('--format', choices=('table','json','csv'), default='table')
    index = advantage.add_mutually_exclusive_group()
    index.add_argument('--build-index', action='store_true')
    index.add_argument('--index-status', action='store_true')


def render_status(result):
    rows = result.get('attempts', [result])
    if not rows:
        return 'No active RL attempts'
    lines = []
    for row in rows:
        latest = row.get('latest_metric') or {}
        lines.extend([f"{row['attempt_id']}  status={row['status']}  {row['observed_liveness']}",
            f"update {row['last_completed_update']}/{row['max_updates']}  inflight={row['inflight_update']}  recoverable={row['recoverable_checkpoint_step']}",
            f"committed groups {row['committed_consumed_groups']}/{row['planned_groups']}  branches={row['consumed_branches']}  tokens={row['consumed_tokens']}",
            f"stage={row['stage']}  age_seconds={row['stage_age_seconds']}  pid={(row.get('process') or {}).get('pid')}",
            f"loss={latest.get('loss')}  policy_loss={latest.get('policy_loss')}  K3(unweighted)={latest.get('k3_penalty_mean')}  clip_fraction={latest.get('clip_fraction')}",
            f"ETA seconds={row['eta']['remaining_seconds']}  confidence={row['eta']['confidence']}  samples={row['eta']['samples']}"])
        lines.extend('warning: '+w for w in row['warnings'])
    return '\n'.join(lines)


def render_advantages(result, fmt):
    if fmt == 'csv':
        output = io.StringIO()
        fields = ('group_id','group_ordinal','native_ordinal','branch_id','score','retained','advantage','rank','sign','provenance','consumption','total','returned','truncated','next_offset')
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for group in result['groups']:
            for branch in group.get('branches', []):
                writer.writerow({**branch,'group_id':group['group_id'],'group_ordinal':group.get('group_ordinal'),
                                 **{key:group.get(key) for key in ('total','returned','truncated','next_offset')}})
        return output.getvalue().rstrip('\r\n')
    lines = []
    for group in result['groups']:
        lines.append(f"group {group['group_id']} ordinal={group.get('group_ordinal')} {group.get('consumption')} provenance={group.get('provenance')}")
        if group.get('exclusion_reason'):
            lines.append('excluded: '+group['exclusion_reason'])
        if 'branches' in group:
            lines.append('native  branch_id  score  retained  advantage  rank  sign')
            for branch in group['branches']:
                lines.append('  '.join(str(branch[k]) for k in ('native_ordinal','branch_id','score','retained','advantage','rank','sign')))
            lines.append(f"returned={group.get('returned',0)} total={group.get('total',0)} truncated={group.get('truncated',False)} next_offset={group.get('next_offset')}")
        else:
            lines.append(f"branches={group['total_branches']} summary={json.dumps(group['summary'])}")
    lines.append(f"has_more={result.get('has_more',False)} next_offset={result.get('next_offset')}")
    return '\n'.join(lines)


def execute(args, cfg):
    if args.command == 'lora-memory':
        from RL.memory_summary import summarize
        return summarize(cfg, args.attempt_id, storage_root=args.storage_root,
                         rank=args.rank, phase=args.phase, limit=args.limit)
    if args.command == 'lora-status':
        from RL.observability import MetricReader, active, snapshot, source
        if bool(args.attempt_id) == args.active:
            raise ValueError('Specify one attempt ID or --active')
        if not 0 <= args.recent_updates <= 256 or args.watch is not None and args.watch < .5:
            raise ValueError('recent-updates must be 0..256; watch must be >= 0.5 seconds')
        reader = None
        if args.attempt_id and args.watch:
            directory, _, _, _ = source(cfg, args.attempt_id, args.storage_root)
            reader = MetricReader(directory/'metrics.jsonl')
            reader.seek_tail(max(10,args.recent_updates))
        try:
            while True:
                result = active(cfg,args.storage_root,args.recent_updates) if args.active else snapshot(
                    cfg,args.attempt_id,args.storage_root,args.recent_updates,reader)
                if args.format == 'json' and not args.watch:
                    return result
                print(json.dumps(result,ensure_ascii=False,allow_nan=False) if args.format == 'json' else render_status(result),flush=True)
                if not args.watch:
                    return None
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return None
    if args.command == 'lora-tensorboard':
        from RL.observability_tensorboard import manage
        return manage(cfg,args.attempt_id,action=args.action,storage_root=args.storage_root,
                      port=args.port,host=args.host,lease_seconds=args.lease_seconds)
    from RL.grpo.observability import build_index, index_status, query
    if args.build_index:
        return build_index(cfg,args.attempt_id,args.storage_root)
    if args.index_status:
        from RL.observability import source
        directory,attempt,_,cache=source(cfg,args.attempt_id,args.storage_root)
        return {'state':'ready','source':'frozen_prepared_artifact'} if 'prepared' in attempt.get('contract',{}) else index_status(cache,directory)
    if args.format == 'csv' and not (args.group_id or args.group_ordinal is not None or args.branch_id):
        raise ValueError('CSV requires an explicit group or branch filter')
    result=query(cfg,args.attempt_id,storage_root=args.storage_root,group_id=args.group_id,
        group_ordinal=args.group_ordinal,branch_id=args.branch_id,offset=args.offset,limit=args.limit,token_preview=args.token_preview)
    if args.format=='json':
        return result
    print(render_advantages(result,args.format))
    return None
