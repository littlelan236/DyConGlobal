"""Training coordinates for explicit probe groups, from immutable logical windows."""
from core.io import hash_json


def positions(contract, state, group_ids, *, segments=None, execution_plan=None):
    """One-based optimizer steps; physical GPU topology never defines coordinates."""
    eligible = contract['input']['eligible_group_ids']
    if len(set(eligible)) != len(eligible):
        raise ValueError('Duplicate frozen eligible group IDs')
    if not group_ids or len(set(group_ids)) != len(group_ids) or not set(group_ids).issubset(eligible):
        raise ValueError('Training coordinates require unique eligible probe groups')
    step = state['global_step']
    if type(step) is not int or step < 0:
        raise ValueError('Invalid checkpoint step for probe coordinates')
    steps = {gid: [] for gid in eligible}
    if execution_plan:
        from RL.grpo.execution import validate, validate_state
        validate(execution_plan)
        validate_state(execution_plan, state)
        windows = execution_plan['windows']
        for index, window in enumerate(windows, 1):
            for chunk in execution_plan['chunks'][window['start']:window['end']]:
                steps[chunk['parent_group_id']].append(index)
        schedule_hash = execution_plan['sha256']
    else:
        if contract['input'].get('max_epochs') != 1:
            raise ValueError('Probe training coordinates require the frozen one-epoch schedule')
        segments = segments or [{'group_ids': eligible,
                                  'group_batch_size': contract['input']['group_batch_size']}]
        if [gid for segment in segments for gid in segment['group_ids']] != eligible:
            raise ValueError('Training coordinate order differs from frozen eligible IDs')
        index = 0
        consumed = 0
        ends = [0]
        for segment in segments:
            batch = segment['group_batch_size']
            if type(batch) is not int or batch < 1:
                raise ValueError('Invalid frozen group batch size')
            ids = segment['group_ids']
            for start in range(0, len(ids), batch):
                index += 1
                window = ids[start:start + batch]
                for gid in window:
                    steps[gid].append(index)
                consumed += len(window)
                ends.append(consumed)
        if step >= len(ends) or state.get('group_cursor') != ends[step]:
            raise ValueError('Checkpoint step/cursor differs from frozen training windows')
        schedule_hash = hash_json(segments)
    ordinals = {gid: i for i, gid in enumerate(eligible)}
    groups = []
    for gid in group_ids:
        scheduled = steps[gid]
        if not scheduled:
            raise ValueError('Probe group is absent from frozen execution windows')
        committed = [s for s in scheduled if s <= step]
        status = ('trained' if len(committed) == len(scheduled)
                  else 'partially_trained' if committed else 'not_yet_trained')
        groups.append({'group_id': gid, 'group_ordinal': ordinals[gid],
                       'scheduled_steps': scheduled, 'scheduled_first_step': scheduled[0],
                       'scheduled_last_step': scheduled[-1],
                       'first_training_step': committed[0] if committed else None,
                       'last_training_step': committed[-1] if committed else None,
                       'status_at_checkpoint': status})
    return {'version': 1, 'checkpoint_step': step, 'step_numbering': 'one_based_optimizer_update',
            'x_coordinate': 'scheduled_first_step', 'schedule_sha256': schedule_hash,
            'groups': groups}


def checkpoint_positions(checkpoint, contract, state, group_ids):
    """Replot historical reports without loading a model, native tokens or probabilities."""
    from RL.grpo.stream_resume import checkpoint_kind
    kind = checkpoint_kind(contract)
    if kind == 'prepared':
        from RL.grpo.prepared import PreparedReader, resolve
        reader = PreparedReader(resolve(contract['prepared']))
        reader.validate_state(state)
        return positions(contract, state, group_ids, segments=reader.schedule['segments'],
                         execution_plan=reader.execution_plan)
    if kind == 'streaming':
        from RL.grpo.stream_resume import resolve_checkpoint
        plan = resolve_checkpoint(checkpoint, contract)['plan']
        execution = plan['proof'].get('execution_plan')
    else:
        execution = contract['input'].get('execution_plan')
    return positions(contract, state, group_ids, execution_plan=execution)
