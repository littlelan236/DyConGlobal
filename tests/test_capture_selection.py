from migration.select_captures import choose


def row(rid, complete=True, submitted=True):
    return {'instance_id': 'task', 'rollout_id': rid, 'capture_integrity_complete': True,
            'rollout_complete': complete, 'submitted': submitted}


def test_complete_candidate_preferred_over_truncated_or_unsubmitted():
    selected = choose([row('a', False, False), row('b', False), row('c')], 42)
    assert selected[0]['rollout_id'] == 'c'
    assert selected[0]['selection_pool'] == ['c']


def test_random_tie_reproducible_without_shortest_or_lexical_preference():
    candidates = [row('a'), row('b')]
    assert choose(candidates, 42) == choose(list(reversed(candidates)), 42)
    assert {choose(candidates, seed)[0]['rollout_id'] for seed in range(20)} == {'a', 'b'}


def test_only_partial_candidate_retained_with_status_unchanged():
    selected = choose([row('a', False, False)], 42)
    assert selected[0]['rollout_complete'] is False
    assert selected[0]['submitted'] is False
