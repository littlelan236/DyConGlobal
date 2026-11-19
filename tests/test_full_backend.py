"""Torch-free full GDN metadata compatibility checks."""
import copy
import pytest
from RL.config import Config
from RL.full_backend import checkpoint_identity
from tests.test_full_policy import full_values


def test_full_bf16_fla_config_and_reference_resume_remain_explicit():
    values = full_values()
    values['model'].update(gdn_backend='fla_chunk', dtype='bfloat16')
    assert Config.model_validate(values).model.gdn_backend == 'fla_chunk'
    assert checkpoint_identity({'schedule': {'model': {}}}) is None


def test_fla_execution_and_anchor_identity_are_required():
    identity = {'configured_backend': 'fla_chunk', 'resolved_backend': 'fla_chunk'}
    contract = {'schedule': {'model': {'gdn_backend': 'fla_chunk'}},
                'gdn_execution': identity, 'initial_policy': {'gdn_execution': identity}}
    assert checkpoint_identity(contract) == identity
    for key in ('gdn_execution', 'initial_policy'):
        bad = copy.deepcopy(contract)
        del bad[key]
        with pytest.raises(ValueError):
            checkpoint_identity(bad)
    bad = copy.deepcopy(contract)
    bad['gdn_execution']['resolved_backend'] = 'reference_pytorch'
    with pytest.raises(ValueError):
        checkpoint_identity(bad)


def test_explicit_complete_parent_selection_and_unknown_parent():
    from RL.grpo.data import source_groups
    from RL.config import schedule_contract
    values = full_values()
    config = Config.model_validate(values)
    assert 'group_ids' not in schedule_contract(config)['algorithm']
    config.algorithm.group_ids = ['synthetic-g1', 'synthetic-g3']
    rows, proof = source_groups(config)
    rows = list(rows)
    assert {r['group']['group_id'] for r in rows} == set(config.algorithm.group_ids)
    assert all(len(r['members']) == r['group']['expected_branch_count'] for r in rows)
    assert proof['explicit_parent_selection']['group_ids'] == sorted(config.algorithm.group_ids)
    config.algorithm.group_ids = ['missing-parent']
    with pytest.raises(ValueError, match='missing from dataset'):
        list(source_groups(config)[0])
    values['algorithm']['group_ids'] = ['a', 'a']
    with pytest.raises(ValueError, match='Duplicate'):
        Config.model_validate(values)


def test_prepared_selection_freezes_all_branches_and_rejects_changed_ids(tmp_path):
    from pathlib import Path
    from RL.grpo.prepared import materialize, validate_config
    values = full_values()
    values['runtime']['output_root'] = str(tmp_path)
    values['algorithm']['group_ids'] = ['synthetic-g1', 'synthetic-g3']
    values['grpo']['group_batch_size'] = 1
    values['training']['max_updates'] = 2
    config = Config.model_validate(values)
    reader = materialize(config, Path(__file__).parents[1], 'full-explicit-selection')
    assert set(reader.ids) == set(config.algorithm.group_ids)
    assert all(len(reader.raw(g)['members']) == 3 for g in reader.ids)
    validate_config(reader, config)
    config.algorithm.group_ids = ['synthetic-g0', 'synthetic-g3']
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_config(reader, config)
