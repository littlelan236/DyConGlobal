"""Full-policy configuration contracts; no model/GPU needed."""
import copy

import pytest

from RL.config import Config, schedule_contract
from tests.test_full_policy import full_values


def managed_values(mode='checkpoint_boundary'):
    values = full_values()
    values['training']['memory'].update(activation_offload=mode,
        host_offload={'enabled': True, 'allocator_budget_gib': .125,
                      'cache_limit_gib': 0., 'reserve_gib': 8., 'reserve_fraction': .03})
    return values


@pytest.mark.parametrize('backend', ['reference_pytorch', 'fla_chunk'])
@pytest.mark.parametrize('mode', ['none', 'checkpoint_boundary', 'auto', 'memory_pressure'])
def test_full_managed_policy_is_frozen(backend, mode):
    values = managed_values(mode)
    values['model'].update(dtype='bfloat16', gdn_backend=backend)
    config = Config.model_validate(values)
    before = schedule_contract(config)
    assert 'scoped_host_reclaim_v2' in str(before)
    changed = copy.deepcopy(values)
    changed['training']['memory']['host_offload']['allocator_budget_gib'] = .25
    assert before != schedule_contract(Config.model_validate(changed))
    changed = copy.deepcopy(values)
    changed['training']['memory']['activation_offload'] = 'auto' if mode != 'auto' else 'none'
    assert before != schedule_contract(Config.model_validate(changed))


def test_long_full_requires_flash_and_managed_host():
    values = full_values()
    values['model']['dtype'] = 'bfloat16'
    values['validation']['max_sequence_length'] = 32768
    values['training']['memory'].update(loss_chunk_tokens=16, exact_length=True, attention_policy='flash_only')
    with pytest.raises(ValueError, match='managed host'):
        Config.model_validate(values)
    values['training']['memory']['host_offload'] = {'enabled': True}
    Config.model_validate(values)
    values['training']['memory']['attention_policy'] = 'auto'
    with pytest.raises(ValueError, match='Long sequences'):
        Config.model_validate(values)


def test_no_offload_contract_retains_legacy_identity():
    from RL.config import memory_contract
    config = Config.model_validate(full_values())
    assert 'host_offload' not in memory_contract(config.training.memory)
    assert 'offload_policy_version' not in str(schedule_contract(config))


@pytest.mark.parametrize('option', ['oom_retry', 'mlp_chunk_tokens'])
def test_full_managed_still_rejects_retry_and_custom_mlp(option):
    values = managed_values()
    values['training']['memory'][option] = {'enabled': True} if option == 'oom_retry' else 16
    with pytest.raises(ValueError, match='no retry/MLP'):
        Config.model_validate(values)
