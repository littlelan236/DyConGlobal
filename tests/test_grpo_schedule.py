"""Versioned deterministic ordering of complete native GRPO groups."""
import random
import pytest

from RL.grpo.schedule import LENGTH_SHUFFLE, LEGACY_SHUFFLE, order_groups


def test_length_shuffle_preserves_all_complete_groups():
    items=[{"id":f"g{i}","length":(i*37)%211} for i in range(388)]
    result=order_groups(items,seed=42,batch_size=4,algorithm=LENGTH_SHUFFLE,
                       prompt_length=lambda item:item["length"])
    assert len(result)==len(items)
    assert len({item["id"] for item in result})==len(items)
    assert sorted(item["id"] for item in result)==sorted(item["id"] for item in items)


def test_legacy_shuffle_is_the_original_seeded_order():
    items=list(range(137))
    expected=list(items)
    random.Random(72).shuffle(expected)
    assert order_groups(items,seed=72,batch_size=4,algorithm=LEGACY_SHUFFLE)==expected


@pytest.mark.parametrize("count,batch",[(0,4),(1,4),(129,4),(259,3),(300,7),(257,128)])
def test_pool_boundaries_keep_full_windows_and_one_tail(count,batch):
    items=list(range(count))
    result=order_groups(items,seed=7,batch_size=batch,prompt_length=lambda n:n)
    assert sorted(result)==items
    assert len(result)==count
    assert len(result)%batch==count%batch
    if count%batch:
        # The final partial window is locally sorted and never shuffled into full windows.
        assert result[-(count%batch):]==sorted(result[-(count%batch):])


def test_new_shuffle_keeps_random_tie_order_and_isolates_rng():
    state=random.getstate()
    result=order_groups(list(range(256)),seed=7,batch_size=4,prompt_length=lambda _:10)
    assert random.getstate()==state
    assert result==order_groups(list(range(256)),seed=7,batch_size=4,prompt_length=lambda _:10)
    assert result!=order_groups(list(range(256)),seed=8,batch_size=4,prompt_length=lambda _:10)
    assert result!=list(range(256))


@pytest.mark.parametrize("algorithm,batch",[("unknown",4),(LENGTH_SHUFFLE,129),(LENGTH_SHUFFLE,0)])
def test_invalid_ordering_rejected(algorithm,batch):
    with pytest.raises(ValueError):
        order_groups([1,2,3],seed=0,batch_size=batch,algorithm=algorithm,prompt_length=lambda n:n)
