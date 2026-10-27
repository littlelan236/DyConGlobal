"""Generated token periods, bounded state, and compatibility of the new config."""

import random
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core.config import ParallelReasoningLoop, load_snapshot, resolve, snapshot
from runtime.serving.reasoning_loop import ReasoningLoopDetector


def options(**kwargs):
    return ParallelReasoningLoop(enabled=True, **kwargs).model_dump()


@pytest.mark.parametrize("period", [1, 2, 7, 99, 132, 412, 565, 744, 2048])
@pytest.mark.parametrize("chunk", [1, 63, 1024, 50000])
def test_exact_periods_detect_across_chunks(period, chunk):
    opts = options()
    ids = list(range(9000, 9311)) + list(range(period)) * max(40, 2200 // period)
    detector = ReasoningLoopDetector(opts, [248069])
    for start in range(0, len(ids), chunk):
        hit = detector.feed(ids[start:start + chunk])
        if hit:
            break
    assert hit is not None
    assert hit["period_tokens"] == period
    assert hit["repeats"] >= 8
    assert hit["repeated_tokens"] >= 1024
    assert hit["detected_at_output_tokens"] >= 2048
    start, end = hit["start_output_token"], hit["end_output_token_exclusive"]
    assert ids[start:end] == ids[end - period:end] * hit["repeats"]
    assert len(detector.tokens) == 16384


def test_nonperiodic_long_reasoning_retains_bounded_history():
    detector = ReasoningLoopDetector(options(), [248069])
    assert detector.feed(list(range(81920))) is None
    assert len(detector.tokens) == 16384
    assert len(detector.prefix) == 16385


def test_close_stops_detection_and_preserves_reasoning_only():
    detector = ReasoningLoopDetector(options(), [248069])
    assert detector.feed(list(range(100)) + [248069] + [1] * 10000) is None
    assert detector.closed
    assert detector.feed([1] * 10000) is None
    assert detector.count == 100


def test_threshold_and_broken_repetitions():
    opts = options(min_generated_tokens=64, min_repeated_tokens=64, max_period_tokens=8,
                   check_interval_tokens=1)
    d = ReasoningLoopDetector(opts, [248069])
    assert d.feed([1, 2] * 31 + [1]) is None
    assert d.feed([2])["detected_at_output_tokens"] == 64
    d = ReasoningLoopDetector(opts, [248069])
    assert d.feed(([1] * 31 + [2]) * 20) is None


def test_hash_collisions_require_exact_native_token_equality(monkeypatch):
    d = ReasoningLoopDetector(options(), [248069])
    monkeypatch.setattr(d, "_span_hash", lambda *args: 0)
    rng = random.Random(4)
    assert d.feed([rng.randrange(20) for _ in range(4096)]) is None


@pytest.mark.parametrize("field,value", [("repeat_count", 1), ("repeat_count", True),
    ("max_period_tokens", 0), ("check_interval_tokens", "64"), ("min_generated_tokens", 0)])
def test_strict_loop_config(field, value):
    with pytest.raises(ValidationError):
        ParallelReasoningLoop(**{field: value})


def test_legacy_config_and_snapshot_do_not_enable_guard(cfg):
    path = Path(cfg["config_source"])
    raw = yaml.safe_load(path.read_text())
    raw["parallel"] = {"enabled": False}
    path.write_text(yaml.safe_dump(raw))
    legacy = resolve(path)
    assert "reasoning_loop" not in legacy["parallel"]
    run = snapshot(legacy, "legacy-reasoning-loop", {"task_keys": []})
    before = (run / "resolved-config.json").read_bytes()
    raw["parallel"]["reasoning_loop"] = options()
    path.write_text(yaml.safe_dump(raw))
    assert resolve(path)["parallel"]["reasoning_loop"]["enabled"]
    restored = load_snapshot(cfg["project_root"], "legacy-reasoning-loop")
    assert "reasoning_loop" not in restored["parallel"]
    assert (run / "resolved-config.json").read_bytes() == before


def test_post_close_loop_preserves_absolute_output_coordinates():
    opts = options(include_post_close=True, min_generated_tokens=64,
                   min_repeated_tokens=32, max_period_tokens=8, check_interval_tokens=8)
    prefix = list(range(100, 132)) + [248069]
    ids = prefix + [4, 5] * 128
    detector = ReasoningLoopDetector(opts, [248069])
    hit = detector.feed(ids)
    assert hit["phase"] == "post_close"
    assert hit["scope"] == "generated_output"
    assert hit["rule"] == "exact_token_period_v2"
    assert hit["start_output_token"] >= len(prefix)
    a, b, period = hit["start_output_token"], hit["end_output_token_exclusive"], hit["period_tokens"]
    assert ids[a:b] == ids[b-period:b] * hit["repeats"]


def test_expanded_period_detects_2566_token_cycle_after_close():
    ids = [41, 248069] + list(range(10000, 12566)) * 9
    opts = options(include_post_close=True, max_period_tokens=8192)
    hit = ReasoningLoopDetector(opts, [248069]).feed(ids)
    assert hit["period_tokens"] == 2566 and hit["phase"] == "post_close"
    assert ReasoningLoopDetector({**opts, "max_period_tokens": 2048}, [248069]).feed(ids) is None


def test_legacy_reasoning_loop_option_preserves_resolved_hash(cfg):
    path = Path(cfg["config_source"])
    raw = yaml.safe_load(path.read_text())
    raw["parallel"] = {"enabled": False, "reasoning_loop": options()}
    raw["parallel"]["reasoning_loop"].pop("include_post_close", None)
    path.write_text(yaml.safe_dump(raw))
    legacy = resolve(path)
    assert "include_post_close" not in legacy["parallel"]["reasoning_loop"]
