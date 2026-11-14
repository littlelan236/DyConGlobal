"""Opt-in bounded preparation/I/O comparison; never a production training launcher."""
import copy
import os
from pathlib import Path
import time

import pytest

from core.io import hash_json, read_json, sha256, write_json
from RL.config import load
from RL.grpo.data import prepare
from RL.grpo.prepared import materialize, PreparedReader

pytestmark=pytest.mark.skipif(os.environ.get("DYCON_GRPO_PREPARED_BENCHMARK")!="1",
                            reason="explicit bounded prepared I/O benchmark")


def test_bounded_long_prompt_preparation_io(tmp_path,monkeypatch):
    from RL.grpo.data import source_groups
    root=Path(__file__).parents[1]
    cfg,_=load(root/"RL/experiments/grpo-offline-k3-smoke.yaml",root)
    fixture=list(source_groups(cfg)[0])
    prompt=[2,3,4,5]*4096
    for group in fixture:
        for row in group["members"]:
            row["prompt_token_ids"]=prompt
            row["prompt_sha256"]=hash_json(prompt)
            row["completion_sha256"]=hash_json(row["completion_token_ids"])
    original=source_groups
    def synthetic_source(config):
        _,proof=original(config)
        return iter(copy.deepcopy(fixture)),proof
    monkeypatch.setattr("RL.grpo.data.source_groups",synthetic_source)
    monkeypatch.setattr("RL.grpo.prepared.source_groups",synthetic_source)
    cfg.runtime.output_root=str(tmp_path/"artifacts")
    cfg.validation.max_sequence_length=32768
    started=time.perf_counter();raw,proof=prepare(cfg)
    legacy=tmp_path/"frozen-groups.json";write_json(legacy,raw)
    legacy_prepare=time.perf_counter()-started
    started=time.perf_counter();read_json(legacy);legacy_read=time.perf_counter()-started
    started=time.perf_counter();reader=materialize(cfg,root,"benchmark",shard_groups=2)
    new_prepare=time.perf_counter()-started
    started=time.perf_counter();reused=PreparedReader(reader.path);new_verify=time.perf_counter()-started
    assert reused.proof["group_order_sha256"]==proof["group_order_sha256"]
    new_bytes=sum(p.stat().st_size for p in reader.path.rglob("*") if p.is_file())
    result={"scope":"5 groups, 3 branches each, shared 16384-token prompts; warm OS cache",
        "legacy_json_bytes":legacy.stat().st_size,"prepared_bytes":new_bytes,
        "legacy_prepare_and_json_seconds":legacy_prepare,"legacy_json_parse_seconds":legacy_read,
        "prepared_materialize_and_verify_seconds":new_prepare,"prepared_reopen_verify_seconds":new_verify,
        "prepared_resume_source_scan_groups":0,"prompt_copies_per_group_before":3,"prompt_copies_per_group_after":1,
        "extrapolates_to_62gb_production":False}
    output=Path(os.environ.get("DYCON_GRPO_BENCHMARK_OUTPUT",str(tmp_path/"benchmark.json")))
    write_json(output,result)
    assert new_bytes<legacy.stat().st_size
