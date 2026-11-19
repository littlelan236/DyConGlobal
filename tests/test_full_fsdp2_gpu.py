"""Opt-in real NCCL/FSDP2 acceptance through the formal CLI on idle GPUs.

FP32 tolerances fixed before execution: parameters/moments 3e-6 absolute,
3e-4 relative; objective/grad-norm 3e-5 absolute, 3e-4 relative.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from core.io import read_json
from tests.test_full_policy import full_values

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("DYCON_FULL_FSDP2_GPU") != "1", reason="Explicit idle GPU acceptance")


def test_large_shard_update_sampling_has_no_float_index_rounding():
    import sys
    # The short-lived child releases its CUDA context before idle-GPU preflight.
    subprocess.run([sys.executable, "-c", '''
import os, torch
from RL.full_policy import parameter_samples
gpu = int(os.environ["DYCON_FULL_GPUS"].split(",")[-1])
module = torch.nn.Module()
module.weight = torch.nn.Parameter(torch.zeros(2**24 + 4, device=f"cuda:{gpu}"))
with torch.no_grad():
    module.weight[-1] = 3.0
samples = parameter_samples(module)["weight"]
assert samples.numel() == 64 and samples[-1].item() == 3.0
'''], cwd=ROOT, check=True, timeout=45, capture_output=True)


def cli(work, label, args, timeout=240, expected=0):
    log = work / (label + ".log")
    with log.open("w") as stream:
        process = subprocess.run([str(ROOT / ".venvs/runner/bin/dycon"), *args], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    assert (process.returncode != 0 if expected is None else process.returncode == expected), log.read_text()[-16000:]


def checkpoint_tensors(path):
    from RL.full_checkpoint import export_parameter
    inventory = read_json(path / "parameters.json")
    return {row["name"]: torch.cat(list(export_parameter(path, row))).reshape(row["shape"])
            for row in inventory["parameters"]}


@pytest.mark.parametrize("dtype,tied,backend", [("float32", False, "reference_pytorch"),
    ("bfloat16", False, "reference_pytorch"), ("float32", True, "reference_pytorch"),
    ("bfloat16", False, "fla_chunk")])
def test_full_fsdp2_math_resume_export_probe(tmp_path, dtype, tied, backend):
    from tests.lora_helpers import raw_model
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.set_num_threads(2)
    model_path = tmp_path / "tiny-base"
    model = raw_model()
    if tied:
        model.config.tie_word_embeddings = model.config.text_config.tie_word_embeddings = True
        model.tie_weights()
    if dtype == "bfloat16":
        model.to(torch.bfloat16)
    model.save_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0")),
                                       unk_token="0", pad_token="0", eos_token="1")
    tokenizer.save_pretrained(model_path)
    gpus = [int(x) for x in os.environ.get("DYCON_FULL_GPUS", "4,5,6,7").split(",")]
    values = full_values()
    values["run"]["mode"] = "smoke"
    values["model"].update(path=str(model_path), dtype=dtype, gdn_backend=backend)
    values["runtime"] = {"gpus": gpus[:1], "output_root": str(tmp_path), "distributed_timeout_seconds": 90}
    values["preflight"] = {"verify_identity": True, "check_resources": True}
    values["training"].update(learning_rate=1e-3, checkpoint_every_updates=2)
    values["algorithm"]["dataset_root"] = str(tmp_path)
    values["grpo"]["top_k"] = 1
    values["grpo"]["k3_cap"] = 1.0
    configs = {}
    checkpoints = {}
    worlds = [w for w in (1, 2, 4) if w <= len(gpus)]
    full_world = max(worlds)
    for world in worlds:
        current = json.loads(json.dumps(values))
        name = f"tiny-full-w{world}"
        current["run"]["attempt_id"] = name
        current["runtime"]["gpus"] = gpus[:world]
        config = tmp_path / (name + ".yaml")
        config.write_text(yaml.safe_dump(current))
        cli(tmp_path, name, ["rl-train", "--training-config", str(config)])
        configs[world] = current
        checkpoints[world] = tmp_path / "RL/checkpoints" / name / "checkpoints/step-000002"
    reference = checkpoint_tensors(checkpoints[1])
    for world in worlds[1:]:
        actual = checkpoint_tensors(checkpoints[world])
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], atol=3e-6 if dtype == "float32" else 2e-4,
                                       rtol=3e-4 if dtype == "float32" else 2e-2, msg=name)
        a = read_json(checkpoints[world] / "state.json")["state"]
        b = read_json(checkpoints[1] / "state.json")["state"]
        assert a == b
    pause = json.loads(json.dumps(configs[full_world]))
    pause["run"]["attempt_id"] = "tiny-full-pause"
    pause["validation"]["stop_after_update"] = 1
    source = tmp_path / "pause.yaml"
    source.write_text(yaml.safe_dump(pause))
    cli(tmp_path, "pause", ["rl-train", "--training-config", str(source)])
    cp1 = tmp_path / "RL/checkpoints/tiny-full-pause/checkpoints/step-000001"
    resume = json.loads(json.dumps(configs[full_world]))
    resume["run"]["attempt_id"] = "tiny-full-resume"
    resume["run"]["start"] = {"mode": "resume", "checkpoint": str(cp1)}
    source = tmp_path / "resume.yaml"
    source.write_text(yaml.safe_dump(resume))
    cli(tmp_path, "resume", ["rl-train", "--training-config", str(source)])
    cp2 = tmp_path / "RL/checkpoints/tiny-full-resume/checkpoints/step-000002"
    continued = checkpoint_tensors(checkpoints[full_world])
    for name, value in checkpoint_tensors(cp2).items():
        torch.testing.assert_close(value, continued[name], atol=0, rtol=0)
    # Exact resume includes all FP32 moments, scheduler, optimizer groups and RNG.
    def identical(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                identical(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                identical(x, y)
        else:
            assert a == b
    left, right = (read_json(p / "receipt.json")["files"] for p in (cp2, checkpoints[full_world]))
    for name in left:
        if name.startswith("ranks/") and name.endswith(".safetensors"):
            assert left[name]["sha256"] == right[name]["sha256"], name
    for rank in range(full_world):
        identical(*(torch.load(p / f"ranks/{rank}/runtime.pt", weights_only=True)
                    for p in (cp2, checkpoints[full_world])))
    wrong = json.loads(json.dumps(resume))
    wrong["run"]["attempt_id"] = "tiny-full-wrong-cap"
    wrong["grpo"]["k3_cap"] = 0.5
    source = tmp_path / "wrong-cap.yaml"
    source.write_text(yaml.safe_dump(wrong))
    cli(tmp_path, "wrong-cap", ["rl-train", "--training-config", str(source)], expected=2)
    warm = json.loads(json.dumps(pause))
    warm["run"].update(attempt_id="tiny-full-warm", start={"mode": "warm_start", "checkpoint": str(cp1)})
    source = tmp_path / "warm.yaml"
    source.write_text(yaml.safe_dump(warm))
    cli(tmp_path, "warm", ["rl-train", "--training-config", str(source)])
    warm_cp = tmp_path / "RL/checkpoints/tiny-full-warm/checkpoints/step-000001"
    warm_meta = read_json(warm_cp / "state.json")
    assert warm_meta["state"]["global_step"] == 1
    assert warm_meta["state"]["group_cursor"] == 4
    assert warm_meta["contract"]["initial_policy"] != read_json(cp1 / "state.json")["contract"]["initial_policy"]
    assert warm_meta["contract"]["initial_policy"]["warm_start_receipt_sha256"]
    for rank in range(full_world):
        manifest = read_json(warm_cp / f"ranks/{rank}/manifest.json")
        assert {v["step"] for v in manifest["parameters"].values()} == {1.0}
    if full_world > 1 and (dtype == 'float32' or backend == 'fla_chunk') and not tied:
        # A rank-local restore failure after process-group initialization must
        # terminate peers and never publish a new checkpoint/committed cursor.
        import shutil
        from RL.artifacts import commit_receipt
        broken = tmp_path / 'broken-source'
        shutil.copytree(cp1, broken)
        (broken / 'ranks/1/runtime.pt').write_bytes(b'injected invalid rank runtime')
        meta = read_json(broken / 'state.json')
        from core.io import hash_json
        commit_receipt(broken, 'rl-full-checkpoint-v1',
                       contract_sha256=hash_json(meta['contract']), global_step=1)
        failed = json.loads(json.dumps(resume))
        failed['run'].update(attempt_id='tiny-full-rank-failure',
                             start={'mode': 'resume', 'checkpoint': str(broken)})
        source = tmp_path / 'rank-failure.yaml'
        source.write_text(yaml.safe_dump(failed))
        cli(tmp_path, 'rank-failure', ['rl-train', '--training-config', str(source)], timeout=90, expected=None)
        location = tmp_path / 'RL/checkpoints/tiny-full-rank-failure'
        assert read_json(location / 'attempt.json')['status'] == 'failed'
        assert not (location / 'receipt.json').exists()
        assert not list((location / 'checkpoints').glob('*/receipt.json'))
    cli(tmp_path, "artifact-verify", ["artifact", "verify", "tiny-full-resume", "--scope", "rl-full-attempt", "--storage-root", str(tmp_path)])
    output = tmp_path / "export"
    cli(tmp_path, "export", ["rl-export", "--checkpoint", str(cp2), "--base-model", str(model_path), "--output", str(output), "--dtype", "float32"])
    group_id = read_json(cp2 / "state.json")["contract"]["input"]["eligible_group_ids"][0]
    report = tmp_path / "probe.json"
    cli(tmp_path, "probe", ["rl-probe", "--checkpoint", str(cp2), "--group-id", group_id,
        "--gpus", *map(str, gpus[:full_world]), "--output", str(report)])
    from transformers import Qwen3_5ForConditionalGeneration
    from RL.full_policy import FullPolicy
    from RL.grpo.prepared import PreparedReader, resolve
    from RL.full_training import _group
    if backend == "fla_chunk":
        import sys
        code = '''
import sys, torch
from core.io import read_json
from RL.config import Config
from RL.gdn_backend import install_gdn_backend
from RL.probe import _configure_numerics
from transformers import Qwen3_5ForConditionalGeneration
from RL.full_policy import FullPolicy
from RL.grpo.prepared import PreparedReader, resolve
from RL.full_training import _group
checkpoint, exported, report, gpu = sys.argv[1:]
from pathlib import Path
config = Config.model_validate(read_json(Path(checkpoint).parent.parent / "resolved-config.json"))
_configure_numerics(config)
identity = read_json(Path(checkpoint) / "state.json")["contract"]["gdn_execution"]
binding = install_gdn_backend(config.model, expected_identity=identity)
model = Qwen3_5ForConditionalGeneration.from_pretrained(exported, local_files_only=True,
    dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:" + gpu)
model.config.use_cache = model.config.text_config.use_cache = False
metadata = read_json(Path(checkpoint) / "state.json")
reader = PreparedReader(resolve(metadata["contract"]["prepared"]))
for item in read_json(report)["groups"]:
    group = _group(reader, item["group_id"])
    for b, branch in enumerate(item["branches"]):
        batch = group.branch_batch(b, "cuda:" + gpu)
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        valid = (batch["labels"][:, 1:] != -100).nonzero()
        bi, position = valid.unbind(1)
        actual = -torch.nn.functional.cross_entropy(logits[bi, position].float(), batch["labels"][bi, position + 1], reduction="none")
        torch.testing.assert_close(actual.cpu(), torch.tensor(branch["target_logprobs"]), atol=0.05, rtol=0.01)
assert binding.observations()["calls"]["chunk"] > 0
'''
        result = subprocess.run([sys.executable, "-c", code, str(cp2), str(output), str(report), str(gpus[-1])],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
        return
    loaded = Qwen3_5ForConditionalGeneration.from_pretrained(output, local_files_only=True, dtype=getattr(torch, dtype))
    policy = FullPolicy(loaded).eval()
    reader = PreparedReader(resolve(read_json(cp2 / "state.json")["contract"]["prepared"]))
    group = _group(reader, group_id)
    observed = read_json(report)["groups"][0]["branches"]
    for b, row in enumerate(observed):
        with torch.no_grad():
            actual = policy(group.branch_batch(b, "cpu"), 16)
        torch.testing.assert_close(actual, torch.tensor(row["target_logprobs"]),
                                   atol=3e-5 if dtype == "float32" else 5e-2,
                                   rtol=3e-4 if dtype == "float32" else 1e-2)
    # Missing shards cannot be accepted as complete checkpoints.
    from RL.full_artifacts import verify_checkpoint
    shard = next((cp2 / "ranks" / str(full_world - 1)).glob("tensor-*.safetensors"))
    saved = shard.read_bytes()
    shard.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_checkpoint(cp2)
    shard.write_bytes(saved)


@pytest.mark.parametrize("backend", ["reference_pytorch", "fla_chunk"])
def test_native_unequal_parents_real_fsdp_worlds(tmp_path, backend):
    """Real sealed native-input path, unequal branch counts and lengths."""
    from tests.lora_helpers import raw_model
    from tests.test_grpo_prepared import dataset
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from core.io import write_json
    from core.model_identity import describe
    from RL.dataset.artifacts import seal
    gpus = [int(x) for x in os.environ['DYCON_FULL_GPUS'].split(',')]
    if len(gpus) < 2:
        pytest.skip('Unequal native real-NCCL comparison needs two idle GPUs')
    base = tmp_path / 'tiny-base'
    raw_model().save_pretrained(base)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token='0')),
                                       unk_token='0', pad_token='0', eos_token='1')
    tokenizer.save_pretrained(base)
    path = dataset(tmp_path, 'unequal-native', range(5), branch_count=4)
    counts = [2, 3, 4, 2, 3]
    groups = [json.loads(line) for line in (path / 'groups.jsonl').read_text().splitlines()]
    for i, group in enumerate(groups):
        group['expected_branch_count'] = counts[i]
        group['branch_ids'] = group['branch_ids'][:counts[i]]
    branches = [json.loads(line) for line in (path / 'branches.jsonl').read_text().splitlines()]
    allowed = {bid for group in groups for bid in group['branch_ids']}
    branches = [row for row in branches if row['branch_sample_id'] in allowed]
    for name, rows in [('groups.jsonl', groups), ('branches.jsonl', branches)]:
        (path / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    metadata = read_json(path / 'dataset.json')
    metadata['cohort']['model'] = {'fingerprint': describe(base)['fingerprint']}
    write_json(path / 'dataset.json', metadata)
    seal(path, 'rl-dataset', 'unequal-native')
    values = full_values()
    values['run']['mode'] = 'smoke'
    values['model'].update(path=str(base), dtype='bfloat16' if backend == 'fla_chunk' else 'float32', gdn_backend=backend)
    values['runtime'] = {'gpus': gpus[:1], 'output_root': str(tmp_path), 'distributed_timeout_seconds': 90}
    values['algorithm'].update(dataset_root=str(tmp_path / 'sources'), dataset_id='unequal-native')
    values['preflight'] = {'verify_identity': True, 'check_resources': True}
    values['grpo'].update(top_k=1, k3_cap=1.0)
    values['training'].update(learning_rate=1e-3, checkpoint_every_updates=2)
    checkpoints = []
    for world in (1, 2):
        values['runtime']['gpus'] = gpus[:world]
        name = f'full-native-w{world}'
        values['run']['attempt_id'] = name
        source = tmp_path / (name + '.yaml')
        source.write_text(yaml.safe_dump(values))
        cli(tmp_path, name, ['rl-train', '--training-config', str(source)])
        checkpoints.append(tmp_path / 'RL/checkpoints' / name / 'checkpoints/step-000002')
    left, right = map(checkpoint_tensors, checkpoints)
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=2e-4 if backend == 'fla_chunk' else 3e-6,
                                   rtol=2e-2 if backend == 'fla_chunk' else 3e-4, msg=name)
    for cp in checkpoints:
        state = read_json(cp / 'state.json')['state']
        assert state['consumed_groups'] == 5 and state['consumed_branches'] == sum(counts)
    metrics = [[json.loads(line) for line in (cp.parent.parent / 'metrics.jsonl').read_text().splitlines()]
               for cp in checkpoints]
    for a, b in zip(*metrics, strict=True):
        assert a['loss'] == pytest.approx(b['loss'], abs=3e-4 if backend == 'fla_chunk' else 3e-5, rel=3e-3 if backend == 'fla_chunk' else 3e-4)
        assert a['global_grad_norm'] == pytest.approx(b['global_grad_norm'], abs=3e-4 if backend == 'fla_chunk' else 3e-5, rel=3e-3 if backend == 'fla_chunk' else 3e-4)
    if backend == 'fla_chunk':
        from safetensors.torch import load_file
        def moments(path, key):
            inventory = read_json(path / 'parameters.json')
            result = {}
            for row in inventory['parameters']:
                pieces = []
                for rank in range(read_json(path / 'state.json')['contract']['world_size']):
                    folder = path / 'ranks' / str(rank)
                    entry = read_json(folder / 'manifest.json')['parameters'][row['name']]['optimizer'][key]
                    pieces.extend(load_file(str(folder / part['file']))['value'] for part in entry['parts'])
                result[row['name']] = torch.cat(pieces).reshape(row['shape'])
            return result
        for key in ('exp_avg', 'exp_avg_sq'):
            a, b = (moments(cp, key) for cp in checkpoints)
            for name in a:
                assert a[name].dtype == b[name].dtype == torch.float32
                assert torch.isfinite(a[name]).all() and torch.isfinite(b[name]).all()
                torch.testing.assert_close(a[name], b[name], atol=3e-5, rtol=2e-2, msg=key + '/' + name)
