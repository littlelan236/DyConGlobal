"""Fixed full-policy probes using the common GPU locks and worker supervisor."""
from contextlib import ExitStack
from pathlib import Path
import tempfile

from core.io import read_json
from core.locking import file_lock
from RL.full_artifacts import verify_checkpoint


def run(root_config, args):
    root = Path(root_config["project_root"])
    checkpoint = (root / args.checkpoint).resolve()
    metadata = verify_checkpoint(checkpoint, deep=not args.dry_run)
    from RL.config import Config
    config = Config.model_validate(read_json(checkpoint.parent.parent / "resolved-config.json"))
    if len(args.gpus) != metadata["contract"]["world_size"] or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Full probe requires the checkpoint world size and distinct GPUs")
    from RL.grpo.prepared import PreparedReader, resolve
    reader = PreparedReader(resolve(metadata["contract"]["prepared"]))
    if len(set(args.group_id)) != len(args.group_id) or not set(args.group_id).issubset(reader.ids):
        raise ValueError("Probe group IDs must be unique frozen complete parents")
    from RL.probe import _output_path, _publish
    output = _output_path(root, args.output, checkpoint, config)
    if args.dry_run:
        return {"status": "validated", "group_ids": args.group_id, "output": str(output),
                "initial_policy": metadata["contract"]["initial_policy"]}
    config.runtime.gpu, config.runtime.gpus = None, args.gpus
    config.preflight.check_resources = True
    config.run.start.mode, config.run.start.checkpoint = "resume", str(checkpoint)
    parent = {"path": str(checkpoint), "metadata": metadata}
    from RL.parallel import launch
    from RL.service import _train
    with ExitStack() as locks, tempfile.TemporaryDirectory(prefix="full-probe-", dir=root / "temp") as work:
        for gpu in sorted(args.gpus):
            locks.enter_context(file_lock(root / "state/artifact-locks/lora-gpu" / f"{gpu}.lock", nonblocking=True))
        attempt = {"operation": "full_probe", "group_ids": args.group_id}
        report = (launch(config, root, Path(work), {}, parent, attempt) if len(args.gpus) > 1 else
                  _train(config, root, Path(work), {}, parent, attempt))
    _publish(output, report)
    return {"output": str(output), "global_step": report["global_step"], "group_count": len(report["groups"])}


def evaluate(config, parent, group_ids, context):
    import torch
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import DeviceMesh
    from RL.full_policy import load
    from RL.full_checkpoint import restore
    from RL.full_training import _group
    from RL.grpo.prepared import PreparedReader, resolve
    from RL.probe import _configure_numerics
    from RL.probe_metrics import group_metrics, TokenChanges
    from RL.memory import attention_context
    _configure_numerics(config)
    metadata = parent["metadata"]
    from RL.full_backend import bind, checkpoint_identity, evidence
    backend_binding = bind(config.model, context, checkpoint_identity(metadata["contract"]))
    mesh = DeviceMesh.from_group(context.data_group, "cuda", mesh_dim_names=("data",))
    policy, _ = load(config, mesh, context.rank, context.world_size)
    restore(parent["path"], policy, None, None, context, training=False)
    policy.eval()
    reader = PreparedReader(resolve(metadata["contract"]["prepared"]))
    anchor = read_json(Path(parent["path"]) / "anchor/manifest.json")
    lookup = {(e["group_id"], e["branch_id"]): e for e in anchor["entries"]}
    reports, changes = [], TokenChanges()
    with torch.no_grad(), attention_context(config.training.memory.attention_policy):
        for gid in group_ids:
            group = _group(reader, gid)
            branches = []
            for b, bid in enumerate(group.branch_ids):
                value = policy(group.branch_batch(b, "cuda:0"), config.training.memory.loss_chunk_tokens or 256).float().cpu()
                entry = lookup[(gid, bid)]
                reference = load_file(str(Path(parent["path"]) / "anchor" / entry["file"]))["values"]
                changes.add(value - reference, group.advantages[b])
                branches.append({"branch_id": bid, "advantage": group.advantages[b], "token_count": len(value),
                    "logprob_mean": float(value.double().mean()),
                    "reference_logprob_mean": float(reference.double().mean()),
                    "logprob_change": float((value.double() - reference.double()).mean()),
                    "target_logprobs": value.tolist()})
            reports.append({"group_id": gid, "branches": branches, "metrics": group_metrics(branches)})
    return {"schema_version": 1, "kind": "rl-full-fixed-probe-v1", "checkpoint": parent["path"],
            "global_step": metadata["state"]["global_step"], "initial_policy": metadata["contract"]["initial_policy"],
            "gdn": evidence(backend_binding),
            "groups": reports, "token_changes": changes.summary()}
