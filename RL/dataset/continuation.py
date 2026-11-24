"""Native multi-turn groups for terminal-reward continuation datasets.

Each turn retains its ACTUAL prompt; only its completion is supervised. This
reader never concatenates changing prompts into a fictitious single prompt.
The legacy estimator/one-step trainer deliberately does not accept this schema.
"""

from core.io import hash_json, owned_path, read_json, sha256
from hidden.artifacts import read_request, response_records
from rollout.key_steps.service import verify


def member_turns(member, root):
    """Load references once; the original branch plus all actual suffix turns."""
    path = owned_path(root, member["trajectory_ref"])
    if sha256(path) != member["trajectory_sha256"]:
        raise ValueError("Child trajectory changed")
    trajectory = read_json(path)
    turns = []
    expected = member["source_ref"]
    for index, message, response in response_records(trajectory):
        source = (message.get("extra") or {}).get("continuation_source")
        if source:
            if (
                source["child_id"] != member["child_id"]
                or source["native_request_id"] != expected["source_native_request_id"]
            ):
                raise ValueError("Injected response does not belong to this member")
            decision = read_json(expected["decision_path"])
            if hash_json(decision) != expected["decision_sha256"]:
                raise ValueError("Source decision changed")
            candidate = next(x for x in decision["candidates"] if x["branch_id"] == member["branch_id"])
            descriptor = {"ledger_ref": candidate["ledger_ref"], "ledger_sha256": candidate["ledger_sha256"]}
            base = expected["ledger_root"]
        else:
            descriptor = response
            base = response.get("ledger_root")
            if not base:
                raise ValueError("Suffix lacks native token evidence")
        native = read_request(base, descriptor)
        if native["request_id"] != (
            source["native_request_id"]
            if source
            else response.get("dycon_native_request_id", response.get("id"))
        ):
            raise ValueError("Turn native identity mismatch")
        p, c = native["prompt_token_ids"], native["output_token_ids"]
        from runtime.serving.raw_logprobs import read_values

        probability = native.get("raw_t1_probability")
        if not probability:
            raise ValueError("Turn lacks native raw T=1 probability coverage")
        read_values(base, probability, output_ids=c, record=native)
        if not p or not c:
            raise ValueError("Native turn is empty")
        turns.append(
            {
                "message_index": index,
                "native_request_id": native["request_id"],
                "prompt_token_ids": p,
                "completion_token_ids": c,
                "loss_mask": [False] * len(p) + [True] * len(c),
                "native_evidence": {"root": base, **descriptor},
                "advantage": member.get("advantage"),
                "raw_t1_probability": native.get("raw_t1_probability"),
            }
        )
    if not turns or turns[0]["native_request_id"] != expected["source_native_request_id"]:
        raise ValueError("Continuation must start with its original candidate")
    return turns


def iter_groups(path, *, method, with_tokens=False):
    payload = verify(read_json(path), "continuation-groups-v1")
    if method not in {"adaptive", "random", "entropy"}:
        raise ValueError("Unknown selector")
    selected = set(payload["methods"].get(method, []))
    for group in payload["groups"]:
        if group["parent_id"] not in selected:
            continue
        if [m["branch_id"] for m in group["members"]] != list(range(8)):
            raise ValueError("Full original branch multiplicity required")
        if not group["train_eligible"]:
            continue
        if group["zero_variance"]:
            continue
        members = []
        for member in group["members"]:
            member = dict(member)
            if with_tokens:
                member["turns"] = member_turns(member, payload["project_root"])
            members.append(member)
        yield {**group, "members": members}


def objective(theta_by_member, reference_by_member, advantages, **options):
    """Reuse GRPO objective after joining token logprobs within each CHILD.

    Arguments are [child][turn] 1D tensors computed at each turn's actual prompt.
    Flattening logprob targets (not token histories) makes long and short turns
    contribute by token count while retaining equal child/parent weights.
    """
    import torch

    from RL.grpo.objective import grpo_objective

    if len(theta_by_member) != 8 or len(reference_by_member) != 8 or len(advantages) != 8:
        raise ValueError("Expected eight logical members")
    theta = []
    ref = []
    for current, anchor in zip(theta_by_member, reference_by_member, strict=True):
        if not current or len(current) != len(anchor):
            raise ValueError("Turn coverage differs")
        for a, b in zip(current, anchor, strict=True):
            if a.ndim != 1 or a.shape != b.shape or not a.numel():
                raise ValueError("Turn logprob alignment differs")
        theta.append(torch.cat(current))
        ref.append(torch.cat(anchor))
    length = max(x.numel() for x in theta)
    masks = torch.stack([torch.arange(length, device=x.device) < x.numel() for x in theta])
    theta = torch.stack([torch.nn.functional.pad(x, (0, length - x.numel())) for x in theta])
    ref = torch.stack([torch.nn.functional.pad(x, (0, length - x.numel())) for x in ref])
    return grpo_objective(theta, ref, advantages, masks, **options)
