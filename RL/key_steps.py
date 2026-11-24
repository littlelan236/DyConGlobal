"""Terminal-reward GRPO parents: one original decision and eight continuations.

This module adapts collection evidence to the existing native trajectory trainer.
It does not launch rollouts, reconstruct containers, or redefine the token loss.
"""

import math
import random
from dataclasses import replace
from pathlib import Path

from core.io import hash_json, identifier, owned_path, read_json, sha256, write_json

SCHEMA = "key_step_grpo_dataset_v1"
REWARD = "loop_m3_unresolved_m1_resolved_1_length_0_2_v1"
LENGTH = "whole_trajectory_generated_tokens_including_common_prefix_v1"
OBJECTIVE = "key_step_continuation_grpo_v1"


def enabled(config):
    return getattr(config, "key_steps", None) is not None


def scores(members):
    """Only resolved members define the min/max; infrastructure stays null."""
    lengths = [m["total_generated_tokens"] for m in members if m["outcome"] == "resolved"]
    shortest, longest = (min(lengths), max(lengths)) if lengths else (None, None)
    result = []
    for member in members:
        outcome = member["outcome"]
        length = member["total_generated_tokens"]
        if type(length) is not int or length < 1:
            raise ValueError("Trajectory generation length must be a positive integer")
        bonus = None
        if outcome == "loop":
            reward = -3.0
        elif outcome == "unresolved":
            reward = -1.0
        elif outcome == "resolved":
            bonus = (
                2.0
                if len(lengths) == 1
                else 1.0
                if shortest == longest
                else 2.0 * (longest - length) / (longest - shortest)
            )
            reward = 1.0 + bonus
        elif outcome == "infrastructure":
            reward = None
        else:
            raise ValueError("Unknown terminal outcome")
        result.append({**member, "reward": reward, "length_bonus": bonus})
    complete = bool(result) and all(m["reward"] is not None for m in result)
    mean = math.fsum(m["reward"] for m in result) / len(result) if complete else None
    std = math.sqrt(math.fsum((m["reward"] - mean) ** 2 for m in result) / len(result)) if complete else None
    for member in result:
        member["advantage"] = (member["reward"] - mean) / (std + 1e-8) if complete else None
    return result, {
        "mean": mean,
        "std": std,
        "complete": complete,
        "zero_variance": std == 0 if complete else None,
    }


def _checked(path, digest=None, size=None):
    path = Path(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError("Trajectory byte count changed")
    if digest is not None and sha256(path) != digest:
        raise ValueError("Trajectory evidence changed")
    return read_json(path)


def _native(root, descriptor):
    from hidden.artifacts import read_request

    value = read_request(root, descriptor)
    if (
        not value.get("prompt_token_ids")
        or not value.get("output_token_ids")
        or not value.get("tokenizer_fingerprint")
    ):
        raise ValueError("Missing exact native turn tokens/tokenizer")
    if any(type(t) is not int or t < 0 for t in value["prompt_token_ids"] + value["output_token_ids"]):
        raise ValueError("Invalid native token ID")
    return value


def _loop(trajectory, terminal_status):
    """A generic step/token/time limit is NOT proof of a loop."""
    info = trajectory.get("info") or {}
    if terminal_status == "reasoning_loop" or info.get("loop_detection"):
        return True
    return any(
        (m.get("extra") or {}).get("loop_detection")
        or (m.get("extra") or {}).get("limit_reason") == "repeated_reasoning_action_loop"
        for m in trajectory.get("messages", [])
    )


def read_member(member, project_root, *, cache=None):
    """Read old injection + actual child turns; common history is context only."""
    from hidden.artifacts import response_records
    from RL.online_runtime import OnlineRow
    from runtime.serving.raw_logprobs import read_values

    cache = {} if cache is None else cache
    child = member["source_ref"]
    source = child["source"]
    trajectory = _checked(owned_path(project_root, member["trajectory_ref"]), member["trajectory_sha256"])
    prefix_key = (source["trajectory_path"], source["trajectory_sha256"], source.get("trajectory_bytes"))
    if prefix_key not in cache:
        cache[prefix_key] = _checked(
            source["trajectory_path"], source["trajectory_sha256"], source.get("trajectory_bytes")
        )
    prefix = cache[prefix_key]
    reference = trajectory.get("continuation") or {}
    if (
        reference.get("child_id") != member["child_id"]
        or reference.get("parent_id") != child["parent_id"]
        or reference.get("source_step") != child["step"]
        or reference.get("prefix_message_count") != child["message_index"]
        or reference.get("trajectory_sha256") != source["trajectory_sha256"]
    ):
        raise ValueError("Child/common-prefix identity mismatch")
    decision = read_json(child["decision_path"])
    if hash_json(decision) != child["decision_sha256"]:
        raise ValueError("Original branch decision changed")
    branches = decision["candidates"]
    if [b["branch_id"] for b in sorted(branches, key=lambda x: x["branch_id"])] != list(range(8)):
        raise ValueError("Original decision must retain all eight branches")
    candidate = next(b for b in branches if b["branch_id"] == member["branch_id"])
    if candidate["native_request_id"] != child["source_native_request_id"]:
        raise ValueError("Injected original branch identity mismatch")

    # Count generated tokens once per historical response, not repeated prompt
    # usage. Losing mother candidates and tool observations are never counted.
    count_key = (prefix_key, child["message_index"], "generated-count")
    if count_key not in cache:
        count = 0
        for _, _, response in response_records({"messages": prefix["messages"][: child["message_index"]]}):
            base = response.get("ledger_root") or child["ledger_root"]
            native = _native(base, response)
            count += len(native["output_token_ids"])
        cache[count_key] = count
    prefix_count = cache[count_key]

    loop = _loop(trajectory, member.get("terminal_status"))
    result_path = member.get("result_ref")
    result = read_json(result_path) if result_path else {}
    outcome = (
        "infrastructure"
        if result.get("disposition") != "accepted"
        else "loop"
        if loop
        else "resolved"
        if member.get("reward") == 1
        else "unresolved"
        if member.get("reward") == 0
        else "infrastructure"
    )
    rows, tokenizers, models, prompts, native_ids = [], set(), set(), [], set()
    for ordinal, (_, message, response) in enumerate(response_records(trajectory)):
        injection = (message.get("extra") or {}).get("continuation_source")
        if ordinal == 0:
            if (
                not injection
                or injection.get("child_id") != member["child_id"]
                or injection.get("native_request_id") != candidate["native_request_id"]
            ):
                raise ValueError("First child response must be its original saved candidate")
            base, descriptor = child["ledger_root"], candidate
        else:
            if injection:
                raise ValueError("Repeated candidate injection")
            base, descriptor = response.get("ledger_root"), response
            if not base:
                raise ValueError("Child turn lacks published native root")
        native = _native(base, descriptor)
        expected = (
            candidate["native_request_id"]
            if ordinal == 0
            else response.get("dycon_native_request_id", response.get("id"))
        )
        if native["request_id"] != expected or expected in native_ids:
            raise ValueError("Duplicate or mismatched native child response")
        native_ids.add(expected)
        if ordinal == 0 and (
            native.get("branch_id") != member["branch_id"] or native.get("step_id") != child["step"]
        ):
            raise ValueError("Original branch/step differs from native evidence")
        completion, prompt = native["output_token_ids"], native["prompt_token_ids"]
        descriptor = native.get("raw_t1_probability")
        if not descriptor:
            raise ValueError("Missing raw T=1 denominator evidence")
        values = read_values(base, descriptor, output_ids=completion, record=native)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Nonfinite historical selected-token probabilities")
        tokenizers.add(native["tokenizer_fingerprint"])
        models.add(descriptor["source_identity"]["model_fingerprint"])
        prompts.append(tuple(prompt))
        rows.append(
            OnlineRow(
                row_id=hash_json([member["child_id"], ordinal]),
                task_id=child["instance_id"],
                source="baseline",
                group_id=child["parent_id"],
                input_token_ids=tuple(prompt + completion),
                completion_token_ids=tuple(completion),
                rollout_logprobs=tuple(float(v) for v in values),
                kl_reference_logprobs=None,
                advantage=-3.0 if loop else 1.0 if outcome == "resolved" else -1.0,
                trajectory_id=member["child_id"],
                agent_step=ordinal,
                resolved=outcome == "resolved",
                reward_reason="reasoning_loop" if loop else None,
            )
        )
    if not rows or len(tokenizers) != 1 or len(models) != 1:
        raise ValueError("Missing or mixed child model/tokenizer evidence")
    supervised = sum(len(r.completion_token_ids) for r in rows)
    rows = tuple(replace(r, trajectory_token_count=supervised) for r in rows)
    return rows, {
        "id": member["child_id"],
        "branch_id": member["branch_id"],
        "outcome": outcome,
        "prefix_generated_tokens": prefix_count,
        "supervised_generated_tokens": supervised,
        "total_generated_tokens": prefix_count + supervised,
        "tokenizer": next(iter(tokenizers)),
        "model": next(iter(models)),
        "first_prompt_sha256": hash_json(prompts[0]),
        "source_trajectory_sha256": member["trajectory_sha256"],
    }


def prepare(groups_path, destination, method):
    """Freeze the exact collection schema; do not silently inherit old rewards."""
    from RL.offline_baseline import _encode_rows

    payload = read_json(groups_path)
    if (
        payload.get("schema") != "continuation-groups-v1"
        or payload.get("branch_count") != 8
        or payload.get("sha256") != hash_json({k: v for k, v in payload.items() if k != "sha256"})
    ):
        raise ValueError("Invalid B=8 collection groups seal")
    if method not in {"adaptive", "random", "entropy"}:
        raise ValueError("Unknown key-step selector")
    selected = set(payload["methods"].get(method, []))
    if not selected:
        raise ValueError("Selector has no collected parents")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "trajectories").mkdir()
    (destination / "prefixes").mkdir()
    entries, audits, failed_sources, by_id = [], [], set(), {}
    seen = set()
    # Validate all three methods before paired selection; errors never turn
    # missing children into unresolved rewards or partial GRPO parents.
    for group in payload["groups"]:
        parent = identifier(group["parent_id"])
        if parent in seen:
            raise ValueError("Duplicate key-step parent")
        seen.add(parent)
        members = sorted(group["members"], key=lambda m: m["branch_id"])
        sources = {m["source_ref"]["source"]["rollout_id"] for m in members}
        try:
            if len(sources) != 1 or [m["branch_id"] for m in members] != list(range(8)):
                raise ValueError("A key-step parent requires eight original branches from one mother")
            origin = {
                (
                    m["source_ref"]["step"],
                    m["source_ref"]["message_index"],
                    m["source_ref"]["decision_sha256"],
                )
                for m in members
            }
            if len(origin) != 1:
                raise ValueError("GRPO parent mixes key steps")
            rows, records = [], []
            cache = {}
            for member in members:
                identifier(member["child_id"])
                if (
                    member["source_ref"]["parent_id"] != parent
                    or member["source_ref"]["branch"] != member["branch_id"]
                ):
                    raise ValueError("Member parent/branch differs from original inventory")
                turns, record = read_member(member, payload["project_root"], cache=cache)
                rows.append(turns)
                records.append(record)
            if len({r["id"] for r in records}) != 8 or any(
                len({r[k] for r in records}) != 1
                for k in ("first_prompt_sha256", "prefix_generated_tokens", "model", "tokenizer")
            ):
                raise ValueError("Group members do not share their exact common prefix/model/tokenizer")
            scored, stats = scores(records)
            if not stats["complete"]:
                raise ValueError("Incomplete terminal rewards")
            common = rows[0][0].input_token_ids[: -len(rows[0][0].completion_token_ids)]
            prefix_path = destination / "prefixes" / (parent + ".json")
            if parent in selected:
                write_json(prefix_path, list(common))
                for turns, record in zip(rows, scored, strict=True):
                    path = destination / "trajectories" / (record["id"] + ".json")
                    write_json(
                        path,
                        _encode_rows(turns, record["supervised_generated_tokens"], initial_prefix=common),
                    )
                    record.update(bytes=path.stat().st_size, sha256=sha256(path))
            entry = {
                "id": parent,
                "task": members[0]["source_ref"]["instance_id"],
                "mother": next(iter(sources)),
                "step": members[0]["source_ref"]["step"],
                "members": scored,
                **stats,
                "prefix_ref": "prefixes/" + parent + ".json",
                "prefix_tokens": len(common),
                "prefix_sha256": sha256(prefix_path) if parent in selected else None,
            }
            by_id[parent] = entry
        except (ValueError, KeyError, OSError, StopIteration) as exc:
            failed_sources.update(sources)
            audits.append({"parent_id": parent, "reason": str(exc)[:1000]})
    if not selected <= seen:
        raise ValueError("Selector references missing parents")
    for parent in sorted(selected):
        entry = by_id.get(parent)
        if entry is not None:
            entry["paired_eligible"] = entry["mother"] not in failed_sources
            entry["train_eligible"] = entry["paired_eligible"] and not entry["zero_variance"]
            entries.append(entry)
    # Discard unselected/intermediate physical child shards; only selected
    # parents are retained in the prepared training artifact.
    keep = {r["id"] for e in entries for r in e["members"]}
    for path in (destination / "trajectories").glob("*.json"):
        if path.stem not in keep:
            path.unlink()
    value = {
        "schema": SCHEMA,
        "schema_version": 2,
        "method": method,
        "reward": REWARD,
        "length": LENGTH,
        "branch_count": 8,
        "denominator": "native_raw_t1_sampled_token_v1",
        "prefix_policy": "context_with_grad_no_direct_loss_v1",
        "entries": entries,
        "source_groups_sha256": sha256(groups_path),
        "excluded": audits,
        "failed_mothers": sorted(failed_sources),
    }
    value["sha256"] = hash_json(value)
    write_json(destination / "manifest.json", value)
    return {
        "parents": len(entries),
        "train_eligible": sum(e["train_eligible"] for e in entries),
        "excluded": len(audits),
        "output": str(destination.resolve()),
    }


def manifest(dataset):
    value = read_json(Path(dataset) / "manifest.json")
    if (
        value.get("schema") != SCHEMA
        or value.get("reward") != REWARD
        or value.get("length") != LENGTH
        or value.get("sha256") != hash_json({k: v for k, v in value.items() if k != "sha256"})
    ):
        raise ValueError("Key-step dataset contract changed")
    parents = set()
    children = set()
    for entry in value["entries"]:
        if entry["id"] in parents or [m["branch_id"] for m in entry["members"]] != list(range(8)):
            raise ValueError("Key-step dataset has duplicate/partial parents")
        parents.add(entry["id"])
        recalculated, stats = scores(entry["members"])
        if any(entry.get(k) != v for k, v in stats.items()):
            raise ValueError("Frozen GRPO reward statistics differ from terminal outcomes")
        for actual, expected in zip(entry["members"], recalculated, strict=True):
            if actual["id"] in children:
                raise ValueError("A child may only belong to its original key-step parent")
            children.add(actual["id"])
            if (
                any(actual.get(k) != expected[k] for k in ("reward", "length_bonus", "advantage"))
                or actual["total_generated_tokens"]
                != actual["prefix_generated_tokens"] + actual["supervised_generated_tokens"]
            ):
                raise ValueError("Frozen child reward/length/advantage changed")
        if entry["train_eligible"] != (
            entry["paired_eligible"] and stats["complete"] and not stats["zero_variance"]
        ):
            raise ValueError("Incorrect key-step training eligibility")
    return value


def schedule(config):
    settings = config.key_steps
    value = manifest(settings.dataset)
    count = sum(e["train_eligible"] for e in value["entries"])
    if not count:
        raise ValueError("No complete nonzero-variance key-step groups")
    steps = math.ceil(count / settings.groups_per_update)
    total = steps * settings.epochs
    if config.training.max_updates is not None and config.training.max_updates != total:
        raise ValueError(f"Key-step max_updates must equal epochs * parent batches = {total}")
    return {"parents": count, "updates_per_epoch": steps, "epochs": settings.epochs, "max_updates": total}


def binding(settings, model_fingerprint, tokenizer_fingerprint):
    value = manifest(settings.dataset)
    if any(
        m["model"] != model_fingerprint or m["tokenizer"] != tokenizer_fingerprint
        for e in value["entries"]
        for m in e["members"]
    ):
        raise ValueError("Key-step source model/tokenizer differs from training")
    return {
        "manifest_sha256": value["sha256"],
        "config": settings.model_dump(exclude={"dataset"}),
        "reward": REWARD,
        "length": LENGTH,
        "grouping": "original_key_step_eight_children_v1",
    }


def validate_resume(config, parent, source_binding):
    if not parent or config.run.start.mode != "resume":
        return
    from RL.parallel import execution_contract

    metadata = read_json(Path(parent["path"]) / "state.json")
    saved, state = metadata["contract"], metadata["state"]
    if (
        saved.get("objective", {}).get("kind") != OBJECTIVE
        or saved.get("input") != source_binding
        or saved.get("schedule") != schedule(config)
        or saved.get("training") != config.training.model_dump()
        or saved.get("model") != config.model.model_dump(exclude={"path"})
        or saved.get("execution") != execution_contract(config)
        or saved.get("seed") != config.run.seed
    ):
        raise ValueError("Key-step resume input/reward/model/schedule/topology changed")
    stop = config.validation.stop_after_update or config.training.max_updates
    if not 0 < state["global_step"] < stop:
        raise ValueError("Key-step resume target must follow the committed update")


class ParentPool:
    def __init__(self, settings):
        self.settings = settings
        self.manifest = manifest(settings.dataset)
        self.entries = sorted(
            (e for e in self.manifest["entries"] if e["train_eligible"]), key=lambda e: e["id"]
        )

    def batch(self, cursor):
        from RL.offline_baseline import _decode_rows

        cycle, offset = cursor.get("cycle", 0), cursor.get("offset", 0)
        if (
            type(cycle) is not int
            or type(offset) is not int
            or cycle < 0
            or not 0 <= offset < len(self.entries)
        ):
            raise ValueError("Invalid key-step parent cursor")
        order = list(self.entries)
        random.Random(hash_json([self.settings.seed, cycle, "key-step-order-v1"])).shuffle(order)
        chosen = order[offset : offset + self.settings.groups_per_update]
        rows, masses = [], []
        for entry in chosen:
            if [m["branch_id"] for m in entry["members"]] != list(range(8)):
                raise ValueError("Partial parent")
            prefix = _checked(owned_path(self.settings.dataset, entry["prefix_ref"]), entry["prefix_sha256"])
            if len(prefix) != entry["prefix_tokens"]:
                raise ValueError("Prepared common prefix changed")
            for member in entry["members"]:
                path = owned_path(self.settings.dataset, "trajectories/" + member["id"] + ".json")
                data = _checked(path, member["sha256"], member["bytes"])
                turns = tuple(_decode_rows(data, 2, initial_prefix=prefix))
                count = sum(len(t.completion_token_ids) for t in turns)
                if count != member["supervised_generated_tokens"] or any(
                    t.group_id != entry["id"] or t.trajectory_id != member["id"] for t in turns
                ):
                    raise ValueError("Prepared child identity/token count changed")
                for turn in turns:
                    rows.append(replace(turn, advantage=member["advantage"]))
                    masses.append(len(turn.completion_token_ids) / count / 8 / len(chosen))
        offset += len(chosen)
        if offset == len(order):
            cycle, offset = cycle + 1, 0
        return (
            tuple(rows),
            masses,
            {"cycle": cycle, "offset": offset},
            {"mode": REWARD, "parents": [e["id"] for e in chosen]},
        )


def build_updater(
    config, model, directory, contract, optimizer, scheduler, *, memory_controller=None, on_stage=None
):
    from RL.online_runtime import OnlineRolloutBatch, SourceMass, TorchOnlineUpdater
    from RL.policy_reference import InitialPolicyScorer, computation

    settings = config.key_steps
    pool = ParentPool(settings)
    identity = contract["grpo"]["reference_identity"]
    if (
        binding(settings, identity["model_fingerprint"], identity["tokenizer_fingerprint"])
        != contract["offline_baseline"]
    ):
        raise ValueError("Key-step training input changed")
    scorer = (
        InitialPolicyScorer(model, Path(directory) / "grpo/reference/adapter") if settings.kl_beta else None
    )
    mass = SourceMass(0.0, 1.0)
    updater = TorchOnlineUpdater(
        model,
        config.training,
        source_mass=mass,
        kl_beta=settings.kl_beta,
        clip_epsilon=settings.clip_epsilon,
        k3_cap=settings.k3_cap,
        reference_scorer=scorer,
        kl_reference_mode=computation(config),
        baseline_training_mode="prefix_trace",
        max_trace_input_tokens=settings.max_trace_input_tokens,
        max_trace_completion_tokens=settings.max_trace_completion_tokens,
        optimizer_pair=(optimizer, scheduler),
        activation_offloader=memory_controller.offloader if memory_controller else None,
        on_stage=on_stage,
    )

    def accumulate(state, parallel, *, oom_replay=False):
        if manifest(settings.dataset)["sha256"] != pool.manifest["sha256"]:
            raise ValueError("Frozen key-step manifest changed during training")
        rows, masses, cursor, advantage = pool.batch(state.get("baseline_cursor", {}))
        batch = OnlineRolloutBatch(
            state["global_step"], tuple(dict.fromkeys(r.task_id for r in rows)), "historical", "unused", rows
        )
        from contextlib import nullcontext

        context = nullcontext()
        if oom_replay and config.model.attention_backend == "sdpa":
            from RL.qwen_boundary import retry_chunked_attention

            context = retry_chunked_attention()
        with context:
            _, metrics, _ = updater.accumulate(
                batch, masses, mass, state["global_step"], attribution=None, parallel=parallel
            )
        metrics["advantage"] = advantage
        return metrics, cursor, None

    return accumulate


def finalize_attempt(directory, contract, state, status, config, attempt):
    """Use the existing attempt/checkpoint receipts with an explicit objective."""
    directory = Path(directory)
    write_json(
        directory / "validation.json",
        {"state": state, "input": contract["input"], "objective": OBJECTIVE, "status": status},
    )
    (directory / "report.md").write_text(
        f"# Key-step continuation GRPO\n\nStatus: {status}\nUpdates: {state['global_step']}\n"
        "Common-prefix targets are masked; full-context parameter gradients are retained.\n",
        encoding="utf8",
    )
    attempt["contract"] = contract
    write_json(directory / "attempt.json", attempt)


def verify_attempt(directory, receipt, config):
    from RL.checkpoints import verify_adapter, verify_checkpoint

    directory = Path(directory)
    contract = read_json(directory / "key-step-contract.json")
    attempt = read_json(directory / "attempt.json")
    validation = read_json(directory / "validation.json")
    if (
        contract["objective"]["kind"] != OBJECTIVE
        or hash_json(contract) != receipt["contract_sha256"]
        or attempt["contract"] != contract
        or validation["state"] != attempt["final_state"]
        or validation["input"] != contract["input"]
        or contract["objective"]["settings"] != config.key_steps.model_dump(exclude={"dataset"})
        or attempt["status"] != receipt["status"]
        or attempt["final_state"]["global_step"]
        != (config.validation.stop_after_update or config.training.max_updates)
    ):
        raise ValueError("Key-step attempt objective/input/cursor mismatch")
    verify_adapter(directory / "adapter", contract["adapter"], verify_owner=False)
    for path in sorted((directory / "checkpoints").glob("step-*")):
        verify_checkpoint(path)
    return {
        "scope": "lora-attempt",
        "artifact_id": config.run.attempt_id,
        "path": str(directory),
        "verified": True,
        "status": attempt["status"],
        "files_checked": len(receipt["files"]),
        "receipt_sha256": sha256(directory / "receipt.json"),
    }
