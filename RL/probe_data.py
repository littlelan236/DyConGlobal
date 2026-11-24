"""Read explicit complete parent groups using the frozen checkpoint data contracts."""
from pathlib import Path

from core.io import hash_json, read_json


class ProbeGroups:
    def __init__(self, config, checkpoint, contract, group_ids):
        from RL.grpo.stream_resume import checkpoint_kind
        self.ids, self.contract = group_ids, contract
        self.kind = checkpoint_kind(contract)
        self.stream = None
        if self.kind == "streaming":
            from RL.grpo.stream_resume import resolve_checkpoint
            from RL.grpo.streaming import StreamingGroups, attach_source
            resolved = resolve_checkpoint(checkpoint, contract)
            self.stream = StreamingGroups(config, resolved["plan"])
            attach_source(config, self.stream, contract["grpo"]["probability_binding"],
                          directory=resolved["evidence"])
            available = contract["input"]["eligible_group_ids"]
        elif self.kind == "prepared":
            from RL.grpo.prepared import PreparedReader, resolve
            from RL.grpo.reference_artifact import ReferenceReader
            self.prepared = PreparedReader(resolve(contract["prepared"]))
            self.reference = ReferenceReader(resolve(contract["grpo"]["reference_artifact"], "rl-reference"),
                                             prepared=self.prepared)
            available = self.prepared.ids
        else:
            available = contract["input"]["eligible_group_ids"]
            self.checkpoint = Path(checkpoint)
        missing = set(group_ids) - set(available)
        if missing:
            self.close()
            raise ValueError("Probe group IDs must be eligible complete parent IDs; unknown IDs: "
                             + ", ".join(sorted(missing)[:10]))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.stream is not None:
            self.stream.close()

    def positions(self, state):
        from RL.probe_schedule import positions
        if self.kind == 'prepared':
            self.prepared.validate_state(state)
            return positions(self.contract, state, self.ids, segments=self.prepared.schedule['segments'],
                             execution_plan=self.prepared.execution_plan)
        execution = (self.stream.proof.get('execution_plan') if self.stream is not None
                     else self.contract['input'].get('execution_plan'))
        return positions(self.contract, state, self.ids, execution_plan=execution)

    def __iter__(self):
        from RL.grpo.data import tensor_groups
        if self.kind == "legacy":
            raw = self._legacy()
        for gid in self.ids:
            if self.stream is not None:
                yield self.stream.load_group(gid)
            else:
                group = next(tensor_groups([self.prepared.raw(gid) if self.kind == "prepared" else raw[gid]]))
                if self.kind == "prepared":
                    reference = self.reference.for_group(group)
                else:
                    from RL.grpo.reference import load_reference_group
                    reference = load_reference_group(self.checkpoint, group, self.contract["input"],
                                                     self.contract["grpo"])
                yield group, reference

    def _legacy(self):
        # Historical files can be tens of GiB. Scan one group at a time and
        # verify the original frozen order/advantages/token digest, never rerank.
        from RL.observability import streaming_json
        from RL.grpo.prepared import hash_sequence
        from RL.grpo.grouping import order_record
        parser = streaming_json()
        proof = self.contract["input"]
        selected, seen = {}, []
        policy = proof["advantage_policy"]
        def records(stream):
            for value in parser.items(stream, "item", use_float=True):
                gid = value["group"]["group_id"]
                seen.append(gid)
                if gid in self.ids:
                    if gid in selected:
                        raise ValueError("Duplicate legacy probe group")
                    selected[gid] = value
                yield order_record(value, policy)
        with (self.checkpoint.parent.parent / "frozen-groups.json").open("rb") as stream:
            digest = hash_sequence(records(stream))
        if seen != proof["eligible_group_ids"] or digest != proof["group_order_sha256"]:
            raise ValueError("Legacy frozen group/token/advantage/order checksum mismatch")
        return selected


def group_identity(group, reference):
    """Bind actual target coordinates and probability bytes, not just ID strings."""
    import hashlib
    return {"group_id": group.group_id, "branches": [
        {"branch_id": bid, "prompt_sha256": hash_json(group.prompt_token_ids[i]),
         "completion_sha256": hash_json(group.completion_token_ids[i]),
         "mask_sha256": hash_json(group.loss_mask[i].tolist()), "advantage": group.advantages[i],
         "reference_sha256": hashlib.sha256(reference[bid].float().contiguous().numpy().tobytes()).hexdigest()}
        for i, bid in enumerate(group.branch_ids)]}
