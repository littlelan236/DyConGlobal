"""Opt-in, detached native target diagnostics; never change scoring or sampling."""
import math


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def branch_diagnostics(current, reference, group, index, settings):
    import torch
    with torch.no_grad():
        theta = current.detach().float().cpu()
        anchor = reference.detach().float().cpu()
        log_ratio = theta.double() - anchor.double()
        ratio = log_ratio.exp()
        k3 = (-log_ratio).expm1() + log_ratio
        finite = torch.isfinite(theta) & torch.isfinite(anchor)
        counts = {}
        for name, values in (("current", theta), ("reference", anchor),
                             ("ratio", ratio), ("k3", k3)):
            for suffix, predicate in (("nan", torch.isnan), ("posinf", torch.isposinf),
                                      ("neginf", torch.isneginf)):
                counts[name + "_" + suffix + "_tokens"] = int(predicate(values).sum())
        counts.update(current_probability_zero_fp32_tokens=int((theta.exp() == 0).sum()),
                      reference_probability_zero_fp32_tokens=int((anchor.exp() == 0).sum()),
                      ratio_zero_fp32_tokens=int(((theta - anchor).exp() == 0).sum()))
        suspicious = (~finite | ~torch.isfinite(k3) | (k3 > settings.k3_threshold))
        positions = suspicious.nonzero().flatten()
        # Nonfinite candidates first, then largest K3. Stable positions break ties.
        ordered = sorted(positions.tolist(), key=lambda p: (
            not bool(finite[p] & torch.isfinite(k3[p])),
            float(k3[p]) if torch.isfinite(k3[p]) else math.inf), reverse=True)
        completion = group.completion_token_ids[index]
        prompt = group.prompt_token_ids[index]
        native_ordinal = group.metadata.get("physical_task", group.metadata.get("execution_chunk", {})).get(
            "native_branch_ordinals", list(range(group.group_size)))[index]
        outliers = []
        for position in ordered[:settings.max_tokens_per_branch]:
            absolute = len(prompt) + position
            start, end = max(0, absolute - settings.context_tokens), min(
                len(prompt) + len(completion), absolute + settings.context_tokens + 1)
            tokens = prompt + completion
            outliers.append({"group_id": group.group_id, "branch_id": group.branch_ids[index],
                "native_branch_ordinal": native_ordinal, "advantage": float(group.advantages[index]),
                "prompt_tokens": len(prompt), "completion_tokens": len(completion),
                "completion_offset": position, "sequence_position": absolute,
                "target_token_id": int(completion[position]),
                "context_start_position": start, "context_token_ids": tokens[start:end],
                "current_logprob": _number(theta[position]), "reference_logprob": _number(anchor[position]),
                "current_probability_fp64": _number(theta.double()[position].exp()),
                "reference_probability_fp64": _number(anchor.double()[position].exp()),
                "log_ratio": _number(log_ratio[position]), "ratio_fp64": _number(ratio[position]),
                "k3_fp64": _number(k3[position]),
                "branch_k3_mean_contribution": _number(k3[position] / len(completion))})
        return {"schema_version": 1, "token_count": len(theta), "branch_count": 1,
                "counts": counts, "threshold_exceeded_tokens": int(suspicious.sum()),
                "truncated_outlier_tokens": max(0, len(ordered) - len(outliers)),
                "outliers": outliers}


def summarize_token_diagnostics(groups):
    rows = [r["_token_diagnostics"] for g in groups for r in g if "_token_diagnostics" in r]
    if not rows:
        return {}
    keys = ("token_count", "branch_count", "threshold_exceeded_tokens", "truncated_outlier_tokens")
    return {"token_diagnostics": {"schema_version": 1,
        **{key: sum(r[key] for r in rows) for key in keys},
        "counts": {key: sum(r["counts"][key] for r in rows) for key in rows[0]["counts"]},
        "outliers": [token for r in rows for token in r["outliers"]]}}


def compare_branch_scores(model, group, index, current, reference, diagnostics, settings, block_size):
    """Independent full teacher-forced forwards on only selected outlier branches.

    Both use the same selected projection as training. Base comparison disables
    the current LoRA temporarily; no optimizer, RNG or reference bytes are changed.
    """
    if not diagnostics["outliers"] or not (settings.compare_full_branch or settings.compare_base):
        return
    import torch
    from .logprobs import model_target_logprobs
    device = next(model.parameters()).device
    batch = group.branch_batch(index, device)
    def observer(key):
        def observe(scoring_model, hidden, labels, projection_block_size):
            distributions = target_distributions(scoring_model, hidden, labels,
                [row["completion_offset"] for row in diagnostics["outliers"]],
                projection_block_size)
            for row in diagnostics["outliers"]:
                row[key] = distributions[row["completion_offset"]]
        return observe
    with torch.no_grad():
        if settings.compare_full_branch:
            full = model_target_logprobs(model, batch, block_size,
                target_observer=observer("full_current_distribution")).float().cpu()
            difference = full - current.detach().float().cpu()
            diagnostics["full_branch_comparison"] = {
                "max_abs_logprob_difference": _number(difference.abs().max()),
                "mean_abs_logprob_difference": _number(difference.abs().mean()),
                "nonfinite_tokens": int((~torch.isfinite(full)).sum())}
            for row in diagnostics["outliers"]:
                value = full[row["completion_offset"]]
                row.update(full_current_logprob=_number(value),
                    full_minus_training_logprob=_number(value - row["current_logprob"]))
        if settings.compare_base:
            with model.disable_adapter():
                base = model_target_logprobs(model, batch, block_size,
                    target_observer=observer("base_bf16_distribution")).float().cpu()
            difference = base - reference.detach().float().cpu()
            diagnostics["base_reference_comparison"] = {
                "max_abs_logprob_difference": _number(difference.abs().max()),
                "mean_abs_logprob_difference": _number(difference.abs().mean()),
                "nonfinite_tokens": int((~torch.isfinite(base)).sum())}
            for row in diagnostics["outliers"]:
                value = base[row["completion_offset"]]
                row.update(base_bf16_logprob=_number(value),
                    base_bf16_minus_reference_logprob=_number(value - row["reference_logprob"]))


def target_distributions(model, hidden, labels, positions, block_size, top_k=8):
    """Bounded alternatives at native targets, using the scorer's block shapes.

    Ranks are one plus strictly larger logits (ties share a rank). Alternatives
    describe a teacher-forced context, not free generation or task success.
    """
    import torch
    import torch.nn.functional as F
    from RL.memory import target_coordinates
    valid = target_coordinates(labels)
    if len(set(positions)) != len(positions) or any(p < 0 or p >= len(valid) for p in positions):
        raise ValueError("Invalid diagnostic target positions")
    selected = set(positions)
    head = model.get_output_embeddings()
    result = {}
    with torch.no_grad():
        offset = 0
        for block in valid.split(block_size or len(valid)):
            wanted = sorted(p for p in selected if offset <= p < offset + len(block))
            if wanted:
                b, pos = block.unbind(1)
                logits = F.linear(hidden[b, pos], head.weight, head.bias).float()
                for target_position in wanted:
                    local = target_position - offset
                    row = logits[local]
                    target_id = int(labels[b[local], pos[local] + 1])
                    logp = row.log_softmax(-1)
                    values, ids = logp.topk(min(top_k, len(logp)))
                    finite = torch.isfinite(row)
                    result[target_position] = {
                        "target_token_id": target_id,
                        "target_logprob": _number(logp[target_id]),
                        "target_rank": int((row > row[target_id]).sum()) + 1,
                        "logits_nan": int(torch.isnan(row).sum()),
                        "logits_posinf": int(torch.isposinf(row).sum()),
                        "logits_neginf": int(torch.isneginf(row).sum()),
                        "entropy_nats": _number(-(logp[finite].exp() * logp[finite]).sum()),
                        "top_tokens": [{"token_id": int(token), "logprob": _number(value)}
                                       for token, value in zip(ids, values)]}
            offset += len(block)
    return result
