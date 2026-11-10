"""Policy clipped ratio and token K3 objective."""

from dataclasses import dataclass
import torch


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    k3_penalty: torch.Tensor
    metrics: dict
    group_count: int = 1
    branch_count: int = 0
    token_count: int = 0


def selected_logprobs(logits, input_ids, labels=None, loss_mask=None):
    """Return teacher-forced log-probabilities at native completion targets.

    Labels are indexed in the unshifted sequence, so the first completion
    token at position ``len(prompt)`` is read from the logit at ``len(prompt)-1``.
    """
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch,length,vocab] and input_ids [batch,length]")
    targets = input_ids[:, 1:] if labels is None else labels[:, 1:]
    mask = (targets != -100) if loss_mask is None else loss_mask[:, 1:].bool()
    if logits.shape[:2] != (input_ids.shape[0], input_ids.shape[1]):
        raise ValueError("logits/input sequence shape mismatch")
    logp = logits[:, :-1].float().log_softmax(-1)
    safe_targets = targets.clamp_min(0)
    selected = logp.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return selected, mask


def _masked_branch_mean(values, mask):
    counts = mask.sum(-1)
    if torch.any(counts <= 0):
        raise ValueError("Every GRPO branch needs at least one completion target")
    return values.masked_fill(~mask, 0).sum(-1) / counts


def grpo_objective(logp_theta, logp_reference, advantages, loss_mask, *, clip_epsilon=0.1,
                   kl_beta=0.02, group_slices=None):
    """Compute one or more groups with branch-then-group mean reduction."""
    if logp_theta.shape != logp_reference.shape or logp_theta.ndim != 2:
        raise ValueError("logprob tensors must have equal [branch,token] shapes")
    mask = loss_mask.to(device=logp_theta.device, dtype=torch.bool)
    if mask.shape != logp_theta.shape:
        raise ValueError("Mask/logprob shape mismatch")
    adv = torch.as_tensor(advantages, device=logp_theta.device, dtype=torch.float32).detach()
    if adv.numel() != logp_theta.shape[0]:
        raise ValueError("advantage count differs from branch count")
    if not torch.isfinite(adv).all():
        raise ValueError("Nonfinite advantages")
    theta = logp_theta.float().masked_fill(~mask, 0)
    ref = logp_reference.to(logp_theta.device).float().detach().masked_fill(~mask, 0)
    ratio = torch.exp(theta - ref)
    weighted = ratio * adv[:, None]
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * adv[:, None]
    surrogate = torch.minimum(weighted, clipped)
    branch_policy = _masked_branch_mean(surrogate, mask)
    delta = ref - theta
    k3 = torch.expm1(delta) - delta
    branch_k3 = _masked_branch_mean(k3, mask)
    if group_slices is None:
        group_slices = [slice(0, logp_theta.shape[0])]
    policy_values = []; k3_values = []
    for span in group_slices:
        if isinstance(span, slice):
            indices = range(*span.indices(logp_theta.shape[0]))
        else:
            indices = span
        indices = list(indices)
        if not indices:
            raise ValueError("Empty group in objective reduction")
        policy_values.append(branch_policy[indices].mean())
        k3_values.append(branch_k3[indices].mean())
    policy_loss = -torch.stack(policy_values).mean()
    k3_penalty = torch.stack(k3_values).mean()
    loss = policy_loss + float(kl_beta) * k3_penalty
    finite = (torch.isfinite(ratio) & torch.isfinite(delta) & torch.isfinite(k3))
    if not torch.all(finite) or not torch.isfinite(loss):
        raise ValueError("Nonfinite GRPO ratio, delta, or K3")
    return ObjectiveResult(loss, policy_loss, k3_penalty, {
        "ratio_mean": float(ratio[mask].mean().detach()),
        "ratio_min": float(ratio[mask].min().detach()),
        "ratio_max": float(ratio[mask].max().detach()),
        "clip_fraction": float(((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon))[mask].float().mean().detach()),
        "k3_penalty_mean": float(k3_penalty.detach()),
        "policy_loss": float(policy_loss.detach()),
        "ratio_quantiles": torch.quantile(ratio[mask].detach(), torch.tensor([0., .25, .5, .75, 1.], device=ratio.device)).cpu().tolist(),
        "_ratio_values": ratio[mask].detach().cpu(),
        "positive_logprob_change": float((theta-ref)[mask & (adv[:, None] > 0)].mean().detach()) if (adv > 0).any() else None,
        "negative_logprob_change": float((theta-ref)[mask & (adv[:, None] < 0)].mean().detach()) if (adv < 0).any() else None,
        "positive_branches": int((adv > 0).sum()), "negative_branches": int((adv < 0).sum()),
        "branch_count": int(logp_theta.shape[0]), "token_count": int(mask.sum()),
    }, len(group_slices), int(logp_theta.shape[0]), int(mask.sum()))
