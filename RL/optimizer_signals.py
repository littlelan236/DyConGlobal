"""Bounded deterministic adapter samples, without RNG or full parameter copies."""
import torch


def gradient_signals(parameters, norm, threshold):
    total = sum(p.numel() for p in parameters)
    covered = sum(p.numel() for p in parameters if p.grad is not None)
    value = float(norm)
    return {"gradient_clip_threshold": float(threshold),
            "gradient_clip_scale": min(1., threshold / (value + 1e-12)),
            "gradient_parameter_coverage": covered / total if total else None}


def parameter_sample(parameters, next_step):
    # Same sample and cadence across fresh/resume. No CUDA host synchronization.
    if next_step != 1 and next_step % 10:
        return None
    with torch.no_grad():
        return torch.cat([p.detach().reshape(-1)[::max(1, p.numel() // 64)][:64].float()
                          for p in parameters if p.numel()]).clone()


def update_signals(parameters, before):
    if before is None:
        return {}
    with torch.no_grad():
        after = parameter_sample(parameters, 1)
        old_norm, delta = float(before.norm()), float((after - before).norm())
    return {"parameter_update_sample_norm": delta, "parameter_before_sample_norm": old_norm,
            "parameter_update_sample_relative_norm": delta / old_norm if old_norm else None,
            "parameter_update_sample_elements": before.numel()}
