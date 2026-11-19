"""Shared optimizer learning-rate schedule for LoRA and full-policy training."""

def factor(step, settings):
    """Return the multiplier at scheduler index ``step``.

    Warmup is measured in optimizer updates. Update 1 uses 1/N of the base
    learning rate and update N reaches the base rate. Linear decay, when
    selected, starts after warmup and reaches zero at max_updates.
    """
    warmup = getattr(settings, "warmup_updates", 0)
    if warmup and step < warmup:
        return (step + 1) / warmup
    if settings.scheduler == "linear":
        if not warmup:
            return max(0.0, 1 - step / settings.max_updates)
        return max(0.0, (settings.max_updates - step) / (settings.max_updates - warmup + 1))
    return 1.0
