"""Selected target logprobs using the existing backbone and causal coordinates."""
import torch
import torch.nn.functional as F

from RL.memory import (text_backbone, target_coordinates, local_saved_tensors,
                       autocast_state, autocast_context)


class _SelectedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, weight, bias, block_size):
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError("GRPO selected projection requires a frozen output head")
        valid = target_coordinates(labels)
        ctx.save_for_backward(hidden, labels, valid)
        ctx.weight, ctx.bias, ctx.block_size = weight, bias, block_size
        ctx.versions = (weight._version, bias._version if bias is not None else None)
        ctx.autocast = autocast_state(hidden)
        pieces = []
        for block in valid.split(block_size):
            b, pos = block.unbind(1)
            logits = F.linear(hidden[b, pos], weight, bias).float()
            pieces.append(-F.cross_entropy(logits, labels[b, pos + 1], reduction="none"))
        return torch.cat(pieces)

    @staticmethod
    def backward(ctx, output_gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("Selected target logprobs support first-order gradients only")
        if ctx.versions != (ctx.weight._version, ctx.bias._version if ctx.bias is not None else None):
            raise RuntimeError("Frozen output head changed")
        hidden, labels, valid = ctx.saved_tensors
        gradient = torch.zeros_like(hidden)
        offset = 0
        for block in valid.split(ctx.block_size):
            b, pos = block.unbind(1)
            with torch.enable_grad(), local_saved_tensors(), autocast_context(ctx.autocast):
                part = hidden[b, pos].detach().requires_grad_(True)
                logits = F.linear(part, ctx.weight, ctx.bias).float()
                values = -F.cross_entropy(logits, labels[b, pos + 1], reduction="none")
                (grad,) = torch.autograd.grad(values, part,
                    grad_outputs=output_gradient[offset:offset + len(block)])
            gradient[b, pos] = grad
            offset += len(block)
        return gradient, None, None, None, None


def model_target_logprobs(model, batch, block_size=0):
    labels = batch["labels"]
    mask = batch["loss_mask"].bool()
    if (not torch.equal(labels != -100, mask) or mask[:, 0].any()
            or (mask & ~batch["attention_mask"].bool()).any()):
        raise ValueError("Native GRPO labels/mask/attention coordinates disagree")
    valid = target_coordinates(labels)
    if not len(valid):
        raise ValueError("No native completion targets")
    hidden = text_backbone(model)(input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("Expected linear frozen output projection")
    return _SelectedLogprobs.apply(hidden, labels, head.weight, head.bias, block_size or len(valid))
