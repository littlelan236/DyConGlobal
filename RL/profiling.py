"""Measured per-update telemetry and optional first-update CPU/CUDA kernel audit."""

import torch

from core.io import write_json
from RL.memory import text_backbone
from RL.resources import host_memory


def transfer_overlap(events):
    """Observed device copy/compute interval intersection, in microseconds."""
    def merge(intervals):
        result = []
        for start, end in sorted(intervals):
            if result and start <= result[-1][1]:
                result[-1][1] = max(end, result[-1][1])
            else:
                result.append([start, end])
        return result
    copies, compute = [], []
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        target = copies if "memcpy" in event.name.lower() else compute
        target.append((event.time_range.start, event.time_range.end))
    copies, compute = merge(copies), merge(compute)
    overlap = 0.0
    index = 0
    for start, end in copies:
        while index < len(compute) and compute[index][1] <= start:
            index += 1
        other = index
        while other < len(compute) and compute[other][0] < end:
            overlap += max(0., min(end, compute[other][1]) - max(start, compute[other][0]))
            other += 1
    copy_time = sum(end - start for start, end in copies)
    return dict(cuda_copy_microseconds=copy_time, cuda_copy_compute_overlap_microseconds=overlap,
                cuda_copy_compute_overlap_fraction=overlap / copy_time if copy_time else None)


class UpdateTelemetry:
    def __init__(self, model, loss, mlps, offloader, directory, profile, require_flash):
        self.loss, self.mlps, self.offloader = loss, mlps, offloader
        self.directory, self.profile, self.require_flash = directory, profile, require_flash
        self.profiler = None
        self.first = True
        self.attention_masks = []
        self.handles = []
        for index, layer in enumerate(text_backbone(model).layers):
            attention = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
            def inspect_mask(module, args, kwargs, layer_index=index):
                mask = kwargs.get("attention_mask")
                shape = list(mask.shape) if mask is not None else None
                record = {"layer": layer_index, "type": type(module).__name__, "mask_shape": shape}
                if record not in self.attention_masks:
                    self.attention_masks.append(record)
                if self.require_flash and shape is not None and len(shape) == 4:
                    raise RuntimeError("flash_only exact-length path constructed a dense attention mask")
            self.handles.append(attention.register_forward_pre_hook(inspect_mask, with_kwargs=True))

    def start(self):
        self.loss.reset()
        for mlp in self.mlps:
            mlp.forward_blocks = mlp.backward_blocks = 0
        if self.offloader:
            self.offloader.reset()
        if self.first and self.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities)
            self.profiler.__enter__()

    def finish(self):
        values = {"projected_tokens": self.loss.tokens, "ce_chunks": self.loss.blocks,
                  "kernel_observation": "profiled_this_update" if self.profiler else "not_profiled",
                  "sequence_lengths": self.loss.sequence_lengths,
                  "mlp_forward_chunks": sum(mlp.forward_blocks for mlp in self.mlps),
                  "mlp_backward_chunks": sum(mlp.backward_blocks for mlp in self.mlps),
                  "host": host_memory()}
        if self.offloader:
            values.update(self.offloader.metrics())
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            events = self.profiler.events()
            operators = sorted({event.name for event in events if event.device_type == torch.autograd.DeviceType.CPU})
            kernels = sorted({event.name for event in events if event.device_type == torch.autograd.DeviceType.CUDA})
            attention = [name for name in operators if any(word in name.lower() for word in
                         ("attention", "delta", "causal_conv", "convolution"))]
            audit = {"operators": attention, "cuda_kernels": kernels,
                     **transfer_overlap(events),
                     "attention_masks": self.attention_masks,
                     "flash_forward": any("flash_attention" in name and "backward" not in name for name in operators),
                     "flash_backward": any("flash_attention" in name and "backward" in name for name in operators),
                     "math_attention": any("attention_math" in name for name in operators),
                     "note": "Linear-attention implementation identity is frozen in environment.json; kernels are measured."}
            write_json(self.directory / "kernel-audit.json", audit)
            self.profiler.export_chrome_trace(str(self.directory / "profile-first-update.json"))
            self.profiler = None
            if self.require_flash and not (audit["flash_forward"] and audit["flash_backward"] and not audit["math_attention"]):
                raise RuntimeError("Profile did not confirm required Flash forward and backward")
        values["attention_masks"] = self.attention_masks
        self.first = False
        return values

    def close(self):
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        for handle in self.handles:
            handle.remove()
