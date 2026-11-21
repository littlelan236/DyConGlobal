"""Weighted, pre-clip gradient attribution for the single-GPU online updater."""
import time


class GradientAttribution:
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.buffers = {name: {} for name in ('dycon', 'baseline', 'kl')}
        self.kl_backward_seconds = 0.0

    def _add(self, name, index, gradient):
        value = gradient.detach().float()
        target = self.buffers[name]
        if index in target:
            target[index].add_(value)
        else:
            target[index] = value.clone()

    def backward(self, loss, source, weighted_kl=None):
        import torch
        if weighted_kl is not None:
            device = loss.device
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.monotonic()
            gradients = torch.autograd.grad(weighted_kl, self.parameters,
                                            retain_graph=True, allow_unused=True)
            for i, gradient in enumerate(gradients):
                if gradient is not None:
                    self._add('kl', i, gradient)
            del gradients
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            self.kl_backward_seconds += time.monotonic() - started
        handles = []
        try:
            for i, parameter in enumerate(self.parameters):
                def capture(gradient, index=i):
                    self._add(source, index, gradient)
                    return gradient
                handles.append(parameter.register_hook(capture))
            loss.backward()
        finally:
            for handle in handles:
                handle.remove()

    def summary(self):
        import torch
        # Reduce per tensor in FP64; never concatenate model-sized vectors.
        def dot(left, right):
            values = [(left[i].double() * right[i].double()).sum()
                      for i in left.keys() & right.keys()]
            return float(torch.stack(values).sum()) if values else 0.0
        total = {i: p.grad.detach().float() for i, p in enumerate(self.parameters)
                 if p.grad is not None}
        total_sq = dot(total, total)
        total_norm = total_sq ** .5
        norms = {k: max(0., dot(v, v)) ** .5 for k, v in self.buffers.items()}
        source_norm_sum = norms['dycon'] + norms['baseline']
        result = {'version': 'weighted_preclip_gradient_v1',
                  'total_norm': total_norm,
                  'kl_backward_seconds': self.kl_backward_seconds,
                  'components': {}}
        for name, vector in self.buffers.items():
            product = dot(vector, total)
            result['components'][name] = {
                'norm': norms[name],
                'norm_over_total': norms[name] / total_norm if total_norm else None,
                'projection_share': product / total_sq if total_sq else None,
                'cosine_with_total': product / (norms[name] * total_norm)
                    if norms[name] and total_norm else None,
                'source_norm_share': norms[name] / source_norm_sum
                    if name != 'kl' and source_norm_sum else None}
        d, b = self.buffers['dycon'], self.buffers['baseline']
        result['source_cosine'] = dot(d, b) / (norms['dycon'] * norms['baseline']) \
            if norms['dycon'] and norms['baseline'] else None
        residual = {i: total[i].clone() for i in total}
        for source in (d, b):
            for i, value in source.items():
                if i in residual:
                    residual[i].sub_(value)
                else:
                    residual[i] = -value
        result['source_sum_residual_norm'] = max(0., dot(residual, residual)) ** .5
        return result
