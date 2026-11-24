"""Online LoRA data parallelism using the common training supervisor.

Rank zero alone owns rollout, estimator transactions and checkpoints. Each rank
builds the same globally weighted units and consumes ordinal modulo world size.
Adapter gradients and attribution vectors are SUM-reduced before clipping.
"""


def finish_contributions(context, model, attribution, metrics, ratio_sum):
    import torch
    import torch.distributed as dist
    parameters = [p for p in model.parameters() if p.requires_grad]
    peers = context.gather({"loss": metrics["loss"], "source_losses": metrics["source_losses"],
                            "source_k3": metrics["source_k3"], "token_count": metrics["token_count"],
                            "ratio_sum": ratio_sum, "source_compute_seconds": metrics["source_compute_seconds"]})
    context.reduce_gradients(parameters)
    reduce_attribution(context, parameters, attribution)
    metrics['loss'] = sum(p['loss'] for p in peers)
    metrics['token_count'] = sum(p['token_count'] for p in peers)
    for field in ('source_losses', 'source_k3', 'source_compute_seconds'):
        metrics[field] = {name: sum(p[field].get(name, 0.) for p in peers)
                          for name in ('dycon', 'baseline')}
    metrics['data_parallel'] = {'world_size': context.world_size,
                               'partition': 'training_unit_ordinal_mod_world_size',
                               'gradient_reduction': 'sum_fp32',
                               'rank_token_counts': [p['token_count'] for p in peers]}
    return sum(p['ratio_sum'] for p in peers)


def reduce_attribution(context, parameters, attribution):
    import torch
    import torch.distributed as dist

    # Never sum norms: attribution describes the actual global gradient vector.
    for name in ('dycon', 'baseline', 'kl'):
        vector = attribution.buffers[name]
        flags = torch.tensor([int(i in vector) for i in range(len(parameters))],
                             dtype=torch.int32, device=parameters[0].device)
        dist.all_reduce(flags, group=context.data_group)
        for i, used in enumerate(flags.tolist()):
            if not used:
                continue
            value = vector.get(i)
            if value is None:
                value = torch.zeros_like(parameters[i], dtype=torch.float32)
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=context.data_group)
            vector[i] = value


class PrimaryUpdater:
    def __init__(self, updater, context):
        object.__setattr__(self, '_updater', updater)
        object.__setattr__(self, '_context', context)
        object.__setattr__(self, '_first_update', True)

    def __getattr__(self, name):
        return getattr(self._updater, name)

    def __setattr__(self, name, value):
        setattr(self._updater, name, value)

    def __call__(self, batch, step, policy_version=None, estimator_version=None):
        message = {'command': 'update', 'batch': batch, 'step': step,
                   'policy_version': policy_version, 'estimator_version': estimator_version}
        if self._first_update:
            # On resume rank zero restores Adam/scheduler once. Peers must receive
            # the moments as well as synchronized adapter weights before updating.
            import torch
            def cpu(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [cpu(v) for v in value]
                if isinstance(value, tuple):
                    return tuple(cpu(v) for v in value)
                return value
            message['optimizer_state'] = cpu(self._updater.optimizer.state_dict())
            message['scheduler_state'] = self._updater.scheduler.state_dict()
            object.__setattr__(self, '_first_update', False)
        self._context.broadcast(message)
        # Restores/resume occur on rank zero; synchronize the complete trainable
        # policy before each update, including the first restored boundary.
        self._context.synchronize_model(self._updater.model)
        result = self._updater(batch, step, policy_version, estimator_version)
        from RL.training import capture_rng
        self._context.rank_rng = self._context.gather(capture_rng())
        return result


def follower(config, model):
    from RL.parallel import current
    from RL.online_runtime import TorchOnlineUpdater
    from RL.online_bootstrap import SourceMass
    from RL.policy_reference import InitialPolicyScorer, computation
    context = current()
    message = context.broadcast()
    if message.get('command') != 'initialize':
        raise ValueError('online follower expected initialization')
    options = config.online_bootstrap
    updater = TorchOnlineUpdater(model, config.training,
        source_mass=SourceMass(options.dycon_weight, options.baseline_weight),
        kl_beta=options.kl_beta, clip_epsilon=options.clip_epsilon, k3_cap=options.k3_cap,
        kl_reference_mode=computation(config),
        baseline_training_mode=options.baseline_training_mode,
        max_trace_input_tokens=options.max_trace_input_tokens)
    if options.kl_beta:
        updater.reference_scorer = InitialPolicyScorer(model, message['initial_adapter'])
    while True:
        message = context.broadcast()
        if message.get('command') == 'stop':
            return {'status': 'completed'}
        if message.get('command') != 'update':
            raise ValueError('unknown online follower command')
        if 'optimizer_state' in message:
            updater.optimizer.load_state_dict(message['optimizer_state'])
            updater.scheduler.load_state_dict(message['scheduler_state'])
        context.synchronize_model(model)
        updater(message['batch'], message['step'], message['policy_version'], message['estimator_version'])
        from RL.training import capture_rng
        context.rank_rng = context.gather(capture_rng())
