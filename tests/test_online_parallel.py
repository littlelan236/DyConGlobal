"""Global online objective/gradient agreement across uneven data-parallel ranks."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import pytest


def _run(rank, store, output, cuda):
    import torch
    import torch.distributed as dist
    import RL.parallel as parallel
    import RL.training
    from RL.online_runtime import OnlineRow, OnlineRolloutBatch, TorchOnlineUpdater
    from RL.online_bootstrap import SourceMass
    from RL.online_parallel import PrimaryUpdater
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=4)
    group = dist.new_group(backend='nccl' if cuda else 'gloo')
    context = parallel.GroupParallel(rank, 4, data_group=group)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.projection = torch.nn.Linear(4, 8)
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.projection(self.embedding(input_ids).cumsum(1)))
    def optimizer(model, settings):
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    RL.training.optimizer_and_scheduler = optimizer
    torch.manual_seed(12)
    model = Model().to(device)
    settings = SimpleNamespace(grad_clip=.1)
    updater = TorchOnlineUpdater(model, settings, source_mass=SourceMass(), kl_beta=.02,
                                k3_cap=1., baseline_training_mode='prefix_trace')
    dense = Model().to(device);dense.load_state_dict(model.state_dict())
    reference = TorchOnlineUpdater(dense, settings, source_mass=SourceMass(), kl_beta=.02,
                                  k3_cap=1., baseline_training_mode='prefix_trace')
    rows = tuple(OnlineRow(str(i), 'a', 'dycon' if i<3 else 'baseline',
                    'g' if i<3 else 'b', (1,2,3), (3,), (-2.,), (-2.2,),
                    1. if i%2 else -1., str(i), 1, 0, False) for i in range(7))
    results = []
    for step in range(3):
        # Seven units: uneven ranks; then one unit: three empty ranks; then
        # an entirely discarded batch, which must not advance Adam/scheduler.
        batch = OnlineRolloutBatch(step, ('a',), 'S0', 'E0', rows if step==0 else rows[3:4] if step==1 else ())
        parallel._context = None
        expected = reference(batch, step)
        parallel._context = context
        if rank == 0:
            actual = PrimaryUpdater(updater, context)(batch, step)
        else:
            command = context.broadcast()
            if 'optimizer_state' in command:
                updater.optimizer.load_state_dict(command['optimizer_state'])
                updater.scheduler.load_state_dict(command['scheduler_state'])
            context.synchronize_model(model)
            actual = updater(command['batch'], command['step'])
            from RL.training import capture_rng
            context.rank_rng = context.gather(capture_rng())
        for p,q in zip(model.parameters(), dense.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-5,atol=2e-7)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-7)
        assert actual['loss'] == pytest.approx(expected['loss'],rel=2e-5,abs=1e-7)
        if step<2:
            for name in ['dycon','baseline','kl']:
                assert actual['gradient_attribution']['components'][name]['norm'] == pytest.approx(expected['gradient_attribution']['components'][name]['norm'],rel=2e-5,abs=1e-7)
            assert sum(actual['data_parallel']['rank_token_counts'])==actual['token_count']
        else:
            assert actual['skipped_update']
        results.append(actual['loss'])
    parallel._context = None
    dist.barrier()
    if rank==0:Path(output).write_text(str(results))
    dist.destroy_process_group()


def test_four_rank_online_matches_single_rank(tmp_path):
    import torch
    import torch.multiprocessing as mp
    cuda = os.environ.get('DYCON_TEST_ONLINE_FOUR_GPU') == '1'
    if cuda and torch.cuda.device_count()<4:pytest.skip('requires four GPUs')
    mp.spawn(_run,args=(str(tmp_path/'store'),str(tmp_path/'result'),cuda),nprocs=4,join=True)
    assert (tmp_path/'result').exists()
