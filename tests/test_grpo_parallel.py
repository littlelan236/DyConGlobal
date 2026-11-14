"""Compare real multiprocess SUM updates with an independent serial oracle."""
import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from RL.grpo.objective import ObjectiveResult
from RL.parallel import GroupParallel
from RL.training import train_grpo, capture_rng, restore_rng


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.3, -.2]))
        self.conditional = torch.nn.Parameter(torch.tensor([.4]))
        self.unused = torch.nn.Parameter(torch.tensor([.7]))


def groups():
    return [SimpleNamespace(group_id=str(i), group_size=32 if i < 4 else 3,
                            loss_mask=torch.ones(32 if i < 4 else 3, 1, dtype=torch.bool),
                            index=i) for i in range(5)]


class Objective:
    def iter_results(self, model, group):
        for branch in range(group.group_size):
            target = torch.tensor([group.index * .2 - .1, branch * .03])
            loss = (model.weight - target).square().sum()
            if group.index % 2:
                loss = loss + model.conditional.square().sum()
            loss = loss / group.group_size
            yield ObjectiveResult(loss, loss, loss * 0, {"policy_loss": float(loss.detach())})


def execute(model, parallel=None, *, state=None, optimizer=None, scheduler=None, stop=None):
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.2)
    scheduler = scheduler or torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
    gradients, updates = [], []
    result = train_grpo(model, groups(), Objective(), optimizer, scheduler, grad_clip=.25,
                        max_updates=stop, group_batch_size=4, state=state, parallel=parallel,
                        on_gradients=lambda: gradients.append({n: p.grad.clone() if p.grad is not None else None
                                                              for n, p in model.named_parameters()}),
                        on_update=lambda state, metrics: updates.append((state, metrics)))
    return result, optimizer, scheduler, gradients, updates


def _worker(rank, world_size, directory, failure):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(directory) / "store").as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=45))
    try:
        parallel = GroupParallel(rank, world_size)
        model = Model()
        parallel.synchronize_model(model)
        if failure:
            original = parallel.finish_window
            def invalid(m, parameters, window, rows, loss, prefix, seconds):
                if rank == world_size - 1:
                    if failure == "nonfinite":
                        next(p for p in parameters if p.grad is not None).grad.fill_(float("inf"))
                    else:
                        rows = []
                return original(m, parameters, window, rows, loss, prefix, seconds)
            parallel.finish_window = invalid
            before = copy.deepcopy(model.state_dict())
            with pytest.raises(ValueError, match="Nonfinite|missing or duplicate"):
                execute(model, parallel)
            assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
            assert all(p.grad is None for p in model.parameters())
            # Both peers must finish checking the expected rejection before the
            # test tears down Gloo and exits its spawned processes.
            dist.barrier()
            return
        # Pause after a complete update, reconstruct optimizer/scheduler and restore rank RNG.
        state, optimizer, scheduler, grad1, metrics1 = execute(model, parallel, stop=1)
        assert len(parallel.rank_rng) == world_size
        saved_rng = parallel.rank_rng[rank]
        expected = torch.rand(4)
        torch.rand(10)
        restore_rng(saved_rng)
        assert torch.equal(torch.rand(4), expected)
        restored = Model()
        restored.load_state_dict(model.state_dict())
        new_optimizer = torch.optim.AdamW(restored.parameters(), lr=.03, weight_decay=.2)
        new_optimizer.load_state_dict(optimizer.state_dict())
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=.8)
        new_scheduler.load_state_dict(scheduler.state_dict())
        state, _, _, grad2, metrics2 = execute(restored, parallel, state=state,
                                              optimizer=new_optimizer, scheduler=new_scheduler)
        torch.save({"model": restored.state_dict(), "state": state,
                    "gradients": grad1 + grad2, "updates": metrics1 + metrics2},
                   Path(directory) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_complete_groups_match_serial_and_resume_with_empty_tail_ranks(tmp_path, world_size):
    torch.set_num_threads(1)
    oracle = Model()
    state, _, _, gradients, updates = execute(oracle)
    mp.spawn(_worker, args=(world_size, str(tmp_path), None), nprocs=world_size, join=True)
    for rank in range(world_size):
        actual = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert actual["state"] == state
        for name, value in oracle.state_dict().items():
            torch.testing.assert_close(actual["model"][name], value, atol=1e-7, rtol=1e-6)
        for want, got in zip(gradients, actual["gradients"]):
            for name in want:
                if want[name] is None:
                    assert got[name] is None
                else:
                    torch.testing.assert_close(got[name], want[name], atol=2e-7, rtol=2e-6)
        assert actual["model"]["unused"].item() == pytest.approx(.7)
        assert [x[1]["branch_count"] for x in actual["updates"]] == [128, 3]
        assert [x[1]["loss"] for x in actual["updates"]] == pytest.approx(
            [x[1]["loss"] for x in updates], abs=1e-7)


@pytest.mark.parametrize("failure", ["nonfinite", "missing"])
def test_all_ranks_reject_invalid_window_without_optimizer_mutation(tmp_path, failure):
    mp.spawn(_worker, args=(2, str(tmp_path), failure), nprocs=2, join=True)


def test_configuration_accepts_only_supported_unambiguous_devices():
    from RL.config import load, Config
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    data = config.model_dump()
    for count in [1, 2, 4]:
        data["runtime"].update(gpu=None, gpus=list(range(count)))
        assert len(Config.model_validate(data).runtime.gpus) == count
    for invalid in [[], [0, 1, 2], [0, 0], [-1, 2], [True, 2], ["MIG-abc", 1]]:
        data["runtime"].update(gpu=None, gpus=invalid)
        with pytest.raises(ValueError):
            Config.model_validate(data)
    data["runtime"].update(gpu=0, gpus=[0, 1])
    with pytest.raises(ValueError, match="exactly one"):
        Config.model_validate(data)


def test_world_size_is_checkpoint_contract_but_device_indexes_are_not():
    from RL.config import load, schedule_contract
    from RL.parallel import execution_contract
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    schedule = schedule_contract(config)
    assert execution_contract(config) is None
    config.runtime.gpu = None
    config.runtime.gpus = [1, 2]
    first = execution_contract(config)
    config.runtime.gpus = [2, 3]
    assert execution_contract(config) == first
    assert schedule_contract(config) == schedule
    config.runtime.gpus = [0, 1, 2, 3]
    assert execution_contract(config) != first


class ChunkObjective:
    """Nonzero K3 and unequal masked completions; frozen parent advantages."""
    def iter_results(self, model, group):
        from RL.grpo.objective import grpo_objective
        chunk = group.metadata["execution_chunk"]
        for i, bid in enumerate(group.branch_ids):
            ordinal = chunk["native_branch_ordinals"][i]
            n = len(group.completion_token_ids[i])
            theta = (model.weight[0] + model.weight[1] * torch.arange(1,n+1) * .03)[None]
            if int(group.group_id[1:]) % 2:
                theta = theta + model.conditional
            ref = torch.full_like(theta,-.2)
            result = grpo_objective(theta,ref,[group.advantages[i]],torch.ones_like(theta,dtype=torch.bool),kl_beta=.13)
            result.loss = result.loss / 16
            result.metrics.update(branch_id=bid,native_branch_ordinal=ordinal)
            yield result


def _chunk_groups(n):
    from RL.grpo.execution import TokenCount,decorate
    from RL.grpo.groups import GroupBatch
    from tests.test_grpo_execution import plan_for
    plan=plan_for(n)
    parents={p["parent_group_id"]:p for p in plan["parents"]}
    class Groups:
        execution_plan=plan
        def __len__(self): return 2*n
        def cursor(self,count):
            from RL.grpo.execution import cursor
            return cursor(plan,count)
        def validate_state(self,state):
            from RL.grpo.execution import validate_state
            validate_state(plan,state)
        def windows(self,start):
            for w in plan["windows"]:
                if w["start"]<start: continue
                window=[]
                for chunk in plan["chunks"][w["start"]:w["end"]]:
                    p=parents[chunk["parent_group_id"]]
                    ordinals=chunk["native_branch_ordinals"]
                    g=GroupBatch(p["parent_group_id"],chunk["branch_ids"],[range(p["prompt_length"])]*16,
                        [range(p["completion_counts"][i]) for i in ordinals],None,None,None,
                        TokenCount(chunk["token_count"]),[float(i) for i in ordinals],
                        [(i-15.5)/16 for i in ordinals])
                    window.append(decorate(g,chunk,w,len(window)%8))
                yield tuple(window)
    return Groups()


def _chunk_run(model,n,parallel=None,*,state=None,optimizer=None,scheduler=None,stop=None):
    optimizer=optimizer or torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.2)
    scheduler=scheduler or torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    grads,updates=[],[]
    state=train_grpo(model,_chunk_groups(n),ChunkObjective(),optimizer,scheduler,group_batch_size=8,
        grad_clip=.07,parallel=parallel,state=state,max_updates=stop,
        on_gradients=lambda:grads.append({k:p.grad.detach().clone() if p.grad is not None else None
                                         for k,p in model.named_parameters()}),
        on_update=lambda s,m:updates.append(m))
    return state,optimizer,scheduler,grads,updates


def _chunk_oracle(n):
    """Independent selected-branch objective: no trainer or SUM helper."""
    from RL.grpo.objective import grpo_objective
    model=Model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.2)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    gradients,losses=[],[]
    for window in _chunk_groups(n).windows(0):
        group_losses=[]
        for g in window:
            branch_losses=[]
            for i,ordinal in enumerate(g.metadata["execution_chunk"]["native_branch_ordinals"]):
                length=len(g.completion_token_ids[i])
                logits=(model.weight[0]+model.weight[1]*torch.arange(1,length+1)*.03)[None]
                if int(g.group_id[1:])%2: logits=logits+model.conditional
                result=grpo_objective(logits,torch.full_like(logits,-.2),[(ordinal-15.5)/16],
                    torch.ones_like(logits,dtype=torch.bool),kl_beta=.13)
                branch_losses.append(result.loss)
            group_losses.append(torch.stack(branch_losses).mean())
        loss=torch.stack(group_losses).mean()
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append({k:p.grad.detach().clone() if p.grad is not None else None for k,p in model.named_parameters()})
        torch.nn.utils.clip_grad_norm_(model.parameters(),.07,error_if_nonfinite=True)
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    return model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),gradients,losses


def _chunk_worker(rank,directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=(Path(directory)/"store").as_uri(),rank=rank,world_size=8,
                            timeout=timedelta(seconds=120))
    try:
        parallel=GroupParallel(rank,8)
        results=[]
        for n in range(8,16):
            model=Model();parallel.synchronize_model(model)
            # Checkpoint after half 0's full window; some parents are unfinished.
            state,opt,sch,g1,m1=_chunk_run(model,n,parallel,stop=1)
            assert state["completed_parent_groups"]==0
            rng=parallel.rank_rng[rank]
            expected=torch.rand(3);restore_rng(rng)
            assert torch.equal(torch.rand(3),expected)
            restored=Model();restored.load_state_dict(model.state_dict())
            new_opt=torch.optim.AdamW(restored.parameters(),lr=.003,weight_decay=.2)
            new_opt.load_state_dict(opt.state_dict())
            new_sch=torch.optim.lr_scheduler.StepLR(new_opt,step_size=1,gamma=.9)
            new_sch.load_state_dict(sch.state_dict())
            state,opt,sch,g2,m2=_chunk_run(restored,n,parallel,state=state,optimizer=new_opt,scheduler=new_sch)
            results.append({"n":n,"model":restored.state_dict(),"optimizer":opt.state_dict(),
                "scheduler":sch.state_dict(),"state":state,"gradients":g1+g2,"updates":m1+m2})
        torch.save(results,Path(directory)/f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_eight_ranks_distinct_parents_all_tail_sizes_match_independent_oracle_and_resume(tmp_path):
    torch.set_num_threads(1)
    expected={n:_chunk_oracle(n) for n in range(8,16)}
    mp.spawn(_chunk_worker,args=(str(tmp_path),),nprocs=8,join=True)
    from tests.test_grpo_lifecycle import assert_tree_close
    for rank in range(8):
        actual=torch.load(tmp_path/f"rank-{rank}.pt",weights_only=True)
        for row in actual:
            model,opt,sch,grads,losses=expected[row["n"]]
            assert_tree_close(row["model"],model)
            assert_tree_close(row["optimizer"],opt)
            assert_tree_close(row["scheduler"],sch)
            for a,b in zip(row["gradients"],grads): assert_tree_close(a,b)
            assert [m["loss"] for m in row["updates"]]==pytest.approx(losses,abs=2e-7,rel=2e-6)
            assert row["state"]["completed_parent_groups"]==row["n"]
            assert row["state"]["consumed_execution_chunks"]==2*row["n"]
            assert row["state"]["consumed_branches"]==32*row["n"]
            assert row["model"]["unused"].item()==pytest.approx(.7)
            assert all(m["parent_group_count"]==m["execution_chunk_count"] for m in row["updates"])


def _chunk_failure_worker(rank,directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=(Path(directory)/"store").as_uri(),rank=rank,world_size=8,
                            timeout=timedelta(seconds=60))
    try:
        for fault in ("duplicate_parent","wrong_half","duplicate_branch","missing_branch","missing_rank","nonfinite","fingerprint"):
            parallel=GroupParallel(rank,8)
            model=Model();parallel.synchronize_model(model)
            before=copy.deepcopy(model.state_dict())
            groups=_chunk_groups(8)
            if fault in ("duplicate_parent","wrong_half"):
                original=groups.windows
                def windows(start):
                    from dataclasses import replace
                    for window in original(start):
                        window=list(window)
                        if rank==7:
                            if fault=="duplicate_parent": window[-1]=replace(window[-1],group_id=window[0].group_id)
                            else:
                                meta=copy.deepcopy(window[-1].metadata)
                                meta["execution_chunk"]["half_index"]=1
                                window[-1]=replace(window[-1],metadata=meta)
                        yield tuple(window)
                groups.windows=windows
            class FaultObjective(ChunkObjective):
                def iter_results(self,m,g):
                    for i,result in enumerate(super().iter_results(m,g)):
                        if rank==7 and fault=="missing_branch" and i==15: break
                        if rank==7 and fault=="duplicate_branch" and i==15:
                            result.metrics["branch_id"]=g.branch_ids[0]
                        yield result
            if fault in ("missing_rank","nonfinite"):
                original_finish=parallel.finish_window
                def finish(m,parameters,window,rows,loss,prefix,seconds):
                    if rank==7:
                        if fault=="missing_rank": rows=[]
                        else: next(p for p in parameters if p.grad is not None).grad.fill_(float("inf"))
                    return original_finish(m,parameters,window,rows,loss,prefix,seconds)
                parallel.finish_window=finish
            if fault=="fingerprint":
                original_after=parallel.after_step
                def after(m):
                    if rank==7:
                        with torch.no_grad(): m.weight.add_(1)
                    original_after(m)
                parallel.after_step=after
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
            committed=[]
            with pytest.raises((ValueError,RuntimeError),match="8x16|missing|Nonfinite|diverged"):
                train_grpo(model,groups,FaultObjective(),optimizer,scheduler,group_batch_size=8,
                           parallel=parallel,on_update=lambda *args:committed.append(args))
            assert not committed
            assert all(p.grad is None for p in model.parameters())
            if fault!="fingerprint":
                assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
                assert scheduler.last_epoch==0
        torch.save({"faults":7},Path(directory)/f"failure-rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_eight_ranks_reject_identity_coverage_nonfinite_and_fingerprint_faults_without_publication(tmp_path):
    mp.spawn(_chunk_failure_worker,args=(str(tmp_path),),nprocs=8,join=True)
    assert len(list(tmp_path.glob("failure-rank-*.pt")))==8
