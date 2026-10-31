import asyncio
import pytest
from core.cancellation import CancellationContext, Cancelled
from core.deadline import Deadline
from runtime.agents.lifecycle import run_cancellable


def test_cancel_during_tool_or_verifier_waits_for_owned_cleanup():
    async def scenario():
        context = CancellationContext()
        events=[]
        class Rollout:
            async def run(self):
                try:
                    events.append('tool_started')
                    context.cancel({'reason':'test cancellation during tool'})
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    events.append('owned_cleanup')
        try:
            with pytest.raises(Cancelled,match='during tool'):
                await run_cancellable(Rollout(),context,Deadline(5),1)
            assert events==['tool_started','owned_cleanup']
            assert not context._callbacks
        finally:
            context.close()
    asyncio.run(scenario())


def test_completed_rollout_retains_result_and_removes_callback():
    async def scenario():
        context=CancellationContext()
        class Rollout:
            async def run(self): return {'reward':1}
        try:
            assert await run_cancellable(Rollout(),context,Deadline(5),1)=={'reward':1}
            assert not context._callbacks
        finally: context.close()
    asyncio.run(scenario())


def test_wall_deadline_cleans_the_same_rollout():
    async def scenario():
        events=[]
        class Rollout:
            async def run(self):
                try: await asyncio.Event().wait()
                finally: events.append('cleaned')
        with pytest.raises(TimeoutError,match='wall-clock'):
            await run_cancellable(Rollout(),None,Deadline(0.02),1)
        assert events==['cleaned']
    asyncio.run(scenario())
