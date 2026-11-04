import asyncio
import json

import pytest

from core.cancellation import CancellationContext, Cancelled
from core.deadline import Deadline
from runtime.agents.lifecycle import run_cancellable


def test_cancel_during_tool_or_verifier_waits_for_owned_cleanup(tmp_path):
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
                await run_cancellable(Rollout(),context,Deadline(5),1,
                    evidence_path=tmp_path / 'process-supervisor.json', binding={'attempt_id': 'a'})
            assert events==['tool_started','owned_cleanup']
            assert not context._callbacks
            assert json.loads((tmp_path / 'process-supervisor.json').read_text())['event'] == 'run_cancelled'
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


def test_wall_deadline_cleans_the_same_rollout(tmp_path):
    async def scenario():
        events=[]
        class Rollout:
            async def run(self):
                try: await asyncio.Event().wait()
                finally: events.append('cleaned')
        with pytest.raises(TimeoutError,match='wall-clock'):
            await run_cancellable(Rollout(),None,Deadline(0.02),1,
                evidence_path=tmp_path / 'process-supervisor.json', binding={'attempt_id': 'a'})
        assert events==['cleaned']
        import json
        assert json.loads((tmp_path / 'process-supervisor.json').read_text())['event'] == 'attempt_deadline'
    asyncio.run(scenario())
