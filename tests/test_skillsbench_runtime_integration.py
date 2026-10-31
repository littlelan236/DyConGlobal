import pytest


def test_benchflow_plane_owns_compose_hooks(tmp_path):
    pytest.importorskip("benchflow")
    from runtime.agents.skillsbench import OwnedPlanes

    class Inner:
        def create_environment(self):
            class Environment:
                async def stop(self, delete=True):
                    return None
            return Environment()

        async def harden_before_verify(self, *args, **kwargs):
            return None

        def verifier(self, *args, **kwargs):
            return "verifier"

        def start(self):
            return "start"

    sample = type("Sample", (), {"instance_id": "t", "sample_key": "s", "task_key": "k"})()
    cfg = {"project_root": str(tmp_path), "execution_context": {"profile": {"cpu_quota": "strict"}},
           "selection": {"tasks": {"k": {"instance": {"metadata": {"sandbox": {}}}}}}}
    planes = OwnedPlanes(Inner(), cfg, "r", sample, "a", {}, tmp_path)
    env = planes.create_environment()
    assert env._mounts_compose_path == tmp_path / "ownership-compose.json"
    assert planes.start() == "start"


def test_benchflow_cleanup_never_removes_shared_leased_images(tmp_path):
    pytest.importorskip('benchflow')
    import asyncio
    from runtime.agents.skillsbench import OwnedPlanes
    from core.io import read_json
    stopped=[]
    class Inner:
        def create_environment(self):
            class Environment:
                async def stop(self,delete=True):stopped.append(delete)
            return Environment()
    sample=type('Sample',(),{'instance_id':'t','sample_key':'s'})()
    cfg={'project_root':str(tmp_path),'execution_context':{'profile':{'cpu_quota':'strict'}}}
    env=OwnedPlanes(Inner(),cfg,'r',sample,'a',{},tmp_path).create_environment()
    asyncio.run(env.stop(delete=True))
    assert stopped==[False]
    assert read_json(tmp_path/'sandbox-cleanup.json')['image_cleanup_owner']=='ImageLeaseManager'
