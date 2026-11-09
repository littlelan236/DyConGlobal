from types import SimpleNamespace

import pytest

from runtime.containers.networks import cleanup_owned_resources


@pytest.mark.parametrize('foreign', [True, False])
def test_never_remove_foreign_or_attached_network(foreign):
    calls = []
    labels = {'dycon.project':'/p','dycon.run':'r','dycon.execution':'001'}
    network = SimpleNamespace(id='n', attrs={'Labels':{**labels,'dycon.run':'other'} if foreign else labels,
        'Containers':{'foreign-container':{}}}, reload=lambda:None, remove=lambda:calls.append('removed'))
    client = SimpleNamespace(containers=SimpleNamespace(list=lambda **kw:[]),
        networks=SimpleNamespace(list=lambda **kw:[network]))
    result = cleanup_owned_resources({'project_root':'/p'},
        {'project':'/p','run':'r','execution':'001'},client=client)
    assert not result['complete'] and not calls


def test_cleanup_prevalidates_containers_before_any_deletion():
    calls = []
    def container(run):
        return SimpleNamespace(id=run,labels={'dycon.project':'/p','dycon.run':run,'dycon.execution':'001'},
            reload=lambda:None,remove=lambda **kw:calls.append(run))
    client = SimpleNamespace(containers=SimpleNamespace(list=lambda **kw:[container('r'),container('other')]),
        networks=SimpleNamespace(list=lambda **kw:[]))
    result = cleanup_owned_resources({'project_root':'/p'},
        {'project':'/p','run':'r','execution':'001'},client=client)
    assert not result['complete'] and not calls
