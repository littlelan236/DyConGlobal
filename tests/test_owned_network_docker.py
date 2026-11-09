"""Opt-in CPU-only check against an explicitly selected Docker daemon."""
import os
import uuid

import docker
import pytest

from runtime.containers.networks import cleanup_owned_resources


@pytest.mark.skipif(not os.environ.get('DYCON_NETWORK_TEST_HOST'), reason='Explicit Docker network test opt-in required')
def test_more_than_thirty_networks_and_scoped_reclamation():
    host = os.environ['DYCON_NETWORK_TEST_HOST']
    project = os.getcwd()
    ownership = {'project':project, 'run':'network-capacity-' + uuid.uuid4().hex, 'execution':'000'}
    labels = {'dycon.'+k:v for k,v in ownership.items()}
    client = docker.DockerClient(base_url=host, timeout=5)
    before = {n.id for n in client.networks.list()}
    created = []
    try:
        for i in range(40):
            network = client.networks.create(ownership['run']+'-'+str(i), driver='bridge', labels=labels)
            created.append(network.id)
            network.reload()
            subnet = network.attrs['IPAM']['Config'][0]['Subnet']
            assert subnet.startswith('172.16.') and subnet.endswith('/23')
        assert len(created) == 40
    finally:
        result = cleanup_owned_resources({'project_root':project},ownership,client=client)
        assert result['complete'], result
        assert before <= {n.id for n in client.networks.list()}
        client.close()
