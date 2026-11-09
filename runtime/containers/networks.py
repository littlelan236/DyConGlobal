"""Reconcile Docker resources inside an explicit ownership boundary."""
import docker


def cleanup_owned_resources(cfg, ownership, *, client=None):
    from core.cancellation import policy
    from core.io import identifier

    expected = {'dycon.' + key: str(value) for key, value in ownership.items()}
    required = {'dycon.project', 'dycon.run', 'dycon.execution'}
    if (not required <= expected.keys() or not all(expected.values())
            or expected['dycon.project'] != cfg['project_root']):
        raise ValueError('Cleanup requires exact project/run/execution ownership')
    identifier(expected['dycon.run'])
    identifier(expected['dycon.execution'])
    # An attempt cleanup must bind its unique epoch, never just a task ID.
    if len(expected) > 3 and 'dycon.epoch' not in expected:
        raise ValueError('Attempt cleanup requires an ownership epoch')
    filters = {'label': [key + '=' + value for key, value in expected.items()]}
    own_client = client is None
    client = client or docker.DockerClient(base_url=cfg['environment']['docker_host'],
        timeout=min(3, policy(cfg)['cleanup_seconds']))
    removed, removed_networks, remaining, remaining_networks, errors = [], [], [], [], []

    def check(labels):
        if any(labels.get(key) != value for key, value in expected.items()):
            raise RuntimeError('Docker resource is outside requested ownership')

    try:
        containers = client.containers.list(all=True, filters=filters)
        networks = client.networks.list(filters=filters)
        # Validate every match before deleting anything, even if Docker's filter
        # implementation returns a resource from another run or execution.
        for container in containers:
            container.reload()
            check(container.labels)
        for network in networks:
            network.reload()
            check(network.attrs.get('Labels') or {})
        for container in containers:
            try:
                container.reload()
                check(container.labels)
                container.remove(force=True)
                removed.append(container.id)
            except docker.errors.NotFound:
                removed.append(container.id)
            except Exception as exc:
                errors.append({'container_id': container.id, 'error': repr(exc)})
        for network in networks:
            try:
                network.reload()
                check(network.attrs.get('Labels') or {})
                if network.attrs.get('Containers'):
                    errors.append({'network_id': network.id, 'error': 'network still has attached containers'})
                    continue
                network.remove()
                removed_networks.append(network.id)
            except docker.errors.NotFound:
                removed_networks.append(network.id)
            except Exception as exc:
                errors.append({'network_id': network.id, 'error': repr(exc)})
        remaining = [c.id for c in client.containers.list(all=True, filters=filters)]
        remaining_networks = [n.id for n in client.networks.list(filters=filters)]
    except Exception as exc:
        errors.append({'error': repr(exc)})
    finally:
        if own_client:
            client.close()
    return {'complete': not errors and not remaining and not remaining_networks,
            'ownership': expected, 'removed': removed, 'removed_networks': removed_networks,
            'remaining': remaining, 'remaining_networks': remaining_networks, 'errors': errors}
