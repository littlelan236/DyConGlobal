"""Register a managed serving process before exec; wait for supervisor ownership ACK."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from core.io import now, read_json, write_json
from core.locking import directory_changes
from core.processes import identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role',choices=['serving','network-proxy'],default='serving')
    parser.add_argument('run')
    parser.add_argument('attempt_id')
    parser.add_argument('resource_id')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run = Path(args.run)
    attempt = run / 'execution/attempts' / args.attempt_id
    resources = attempt / 'resources'
    resources.mkdir(parents=True, exist_ok=True)
    record = {**identity(os.getpid()), 'run_id': run.name, 'execution_attempt_id': args.attempt_id,
              'project_root': str(run.parent.parent), 'role': args.role, 'managed': True,
              'argv': args.command, 'registered_at': now()}
    write_json(resources / (args.resource_id + '.json'), record)
    write_json(run / 'execution/resource-notification.json', {'resource_id': args.resource_id, 'at': now()})
    ack = resources / (args.resource_id + '.ack.json')
    deadline = time.monotonic() + 30
    with directory_changes(resources) as changed:
        while not ack.exists():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Supervisor did not acknowledge serving ownership')
            changed(remaining)
    if read_json(ack).get('allowed') is not True:
        raise SystemExit(125)
    os.execvpe(args.command[0], args.command, os.environ)


if __name__ == '__main__':
    main()
