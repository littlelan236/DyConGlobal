"""Linux process identities and narrowly owned process-group signalling."""
from __future__ import annotations

import os
import socket
from pathlib import Path

IDENTITY_KEYS = ('pid', 'start_ticks', 'host', 'boot_id')


def enable_subreaper():
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': int(pid), 'start_ticks': stat[19], 'host': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pgid': int(stat[2]), 'sid': int(stat[3]), 'state': stat[0]}


def matches(record):
    try:
        actual = identity(record['pid'])
        return all(actual[k] == record.get(k) for k in IDENTITY_KEYS)
    except (FileNotFoundError, ProcessLookupError):
        return False


def members(record, known=()):
    """Discover a group only while a verified member anchors its ownership."""
    if record.get('pgid') != record.get('sid') or record.get('pgid') != record.get('pid'):
        raise RuntimeError('Refusing a non-isolated process group')
    anchor = any(matches(item) for item in (record, *known))
    if not anchor:
        return [item for item in known if matches(item)]
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            item = identity(int(entry.name))
            if item['pgid'] == record['pgid'] and item['sid'] == record['sid']:
                result.append(item)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def signal_group(record, signum, known=()):
    group = members(record, known)
    living = [item for item in group if item['state'] != 'Z']
    if not living:
        return {'signal': int(signum), 'sent': False, 'reason': 'already_exited'}
    if record['pgid'] in (os.getpgrp(), os.getsid(0)):
        raise RuntimeError('Refusing to signal the controller process group')
    # A verified live member plus an isolated SID/PGID anchors the group.
    if not any(matches(item) for item in living):
        return {'signal': int(signum), 'sent': False, 'reason': 'identity_changed'}
    try:
        os.killpg(record['pgid'], signum)
        return {'signal': int(signum), 'sent': True, 'members': [i['pid'] for i in living]}
    except ProcessLookupError:
        return {'signal': int(signum), 'sent': False, 'reason': 'already_exited'}
