"""Reconcile only a stopped run's identity-checked legacy proxy process."""
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from core.io import now, read_json, write_json
from core.processes import identity, matches, signal_group


def cleanup_network(root,run_id):
    from application.execution.run_plan import load_plan
    loaded=load_plan(root,run_id);run=loaded['run']
    previous=run/'agent/network/cleanup-reconciliation.json'
    if previous.is_file():
        proof=read_json(previous)
        owner=proof.get('process_identity')
        if proof.get('cleanup_complete') and (not owner or not matches(owner) or identity(owner['pid'])['state']=='Z'):
            return proof
    if read_json(run/'status.json').get('stage') not in {'complete','failed','cancelled'}:
        raise ValueError('Network cleanup requires a terminal run')
    attempts=sorted(p for p in (run/'execution/attempts').iterdir() if p.is_dir())
    termination=read_json(attempts[-1]/'termination.json')
    if not termination.get('processes_stopped') or not termination.get('cleanup_complete'):
        raise ValueError('Execution resources have not stopped')
    receipt=read_json(run/'agent/network/proxy.json')
    pid=receipt['pid'];path=Path('/proc')/str(pid)
    result={'run_id':run_id,'proxy_pid':pid,'reconciled_at':now(),'signals':[]}
    if path.exists():
        cmd=path.joinpath('cmdline').read_bytes().split(b'\0')
        actual=[s.decode() for s in cmd if s]
        owner=identity(pid)
        boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines()
                      if line.startswith('btime ')))
        born=boot+int(owner['start_ticks'])/os.sysconf('SC_CLK_TCK')
        if (actual!=receipt['args'] or owner['pgid']!=pid or owner['sid']!=pid
                or abs(born-datetime.fromisoformat(receipt['started_at']).timestamp())>5):
            raise ValueError('Proxy command, session or birth identity differs; refusing cleanup')
        result['process_identity']=owner
        result['signals'].append(signal_group(owner,signal.SIGTERM))
        deadline=time.monotonic()+5
        while matches(owner) and identity(pid)['state']!='Z' and time.monotonic()<deadline:
            time.sleep(.1)
        if matches(owner) and identity(pid)['state']!='Z':
            result['signals'].append(signal_group(owner,signal.SIGKILL))
    result['cleanup_complete']=not path.exists() or identity(pid)['state']=='Z'
    if not result['cleanup_complete']:raise RuntimeError('Proxy did not exit')
    write_json(run/'agent/network/cleanup-reconciliation.json',result)
    return result
