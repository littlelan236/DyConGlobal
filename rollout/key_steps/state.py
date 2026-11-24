"""Lightweight verification records, never a restorable filesystem snapshot."""

import copy
import json
import shlex
import time
from dataclasses import asdict
from pathlib import Path

from core.io import hash_json, read_json, write_json


class ReplayMismatch(RuntimeError):
    """Infrastructure-null: historical context does not match this environment."""


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def rows(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


# This program executes inside the existing task container through its factory.
# Paths are fixed by the v1 schema. No file contents are exported. Large files
# have metadata only and conservatively invalidate changed boundaries.
PROBE = r"""
import os, stat, json, hashlib, subprocess, sys, importlib.metadata
options=json.loads(sys.argv[1]); files={}; errors=[]
# Only the immutable container entrypoint and this probe's ancestor chain may
# survive a tool call. Process memory and sockets cannot be validated by files.
ancestors={1}; current=os.getpid()
while current>1 and current not in ancestors:
 ancestors.add(current)
 try: current=int(open('/proc/%d/stat'%current).read().rsplit(')',1)[1].split()[1])
 except (OSError,ValueError): break
for name in os.listdir('/proc'):
 if not name.isdigit() or int(name) in ancestors: continue
 try:
  status=open('/proc/'+name+'/stat').read().rsplit(')',1)[1].split()[0]
  if status not in {'Z','X'}: errors.append('resident-process:'+name)
 except OSError: pass
packages=sorted((d.metadata.get('Name','').lower(),d.version,
 hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest()) for d in importlib.metadata.distributions())
files['/__dycon_validation__/packages']={'sha256':hashlib.sha256(json.dumps(packages).encode()).hexdigest()}

for root in options['state_roots']:
 if not os.path.isdir(root):
  errors.append('missing-root:'+root); continue
 for directory,dirs,names in os.walk(root,followlinks=False):
  for name in sorted(set(dirs+names)):
   path=os.path.join(directory,name)
   if path=='/testbed/.git/index': continue
   try:
    s=os.lstat(path); mode=stat.S_IFMT(s.st_mode)
    item={'type':mode,'permissions':stat.S_IMODE(s.st_mode)}
    if stat.S_ISLNK(s.st_mode): item['target']=os.readlink(path)
    elif stat.S_ISREG(s.st_mode):
     item['bytes']=s.st_size
     if s.st_size<=options['small_file_bytes']:
      with open(path,'rb') as f: item['content_sha256']=hashlib.sha256(f.read()).hexdigest()
     else:
      item['large_unverified']=True
      item['mtime_ns']=s.st_mtime_ns
    elif not stat.S_ISDIR(s.st_mode): errors.append('unsupported-file:'+path)
    files[path]=item
    if len(files)>options['max_state_paths']: raise RuntimeError('state path budget exceeded')
   except (OSError,IOError) as e: errors.append(path+':'+type(e).__name__)
def git(args):
 r=subprocess.run(['git','-C','/testbed']+args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
 if r.returncode: errors.append('git:'+str(args))
 return hashlib.sha256(r.stdout).hexdigest()
print(json.dumps({'files':files,'errors':errors,'git':{k:git(v) for k,v in {
 'head':['rev-parse','HEAD'],'index':['ls-files','--stage','-z'],
 'diff':['diff','--binary'],'cached':['diff','--cached','--binary']}.items()}},sort_keys=True))
"""


def fingerprint(env, settings):
    options = {k: settings[k] for k in ("state_roots", "small_file_bytes", "max_state_paths")}
    command = (
        "source /opt/miniconda3/bin/activate && conda activate testbed && python -c "
        + shlex.quote(PROBE)
        + " "
        + shlex.quote(json.dumps(options))
    )
    result = env.factory.execute(env.container, command, 120, env.config.cwd)
    if result["returncode"] != 0:
        raise ReplayMismatch("State probe failed: " + result["output"][-500:])
    value = json.loads(result["output"])
    if value["errors"]:
        raise ReplayMismatch("Incomplete state probe: " + str(value["errors"][:4]))
    return value


def contract(env):
    """Image and effective policy identity; no image byte hashing."""
    spec = env.container.spec
    bootstrap = read_json(env.context.output_dir / "bootstrap-receipt.json")
    if any(v.get("mode") != "ro" for v in spec.get("volumes", {}).values()):
        raise ReplayMismatch("Pure replay does not support writable external mounts")
    return {
        "image_id": spec["image_id"],
        "policy_hash": spec["policy_hash"],
        "packages_hash": bootstrap["packages_hash"],
        "network_mode": spec.get("network_mode"),
        "environment": env.config.env,
        "cwd": env.config.cwd,
    }


def compare_state(expected, actual):
    if expected != actual:
        paths = sorted(
            k
            for k in expected["files"].keys() | actual["files"].keys()
            if expected["files"].get(k) != actual["files"].get(k)
        )
        raise ReplayMismatch(
            "State differs at " + str(paths[:12]) + "; git_equal=" + str(expected["git"] == actual["git"])
        )


class Recorder:
    def __init__(self, env, settings, out):
        self.env, self.settings, self.out = env, settings, Path(out)
        self.actions = []
        self.states = []
        self.previous = None
        self.original_execute = env.execute
        env.execute = self.execute
        write_json(self.out / "replay-contract.json", contract(env))

    def execute(self, action, cwd="", *, timeout=None):
        item = {
            "index": len(self.actions),
            "action": copy.deepcopy(action),
            "cwd": cwd,
            "timeout": timeout,
            "status": "returned",
        }
        try:
            result = self.original_execute(action, cwd, timeout=timeout)
        except Exception as exc:
            item.update(status="raised", error_type=type(exc).__name__)
            # Submitted is terminal; no subsequent recoverable source boundary.
            self.actions.append(item)
            append(self.out / "replay-actions.jsonl", item)
            raise
        item.update(result_sha256=hash_json(result), returncode=result.get("returncode"))
        self.actions.append(item)
        append(self.out / "replay-actions.jsonl", item)
        return result

    def before_query(self, agent):
        try:
            state = fingerprint(self.env, self.settings)
        except ReplayMismatch as exc:
            # Mother collection continues; this boundary is explicitly unusable.
            append(
                self.out / "replay-boundaries.jsonl",
                {
                    "step": agent.n_calls,
                    "message_index": len(agent.messages),
                    "action_count": len(self.actions),
                    "unverifiable": str(exc)[:1000],
                    "state": {"changed": {}, "removed": [], "git": {}},
                    "state_sha256": None,
                },
            )
            return
        changed = {
            k: v for k, v in state["files"].items() if not self.previous or self.previous["files"].get(k) != v
        }
        removed = sorted(self.previous["files"].keys() - state["files"].keys()) if self.previous else []
        row = {
            "step": agent.n_calls,
            "message_index": len(agent.messages),
            "action_count": len(self.actions),
            "history_sha256": hash_json(agent.model._prepare_messages_for_api(agent.messages)),
            "state": {"changed": changed, "removed": removed, "git": state["git"]},
            "state_sha256": hash_json(state),
            "agent": {
                "n_calls": agent.n_calls,
                "cost": logical_cost(agent.messages),
                "format_errors": agent.n_consecutive_format_errors,
                "loop_history": list(agent._loop_history),
                "loop_step": agent._loop_step,
                "elapsed_seconds": time.time() - agent._start_time,
                "extra_template_vars": copy.deepcopy(agent.extra_template_vars),
                "loop_escape": asdict(agent.model._dycon_loop_escape),
            },
        }
        self.states.append(row)
        append(self.out / "replay-boundaries.jsonl", row)
        self.previous = state


def logical_cost(messages):
    total = 0.0
    for message in messages:
        extra = message.get("extra") or {}
        decision = extra.get("dycon_parallel")
        if decision:
            selected = decision.get("selected_branch_id")
            total += next(
                (b.get("cost", 0.0) for b in decision.get("candidates", []) if b["branch_id"] == selected),
                0.0,
            )
        elif message.get("role") == "assistant":
            total += extra.get("cost", 0.0)
    return total


def expand_states(boundaries):
    files = {}
    for row in boundaries:
        for path in row["state"]["removed"]:
            files.pop(path, None)
        files.update(row["state"]["changed"])
        yield {"files": dict(files), "git": row["state"]["git"], "errors": []}


def replay(env, source_dir, settings, target, receipt):
    """Replay tools only. A mismatch stops BEFORE any candidate/model request."""
    source_dir = Path(source_dir)
    boundaries = rows(source_dir / "replay-boundaries.jsonl")
    actions = (
        rows(source_dir / "replay-actions.jsonl") if (source_dir / "replay-actions.jsonl").exists() else []
    )
    if contract(env) != read_json(source_dir / "replay-contract.json"):
        raise ReplayMismatch("Initial image/bootstrap/network contract differs")
    executed = 0
    results = []
    for boundary, state in zip(boundaries, expand_states(boundaries), strict=True):
        if boundary["step"] > target:
            break
        try:
            if boundary.get("unverifiable"):
                raise ReplayMismatch("Source boundary unverified: " + boundary["unverifiable"])
            while executed < boundary["action_count"]:
                item = actions[executed]
                if item["status"] != "returned":
                    raise ReplayMismatch("Nonreturning historical action in prefix")
                result = env.execute(item["action"], item["cwd"], timeout=item["timeout"])
                if result.get("exception_info") or result.get("returncode") in {124, 137}:
                    raise ReplayMismatch("Timed out/interrupted replay action " + str(executed))
                if hash_json(result) != item["result_sha256"]:
                    raise ReplayMismatch("Tool observation mismatch at action " + str(executed))
                executed += 1
            actual = fingerprint(env, settings)
            compare_state(state, actual)
            # Metadata alone cannot verify large files changed since the image.
            if boundary["step"] and any(
                v.get("large_unverified") for v in boundary["state"]["changed"].values()
            ):
                raise ReplayMismatch("Changed large file lacks content identity")
            if hash_json(actual) != boundary["state_sha256"]:
                raise ReplayMismatch("Source state evidence hash mismatch")
            results.append({"step": boundary["step"], "eligible": True, "actions": executed})
        except Exception as exc:
            results.append({"step": boundary["step"], "eligible": False, "reason": str(exc)[:1000]})
            write_json(
                receipt,
                {"schema_version": 1, "backend": "initial_replay_v1", "boundaries": results, "passed": False},
            )
            if isinstance(exc, ReplayMismatch):
                raise
            raise ReplayMismatch(str(exc)) from exc
        if boundary["step"] == target:
            write_json(
                receipt,
                {"schema_version": 1, "backend": "initial_replay_v1", "boundaries": results, "passed": True},
            )
            return boundary
    raise ReplayMismatch("Target boundary not found")
