"""Recover published artifacts from local bytes without any model requests."""

from pathlib import Path

from core.config import load_snapshot
from core.io import now, read_json, sha256, write_json
from core.locking import exclusive_run, file_lock
from hidden.artifacts import read_branch_index
from runtime.serving.spool import CaptureSpool


@exclusive_run
def recover_online(cfg, run_id):
    return recover_online_service(cfg, run_id)


def recover_online_service(cfg, run_id, *, snapshot=None, checkpoint=None):
    """Caller owns the run lock; a snapshot restricts recovery to old files."""
    check = checkpoint or (lambda: None)
    frozen = load_snapshot(cfg["project_root"], run_id)
    spool = CaptureSpool(
        frozen["project_root"],
        frozen["capture"]["export"],
        run_id,
        frozen["paths"]["nas_root"],
    )
    sessions = {}
    try:
        jobs = spool.recover(snapshot=snapshot, checkpoint=checkpoint) if snapshot is not None else spool.recover()
        metadata = None
        if snapshot is not None:
            # One inventory pass, rather than full-run scans for every session.
            metadata = {}
            for path in (*snapshot['requests'], *snapshot['index']):
                check()
                row = read_json(path)
                sid = row[0].get('session_id') if isinstance(row, list) and row else row.get('session_id') if isinstance(row, dict) else None
                if sid is None or sid in snapshot['deferred_sessions']:
                    continue
                paths = metadata.setdefault(sid, set())
                paths.add(path)
                shard = (row.get('tensor_locator') or {}).get('shard') if isinstance(row, dict) else None
                if shard:
                    paths.add(spool.root / 'index' / ('shard-' + Path(shard).stem + '.json'))
        for path in (snapshot['finalizations'] if snapshot is not None else sorted((spool.root / "sessions").glob("*/finalization.json"))):
            check()
            sid = path.parent.name
            if snapshot is not None and sid in snapshot['deferred_sessions']:
                continue
            finalization = read_json(path)
            from application.execution.evidence import verified_trajectory
            trajectory = verified_trajectory(frozen["project_root"], {**finalization, "producer_run_id": run_id})
            if sha256(trajectory) != finalization["trajectory_sha256"]:
                raise ValueError("Recovery trajectory hash mismatch")
            coverage = read_json(path.parent / "coverage.json")
            branch_path = path.parent / "parallel/branch-coverage.json"
            branch_complete = not branch_path.exists() or not read_branch_index(spool.root, sid)["coverage"]["missing"]
            options = {'metadata_paths': sorted(metadata.get(sid, ())), 'checkpoint': checkpoint} if metadata is not None else {}
            published = spool.publish_session(sid, frozen["capture"]["export"]["upload_max_elapsed_seconds"], **options)
            state = (
                "complete"
                if published and not coverage["missing"] and branch_complete
                else "pending_upload"
                if not published
                else "partial"
            )
            sessions[sid] = state
            with file_lock(Path(cfg["project_root"]) / "state/product-manifests.lock"):
                for owner in ("hidden", "enrichment"):
                    manifest_path = Path(cfg["project_root"]) / owner / "manifest.json"
                    manifest = read_json(manifest_path)
                    entry = manifest.get("online_captures", {}).get(sid)
                    if entry:
                        entry.update(status=state, published=state == "complete",
                                     eligible_for_training=state == "complete" and entry.get("dataset_role") != "diagnostic")
                        write_json(manifest_path, manifest)
            if state == "complete":
                spool.release_session_metadata(sid)
        run_hidden = Path(cfg["project_root"]) / "runs" / run_id / "hidden"
        run_hidden.mkdir(parents=True, exist_ok=True)
        write_json(run_hidden / "publication.json", {
            "schema_version": 1, "run_id": run_id, "sessions": sessions,
            "nas_root": frozen["paths"]["nas_root"], "updated_at": now(),
        })
        return {"retried_jobs": jobs, "sessions": sessions}
    finally:
        # On a cooperative stop retain unacknowledged bytes for the next resume.
        try:
            check()
        except BaseException:
            spool.close(0)
            # shutdown terminates upload children and cancels queued futures.
            # Their ACK callbacks must finish before releasing the run lock.
            with spool.condition:
                spool.condition.wait_for(lambda: not spool.pending)
        else:
            spool.close(frozen["capture"]["export"]["upload_max_elapsed_seconds"])



class RecoveryStopped(Exception):
    """The owning rollout is tearing down or was cancelled."""


class BackgroundCaptureRecovery:
    """Run-owned recovery; no detached writer may outlive the run lock."""

    def __init__(self, cfg, run_id, cancellation, *, enabled=True):
        import threading
        self.cfg, self.run_id, self.cancellation = cfg, run_id, cancellation
        self.run = Path(cfg['project_root']) / 'runs' / run_id
        self.enabled = enabled and cfg.get('capture', {}).get('enabled', False)
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.summary = {'schema_version': 1, 'run_id': self.run_id,
                        'execution_attempt_id': self.cancellation.attempt_id,
                        'started_at': now(), 'status': 'running'}
        try:
            self._start()
        except Exception as exc:
            self.summary.update(status='failed', error=repr(exc), finished_at=now())
            write_json(self.run / 'capture-recovery-error.json', self.summary)
            write_json(self.run / 'capture-recovery.json', self.summary)
        return self

    def _start(self):
        import threading
        root = Path(self.cfg['project_root']) / self.cfg['capture']['export']['spool_root'] / self.run_id
        # List paths before any live capture producer can create a new session.
        # Spill replay belongs to the live export queue; never seal its sessions
        # concurrently with that writer. Final publication will retry them.
        deferred = {read_json(p)['session_id'] for p in (root / 'spill').glob('*.json')}
        self.snapshot = {name: tuple(sorted((root / name).glob('*.json')))
                         for name in ('uploads', 'index', 'requests')}
        self.snapshot.update(finalizations=tuple(sorted((root / 'sessions').glob('*/finalization.json'))),
                             deferred_sessions=deferred)
        if not any(self.snapshot[k] for k in ('uploads', 'index', 'finalizations')):
            self.summary.update(status='complete', reason='no_existing_capture', finished_at=now())
            write_json(self.run / 'capture-recovery.json', self.summary)
            (self.run / 'capture-recovery-error.json').unlink(missing_ok=True)
            return
        self.summary['deferred_sessions'] = sorted(deferred)
        write_json(self.run / 'capture-recovery.json', self.summary)
        self.thread = threading.Thread(target=self._run, name='dycon-capture-recovery', daemon=True)
        self.thread.start()
        return self

    def _check(self):
        if self.stop.is_set() or self.cancellation.event.is_set():
            raise RecoveryStopped()

    def _run(self):
        try:
            result = recover_online_service(self.cfg, self.run_id, snapshot=self.snapshot, checkpoint=self._check)
            from collections import Counter
            self.summary.update(status='complete', recovered_sessions=len(result['sessions']),
                                session_states=dict(Counter(result['sessions'].values())),
                                retried_jobs=len(result['retried_jobs']))
            (self.run / 'capture-recovery-error.json').unlink(missing_ok=True)
        except RecoveryStopped:
            self.summary.update(status='stopped', reason='rollout_teardown_or_cancellation')
        except Exception as exc:
            self.summary.update(status='failed', error=repr(exc))
            write_json(self.run / 'capture-recovery-error.json', self.summary)
        finally:
            self.summary['finished_at'] = now()
            write_json(self.run / 'capture-recovery.json', self.summary)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            # Retain the run lock until the recovery writer has exited. The
            # execution supervisor bounds stuck I/O during cancellation.
            self.thread.join()

    def __exit__(self, *_):
        self.close()
