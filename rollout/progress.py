"""Thread-safe phase projections; GPU work is measured at transport boundaries."""
from collections import Counter
from contextlib import contextmanager
from threading import RLock

from core.io import now, write_json
from rollout.retry import count_projection, sample_accepted


class Progress:
    def __init__(self, run, cfg, items, settled=None):
        self.run, self.cfg = run, cfg
        self.items = {i['sample_key']: i for i in items}
        self.samples = {i['sample_key']: {
            **i, 'seed_round': 0, 'effective_seed': i['sample_seed'],
            'exit_status': 'Pending', 'disposition': 'pending', 'physical_attempts': 0,
            'resample_retry_count': 0, 'infrastructure_retry_count': 0,
        } for i in items}
        self.samples.update(settled or {})
        self.phases = {}
        self.generating = Counter()
        self.lock = RLock()
        self.errors = []

    def started(self, context, retry_kind):
        with self.lock:
            key = context['sample_key']
            previous = self.samples.get(key, {})
            self.samples[key] = {
                **previous, **context, 'exit_status': 'Running', 'disposition': 'pending',
                'last_status': previous.get('exit_status'),
                'physical_attempts': previous.get('physical_attempts', 0) + 1,
                'resample_retry_count': previous.get('resample_retry_count', 0) + (retry_kind == 'resample'),
                'infrastructure_retry_count': previous.get('infrastructure_retry_count', 0) + (retry_kind == 'infrastructure'),
            }
            self._write()

    def phase(self, key, value):
        with self.lock:
            self.phases[key] = value
            self._write_live()

    def result(self, key, value):
        with self.lock:
            self.samples[key] = value
            self.phases.pop(key, None)
            self._write()

    @contextmanager
    def generation(self, key):
        with self.lock:
            self.generating[key] += 1
            self._generation_write()
        try:
            yield
        finally:
            with self.lock:
                self.generating[key] -= 1
                if not self.generating[key]:
                    del self.generating[key]
                self._generation_write()

    def _generation_write(self):
        try:
            self._write_live()
        except Exception as exc:
            self.errors.append(repr(exc))

    def _live(self):
        phases = Counter(self.phases.values())
        return {
            'ready_samples': phases['ready'],
            'running_attempts': sum(n for p, n in phases.items() if p != 'ready'),
            'active_generation_samples': len(self.generating),
            'finalizing_attempts': phases['finalizing'],
            'admission_waiting_samples': phases['admission'],
        }

    def _write_live(self):
        # Model requests publish only a small live projection, never the full cohort.
        write_json(self.run / 'rollout/phases.json', {'updated_at': now(), **self._live()})

    def _write(self):
        from runtime.containers.dependencies import failure_status

        accepted = sum(sample_accepted(self.cfg, v) for v in self.samples.values())
        failed = sum(v.get('disposition') in {'failed', 'blocked'} for v in self.samples.values())
        summary = {
            'planned_samples': len(self.items), 'accepted_samples': accepted,
            'failed_samples': failed, 'pending_samples': len(self.items) - accepted - failed,
            'physical_attempts': sum(v.get('physical_attempts', 0) for v in self.samples.values()),
            'resample_retry_count': sum(v.get('resample_retry_count', 0) for v in self.samples.values()),
            'infrastructure_retry_count': sum(v.get('infrastructure_retry_count', 0) for v in self.samples.values()),
            **self._live(),
        }
        counts = count_projection(self.run)
        for field in ('limits_exceeded_count', 'reasoning_loop_count', 'resample_trigger_count'):
            summary[field] = counts.get(field)
        write_json(self.run / 'progress.json', {
            'schema_version': 2, 'updated_at': now(), 'samples': self.samples,
            **summary, 'counts_path': counts['path'],
            'observability_errors': self.errors,
            'environment_dependency_failure_count': failure_status(self.cfg['project_root'], self.run.name)['count'],
            'phase_projection_path': str(self.run / 'rollout/phases.json'),
        })
        self._write_live()


@contextmanager
def generation_phase(runtime):
    tracker = (runtime or {}).get('progress_tracker')
    key = (runtime or {}).get('sample_key')
    if tracker is not None and key is not None:
        with tracker.generation(key):
            yield
    else:
        yield
