"""Frozen, run-local circuit breaker for repeated infrastructure startup failures."""
class StartupFailureGate:
    def __init__(self, limit):
        self.limit = limit
        self.consecutive = 0
        self.seen = set()
        self.failures = []
        self.blocked = False

    def observe(self, value):
        status = value.get('exit_status')
        if status in {'Pending', 'AdmissionWaiting', 'Cancelled'}:
            return False
        identity = (value.get('sample_key'), value.get('attempt_id'))
        if identity in self.seen:
            return self.blocked
        self.seen.add(identity)
        failed = (value.get('failure_type') == 'dependency_error' or status in
                  {'EnvironmentSetupFailed', 'WorkerFailure', 'CaptureError', 'CleanupError', 'ContainerCorrupted'})
        self.consecutive = self.consecutive + 1 if failed else 0
        if failed:
            self.failures.append({'sample_key': value.get('sample_key'), 'attempt_id': value.get('attempt_id'),
                                  'exit_status': status, 'failure_type': value.get('failure_type')})
        self.blocked = self.blocked or bool(self.limit and self.consecutive >= self.limit)
        return self.blocked

    def receipt(self, run_id):
        return {'schema_version': 1, 'run_id': run_id, 'limit': self.limit,
                'consecutive_startup_failures': self.consecutive, 'failures': self.failures,
                'blocked': self.blocked,
                'scope': 'stop dispatching this run; finish already owned attempts; preserve pending samples',
                'model_limits_protocol_and_reasoning_loops_are_startup_failures': False}
