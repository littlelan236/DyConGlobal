"""Durable sample usage and conservative reservations across physical attempts."""
from __future__ import annotations

from core.io import identifier, now, read_json, run_path, write_json
from core.locking import file_lock


class BudgetStopped(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class SampleTokenBudget:
    def __init__(self, cfg, binding):
        self.limit = cfg.get('execution', {}).get('sample_completion_token_budget')
        self.binding = {k: binding.get(k) for k in
                        ('run_id', 'task_key', 'sample_key', 'repetition_id')}
        self.attempt = binding.get('attempt_id')
        self.path = self.lock_path = None
        if self.limit is not None:
            if type(self.limit) is not int or self.limit < 1:
                raise ValueError('sample_completion_token_budget must be a positive integer')
            if not all(self.binding.values()) or not self.attempt:
                raise ValueError('Sample token budget requires complete sample and attempt bindings')
            self.path = (run_path(cfg, self.binding['run_id']) / 'rollout/sample-token-budgets'
                         / (identifier(self.binding['sample_key']) + '.json'))
            self.lock_path = self.path.with_suffix('.lock')

    def _read(self):
        if self.path.exists():
            row = read_json(self.path)
            if row['binding'] != self.binding or row['limit'] != self.limit:
                raise ValueError('Sample token budget binding or frozen limit changed')
            return row
        return {'schema_version': 1, 'binding': self.binding, 'limit': self.limit,
                'completion_tokens': 0, 'by_role': {}, 'requests': {}, 'created_at': now()}

    @staticmethod
    def _pending(row):
        return [r for r in row['requests'].values() if r.get('status') == 'reserved']

    def reserve(self, api_id, role, requested, request_id=None):
        """Return None while other candidates hold the remaining allowance."""
        if self.limit is None:
            return requested
        with file_lock(self.lock_path):
            row = self._read()
            pending = self._pending(row)
            if row.get('usage_unknown') or any(r['attempt_id'] != self.attempt for r in pending):
                raise BudgetStopped('sample_token_usage_unknown')
            if row['completion_tokens'] >= self.limit:
                raise BudgetStopped('sample_completion_token_budget')
            if api_id in row['requests']:
                raise ValueError('Physical API attempt cannot be admitted twice')
            available = self.limit - row['completion_tokens'] - sum(r['reserved_tokens'] for r in pending)
            if available <= 0:
                return None
            amount = min(requested, available)
            row['requests'][api_id] = {'role': role, 'attempt_id': self.attempt,
                'request_id': request_id, 'status': 'reserved', 'reserved_tokens': amount,
                'started_at': now()}
            write_json(self.path, row)
            return amount

    def record(self, api_id, role, usage):
        if self.limit is None:
            return None
        tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens < 0:
            self.record_unknown(api_id, role)
            raise BudgetStopped('sample_token_usage_unknown')
        with file_lock(self.lock_path):
            row = self._read()
            prior = row['requests'].get(api_id)
            if prior and prior.get('status') == 'completed':
                if prior['role'] != role or prior['completion_tokens'] != tokens:
                    raise ValueError('Conflicting sample usage receipt')
                return row
            if prior and (prior['role'] != role or prior['attempt_id'] != self.attempt):
                raise ValueError('Sample usage receipt binding changed')
            row['requests'][api_id] = {**(prior or {}), 'role': role, 'attempt_id': self.attempt,
                'completion_tokens': tokens, 'status': 'completed', 'recorded_at': now()}
            row['completion_tokens'] += tokens
            row['by_role'][role] = row['by_role'].get(role, 0) + tokens
            row['exhausted'] = row['completion_tokens'] >= self.limit
            row['updated_at'] = now()
            write_json(self.path, row)
            return row

    def record_unknown(self, api_id, role):
        if self.limit is None:
            return
        with file_lock(self.lock_path):
            row = self._read()
            prior = row['requests'].get(api_id, {})
            if prior.get('status') == 'completed':
                return
            row['requests'][api_id] = {**prior, 'role': role, 'attempt_id': self.attempt,
                'completion_tokens': None, 'status': 'usage_unknown', 'recorded_at': now()}
            row['usage_unknown'] = True
            write_json(self.path, row)

    def remaining(self):
        row = self.snapshot()
        return None if row is None else max(0, self.limit - row['completion_tokens'])

    def snapshot(self):
        if self.limit is None:
            return None
        with file_lock(self.lock_path):
            return self._read()
