"""Durable completion-token usage and reservations for one physical attempt.

The class and configuration field retain their historical sample-level names.
Only schema 2 attempt ledgers are read; legacy sample ledgers are evidence only.
"""
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
                        ('run_id', 'task_key', 'sample_key', 'repetition_id', 'attempt_id')}
        self.attempt = self.binding['attempt_id']
        self.path = self.lock_path = None
        if self.limit is not None:
            if type(self.limit) is not int or self.limit < 1:
                raise ValueError('sample_completion_token_budget must be a positive integer')
            if not all(self.binding.values()):
                raise ValueError('Physical attempt token budget requires complete bindings')
            for value in self.binding.values():
                identifier(value)
            self.path = (run_path(cfg, self.binding['run_id']) / 'rollout/sample-token-budgets'
                         / self.binding['sample_key'] / (self.attempt + '.json'))
            self.lock_path = self.path.with_suffix('.lock')

    def _read(self):
        if self.path.exists():
            row = read_json(self.path)
            self._validate(row)
            return row
        return {'schema_version': 2, 'scope': 'physical_attempt',
                'binding': dict(self.binding), 'limit': self.limit,
                'completion_tokens': 0, 'by_role': {}, 'requests': {}, 'created_at': now()}

    def _validate(self, row):
        """Fail closed on corrupt state, including evidence from another attempt."""
        if (not isinstance(row, dict) or type(row.get('schema_version')) is not int
                or row['schema_version'] != 2 or row.get('scope') != 'physical_attempt'):
            raise ValueError('Invalid physical attempt token budget schema or scope')
        if (row.get('binding') != self.binding or type(row.get('limit')) is not int
                or row['limit'] != self.limit):
            raise ValueError('Physical attempt token budget binding or frozen limit changed')
        if (type(row.get('completion_tokens')) is not int or row['completion_tokens'] < 0
                or not isinstance(row.get('requests'), dict)
                or not isinstance(row.get('by_role'), dict)
                or type(row.get('usage_unknown', False)) is not bool):
            raise ValueError('Invalid physical attempt token budget counters')
        total, by_role, reserved = 0, {}, 0
        for api_id, request in row['requests'].items():
            identifier(api_id)
            if not isinstance(request, dict) or request.get('attempt_id') != self.attempt:
                raise ValueError('Physical attempt request binding changed')
            role = identifier(request.get('role'))
            if type(request.get('reserved_tokens')) is not int or request['reserved_tokens'] < 1:
                raise ValueError('Invalid physical attempt reservation')
            status = request.get('status')
            if status == 'completed':
                tokens = request.get('completion_tokens')
                if type(tokens) is not int or tokens < 0:
                    raise ValueError('Invalid physical attempt usage receipt')
                total += tokens
                by_role[role] = by_role.get(role, 0) + tokens
            elif status == 'reserved':
                reserved += request['reserved_tokens']
            elif status == 'usage_unknown':
                if not row.get('usage_unknown') or request.get('completion_tokens') is not None:
                    raise ValueError('Invalid physical attempt unknown usage receipt')
            else:
                raise ValueError('Invalid physical attempt request status')
        if (total != row['completion_tokens'] or by_role != row['by_role']
                or any(type(value) is not int for value in row['by_role'].values())
                or reserved > self.limit):
            raise ValueError('Inconsistent physical attempt token budget counters')

    def _prior(self, row, api_id, role):
        prior = row['requests'].get(api_id)
        if prior is None:
            raise ValueError('Physical API attempt has no reservation in this ledger')
        if prior['role'] != role or prior['attempt_id'] != self.attempt:
            raise ValueError('Physical attempt usage receipt binding changed')
        return prior

    @staticmethod
    def _pending(row):
        return [r for r in row['requests'].values() if r.get('status') == 'reserved']

    def reserve(self, api_id, role, requested, request_id=None):
        """Return None while other candidates hold the remaining allowance."""
        if self.limit is None:
            return requested
        identifier(api_id)
        identifier(role)
        if type(requested) is not int or requested < 1:
            raise ValueError('Requested completion tokens must be a positive integer')
        with file_lock(self.lock_path):
            row = self._read()
            pending = self._pending(row)
            if api_id in row['requests']:
                raise ValueError('Physical API attempt cannot be admitted twice')
            if row.get('usage_unknown'):
                raise BudgetStopped('sample_token_usage_unknown')
            if row['completion_tokens'] >= self.limit:
                raise BudgetStopped('sample_completion_token_budget')
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
            prior = self._prior(row, api_id, role)
            if prior.get('status') == 'completed':
                if prior['role'] != role or prior['completion_tokens'] != tokens:
                    raise ValueError('Conflicting physical attempt usage receipt')
                return row
            if prior.get('status') != 'reserved':
                raise ValueError('Physical attempt usage cannot replace an unknown receipt')
            row['requests'][api_id] = {**prior, 'role': role, 'attempt_id': self.attempt,
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
            prior = self._prior(row, api_id, role)
            if prior.get('status') in {'completed', 'usage_unknown'}:
                return
            row['requests'][api_id] = {**prior, 'role': role, 'attempt_id': self.attempt,
                'completion_tokens': None, 'status': 'usage_unknown', 'recorded_at': now()}
            row['usage_unknown'] = True
            write_json(self.path, row)

    def remaining(self):
        row = self.snapshot()
        if row is None:
            return None
        if row.get('usage_unknown'):
            return 0
        reserved = sum(r['reserved_tokens'] for r in self._pending(row))
        return max(0, self.limit - row['completion_tokens'] - reserved)

    def snapshot(self):
        if self.limit is None:
            return None
        with file_lock(self.lock_path):
            return self._read()
