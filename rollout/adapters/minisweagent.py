"""Preserve mini-swe-agent's exact exception and format-correction semantics."""
class MiniPolicy:
    def limit_error(self, payload):
        from minisweagent.exceptions import LimitsExceeded
        return LimitsExceeded(payload)

    def failure(self, candidates, records, baseline, error_type):
        from litellm.exceptions import ContextWindowExceededError
        from minisweagent.exceptions import FormatError, LimitsExceeded
        errors = [c for c in candidates if isinstance(c.error, FormatError)]
        branch_errors = [c.error for c in candidates]
        if all(isinstance(e, (ContextWindowExceededError, LimitsExceeded)) for e in branch_errors):
            selected = next((e for e in branch_errors if isinstance(e, ContextWindowExceededError)),
                            branch_errors[0])
            if all(row["reason"] == "reasoning_loop" for row in records):
                selected = self.limit_error({
                    "role": "exit", "content": "All parallel branches ended in reasoning loops",
                    "extra": {"exit_status": "reasoning_loop" if baseline else "LimitsExceeded",
                              "submission": "", "limit_reason": "all_parallel_branches_reasoning_loop"}})
            return None, selected, None
        if errors:
            return min(errors, key=lambda c: c.branch_id), None, "all_branches_format_error"
        return None, error_type("All parallel API requests failed"), None
