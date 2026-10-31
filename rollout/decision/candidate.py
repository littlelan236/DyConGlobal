"""Stable normalized candidate and winner contracts shared by both adapters."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizedCandidate:
    decision_id: str
    branch_id: int
    request_sha256: str
    content: object
    reasoning: object
    tool_calls: list
    finish_reason: str
    usage: dict
    native_request_id: str | None
    adapter_version: str
    schema_version: int = 1
    raw_response: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WinnerReceipt:
    decision_id: str
    branch_id: int
    request_sha256: str
    response_sha256: str
    adapter_version: str
    schema_version: int = 1
