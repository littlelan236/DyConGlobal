"""OpenHands OpenAI request/response/event adapter; owns no ACP lifecycle."""
from __future__ import annotations

import copy
from types import SimpleNamespace

from core.action_identity import canonical_action
from core.io import hash_json
from rollout.decision.candidate import NormalizedCandidate

VERSION = "openhands-openai-v3"
CONDENSATION_PREFIX = ("You are maintaining a context-aware state summary for an interactive agent.\n"
    "You will be given a list of events corresponding to actions taken by the agent, which will include previous summaries.")
CONDENSATION_SUFFIX = "Now summarize the events using the rules above."


class OpenHandsProtocolError(ValueError):
    pass


class OpenHandsLimit(RuntimeError):
    def __init__(self, payload):
        self.messages = [payload]
        super().__init__(payload.get("content", "OpenHands generation limit"))


class OpenHandsPolicy:
    def limit_error(self, payload):
        return OpenHandsLimit(payload)

    def failure(self, candidates, records, baseline, error_type):
        limits = [c.error for c in candidates if isinstance(c.error, OpenHandsLimit)]
        if limits and len(limits) == len(candidates):
            return None, limits[0], None
        return None, error_type("No complete OpenHands candidate: " + "; ".join(
            str(c.error) for c in candidates)), None


class Response:
    """Small transport envelope matching the shared coordinator protocol."""
    def __init__(self, raw):
        self.raw = copy.deepcopy(raw)
        choices = raw.get("choices") or []
        self.choices = [SimpleNamespace(finish_reason=c.get("finish_reason")) for c in choices]

    def model_dump(self, mode="json"):
        return copy.deepcopy(self.raw)


def request_identity(request):
    return hash_json({k: v for k, v in request.items() if k not in {"stream", "stream_options"}})


def preserve_reasoning_history(request, records):
    """Restore only same-attempt winner reasoning omitted by SDK serialization."""
    known={}
    for record in records:
        if record.get('role')!='agent_decision' or not record.get('selected_response'):continue
        choices=record['selected_response'].get('choices') or []
        message=(choices[0].get('message') or {}) if choices else {}
        ids=tuple(c['id'] for c in message.get('tool_calls') or [])
        reasoning=message.get('reasoning_content')
        if ids and isinstance(reasoning,str):known[ids]=reasoning
    restored=0
    for message in request['messages']:
        if message.get('role')!='assistant':continue
        ids=tuple(c['id'] for c in message.get('tool_calls') or [])
        if ids in known and not message.get('reasoning_content'):
            message['reasoning_content']=known[ids]
            restored+=1
    return restored


def request_role(request, profile):
    if profile.get("request_roles") != "openhands-1.22.1":
        return "unqualified"
    # Pinned SDK agent calls carry the declared tool set. Condensation is a
    # separate text-only completion. Unknown request shapes stay unqualified.
    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        return "agent_decision"
    messages=request.get('messages') or []
    if len(messages)==1 and messages[0].get('role')=='user':
        content=messages[0].get('content')
        if isinstance(content,list):
            content=''.join(p.get('text','') for p in content if p.get('type')=='text')
        if (isinstance(content,str) and content.startswith(CONDENSATION_PREFIX)
                and content.rstrip().endswith(CONDENSATION_SUFFIX)
                and '<EVENT>' in content and '</EVENT>' in content):
            return 'condensation'
    metadata = request.get("metadata") or {}
    if metadata.get("dycon_request_role") in {"condensation", "auxiliary"}:
        return metadata["dycon_request_role"]
    return "unqualified"


def parse_response(response, request, *, decision_id="single", branch_id=0):
    raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else copy.deepcopy(response)
    choices = raw.get("choices") or []
    if len(choices) != 1:
        raise OpenHandsProtocolError("Expected exactly one complete response choice")
    choice = choices[0]
    reason = choice.get("finish_reason")
    loop = raw.get('dycon_reasoning_loop') or {}
    if loop.get('abort_completed'):
        raise OpenHandsLimit({'content': 'Native repeated reasoning loop detected',
            'extra': {'exit_status': 'reasoning_loop', 'limit_reason': 'native_reasoning_loop', 'reasoning_loop': loop}})
    native_reason = raw.get('dycon_native_finish_reason')
    if native_reason in {'length', 'abort'}:
        raise OpenHandsLimit({'content': 'Native generation ended: ' + native_reason,
            'extra': {'exit_status': 'LimitsExceeded', 'finish_reason': native_reason}})
    if reason in {"length", "abort", "content_filter"}:
        raise OpenHandsLimit({"content": "OpenHands response ended: " + reason,
                              "extra": {"exit_status": "LimitsExceeded", "finish_reason": reason}})
    if reason not in {"stop", "tool_calls"}:
        raise OpenHandsProtocolError("Unknown OpenHands finish reason")
    message = copy.deepcopy(choice.get("message") or {})
    if message.get("role") != "assistant":
        raise OpenHandsProtocolError("Expected assistant response")
    calls = message.get("tool_calls") or []
    offered = {t.get("function", {}).get("name") for t in request.get("tools", [])}
    ids = set()
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or call["id"] in ids:
            raise OpenHandsProtocolError("Missing or duplicate tool call ID")
        ids.add(call["id"])
        if call.get("type") != "function" or call.get("function", {}).get("name") not in offered:
            raise OpenHandsProtocolError("Response used an undeclared tool")
    if calls:
        canonical_action(message)  # Validates JSON arguments, preserving original bytes and IDs.
    elif reason == "tool_calls" or not message.get("content"):
        raise OpenHandsProtocolError("Empty or incomplete assistant response")
    candidate = NormalizedCandidate(decision_id, branch_id, request_identity(request),
                                    message.get("content"), message.get("reasoning_content"),
                                    copy.deepcopy(calls), reason, raw.get("usage") or {},
                                    raw.get("native_request_id"), VERSION, raw_response=raw)
    message["extra"] = {"response": raw, "request": copy.deepcopy(request),
                        "request_role": "agent_decision", "adapter_version": VERSION,
                        "candidate_request_sha256": candidate.request_sha256}
    return message


def canonical_trajectory(instance_id, binding, records, events, terminal):
    messages = []
    for record in records:
        if record.get("role") != "agent_decision" or not record.get("selected_response"):
            continue
        request = record.get("effective_request", record["request"])
        # The effective prefix is stored per decision, including condensations.
        # Only tool observations after the preceding assistant are projected.
        prefix = request["messages"]
        if not messages:
            messages.extend(copy.deepcopy([m for m in prefix if m["role"] in {"system", "user"}]))
        last_assistant = max((i for i, m in enumerate(prefix) if m.get("role") == "assistant"), default=-1)
        messages.extend(copy.deepcopy([m for m in prefix[last_assistant + 1:] if m.get("role") == "tool"]))
        msg = parse_response(record["selected_response"], request)
        msg["extra"].update(record.get("extra", {}))
        msg["extra"]["request"]["dycon_capture"] = record.get("capture")
        messages.append(msg)
    return {"schema_version": 2, "agent": "openhands", "adapter_version": VERSION,
            "instance_id": instance_id, "messages": messages, "raw_acp_events": events,
            "info": {"exit_status": terminal, "attempt_id": binding["attempt_id"],
                     "model_stats": {"api_calls": sum(r.get("role") == "agent_decision"
                         and bool(r.get("selected_response")) for r in records)},
                     "dycon": {"attempt_context": binding}, "terminal_is_evaluation": False}}
