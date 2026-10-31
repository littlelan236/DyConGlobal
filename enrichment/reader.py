from __future__ import annotations

from pathlib import Path

from core.io import read_json


class TrajectoryReader:
    def __init__(self, path: Path, *, hydrate_responses: bool = True):
        self.path = Path(path)
        self.data = read_json(self.path)
        self.messages = self.data.get("messages", [])
        if not hydrate_responses:
            # Compact canonical responses retain usage; reports need no native token arrays.
            return
        from hidden.artifacts import read_request, response_records

        for _, message, response in response_records(self.data):
            if response.get("ledger_ref"):
                message["extra"]["response"] = read_request(response["ledger_root"], response)

    def token_steps(self):
        """Every completed response, including native FormatError evidence."""
        from hidden.artifacts import response_records

        action_step = 0
        for query_index, (index, message, response) in enumerate(response_records(self.data)):
            actual = (response.get("choices") or [{}])[0].get("message") or message
            valid = message.get("role") == "assistant"
            yield {
                "step": action_step,
                "query_index": query_index,
                "message_index": index,
                "reasoning": actual.get("reasoning_content") or "",
                "content": actual.get("content") or "",
                "message": message,
                "tool_calls": actual.get("tool_calls") or [],
                "observations": [],
                "response": response,
                "valid_action": valid,
            }
            action_step += int(valid)

    def steps(self):
        ordinal = 0
        for index, message in enumerate(self.messages):
            if message.get("role") != "assistant":
                continue
            observations = []
            for following in self.messages[index + 1 :]:
                if following.get("role") == "assistant":
                    break
                if following.get("role") == "tool":
                    observations.append(following)
            yield {
                "step": ordinal,
                "message_index": index,
                "reasoning": message.get("reasoning_content") or "",
                "content": message.get("content") or "",
                "message": message,
                "tool_calls": message.get("tool_calls") or [],
                "observations": observations,
                "response": message.get("extra", {}).get("response"),
            }
            ordinal += 1

    def responses(self):
        seen = set()
        for index, message in enumerate(self.messages):
            response = (message.get("extra") or {}).get("response")
            if not isinstance(response, dict):
                if response is not None:
                    yield {"message_index": index, "response": None, "malformed": True}
                continue
            # A response ID identifies one persisted call; without it, preserve each record.
            key = response.get("id")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            yield {"message_index": index, "response": response, "malformed": False}

    def prepared_prefix(self, message_index: int) -> list[dict]:
        if self.data.get("agent") == "openhands":
            request = self.messages[message_index].get("extra", {}).get("request", {})
            if not isinstance(request.get("messages"), list):
                raise ValueError("OpenHands canonical request prefix is unavailable")
            return request["messages"]
        # OpenAI text-only runtime: no cache-control or multimodal rewriting enabled.
        prefix = [{k: v for k, v in m.items() if k != "extra"} for m in self.messages[:message_index]]
        if any(m.get("role") == "exit" for m in prefix):
            raise ValueError("A terminal message cannot occur before an assistant request")
        return prefix

    def usage(self) -> dict:
        prompt, completion, missing, count = 0, 0, 0, 0
        for item in self.responses():
            count += 1
            usage = (item["response"] or {}).get("usage") or {}
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if type(pt) is int and pt >= 0:
                prompt += pt
            if type(ct) is int and ct >= 0:
                completion += ct
            if type(pt) is not int or type(ct) is not int or item["malformed"]:
                missing += 1
        calls = self.data.get("info", {}).get("model_stats", {}).get("api_calls")
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "persisted_responses": count,
            "missing_usage": missing,
            "api_calls": calls,
            "response_coverage_complete": missing == 0 and count > 0 and calls == count,
            "coverage_scope": "persisted agent query responses; unreturned HTTP retries unknown",
        }
