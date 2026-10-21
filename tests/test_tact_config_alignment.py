"""Exercise the imported prompt contract through native formatting helpers."""

from types import SimpleNamespace

import pytest
from jinja2 import StrictUndefined, Template

from rollout.model import model_config


@pytest.mark.parametrize("size", [9999, 10000, 12345])
def test_observation_bounds_and_real_newlines(cfg, size):
    text = "H" * 5000 + "M" * max(0, size - 10000) + "T" * min(5000, size - 5000)
    output = {"output": text, "returncode": 1, "exception_info": "fixture exception"}
    rendered = Template(cfg["agent"]["observation_template"], undefined=StrictUndefined).render(output=output)
    assert "<exception>fixture exception</exception>\n<returncode>1</returncode>" in rendered
    assert "\\n" not in rendered
    if size < 10000:
        assert text in rendered and "<warning>" not in rendered
    else:
        assert "<warning>" in rendered
        assert "<output_head>\n" + text[:5000] in rendered
        assert "<output_tail>\n" + text[-5000:] in rendered
        assert f"<elided_chars>\n{size - 10000} characters elided" in rendered
        assert "M" not in rendered


@pytest.mark.parametrize("finish", ["length", "tool_calls", "stop"])
def test_native_format_error_uses_finish_reason(cfg, finish):
    pytest.importorskip("minisweagent")
    from minisweagent.exceptions import FormatError
    from minisweagent.models.litellm_model import LitellmModel

    model = LitellmModel(**model_config(cfg))
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(tool_calls=[])
    )])
    with pytest.raises(FormatError) as caught:
        model._parse_actions(response)
    text = caught.value.messages[0]["content"]
    if finish in {"length", "tool_calls"}:
        assert "Respond more concisely" in text and "exactly one bash tool call" in text
    else:
        assert "No tool calls found" in text and "Arguments:" in text


@pytest.mark.parametrize("mode", ["plain", "capture", "steering"])
def test_modes_share_explicit_sampling_and_prompt_contract(cfg, monkeypatch, mode):
    import rollout.model as module

    cfg["capture"]["enabled"] = mode == "capture"
    cfg["steering"]["enabled"] = mode == "steering"
    monkeypatch.setattr(module, "steering_payload", lambda config: mode == "steering")
    context = {"session_id": "fixture", "step_id": 0} if mode == "capture" else None
    settings = model_config(cfg, context)
    request = settings["model_kwargs"]
    assert request["timeout"] == 28800
    assert request["max_tokens"] == 81920
    assert request["presence_penalty"] == 0.0
    assert request["extra_body"]["min_p"] == 0.0
    assert request["extra_body"]["repetition_penalty"] == 1.0
    assert settings["observation_template"] == cfg["agent"]["observation_template"]
    assert settings["format_error_template"] == cfg["agent"]["format_error_template"]
