"""S7-T02: the LLM client sends the temperature of config.yaml and can ask for JSON.

Fast tests use a fake OpenAI client (no network): they check WHAT is sent. One slow test asks
the real provider (Gemini) for a JSON object, to know before S7-T04 whether it accepts the
JSON mode; run it with ``py -m pytest tests/test_llm_client.py -m slow -s -q``.
"""
import json
import types

import pytest

from src.ai.llm_client import (DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, SYSTEM_PROMPT, LLMClient,
                               is_json_mode_rejected, llm_settings)

CFG = {"llm": {"temperature": 0.0, "max_tokens": 1024}}


class FakeCompletions:
    def __init__(self, reject_json=None, answer="respuesta"):
        self.calls = []
        self.reject_json = reject_json
        self.answer = answer

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs and self.reject_json is not None:
            raise self.reject_json
        message = types.SimpleNamespace(content=self.answer)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _client(cfg=CFG, **kwargs):
    completions = FakeCompletions(**kwargs)
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    return LLMClient(client=fake, model="modelo-de-prueba", cfg=cfg), completions


# --------------------------------------------------------------------------- settings

def test_settings_come_from_the_llm_block():
    assert llm_settings({"llm": {"temperature": 0.3, "max_tokens": 50}}) == {
        "temperature": 0.3, "max_tokens": 50}
    assert llm_settings({}) == {"temperature": DEFAULT_TEMPERATURE,
                                "max_tokens": DEFAULT_MAX_TOKENS}


@pytest.mark.parametrize("block", [{"temperature": 2.5}, {"temperature": -0.1},
                                   {"max_tokens": 0}])
def test_invalid_settings_are_rejected(block):
    with pytest.raises(ValueError, match="bloque llm"):
        llm_settings({"llm": block})


def test_project_config_uses_temperature_zero():
    assert llm_settings()["temperature"] == 0.0


# --------------------------------------------------------------------------- what is sent

def test_temperature_and_max_tokens_are_sent():
    client, calls = _client()
    assert client.ask("¿Qué es el consentimiento?") == "respuesta"
    sent = calls.calls[0]
    assert sent["temperature"] == 0.0 and sent["max_tokens"] == 1024
    assert sent["model"] == "modelo-de-prueba"
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "response_format" not in sent


def test_explicit_values_win_over_config():
    completions = FakeCompletions()
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    client = LLMClient(client=fake, model="m", temperature=0.7, cfg=CFG)
    client.ask("p", max_tokens=99)
    assert completions.calls[0]["temperature"] == 0.7
    assert completions.calls[0]["max_tokens"] == 99


def test_context_is_wrapped_as_before():
    client, calls = _client()
    client.ask("PREGUNTA", context="CONTEXTO")
    assert calls.calls[0]["messages"][1]["content"] == "Contexto:\nCONTEXTO\n\nPregunta:\nPREGUNTA"


def test_model_comes_from_env_when_not_given(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-de-prueba")
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
    assert LLMClient(client=fake, cfg=CFG).model == "gemini-de-prueba"


# --------------------------------------------------------------------------- JSON mode

def test_json_mode_asks_for_a_json_object():
    client, calls = _client(answer='{"ok": true}')
    assert json.loads(client.ask("p", json_mode=True)) == {"ok": True}
    assert calls.calls[0]["response_format"] == {"type": "json_object"}
    assert calls.calls[0]["temperature"] == 0.0
    assert client.json_mode_supported is True


def test_rejected_json_mode_falls_back_and_is_remembered():
    rejection = Exception("Error code: 400 - Invalid JSON payload: unknown field response_format")
    client, calls = _client(reject_json=rejection, answer='{"ok": true}')
    assert client.ask("p", json_mode=True) == '{"ok": true}'
    assert [("response_format" in c) for c in calls.calls] == [True, False]
    assert client.json_mode_supported is False
    client.ask("p", json_mode=True)                       # it does not try again
    assert [("response_format" in c) for c in calls.calls] == [True, False, False]


def test_quota_error_in_json_mode_is_not_hidden():
    client, calls = _client(reject_json=Exception("Error code: 429 RESOURCE_EXHAUSTED quota"))
    with pytest.raises(Exception, match="429"):
        client.ask("p", json_mode=True)
    assert len(calls.calls) == 1 and client.json_mode_supported is None


@pytest.mark.parametrize("message,expected", [
    ("Error code: 400 - unknown field response_format", True),
    ("400 Invalid value for json_object", True),
    ("response_format is not supported by this model", True),
    ("Error code: 429 - quota exceeded for response_format", False),
    ("Error code: 429 RESOURCE_EXHAUSTED - invalid rate for json_object answers", False),
    ("503 UNAVAILABLE: the model is overloaded", False),
    ("400 Invalid API key", False),
])
def test_which_errors_mean_json_mode_was_rejected(message, expected):
    assert is_json_mode_rejected(Exception(message)) is expected


# --------------------------------------------------------------------------- real provider

@pytest.mark.slow
def test_real_provider_answers_a_json_object():
    client = LLMClient()
    raw = client.ask('Devuelve un objeto JSON con la clave "ok" y el valor true.', json_mode=True)
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    assert json.loads(text).get("ok") is True
    print(f"\nModo JSON aceptado por el proveedor: {client.json_mode_supported}")