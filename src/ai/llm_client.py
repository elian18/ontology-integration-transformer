"""Client of the LLM (Sprint 0, extended in S7-T02).

OpenAI-compatible API (today Gemini, model read from .env: LLM_MODEL, LLM_API_KEY,
LLM_BASE_URL). Two changes in S7-T02:
- TEMPERATURE: it now sends ``llm.temperature`` from config.yaml (0.0), so the same prompt with
  the same input gives (as far as the provider allows) the same answer. Until Sprint 6 the
  value was in config.yaml but never sent, so the provider's default was used.
- JSON ANSWERS: ``ask(..., json_mode=True)`` asks the provider for one JSON object
  (``response_format={"type": "json_object"}``). If the provider rejects that option, the
  client remembers it and asks again without it, so the caller's tolerant parser still works.
  The prompts of Sprints 3-6 (extract, justify) ask for JSONL and keep ``json_mode=False``;
  the decider of S7-T04 uses JSON mode.

The OpenAI client can be injected (``client=``), so the tests use a fake and never touch the
network.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = "Responde únicamente con base en el contexto entregado."
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
_JSON_FORMAT = {"type": "json_object"}


def llm_settings(cfg: dict | None = None) -> dict:
    """``llm`` block of config.yaml: temperature (0 to 2) and max_tokens."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("llm", {}) or {}
    temperature = float(block.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(block.get("max_tokens", DEFAULT_MAX_TOKENS))
    if not 0.0 <= temperature <= 2.0 or max_tokens < 1:
        raise ValueError("Valores inválidos en el bloque llm de config.yaml "
                         "(temperature entre 0 y 2, max_tokens 1 o más).")
    return {"temperature": temperature, "max_tokens": max_tokens}


def _is_quota_or_server_error(message: str) -> bool:
    m = message.lower()
    return any(k in m for k in ("429", "resource_exhausted", "quota", "503", "500",
                                "unavailable", "overloaded", "timeout", "timed out"))


def is_json_mode_rejected(exc: Exception) -> bool:
    """True if the provider refused the JSON-mode option itself (not a quota or server error)."""
    message = str(exc)
    if _is_quota_or_server_error(message):
        return False
    m = message.lower()
    return ("response_format" in m or "json" in m) and any(
        k in m for k in ("400", "invalid", "unsupported", "not supported", "unknown"))


class LLMClient:
    def __init__(self, client=None, model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, cfg: dict | None = None):
        settings = llm_settings(cfg)
        self.client = client if client is not None else OpenAI(
            api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
        self.model = model or os.environ["LLM_MODEL"]
        self.temperature = settings["temperature"] if temperature is None else float(temperature)
        self.max_tokens = settings["max_tokens"] if max_tokens is None else int(max_tokens)
        self.json_mode_supported: bool | None = None     # unknown until the first JSON request

    def ask(self, prompt: str, context: str = "", max_tokens: int | None = None,
            json_mode: bool = False) -> str:
        content = f"Contexto:\n{context}\n\nPregunta:\n{prompt}" if context else prompt
        request = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
        }
        if json_mode and self.json_mode_supported is not False:
            try:
                resp = self.client.chat.completions.create(**request,
                                                           response_format=_JSON_FORMAT)
                self.json_mode_supported = True
                return resp.choices[0].message.content
            except Exception as exc:                    # noqa: BLE001 - decide below
                if not is_json_mode_rejected(exc):
                    raise
                self.json_mode_supported = False         # ask again, without the option
        resp = self.client.chat.completions.create(**request)
        return resp.choices[0].message.content