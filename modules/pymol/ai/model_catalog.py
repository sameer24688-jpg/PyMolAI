"""Remote/static model catalogs per LLM provider (Phase 2)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .providers import ProviderSpec, get_provider_spec, resolve_openai_compat_base_url


class ModelCatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    display_name: str
    provider_id: str = ""

    def as_pair(self) -> Tuple[str, str]:
        return (self.model_id, self.display_name or self.model_id)


ANTHROPIC_STATIC: Tuple[Tuple[str, str], ...] = (
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5", "Claude Haiku 4.5"),
    ("claude-opus-4-6", "Claude Opus 4.6"),
)

OPENROUTER_FAVORITES: Tuple[Tuple[str, str], ...] = (
    ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
    ("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6"),
    ("z-ai/glm-5", "GLM-5"),
    ("minimax/minimax-m2.5", "MiniMax M2.5"),
    ("moonshotai/kimi-k2.5", "Kimi K2.5"),
    ("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
    ("openai/gpt-5.2", "GPT-5.2"),
)

FIREWORKS_FAVORITES: Tuple[Tuple[str, str], ...] = (
    ("accounts/fireworks/models/glm-5p2", "GLM 5.2"),
    ("accounts/fireworks/models/gpt-oss-120b", "GPT-OSS 120B"),
    ("accounts/fireworks/models/deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("accounts/fireworks/models/llama-v3p3-70b-instruct", "Llama 3.3 70B"),
)

OPENAI_FAVORITES: Tuple[Tuple[str, str], ...] = (
    ("gpt-4o-mini", "GPT-4o mini"),
    ("gpt-4o", "GPT-4o"),
    ("gpt-4.1-mini", "GPT-4.1 mini"),
    ("o4-mini", "o4-mini"),
)

DEEPSEEK_FAVORITES: Tuple[Tuple[str, str], ...] = (
    ("deepseek-chat", "DeepSeek Chat"),
    ("deepseek-reasoner", "DeepSeek Reasoner"),
)

KIMI_FAVORITES: Tuple[Tuple[str, str], ...] = (
    ("moonshot-v1-auto", "Moonshot Auto"),
    ("kimi-k2.5", "Kimi K2.5"),
    ("moonshot-v1-128k", "Moonshot 128k"),
)


def favorites_for_provider(provider_id: str) -> List[ModelEntry]:
    pid = str(provider_id or "openrouter").strip().lower()
    mapping = {
        "openrouter": OPENROUTER_FAVORITES,
        "fireworks": FIREWORKS_FAVORITES,
        "anthropic": ANTHROPIC_STATIC,
        "openai": OPENAI_FAVORITES,
        "deepseek": DEEPSEEK_FAVORITES,
        "kimi": KIMI_FAVORITES,
        "custom": (),
    }
    rows = mapping.get(pid, OPENROUTER_FAVORITES)
    return [ModelEntry(model_id=mid, display_name=name, provider_id=pid) for mid, name in rows]


def _http_get_json(url: str, *, api_key: str = "", timeout_sec: float = 20.0) -> object:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % (api_key,)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=float(max(0.1, timeout_sec))) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = str(exc)
        raise ModelCatalogError("HTTP %s: %s" % (exc.code, body[:400])) from exc
    except Exception as exc:  # noqa: BLE001
        raise ModelCatalogError(str(exc)) from exc
    try:
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise ModelCatalogError("Invalid JSON from models endpoint.") from exc


def _display_name_from_id(model_id: str) -> str:
    text = str(model_id or "").strip()
    if not text:
        return ""
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.replace("-", " ").replace("_", " ").strip() or model_id


def _parse_openai_models_payload(payload: object, provider_id: str) -> List[ModelEntry]:
    rows: List[ModelEntry] = []
    data = []
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("models") or []
    elif isinstance(payload, list):
        data = payload
    if not isinstance(data, list):
        return rows
    for item in data:
        if isinstance(item, str):
            mid = item.strip()
            if mid:
                rows.append(ModelEntry(mid, _display_name_from_id(mid), provider_id))
            continue
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or item.get("name") or "").strip()
        if not mid:
            continue
        name = str(item.get("name") or item.get("display_name") or "").strip() or _display_name_from_id(mid)
        rows.append(ModelEntry(mid, name, provider_id))
    return rows


def _parse_fireworks_models_payload(payload: object, provider_id: str) -> List[ModelEntry]:
    rows: List[ModelEntry] = []
    models = []
    if isinstance(payload, dict):
        models = payload.get("models") or payload.get("data") or []
    elif isinstance(payload, list):
        models = payload
    if not isinstance(models, list):
        return rows
    for item in models:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("name") or item.get("id") or "").strip()
        if not mid:
            continue
        # Fireworks returns resource names; normalize to inference ids when needed.
        if mid.startswith("accounts/") and "/models/" in mid:
            pass
        elif "/" not in mid:
            mid = "accounts/fireworks/models/%s" % (mid,)
        name = str(item.get("displayName") or item.get("display_name") or "").strip()
        if not name:
            name = _display_name_from_id(mid)
        rows.append(ModelEntry(mid, name, provider_id))
    return rows


def fetch_provider_catalog(
    provider_id: str,
    *,
    api_key: str = "",
    timeout_sec: float = 20.0,
    http_get_json: Optional[Callable[..., object]] = None,
) -> List[ModelEntry]:
    """Fetch remote catalog. Failures raise ModelCatalogError (caller must not wipe local state)."""
    spec = get_provider_spec(provider_id)
    getter = http_get_json or _http_get_json
    pid = spec.id

    if pid == "anthropic":
        return list(favorites_for_provider(pid))

    if pid == "fireworks":
        url = (
            "https://api.fireworks.ai/v1/accounts/fireworks/models"
            "?filter=supports_serverless%3Dtrue&pageSize=200"
        )
        payload = getter(url, api_key=api_key, timeout_sec=timeout_sec)
        rows = _parse_fireworks_models_payload(payload, pid)
        if not rows:
            raise ModelCatalogError("Fireworks catalog returned no models.")
        return rows

    if spec.api_style == "openai_compat" or pid == "openrouter":
        base = resolve_openai_compat_base_url(spec).rstrip("/")
        if not base:
            raise ModelCatalogError("Provider base URL is empty; set a Custom URL first.")
        url = base + "/models"
        payload = getter(url, api_key=api_key, timeout_sec=timeout_sec)
        rows = _parse_openai_models_payload(payload, pid)
        if not rows:
            raise ModelCatalogError("Models endpoint returned no models.")
        return rows

    raise ModelCatalogError("Catalog refresh is not supported for this provider.")


def merge_favorites_and_catalog(
    provider_id: str,
    catalog: Optional[List[ModelEntry]] = None,
) -> List[ModelEntry]:
    """Favorites first, then catalog entries not already present."""
    merged: List[ModelEntry] = []
    seen = set()
    for entry in favorites_for_provider(provider_id) + list(catalog or []):
        mid = str(entry.model_id or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        merged.append(
            ModelEntry(
                model_id=mid,
                display_name=str(entry.display_name or mid),
                provider_id=str(entry.provider_id or provider_id),
            )
        )
    return merged


def filter_models(entries: List[ModelEntry], query: str) -> List[ModelEntry]:
    q = str(query or "").strip().lower()
    if not q:
        return list(entries)
    out: List[ModelEntry] = []
    for entry in entries:
        hay = ("%s %s" % (entry.model_id, entry.display_name)).lower()
        if q in hay:
            out.append(entry)
    return out
