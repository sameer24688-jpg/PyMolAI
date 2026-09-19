"""Per-provider API key storage (keyring + env), ONLYOFFICE-style."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from .models import DEFAULT_MODEL
from .providers import (
    ProviderSpec,
    get_provider_spec,
    resolve_openai_compat_base_url,
)

SERVICE_NAME = "pymol.ai"
_ENV_KEY_SOURCE_SAVED = "saved_keyring"


class ProviderKeyStoreError(RuntimeError):
    pass


class ProviderKeyValidationError(ProviderKeyStoreError):
    pass


@dataclass
class ProviderKeyStatus:
    provider_id: str
    has_key: bool
    source: Literal["env", "saved", "none"]
    masked_key: str
    keyring_available: bool


def _sanitize_error_message(text: str, key: str) -> str:
    message = str(text or "").strip() or "Unknown error"
    if key:
        message = message.replace(key, "***")
    return message


def _mask_key(key: str) -> str:
    raw = str(key or "").strip()
    if not raw:
        return ""
    suffix = raw[-4:] if len(raw) >= 4 else raw
    return "****%s" % (suffix,)


def _load_keyring() -> Tuple[Optional[object], bool]:
    try:
        import keyring  # type: ignore[import-not-found]
    except Exception:
        return None, False

    try:
        backend = keyring.get_keyring()
        priority = float(getattr(backend, "priority", 0.0))
    except Exception:
        priority = 0.0
    return keyring, bool(priority > 0.0)


def _require_keyring() -> object:
    keyring_mod, available = _load_keyring()
    if keyring_mod is None:
        raise ProviderKeyStoreError(
            "Secure key storage requires the 'keyring' package, but it is unavailable."
        )
    if not available:
        raise ProviderKeyStoreError(
            "No system keyring backend is available. Configure an OS keychain and retry."
        )
    return keyring_mod


def _spec(provider_id: Optional[str] = None) -> ProviderSpec:
    return get_provider_spec(provider_id)


def _env_key_for(spec: ProviderSpec) -> str:
    primary = str(os.getenv(spec.key_env) or "").strip()
    if primary:
        return primary
    # OpenRouter historically also accepted ANTHROPIC_AUTH_TOKEN.
    if spec.id == "openrouter":
        return str(os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
    if spec.id == "anthropic":
        return str(os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
    return ""


def _get_saved_key(spec: ProviderSpec) -> str:
    keyring_mod, available = _load_keyring()
    if keyring_mod is None or not available:
        return ""
    try:
        value = keyring_mod.get_password(SERVICE_NAME, spec.keyring_account)
    except Exception:
        return ""
    return str(value or "").strip()


def get_status(provider_id: Optional[str] = None) -> ProviderKeyStatus:
    spec = _spec(provider_id)
    env_key = _env_key_for(spec)
    _, keyring_available = _load_keyring()
    if env_key:
        return ProviderKeyStatus(
            provider_id=spec.id,
            has_key=True,
            source="env",
            masked_key=_mask_key(env_key),
            keyring_available=keyring_available,
        )
    saved_key = _get_saved_key(spec)
    if saved_key:
        return ProviderKeyStatus(
            provider_id=spec.id,
            has_key=True,
            source="saved",
            masked_key=_mask_key(saved_key),
            keyring_available=keyring_available,
        )
    return ProviderKeyStatus(
        provider_id=spec.id,
        has_key=False,
        source="none",
        masked_key="",
        keyring_available=keyring_available,
    )


def save_key(provider_id: str, key: str) -> None:
    spec = _spec(provider_id)
    value = str(key or "").strip()
    if not value:
        raise ProviderKeyStoreError("API key cannot be empty.")
    keyring_mod = _require_keyring()
    try:
        keyring_mod.set_password(SERVICE_NAME, spec.keyring_account, value)
    except Exception as exc:  # noqa: BLE001
        raise ProviderKeyStoreError("Failed to save API key to system keychain.") from exc


def clear_saved_key(provider_id: str) -> None:
    spec = _spec(provider_id)
    keyring_mod = _require_keyring()
    try:
        keyring_mod.delete_password(SERVICE_NAME, spec.keyring_account)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc or "").lower()
        if "not found" in msg or "no such password" in msg:
            return
        raise ProviderKeyStoreError("Failed to clear API key from system keychain.") from exc


def load_saved_key_into_env_if_needed(provider_id: Optional[str] = None) -> ProviderKeyStatus:
    spec = _spec(provider_id)
    env_key = _env_key_for(spec)
    _, keyring_available = _load_keyring()
    if env_key:
        return ProviderKeyStatus(
            provider_id=spec.id,
            has_key=True,
            source="env",
            masked_key=_mask_key(env_key),
            keyring_available=keyring_available,
        )

    saved_key = _get_saved_key(spec)
    if not saved_key:
        os.environ.pop(spec.key_source_env, None)
        return ProviderKeyStatus(
            provider_id=spec.id,
            has_key=False,
            source="none",
            masked_key="",
            keyring_available=keyring_available,
        )

    os.environ[spec.key_env] = saved_key
    os.environ[spec.key_source_env] = _ENV_KEY_SOURCE_SAVED
    return ProviderKeyStatus(
        provider_id=spec.id,
        has_key=True,
        source="saved",
        masked_key=_mask_key(saved_key),
        keyring_available=keyring_available,
    )


def clear_saved_key_and_loaded_env_if_needed(provider_id: str) -> bool:
    spec = _spec(provider_id)
    saved_key = _get_saved_key(spec)
    clear_saved_key(provider_id)

    current = str(os.getenv(spec.key_env) or "").strip()
    source = str(os.getenv(spec.key_source_env) or "").strip()
    if saved_key and current and source == _ENV_KEY_SOURCE_SAVED and current == saved_key:
        os.environ.pop(spec.key_env, None)
        os.environ.pop(spec.key_source_env, None)
        return True
    return False


def load_all_saved_keys_into_env() -> List[ProviderKeyStatus]:
    """Load keyring keys for every builtin provider when env is unset."""
    statuses: List[ProviderKeyStatus] = []
    for pid in (
        "openrouter",
        "fireworks",
        "anthropic",
        "openai",
        "deepseek",
        "kimi",
        "custom",
    ):
        statuses.append(load_saved_key_into_env_if_needed(pid))
    return statuses


def resolve_api_key(provider_id: Optional[str] = None) -> str:
    spec = _spec(provider_id)
    load_saved_key_into_env_if_needed(spec.id)
    return _env_key_for(spec)


def _model_looks_compatible(spec: ProviderSpec, model: str) -> bool:
    mid = str(model or "").strip()
    if not mid:
        return False
    if spec.id == "fireworks":
        return mid.startswith("accounts/") or "/" not in mid
    if spec.id == "openrouter":
        return "/" in mid or mid.startswith("openrouter/")
    if spec.id == "anthropic":
        return mid.startswith("claude-")
    if spec.id == "openai":
        return mid.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
    if spec.id == "deepseek":
        return mid.startswith("deepseek")
    if spec.id == "kimi":
        return mid.startswith(("moonshot", "kimi"))
    return True


def resolve_validation_model(provider_id: str, model: str = "") -> str:
    """Pick a provider-local model for key tests; ignore cross-provider leftovers."""
    spec = _spec(provider_id)
    candidate = str(model or "").strip()
    if candidate and _model_looks_compatible(spec, candidate):
        return candidate
    return str(spec.default_model or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def validate_key_live(
    provider_id: str,
    key: str,
    model: str = "",
    timeout_sec: float = 10.0,
) -> None:
    spec = _spec(provider_id)
    value = str(key or "").strip()
    if not value:
        raise ProviderKeyValidationError("API key is empty.")
    model_id = resolve_validation_model(spec.id, model)

    if spec.api_style == "openai_compat" or spec.id in ("openrouter", "fireworks"):
        _validate_openai_compat(spec, value, model_id, timeout_sec)
        return
    _validate_anthropic_compat(spec, value, model_id, timeout_sec)


def _validate_openai_compat(spec: ProviderSpec, key: str, model: str, timeout_sec: float) -> None:
    client = None
    try:
        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise ProviderKeyValidationError(
                "Live validation requires the 'openai' package, which is unavailable."
            ) from exc

        base_url = resolve_openai_compat_base_url(spec)
        client = OpenAI(api_key=key, base_url=base_url, timeout=float(max(0.1, timeout_sec)))
        # Prefer models.list for auth checks so stale/retired chat model IDs do not
        # fail a valid key (common on Fireworks serverless churn).
        listed = False
        models_api = getattr(client, "models", None)
        list_fn = getattr(models_api, "list", None) if models_api is not None else None
        if callable(list_fn):
            try:
                list_fn()
                listed = True
            except Exception:
                listed = False
        if listed:
            return
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
    except ProviderKeyValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ProviderKeyValidationError(_sanitize_error_message(str(exc), key)) from exc
    finally:
        closer = getattr(client, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def _validate_anthropic_compat(spec: ProviderSpec, key: str, model: str, timeout_sec: float) -> None:
    try:
        import urllib.error
        import urllib.request
    except Exception as exc:  # noqa: BLE001
        raise ProviderKeyValidationError("urllib is unavailable for Anthropic key validation.") from exc

    import json

    url = spec.effective_base_url().rstrip("/") + "/v1/messages"
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "authorization": "Bearer %s" % (key,),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(max(0.1, timeout_sec))) as resp:
            resp.read(256)
    except urllib.error.HTTPError as exc:
        # 401/403 => bad key; other 4xx may still prove reachability with auth accepted partially
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = str(exc)
        if exc.code in (401, 403):
            raise ProviderKeyValidationError(_sanitize_error_message(body or str(exc), key)) from exc
        # 400 with auth present often means key accepted but request shape rejected — treat as OK for ping
        if exc.code == 400:
            return
        raise ProviderKeyValidationError(_sanitize_error_message(body or str(exc), key)) from exc
    except Exception as exc:  # noqa: BLE001
        raise ProviderKeyValidationError(_sanitize_error_message(str(exc), key)) from exc
