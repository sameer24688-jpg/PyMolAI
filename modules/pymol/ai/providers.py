"""LLM provider registry (ONLYOFFICE-inspired: name, url, key, addon, api_style)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

ApiStyle = Literal["anthropic_compat", "openai_compat"]
ProviderId = str

ENV_PROVIDER = "PYMOL_AI_PROVIDER"
ENV_CUSTOM_NAME = "PYMOL_AI_CUSTOM_NAME"
ENV_CUSTOM_URL = "PYMOL_AI_BASE_URL"
ENV_CUSTOM_ADDON = "PYMOL_AI_CUSTOM_ADDON"
ENV_CUSTOM_API_STYLE = "PYMOL_AI_API_STYLE"

DEFAULT_PROVIDER_ID = "openrouter"
API_STYLES: Tuple[ApiStyle, ...] = ("anthropic_compat", "openai_compat")


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    url: str
    key_env: str
    key_source_env: str
    keyring_account: str
    addon: str = ""
    api_style: ApiStyle = "anthropic_compat"
    default_model: str = ""
    builtin: bool = True

    def effective_base_url(self) -> str:
        base = str(self.url or "").rstrip("/")
        addon = str(self.addon or "").strip().strip("/")
        if not addon:
            return base
        if base.endswith("/" + addon) or base.endswith(addon):
            return base
        return "%s/%s" % (base, addon)


_BUILTINS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openrouter",
        name="OpenRouter",
        url="https://openrouter.ai/api",
        key_env="OPENROUTER_API_KEY",
        key_source_env="PYMOL_AI_OPENROUTER_KEY_SOURCE",
        keyring_account="openrouter_api_key",
        addon="",
        api_style="anthropic_compat",
        default_model="anthropic/claude-sonnet-4.6",
    ),
    ProviderSpec(
        id="fireworks",
        name="Fireworks",
        url="https://api.fireworks.ai/inference",
        key_env="FIREWORKS_API_KEY",
        key_source_env="PYMOL_AI_FIREWORKS_KEY_SOURCE",
        keyring_account="fireworks_api_key",
        addon="",
        api_style="anthropic_compat",
        default_model="accounts/fireworks/models/glm-5p2",
    ),
    ProviderSpec(
        id="anthropic",
        name="Anthropic",
        url="https://api.anthropic.com",
        key_env="ANTHROPIC_API_KEY",
        key_source_env="PYMOL_AI_ANTHROPIC_KEY_SOURCE",
        keyring_account="anthropic_api_key",
        addon="",
        api_style="anthropic_compat",
        default_model="claude-sonnet-4-6",
    ),
    ProviderSpec(
        id="openai",
        name="OpenAI",
        url="https://api.openai.com",
        key_env="OPENAI_API_KEY",
        key_source_env="PYMOL_AI_OPENAI_KEY_SOURCE",
        keyring_account="openai_api_key",
        addon="v1",
        api_style="openai_compat",
        default_model="gpt-4o-mini",
    ),
    ProviderSpec(
        id="deepseek",
        name="DeepSeek",
        url="https://api.deepseek.com",
        key_env="DEEPSEEK_API_KEY",
        key_source_env="PYMOL_AI_DEEPSEEK_KEY_SOURCE",
        keyring_account="deepseek_api_key",
        addon="",
        api_style="openai_compat",
        default_model="deepseek-chat",
    ),
    ProviderSpec(
        id="kimi",
        name="Kimi",
        url="https://api.moonshot.ai",
        key_env="MOONSHOT_API_KEY",
        key_source_env="PYMOL_AI_KIMI_KEY_SOURCE",
        keyring_account="kimi_api_key",
        addon="v1",
        api_style="openai_compat",
        default_model="moonshot-v1-auto",
    ),
    ProviderSpec(
        id="custom",
        name="Custom URL",
        url="",
        key_env="PYMOL_AI_CUSTOM_API_KEY",
        key_source_env="PYMOL_AI_CUSTOM_KEY_SOURCE",
        keyring_account="custom_api_key",
        addon="",
        api_style="openai_compat",
        default_model="",
        builtin=False,
    ),
)


def builtin_providers() -> List[ProviderSpec]:
    return list(_BUILTINS)


def provider_ids() -> List[str]:
    return [spec.id for spec in _BUILTINS]


def normalize_api_style(value: object) -> ApiStyle:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in ("anthropic", "anthropic_compat", "claude", "messages"):
        return "anthropic_compat"
    if text in ("openai", "openai_compat", "chat_completions", "chat"):
        return "openai_compat"
    return "openai_compat"


def normalize_provider_id(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return DEFAULT_PROVIDER_ID
    aliases = {
        "or": "openrouter",
        "fw": "fireworks",
        "firework": "fireworks",
        "fireworks.ai": "fireworks",
        "claude": "anthropic",
        "moonshot": "kimi",
        "moonshotai": "kimi",
    }
    text = aliases.get(text, text)
    if text in provider_ids():
        return text
    return DEFAULT_PROVIDER_ID


def _config_path() -> Path:
    override = str(os.getenv("PYMOL_AI_PROVIDER_CONFIG") or "").strip()
    if override:
        return Path(override)
    base = Path(os.getenv("APPDATA") or Path.home() / ".pymolai")
    return Path(base) / "PyMolAI" / "provider_config.json"


def _load_provider_config() -> Dict[str, object]:
    path = _config_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_provider_config(payload: Dict[str, object]) -> Path:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_custom_overrides() -> Dict[str, str]:
    payload = _load_provider_config()
    if not payload:
        return {}
    if isinstance(payload.get("custom"), dict):
        custom = payload["custom"]
    elif "active_provider" in payload or "models" in payload:
        return {}
    else:
        # Legacy: entire file was only custom fields.
        custom = payload
    if not isinstance(custom, dict):
        return {}
    return {
        "name": str(custom.get("name") or "").strip(),
        "url": str(custom.get("url") or "").strip(),
        "addon": str(custom.get("addon") or "").strip(),
        "api_style": normalize_api_style(custom.get("api_style")),
        "default_model": str(custom.get("default_model") or "").strip(),
    }


def save_custom_overrides(
    *,
    name: str = "",
    url: str = "",
    addon: str = "",
    api_style: object = "openai_compat",
    default_model: str = "",
) -> Path:
    payload = _load_provider_config()
    payload["custom"] = {
        "name": str(name or "").strip() or "Custom URL",
        "url": str(url or "").strip(),
        "addon": str(addon or "").strip(),
        "api_style": normalize_api_style(api_style),
        "default_model": str(default_model or "").strip(),
    }
    return _save_provider_config(payload)


def load_active_provider_preference() -> Optional[str]:
    payload = _load_provider_config()
    raw = str(payload.get("active_provider") or "").strip()
    if not raw:
        return None
    pid = normalize_provider_id(raw)
    return pid if pid in provider_ids() else None


def save_active_provider_preference(provider_id: str) -> Path:
    pid = normalize_provider_id(provider_id)
    payload = _load_provider_config()
    payload["active_provider"] = pid
    return _save_provider_config(payload)


def load_preferred_model(provider_id: Optional[str] = None) -> str:
    pid = normalize_provider_id(provider_id if provider_id is not None else os.getenv(ENV_PROVIDER))
    payload = _load_provider_config()
    models = payload.get("models")
    if not isinstance(models, dict):
        return ""
    return str(models.get(pid) or "").strip()


def save_preferred_model(provider_id: str, model_id: str) -> Path:
    pid = normalize_provider_id(provider_id)
    mid = str(model_id or "").strip()
    payload = _load_provider_config()
    models = payload.get("models")
    if not isinstance(models, dict):
        models = {}
    if mid:
        models[pid] = mid
    elif pid in models:
        del models[pid]
    payload["models"] = models
    return _save_provider_config(payload)


def bootstrap_active_provider_env() -> str:
    """Load saved provider into env when PYMOL_AI_PROVIDER is unset (process restart)."""
    existing = str(os.getenv(ENV_PROVIDER) or "").strip()
    if existing:
        return active_provider_id()
    preferred = load_active_provider_preference()
    if preferred:
        os.environ[ENV_PROVIDER] = preferred
    return active_provider_id()


def get_provider_spec(provider_id: Optional[str] = None) -> ProviderSpec:
    pid = normalize_provider_id(provider_id if provider_id is not None else os.getenv(ENV_PROVIDER))
    for spec in _BUILTINS:
        if spec.id == pid:
            if pid != "custom":
                return spec
            overrides = load_custom_overrides()
            name = (
                str(os.getenv(ENV_CUSTOM_NAME) or "").strip()
                or overrides.get("name")
                or spec.name
            )
            url = (
                str(os.getenv(ENV_CUSTOM_URL) or "").strip()
                or overrides.get("url")
                or spec.url
            )
            addon = (
                str(os.getenv(ENV_CUSTOM_ADDON) or "").strip()
                or overrides.get("addon")
                or spec.addon
            )
            style = normalize_api_style(
                os.getenv(ENV_CUSTOM_API_STYLE)
                or overrides.get("api_style")
                or spec.api_style
            )
            default_model = overrides.get("default_model") or spec.default_model
            return replace(
                spec,
                name=name or "Custom URL",
                url=url,
                addon=addon,
                api_style=style,
                default_model=default_model,
            )
    return _BUILTINS[0]


def active_provider_id() -> str:
    return normalize_provider_id(os.getenv(ENV_PROVIDER))


def set_active_provider(provider_id: str, *, persist: bool = True) -> str:
    pid = normalize_provider_id(provider_id)
    os.environ[ENV_PROVIDER] = pid
    if persist:
        try:
            save_active_provider_preference(pid)
        except Exception:
            pass
    return pid


def provider_menu_entries() -> List[Tuple[str, str]]:
    return [(spec.id, spec.name) for spec in _BUILTINS]


def join_base_url(url: str, addon: str = "") -> str:
    return replace(
        ProviderSpec(
            id="tmp",
            name="tmp",
            url=url,
            key_env="",
            key_source_env="",
            keyring_account="",
            addon=addon,
        ),
        url=url,
        addon=addon,
    ).effective_base_url()


def resolve_openai_compat_base_url(spec: Optional[ProviderSpec] = None) -> str:
    """Base URL suitable for the OpenAI Python client (usually includes /v1)."""
    active = spec or get_provider_spec()
    if active.api_style == "openai_compat":
        return active.effective_base_url()
    # Anthropic-compat hosts: OpenAI client validation may use a sibling /v1 path.
    base = active.effective_base_url().rstrip("/")
    if base.endswith("/v1"):
        return base
    if "openrouter.ai" in base:
        return base + "/v1" if not base.endswith("/api/v1") else base
    if "fireworks.ai" in base:
        return base + "/v1"
    return base


def resolve_anthropic_compat_base_url(spec: Optional[ProviderSpec] = None) -> str:
    active = spec or get_provider_spec()
    return active.effective_base_url()


def provider_default_model(provider_id: Optional[str] = None) -> str:
    return str(get_provider_spec(provider_id).default_model or "").strip()


def backend_for_provider(provider_id: Optional[str] = None) -> str:
    style = get_provider_spec(provider_id).api_style
    return "claude_sdk" if style == "anthropic_compat" else "openai_compat"


def iter_provider_specs() -> Iterable[ProviderSpec]:
    for spec in _BUILTINS:
        yield get_provider_spec(spec.id) if spec.id == "custom" else spec


def provider_spec_as_dict(spec: ProviderSpec) -> Dict[str, object]:
    return asdict(spec)
