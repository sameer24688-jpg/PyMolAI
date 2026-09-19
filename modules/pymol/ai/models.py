from __future__ import annotations

from typing import List, Tuple

from .model_catalog import OPENROUTER_FAVORITES, favorites_for_provider
from .model_store import models_for_menu
from .providers import active_provider_id

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

# Backward-compatible OpenRouter favorites (legacy callers / tests).
SUPPORTED_MODELS: Tuple[Tuple[str, str], ...] = tuple(OPENROUTER_FAVORITES)


def supported_model_ids() -> List[str]:
    return [model_id for model_id, _ in SUPPORTED_MODELS]


def model_menu_entries(provider_id: str | None = None) -> List[Tuple[str, str]]:
    pid = str(provider_id or active_provider_id() or "openrouter")
    return [(e.model_id, e.display_name) for e in models_for_menu(pid)]


def provider_favorite_entries(provider_id: str) -> List[Tuple[str, str]]:
    return [(e.model_id, e.display_name) for e in favorites_for_provider(provider_id)]


def is_supported_model(model_id: str) -> bool:
    candidate = str(model_id or "").strip()
    if not candidate:
        return False
    return candidate in supported_model_ids()


def canonical_default_model() -> str:
    return DEFAULT_MODEL
