"""Persisted user AI model list (ONLYOFFICE-style Edit AI models store)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .model_catalog import ModelEntry, favorites_for_provider


@dataclass
class SavedModel:
    model_id: str
    display_name: str
    provider_id: str

    def to_entry(self) -> ModelEntry:
        return ModelEntry(
            model_id=self.model_id,
            display_name=self.display_name or self.model_id,
            provider_id=self.provider_id,
        )


def _store_path() -> Path:
    override = str(os.getenv("PYMOL_AI_MODELS_CONFIG") or "").strip()
    if override:
        return Path(override)
    base = Path(os.getenv("APPDATA") or Path.home() / ".pymolai")
    return Path(base) / "PyMolAI" / "ai_models.json"


def load_saved_models(provider_id: Optional[str] = None) -> List[SavedModel]:
    path = _store_path()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    out: List[SavedModel] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("model_id") or "").strip()
        pid = str(item.get("provider_id") or "").strip()
        name = str(item.get("display_name") or mid).strip() or mid
        if not mid or not pid:
            continue
        if provider_id and pid != provider_id:
            continue
        out.append(SavedModel(model_id=mid, display_name=name, provider_id=pid))
    return out


def save_all_models(models: List[SavedModel]) -> Path:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": [
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "provider_id": m.provider_id,
            }
            for m in models
        ]
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def upsert_saved_model(*, provider_id: str, model_id: str, display_name: str = "") -> SavedModel:
    pid = str(provider_id or "").strip()
    mid = str(model_id or "").strip()
    if not pid or not mid:
        raise ValueError("provider_id and model_id are required")
    name = str(display_name or mid).strip() or mid
    existing = load_saved_models()
    updated: List[SavedModel] = []
    found = False
    for item in existing:
        if item.provider_id == pid and item.model_id == mid:
            updated.append(SavedModel(model_id=mid, display_name=name, provider_id=pid))
            found = True
        else:
            updated.append(item)
    if not found:
        updated.append(SavedModel(model_id=mid, display_name=name, provider_id=pid))
    save_all_models(updated)
    return SavedModel(model_id=mid, display_name=name, provider_id=pid)


def delete_saved_model(*, provider_id: str, model_id: str) -> bool:
    pid = str(provider_id or "").strip()
    mid = str(model_id or "").strip()
    existing = load_saved_models()
    kept = [m for m in existing if not (m.provider_id == pid and m.model_id == mid)]
    if len(kept) == len(existing):
        return False
    save_all_models(kept)
    return True


def models_for_menu(provider_id: str) -> List[ModelEntry]:
    """Favorites + user-saved models for the active provider (favorites first)."""
    pid = str(provider_id or "openrouter").strip() or "openrouter"
    merged: List[ModelEntry] = []
    seen = set()
    for entry in favorites_for_provider(pid) + [m.to_entry() for m in load_saved_models(pid)]:
        if entry.model_id in seen:
            continue
        seen.add(entry.model_id)
        merged.append(entry)
    return merged
