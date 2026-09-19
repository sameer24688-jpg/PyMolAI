"""Smoke checks for Phase 2 model edit flow helpers."""

from pymol.ai.model_catalog import ModelEntry, fetch_provider_catalog, merge_favorites_and_catalog
from pymol.ai.model_store import models_for_menu, upsert_saved_model
from pymol.ai.models import model_menu_entries


def test_smoke_edit_models_flow_preserves_favorites_on_failed_refresh(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMOL_AI_MODELS_CONFIG", str(tmp_path / "ai_models.json"))
    monkeypatch.setenv("PYMOL_AI_PROVIDER", "openai")

    before = model_menu_entries("openai")
    assert before

    def boom(*_a, **_k):
        raise RuntimeError("offline")

    raised = False
    try:
        fetch_provider_catalog("openai", api_key="x", http_get_json=boom)
    except Exception:
        raised = True
    assert raised is True
    after = model_menu_entries("openai")
    assert after == before


def test_smoke_custom_model_appears_in_menu(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMOL_AI_MODELS_CONFIG", str(tmp_path / "ai_models.json"))
    upsert_saved_model(provider_id="deepseek", model_id="deepseek-chat-custom", display_name="Custom DS")
    menu = models_for_menu("deepseek")
    assert any(m.model_id == "deepseek-chat-custom" for m in menu)


def test_smoke_merge_keeps_favorites_ahead_of_remote():
    from pymol.ai.model_catalog import favorites_for_provider

    remote = merge_favorites_and_catalog(
        "kimi",
        [ModelEntry("brand-new-kimi", "Brand New", "kimi")],
    )
    favorite_ids = {e.model_id for e in favorites_for_provider("kimi")}
    assert remote[0].model_id in favorite_ids
    assert remote[-1].model_id == "brand-new-kimi"
