from pymol.ai.model_store import (
    delete_saved_model,
    load_saved_models,
    models_for_menu,
    upsert_saved_model,
)


def test_upsert_and_delete_saved_models(monkeypatch, tmp_path):
    path = tmp_path / "ai_models.json"
    monkeypatch.setenv("PYMOL_AI_MODELS_CONFIG", str(path))

    upsert_saved_model(provider_id="fireworks", model_id="accounts/fireworks/models/x", display_name="X")
    upsert_saved_model(provider_id="openai", model_id="gpt-4o-mini", display_name="Mini")
    rows = load_saved_models("fireworks")
    assert len(rows) == 1
    assert rows[0].display_name == "X"

    upsert_saved_model(provider_id="fireworks", model_id="accounts/fireworks/models/x", display_name="X2")
    assert load_saved_models("fireworks")[0].display_name == "X2"

    assert delete_saved_model(provider_id="fireworks", model_id="accounts/fireworks/models/x") is True
    assert load_saved_models("fireworks") == []
    assert load_saved_models("openai")


def test_models_for_menu_merges_favorites_and_saved(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMOL_AI_MODELS_CONFIG", str(tmp_path / "ai_models.json"))
    upsert_saved_model(provider_id="openai", model_id="my-custom-model", display_name="Custom")
    menu = models_for_menu("openai")
    ids = [m.model_id for m in menu]
    assert "gpt-4o-mini" in ids
    assert "my-custom-model" in ids
    assert ids.index("gpt-4o-mini") < ids.index("my-custom-model")
