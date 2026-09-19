import pytest

from pymol.ai.model_catalog import (
    ModelCatalogError,
    ModelEntry,
    favorites_for_provider,
    fetch_provider_catalog,
    filter_models,
    merge_favorites_and_catalog,
)


def test_favorites_exist_for_builtin_providers():
    for pid in ("openrouter", "fireworks", "anthropic", "openai", "deepseek", "kimi"):
        rows = favorites_for_provider(pid)
        assert rows
        assert all(isinstance(r, ModelEntry) for r in rows)


def test_merge_favorites_first_then_catalog():
    catalog = [
        ModelEntry("accounts/fireworks/models/glm-5p2", "GLM dup", "fireworks"),
        ModelEntry("accounts/fireworks/models/new-model", "New Model", "fireworks"),
    ]
    merged = merge_favorites_and_catalog("fireworks", catalog)
    ids = [m.model_id for m in merged]
    assert ids[0] == "accounts/fireworks/models/glm-5p2"
    assert "accounts/fireworks/models/new-model" in ids
    assert ids.count("accounts/fireworks/models/glm-5p2") == 1


def test_filter_models_case_insensitive():
    rows = [
        ModelEntry("a/b", "Alpha", "openrouter"),
        ModelEntry("c/d", "Beta", "openrouter"),
    ]
    assert [m.model_id for m in filter_models(rows, "alp")] == ["a/b"]
    assert filter_models(rows, "") == rows


def test_fetch_openai_catalog_parses_data(monkeypatch):
    def fake_get(url, api_key="", timeout_sec=20.0):
        assert url.endswith("/models")
        assert api_key == "sk-test"
        return {"data": [{"id": "gpt-4o-mini", "name": "GPT-4o mini"}, {"id": "o4-mini"}]}

    rows = fetch_provider_catalog(
        "openai",
        api_key="sk-test",
        http_get_json=fake_get,
    )
    assert rows[0].model_id == "gpt-4o-mini"
    assert any(r.model_id == "o4-mini" for r in rows)


def test_fetch_fireworks_catalog_normalizes_names():
    def fake_get(url, api_key="", timeout_sec=20.0):
        assert "accounts/fireworks/models" in url
        return {
            "models": [
                {"name": "accounts/fireworks/models/kimi-k2p5", "displayName": "Kimi"},
                {"name": "deepseek-v3p2", "displayName": "DeepSeek"},
            ]
        }

    rows = fetch_provider_catalog("fireworks", api_key="fw", http_get_json=fake_get)
    assert rows[0].model_id == "accounts/fireworks/models/kimi-k2p5"
    assert rows[1].model_id == "accounts/fireworks/models/deepseek-v3p2"


def test_fetch_catalog_failure_raises_without_side_effects():
    def boom(*_a, **_k):
        raise ModelCatalogError("network down")

    with pytest.raises(ModelCatalogError, match="network down"):
        fetch_provider_catalog("openai", api_key="x", http_get_json=boom)


def test_anthropic_catalog_is_static():
    rows = fetch_provider_catalog("anthropic", api_key="")
    assert any(r.model_id.startswith("claude-") for r in rows)
