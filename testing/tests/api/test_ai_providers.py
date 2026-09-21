import os

import pytest

from pymol.ai import providers as providers_mod


@pytest.fixture
def clean_provider_env(monkeypatch, tmp_path):
    monkeypatch.delenv(providers_mod.ENV_PROVIDER, raising=False)
    monkeypatch.delenv(providers_mod.ENV_CUSTOM_NAME, raising=False)
    monkeypatch.delenv(providers_mod.ENV_CUSTOM_URL, raising=False)
    monkeypatch.delenv(providers_mod.ENV_CUSTOM_ADDON, raising=False)
    monkeypatch.delenv(providers_mod.ENV_CUSTOM_API_STYLE, raising=False)
    monkeypatch.setenv("PYMOL_AI_PROVIDER_CONFIG", str(tmp_path / "provider_config.json"))


def test_default_provider_is_openrouter(clean_provider_env):
    assert providers_mod.active_provider_id() == "openrouter"
    assert providers_mod.backend_for_provider() == "claude_sdk"


def test_normalize_provider_aliases(clean_provider_env):
    assert providers_mod.normalize_provider_id("Fireworks") == "fireworks"
    assert providers_mod.normalize_provider_id("firework") == "fireworks"
    assert providers_mod.normalize_provider_id("moonshot") == "kimi"
    assert providers_mod.normalize_provider_id("unknown-xyz") == "openrouter"


def test_builtin_provider_api_styles(clean_provider_env):
    assert providers_mod.get_provider_spec("fireworks").api_style == "anthropic_compat"
    assert providers_mod.get_provider_spec("anthropic").api_style == "anthropic_compat"
    assert providers_mod.get_provider_spec("openai").api_style == "openai_compat"
    assert providers_mod.get_provider_spec("deepseek").api_style == "openai_compat"
    assert providers_mod.get_provider_spec("kimi").api_style == "openai_compat"
    assert providers_mod.backend_for_provider("openai") == "openai_compat"
    assert providers_mod.backend_for_provider("fireworks") == "claude_sdk"


def test_effective_base_url_joins_addon(clean_provider_env):
    openai = providers_mod.get_provider_spec("openai")
    assert openai.effective_base_url().endswith("/v1")
    assert providers_mod.join_base_url("https://example.com", "v1") == "https://example.com/v1"


def test_custom_provider_overrides_persist(clean_provider_env, monkeypatch):
    providers_mod.save_custom_overrides(
        name="Local LLM",
        url="http://127.0.0.1:1234",
        addon="v1",
        api_style="openai_compat",
        default_model="local-model",
    )
    monkeypatch.setenv(providers_mod.ENV_PROVIDER, "custom")
    spec = providers_mod.get_provider_spec("custom")
    assert spec.name == "Local LLM"
    assert spec.url == "http://127.0.0.1:1234"
    assert spec.addon == "v1"
    assert spec.api_style == "openai_compat"
    assert spec.effective_base_url() == "http://127.0.0.1:1234/v1"


def test_set_active_provider(clean_provider_env):
    assert providers_mod.set_active_provider("deepseek") == "deepseek"
    assert os.getenv(providers_mod.ENV_PROVIDER) == "deepseek"
    assert providers_mod.active_provider_id() == "deepseek"
    assert providers_mod.load_active_provider_preference() == "deepseek"


def test_active_provider_survives_env_clear_via_config(clean_provider_env, monkeypatch):
    providers_mod.set_active_provider("fireworks")
    providers_mod.save_preferred_model("fireworks", "accounts/fireworks/models/glm-5p2")
    monkeypatch.delenv(providers_mod.ENV_PROVIDER, raising=False)
    assert providers_mod.active_provider_id() == "openrouter"
    restored = providers_mod.bootstrap_active_provider_env()
    assert restored == "fireworks"
    assert providers_mod.active_provider_id() == "fireworks"
    assert providers_mod.load_preferred_model("fireworks") == "accounts/fireworks/models/glm-5p2"


def test_save_custom_overrides_preserves_active_provider(clean_provider_env):
    providers_mod.set_active_provider("fireworks")
    providers_mod.save_custom_overrides(
        name="Local LLM",
        url="http://127.0.0.1:1234",
        addon="v1",
        api_style="openai_compat",
        default_model="local-model",
    )
    assert providers_mod.load_active_provider_preference() == "fireworks"
    spec = providers_mod.get_provider_spec("custom")
    assert spec.url == "http://127.0.0.1:1234"


def test_resolve_model_for_provider_drops_stale_unlisted_ids(clean_provider_env, tmp_path, monkeypatch):
    monkeypatch.setenv("PYMOL_AI_MODELS_CONFIG", str(tmp_path / "ai_models.json"))
    providers_mod.save_preferred_model("openrouter", "openai/gpt-luna-latest")
    resolved = providers_mod.resolve_model_for_provider("openrouter", allow_unlisted=False)
    assert resolved == providers_mod.provider_default_model("openrouter")
    assert resolved != "openai/gpt-luna-latest"


def test_model_compatible_rejects_cross_provider_ids(clean_provider_env):
    assert providers_mod.model_compatible_with_provider("openrouter", "accounts/fireworks/models/glm-5p2") is False
    assert providers_mod.model_compatible_with_provider("fireworks", "anthropic/claude-sonnet-4.6") is False
    assert providers_mod.normalize_model_id("~openai/gpt-luna-latest~") == "openai/gpt-luna-latest"


def test_provider_menu_entries_include_custom(clean_provider_env):
    ids = [pid for pid, _ in providers_mod.provider_menu_entries()]
    for expected in ("openrouter", "fireworks", "anthropic", "openai", "deepseek", "kimi", "custom"):
        assert expected in ids
