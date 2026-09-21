"""Smoke / integration checks for Phase 1 provider routing (no live network)."""

from types import SimpleNamespace

from pymol.ai.claude_sdk_loop import ClaudeSdkLoop
from pymol.ai.openai_compat_loop import OpenAICompatLoop
from pymol.ai.providers import backend_for_provider, get_provider_spec, set_active_provider
from pymol.ai.runtime import AiRuntime
from pymol.shortcut import Shortcut


class DummyParser:
    def parse(self, command):
        return 1


class DummyCmd:
    def __init__(self):
        self.kwhash = Shortcut(["show", "hide"])
        self._parser = DummyParser()
        self._pymol = SimpleNamespace()
        self._call_in_gui_thread = lambda fn: fn()

    def get_names(self, *args, **kwargs):
        return []

    def count_atoms(self, selection):
        return 0

    def get_vis(self):
        return {}

    def get_view(self, output=0, quiet=1):
        return [0.0] * 18

    def get_viewport(self, output=0, quiet=1):
        return [800, 600]

    def get_object_list(self, selection="(all)", quiet=1):
        return []


def test_smoke_openrouter_env_mapping_unchanged(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-smoke-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("PYMOL_AI_PROVIDER", raising=False)

    loop = ClaudeSdkLoop()
    env = loop.map_openrouter_env()
    assert env["ANTHROPIC_AUTH_TOKEN"] == "or-smoke-key"
    assert "openrouter.ai" in env["ANTHROPIC_BASE_URL"]


def test_smoke_fireworks_provider_maps_inference_base(monkeypatch):
    monkeypatch.setenv("PYMOL_AI_PROVIDER", "fireworks")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-smoke-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    loop = ClaudeSdkLoop()
    env = loop.map_provider_env("fireworks")
    assert env["ANTHROPIC_AUTH_TOKEN"] == "fw-smoke-key"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.fireworks.ai/inference"
    assert env["PYMOL_AI_PROVIDER"] == "fireworks"


def test_smoke_runtime_switches_backend_with_provider(monkeypatch):
    monkeypatch.setenv("PYMOL_AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    monkeypatch.delenv("PYMOL_AI_DISABLE", raising=False)
    monkeypatch.setattr(
        "pymol.ai.runtime.load_saved_key_into_env_if_needed",
        lambda: SimpleNamespace(source="env", has_key=True, masked_key="****", keyring_available=True),
    )
    monkeypatch.setattr("pymol.ai.runtime.load_all_saved_keys_into_env", lambda: [])
    monkeypatch.setattr(
        "pymol.ai.runtime.load_openbio_saved_key_into_env_if_needed",
        lambda: SimpleNamespace(source="none", has_key=False, masked_key="", keyring_available=True),
    )

    runtime = AiRuntime(DummyCmd())
    assert runtime.provider == "openrouter"
    assert runtime._agent_backend == "claude_sdk"

    runtime.set_provider("openai", emit_notice=False)
    assert runtime.provider == "openai"
    assert runtime._agent_backend == "openai_compat"
    assert backend_for_provider("openai") == "openai_compat"
    assert runtime.model == "gpt-4o-mini"

    runtime.set_provider("openrouter", emit_notice=False)
    assert runtime._agent_backend == "claude_sdk"
    assert runtime.model == "anthropic/claude-sonnet-4.6"


def test_smoke_provider_switch_resets_stale_model_and_session(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMOL_AI_PROVIDER_CONFIG", str(tmp_path / "provider_config.json"))
    monkeypatch.setenv("PYMOL_AI_PROVIDER", "fireworks")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("PYMOL_AI_DISABLE", raising=False)
    monkeypatch.setattr(
        "pymol.ai.runtime.load_saved_key_into_env_if_needed",
        lambda: SimpleNamespace(source="env", has_key=True, masked_key="****", keyring_available=True),
    )
    monkeypatch.setattr("pymol.ai.runtime.load_all_saved_keys_into_env", lambda: [])
    monkeypatch.setattr(
        "pymol.ai.runtime.load_openbio_saved_key_into_env_if_needed",
        lambda: SimpleNamespace(source="none", has_key=False, masked_key="", keyring_available=True),
    )
    from pymol.ai.providers import save_preferred_model

    save_preferred_model("openrouter", "openai/gpt-luna-latest")
    runtime = AiRuntime(DummyCmd())
    runtime._sdk_session_id = "old-fireworks-session"
    runtime.set_provider("openrouter", emit_notice=False, reset_model=True)
    assert runtime.provider == "openrouter"
    assert runtime.model == "anthropic/claude-sonnet-4.6"
    assert runtime._sdk_session_id is None


def test_smoke_openai_compat_loop_auth_error_without_key():
    loop = OpenAICompatLoop()
    result = loop.run_turn(prompt="hi", model="gpt-4o-mini", api_key="", provider_id="openai")
    assert result.error_class == "auth_error"


def test_smoke_provider_specs_have_onlyoffice_fields():
    for pid in ("openrouter", "fireworks", "anthropic", "openai", "deepseek", "kimi", "custom"):
        spec = get_provider_spec(pid)
        assert spec.name
        assert spec.key_env
        assert spec.keyring_account
        assert spec.api_style in ("anthropic_compat", "openai_compat")
        # url may be empty only for unset custom
        if pid != "custom":
            assert spec.url
