import os

import pytest

from pymol.ai import provider_key_store as store


class _FakeKeyring:
    def __init__(self):
        self._data = {}

    def get_password(self, service, account):
        return self._data.get((service, account))

    def set_password(self, service, account, value):
        self._data[(service, account)] = value

    def delete_password(self, service, account):
        key = (service, account)
        if key not in self._data:
            raise RuntimeError("not found")
        del self._data[key]


@pytest.fixture
def fake_keyring(monkeypatch):
    keyring = _FakeKeyring()
    monkeypatch.setattr(store, "_load_keyring", lambda: (keyring, True))
    return keyring


@pytest.fixture
def clean_keys(monkeypatch):
    for env in (
        "OPENROUTER_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "FIREWORKS_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
        "PYMOL_AI_CUSTOM_API_KEY",
        "PYMOL_AI_OPENROUTER_KEY_SOURCE",
        "PYMOL_AI_FIREWORKS_KEY_SOURCE",
        "PYMOL_AI_OPENAI_KEY_SOURCE",
    ):
        monkeypatch.delenv(env, raising=False)


def test_provider_keys_are_isolated(fake_keyring, clean_keys, monkeypatch):
    store.save_key("openrouter", "or-key-1111")
    store.save_key("fireworks", "fw-key-2222")
    assert fake_keyring.get_password(store.SERVICE_NAME, "openrouter_api_key") == "or-key-1111"
    assert fake_keyring.get_password(store.SERVICE_NAME, "fireworks_api_key") == "fw-key-2222"

    status_or = store.load_saved_key_into_env_if_needed("openrouter")
    status_fw = store.load_saved_key_into_env_if_needed("fireworks")
    assert status_or.source == "saved"
    assert status_fw.source == "saved"
    assert os.getenv("OPENROUTER_API_KEY") == "or-key-1111"
    assert os.getenv("FIREWORKS_API_KEY") == "fw-key-2222"


def test_clear_does_not_wipe_other_provider(fake_keyring, clean_keys):
    store.save_key("openrouter", "or-key-aaaa")
    store.save_key("openai", "oai-key-bbbb")
    store.clear_saved_key("openai")
    assert fake_keyring.get_password(store.SERVICE_NAME, "openrouter_api_key") == "or-key-aaaa"
    assert fake_keyring.get_password(store.SERVICE_NAME, "openai_api_key") is None


def test_status_masks_key(fake_keyring, clean_keys):
    store.save_key("deepseek", "sk-deepseek-SECRET99")
    status = store.get_status("deepseek")
    assert status.has_key is True
    assert "SECRET" not in status.masked_key
    assert status.masked_key.endswith("T99")


def test_validate_openai_compat_uses_provider_base(monkeypatch, clean_keys):
    calls = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            return {"ok": True}

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.chat = _FakeChat()

        def close(self):
            return None

    import types
    import sys

    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    store.validate_key_live("openai", "sk-test", model="gpt-4o-mini", timeout_sec=1.0)
    assert calls["init"]["api_key"] == "sk-test"
    assert str(calls["init"]["base_url"]).endswith("/v1")
    assert calls["create"]["model"] == "gpt-4o-mini"


def test_resolve_validation_model_rejects_cross_provider_ids(clean_keys):
    assert store.resolve_validation_model("fireworks", "anthropic/claude-sonnet-4.6").startswith(
        "accounts/fireworks/models/"
    )
    assert store.resolve_validation_model("fireworks", "accounts/fireworks/models/glm-5p2") == (
        "accounts/fireworks/models/glm-5p2"
    )


def test_validate_fireworks_prefers_models_list(monkeypatch, clean_keys):
    calls = {"list": 0, "create": 0}

    class _FakeModels:
        def list(self):
            calls["list"] += 1
            return {"data": []}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["create"] += 1
            return {"ok": True}

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.models = _FakeModels()
            self.chat = _FakeChat()

        def close(self):
            return None

    import types
    import sys

    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    store.validate_key_live(
        "fireworks",
        "fw-test",
        model="accounts/fireworks/models/kimi-k2p5",
        timeout_sec=1.0,
    )
    assert calls["list"] == 1
    assert calls["create"] == 0
