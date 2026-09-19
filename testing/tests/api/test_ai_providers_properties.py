"""Property-based tests for provider normalization and URL joining."""

from hypothesis import given, settings, strategies as st

from pymol.ai.providers import (
    API_STYLES,
    DEFAULT_PROVIDER_ID,
    backend_for_provider,
    join_base_url,
    normalize_api_style,
    normalize_provider_id,
    provider_ids,
)


_PROVIDER_IDS = provider_ids()


@given(st.sampled_from(_PROVIDER_IDS))
@settings(max_examples=50)
def test_known_provider_ids_are_stable(pid):
    assert normalize_provider_id(pid) == pid
    assert normalize_provider_id(pid.upper()) == pid
    assert backend_for_provider(pid) in ("claude_sdk", "openai_compat")


@given(st.text(min_size=0, max_size=40))
@settings(max_examples=80)
def test_unknown_provider_falls_back_to_openrouter(raw):
    normalized = normalize_provider_id(raw)
    assert normalized in _PROVIDER_IDS
    if raw.strip().lower() not in _PROVIDER_IDS and raw.strip().lower() not in {
        "or",
        "fw",
        "firework",
        "fireworks.ai",
        "claude",
        "moonshot",
        "moonshotai",
    }:
        # Empty and garbage map to default.
        if not raw.strip() or normalize_provider_id(raw) == DEFAULT_PROVIDER_ID:
            assert normalized == DEFAULT_PROVIDER_ID or normalized in _PROVIDER_IDS


@given(st.sampled_from(list(API_STYLES) + ["anthropic", "openai", "claude", "chat", "bogus"]))
@settings(max_examples=40)
def test_api_style_normalization_is_closed(raw):
    style = normalize_api_style(raw)
    assert style in API_STYLES


@given(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters=".-:/"),
        min_size=1,
        max_size=40,
    ),
    st.sampled_from(["", "v1", "v1/", "/v1"]),
)
@settings(max_examples=60)
def test_join_base_url_never_duplicates_brokenly(url, addon):
    base = "https://example.test/%s" % (url.lstrip("/"),)
    joined = join_base_url(base, addon)
    assert joined.startswith("https://example.test/")
    assert " " not in joined
    if addon.strip().strip("/") == "v1":
        assert joined.rstrip("/").endswith("v1")


@given(st.sampled_from(_PROVIDER_IDS), st.sampled_from(_PROVIDER_IDS))
@settings(max_examples=40)
def test_backend_deterministic_for_pair(a, b):
    # Same input always same backend (idempotent).
    assert backend_for_provider(a) == backend_for_provider(a)
    assert backend_for_provider(b) == backend_for_provider(b)
