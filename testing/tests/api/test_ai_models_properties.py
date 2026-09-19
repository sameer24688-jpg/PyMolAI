"""Property tests for model catalog merge/filter invariants."""

from hypothesis import given, settings, strategies as st

from pymol.ai.model_catalog import ModelEntry, filter_models, merge_favorites_and_catalog
from pymol.ai.providers import provider_ids


@st.composite
def model_entries(draw):
    n = draw(st.integers(min_value=0, max_value=8))
    rows = []
    for i in range(n):
        mid = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_/."),
                min_size=1,
                max_size=24,
            )
        )
        name = draw(st.text(min_size=0, max_size=20))
        rows.append(ModelEntry(model_id=mid.strip() or ("m%d" % i), display_name=name or mid, provider_id="x"))
    return rows


@given(st.sampled_from(provider_ids()), model_entries())
@settings(max_examples=40)
def test_merge_never_duplicates_ids(provider_id, catalog):
    merged = merge_favorites_and_catalog(provider_id, catalog)
    ids = [m.model_id for m in merged]
    assert len(ids) == len(set(ids))


@given(model_entries(), st.text(min_size=0, max_size=12))
@settings(max_examples=40)
def test_filter_is_subset(entries, query):
    filtered = filter_models(entries, query)
    assert all(item in entries for item in filtered)


@given(st.sampled_from(provider_ids()))
@settings(max_examples=20)
def test_merge_with_empty_catalog_at_least_favorites_or_empty(provider_id):
    merged = merge_favorites_and_catalog(provider_id, [])
    if provider_id == "custom":
        assert merged == []
    else:
        assert len(merged) >= 1
