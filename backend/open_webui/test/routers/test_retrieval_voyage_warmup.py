from open_webui.config import VOYAGE_TOKENIZER_MODEL as DEFAULT_VOYAGE_TOKENIZER_MODEL
from open_webui.routers.retrieval import (
    _normalize_voyage_tokenizer_model,
    _should_warm_voyage_tokenizer,
)


def test_should_warm_when_switching_splitter_to_voyage():
    should_warm, splitter_enabled, voyage_model_changed = _should_warm_voyage_tokenizer(
        previous_text_splitter="token",
        current_text_splitter="token_voyage",
        previous_voyage_tokenizer_model_raw="voyageai/voyage-3-lite",
        current_voyage_tokenizer_model_raw="voyageai/voyage-3-lite",
    )

    assert should_warm is True
    assert splitter_enabled is True
    assert voyage_model_changed is False


def test_should_warm_when_model_changes_without_splitter_change():
    should_warm, splitter_enabled, voyage_model_changed = _should_warm_voyage_tokenizer(
        previous_text_splitter="token_voyage",
        current_text_splitter="token_voyage",
        previous_voyage_tokenizer_model_raw="voyageai/voyage-3-lite",
        current_voyage_tokenizer_model_raw="voyageai/voyage-3",
    )

    assert should_warm is True
    assert splitter_enabled is False
    assert voyage_model_changed is True


def test_should_not_warm_when_model_is_effectively_unchanged():
    should_warm, splitter_enabled, voyage_model_changed = _should_warm_voyage_tokenizer(
        previous_text_splitter="token_voyage",
        current_text_splitter="token_voyage",
        previous_voyage_tokenizer_model_raw="voyageai/voyage-3-lite",
        current_voyage_tokenizer_model_raw="  voyageai/voyage-3-lite  ",
    )

    assert should_warm is False
    assert splitter_enabled is False
    assert voyage_model_changed is False


def test_should_not_warm_for_empty_and_default_model_equivalence():
    default_model = DEFAULT_VOYAGE_TOKENIZER_MODEL.value

    should_warm, splitter_enabled, voyage_model_changed = _should_warm_voyage_tokenizer(
        previous_text_splitter="token_voyage",
        current_text_splitter="token_voyage",
        previous_voyage_tokenizer_model_raw="",
        current_voyage_tokenizer_model_raw=default_model,
    )

    assert should_warm is False
    assert splitter_enabled is False
    assert voyage_model_changed is False


def test_should_warm_on_model_change_even_when_splitter_not_voyage():
    should_warm, splitter_enabled, voyage_model_changed = _should_warm_voyage_tokenizer(
        previous_text_splitter="character",
        current_text_splitter="character",
        previous_voyage_tokenizer_model_raw="voyageai/voyage-3-lite",
        current_voyage_tokenizer_model_raw="voyageai/voyage-3",
    )

    assert should_warm is True
    assert splitter_enabled is False
    assert voyage_model_changed is True


def test_normalize_voyage_tokenizer_model_uses_default_for_empty_values():
    assert _normalize_voyage_tokenizer_model(None) == DEFAULT_VOYAGE_TOKENIZER_MODEL.value
    assert _normalize_voyage_tokenizer_model("   ") == DEFAULT_VOYAGE_TOKENIZER_MODEL.value
