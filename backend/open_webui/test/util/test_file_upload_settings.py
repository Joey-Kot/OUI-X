from types import SimpleNamespace

from open_webui.utils.file_upload_settings import (
    is_conversation_file_upload_embedding_enabled,
)


def test_conversation_embedding_disabled_when_global_false_and_no_user_override():
    user = SimpleNamespace(settings={"ui": {}})
    assert (
        is_conversation_file_upload_embedding_enabled(user=user, global_enabled=False)
        is False
    )


def test_conversation_embedding_enabled_when_user_override_true_and_global_false():
    user = SimpleNamespace(settings={"ui": {"conversationFileUploadEmbedding": True}})
    assert (
        is_conversation_file_upload_embedding_enabled(user=user, global_enabled=False)
        is True
    )


def test_conversation_embedding_enabled_when_global_true_even_if_user_false():
    user = SimpleNamespace(settings={"ui": {"conversationFileUploadEmbedding": False}})
    assert (
        is_conversation_file_upload_embedding_enabled(user=user, global_enabled=True)
        is True
    )


def test_conversation_embedding_enabled_with_model_style_settings_object():
    settings = SimpleNamespace(ui={"conversationFileUploadEmbedding": True})
    user = SimpleNamespace(settings=settings)
    assert (
        is_conversation_file_upload_embedding_enabled(user=user, global_enabled=False)
        is True
    )
