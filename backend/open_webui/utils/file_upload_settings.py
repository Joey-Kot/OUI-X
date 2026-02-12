from typing import Any


def _extract_ui_settings(user: Any) -> dict:
    settings = getattr(user, "settings", None)
    if not settings:
        return {}

    if isinstance(settings, dict):
        ui_settings = settings.get("ui")
        return ui_settings if isinstance(ui_settings, dict) else {}

    ui_settings = getattr(settings, "ui", None)
    return ui_settings if isinstance(ui_settings, dict) else {}


def is_conversation_file_upload_embedding_enabled(
    user: Any, global_enabled: bool
) -> bool:
    """Resolve per-user override for conversation file upload embedding.

    User-level setting only overrides when explicitly enabled (True); otherwise
    the behavior follows the global configuration.
    """

    ui_settings = _extract_ui_settings(user)
    user_override_enabled = ui_settings.get("conversationFileUploadEmbedding") is True
    return bool(global_enabled) or user_override_enabled
