import mimetypes
from pathlib import Path
from typing import Optional, Any

DEFAULT_IMAGE_MIME_TYPE = "image/webp"


def resolve_image_content_type(
    file_obj: Optional[Any] = None,
    file_path: Optional[str | Path] = None,
    default: str = DEFAULT_IMAGE_MIME_TYPE,
) -> str:
    if file_obj is not None:
        meta = getattr(file_obj, "meta", None) or {}
        content_type = meta.get("content_type")
        if isinstance(content_type, str) and content_type.startswith("image/"):
            return content_type

    if file_path is not None:
        guessed_type, _ = mimetypes.guess_type(str(file_path))
        if isinstance(guessed_type, str) and guessed_type.startswith("image/"):
            return guessed_type

    return default
