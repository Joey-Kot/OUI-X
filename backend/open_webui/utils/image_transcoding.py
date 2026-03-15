import logging
import math
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

DEFAULT_IMAGE_COMPRESSION_QUALITY = 0.75
DEFAULT_IMAGE_TRANSCODE_MAX_CONCURRENCY_PER_USER = 2
WEBP_COMPRESSION_LEVEL = 6
FFMPEG_THREADS = 1
HEIC_EXTENSIONS = {"heic", "heif"}

_user_semaphore_lock = threading.Lock()
_user_semaphores: dict[tuple[str, int], threading.BoundedSemaphore] = {}


class ImageTranscodeError(RuntimeError):
    pass


class ImageTranscodeCapabilityError(ImageTranscodeError):
    pass


def clamp_quality(value: Any, default: float = DEFAULT_IMAGE_COMPRESSION_QUALITY) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, round(numeric, 2)))


def parse_image_compression_metadata(metadata: Optional[dict]) -> Optional[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None

    raw = metadata.get("image_compression")
    if not isinstance(raw, dict):
        return None

    enabled = raw.get("enabled") is True
    if not enabled:
        return None

    width = raw.get("width")
    height = raw.get("height")
    if width in (None, "") or height in (None, ""):
        raise ImageTranscodeError("Image compression width and height are required.")

    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError) as exc:
        raise ImageTranscodeError("Image compression width and height must be integers.") from exc

    if width <= 0 or height <= 0:
        raise ImageTranscodeError("Image compression width and height must be greater than zero.")

    return {
        "enabled": True,
        "width": width,
        "height": height,
        "quality": clamp_quality(raw.get("quality")),
    }


def normalize_orientation_to_tempfile(input_path: str) -> tuple[str, int, int, bool]:
    try:
        with Image.open(input_path) as image:
            normalized = ImageOps.exif_transpose(image)
            normalized.load()
            orientation_applied = normalized.size != image.size or normalized.tobytes() != image.tobytes()
            suffix = Path(input_path).suffix or ".img"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                normalized.save(temp_file, format=normalized.format or image.format)
                return temp_file.name, normalized.width, normalized.height, orientation_applied
    except OSError as exc:
        if Path(input_path).suffix.lower().lstrip(".") in HEIC_EXTENSIONS:
            raise ImageTranscodeCapabilityError(
                "HEIC/HEIF decode support is unavailable on this server."
            ) from exc
        raise ImageTranscodeError(f"Failed to normalize image orientation: {exc}") from exc


def compute_target_dimensions(
    original_width: int,
    original_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    if original_width <= 0 or original_height <= 0:
        raise ImageTranscodeError("Invalid input dimensions.")

    if original_width <= max_width and original_height <= max_height:
        return original_width, original_height

    scale = min(max_width / original_width, max_height / original_height)
    target_width = max(1, int(math.floor(original_width * scale)))
    target_height = max(1, int(math.floor(original_height * scale)))

    if target_width % 2 != 0:
        target_width = max(1, target_width - 1)
    if target_height % 2 != 0:
        target_height = max(1, target_height - 1)

    return target_width, target_height


def build_ffmpeg_webp_command(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    quality: float,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-threads",
        str(FFMPEG_THREADS),
        "-i",
        input_path,
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-frames:v",
        "1",
        "-c:v",
        "libwebp",
        "-quality",
        str(int(round(clamp_quality(quality) * 100))),
        "-compression_level",
        str(WEBP_COMPRESSION_LEVEL),
        output_path,
    ]


def transcode_image_to_webp(
    *,
    input_path: str,
    output_path: str,
    max_width: int,
    max_height: int,
    quality: float,
) -> dict[str, Any]:
    normalized_input_path = None
    try:
        normalized_input_path, normalized_width, normalized_height, orientation_applied = (
            normalize_orientation_to_tempfile(input_path)
        )

        target_width, target_height = compute_target_dimensions(
            original_width=normalized_width,
            original_height=normalized_height,
            max_width=max_width,
            max_height=max_height,
        )

        command = build_ffmpeg_webp_command(
            input_path=normalized_input_path,
            output_path=output_path,
            width=target_width,
            height=target_height,
            quality=quality,
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise ImageTranscodeError(stderr or "Failed to transcode image.")

        if not os.path.exists(output_path):
            raise ImageTranscodeError("Image transcoding did not produce an output file.")

        return {
            "original_width": normalized_width,
            "original_height": normalized_height,
            "orientation_applied": orientation_applied,
            "width": target_width,
            "height": target_height,
            "resized": target_width != normalized_width or target_height != normalized_height,
            "quality": clamp_quality(quality),
        }
    finally:
        if normalized_input_path and os.path.exists(normalized_input_path):
            os.unlink(normalized_input_path)


def get_user_transcode_semaphore(user_id: str, limit: int) -> threading.BoundedSemaphore:
    normalized_limit = max(1, int(limit))
    key = (user_id, normalized_limit)
    with _user_semaphore_lock:
        semaphore = _user_semaphores.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(normalized_limit)
            _user_semaphores[key] = semaphore
        return semaphore
