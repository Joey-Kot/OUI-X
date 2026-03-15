from pathlib import Path

import pytest
from PIL import Image

from open_webui.utils.image_transcoding import (
    ImageTranscodeError,
    build_ffmpeg_webp_command,
    clamp_quality,
    compute_target_dimensions,
    normalize_orientation_to_tempfile,
    parse_image_compression_metadata,
)


def test_parse_image_compression_metadata_disabled_returns_none():
    assert parse_image_compression_metadata({"image_compression": {"enabled": False}}) is None


def test_parse_image_compression_metadata_validates_dimensions_and_quality():
    metadata = parse_image_compression_metadata(
        {
            "image_compression": {
                "enabled": True,
                "width": "2048",
                "height": 1024,
                "quality": 0.756,
            }
        }
    )

    assert metadata == {
        "enabled": True,
        "width": 2048,
        "height": 1024,
        "quality": 0.76,
    }


@pytest.mark.parametrize("quality,expected", [(-1, 0.0), (2, 1.0), (0.754, 0.75), (None, 0.75)])
def test_clamp_quality_bounds_values(quality, expected):
    assert clamp_quality(quality) == expected


def test_parse_image_compression_metadata_requires_positive_dimensions():
    with pytest.raises(ImageTranscodeError):
        parse_image_compression_metadata(
            {"image_compression": {"enabled": True, "width": 0, "height": 512}}
        )


def test_compute_target_dimensions_keeps_original_when_under_limit():
    assert compute_target_dimensions(1200, 800, 2048, 2048) == (1200, 800)


def test_compute_target_dimensions_scales_down_with_aspect_ratio():
    width, height = compute_target_dimensions(4032, 3024, 2048, 2048)
    assert width == 2048
    assert height == 1536


def test_compute_target_dimensions_scales_rotated_dimensions():
    width, height = compute_target_dimensions(3060, 4080, 2048, 2048)
    assert width == 1536
    assert height == 2048


def test_build_ffmpeg_webp_command_uses_webp_and_fixed_threads():
    command = build_ffmpeg_webp_command(
        input_path="/tmp/input.jpg",
        output_path="/tmp/output.webp",
        width=1536,
        height=2048,
        quality=0.75,
    )

    assert command[:8] == ["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", "/tmp/input.jpg"]
    assert command[8:10] == ["-vf", "scale=1536:2048:flags=lanczos"]
    assert "libwebp" in command
    assert "75" in command
    assert "/tmp/output.webp" == command[-1]


def test_normalize_orientation_to_tempfile_keeps_image_when_no_exif(tmp_path):
    source = tmp_path / "plain.jpg"
    Image.new("RGB", (100, 50), color="red").save(source, format="JPEG")

    normalized_path, width, height, orientation_applied = normalize_orientation_to_tempfile(str(source))
    try:
        assert Path(normalized_path).is_file()
        assert (width, height) == (100, 50)
        assert orientation_applied is False
    finally:
        Path(normalized_path).unlink(missing_ok=True)
