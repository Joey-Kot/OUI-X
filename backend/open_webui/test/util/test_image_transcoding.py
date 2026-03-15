import pytest

from open_webui.utils.image_transcoding import (
    ImageTranscodeError,
    build_ffmpeg_webp_command,
    clamp_quality,
    compute_target_dimensions,
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


def test_build_ffmpeg_webp_command_uses_webp_and_fixed_threads():
    command = build_ffmpeg_webp_command(
        input_path="/tmp/input.heic",
        output_path="/tmp/output.webp",
        width=2048,
        height=1536,
        quality=0.75,
    )

    assert command[:8] == ["ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", "/tmp/input.heic"]
    assert "libwebp" in command
    assert "75" in command
    assert "/tmp/output.webp" == command[-1]
