from types import SimpleNamespace

from open_webui.utils.image_mime import resolve_image_content_type


def test_resolve_image_content_type_prefers_file_meta():
    file_obj = SimpleNamespace(meta={"content_type": "image/webp"})
    assert resolve_image_content_type(file_obj=file_obj, file_path="/tmp/no-extension") == "image/webp"


def test_resolve_image_content_type_falls_back_to_filename_guess():
    file_obj = SimpleNamespace(meta={})
    assert resolve_image_content_type(file_obj=file_obj, file_path="/tmp/image.webp") == "image/webp"


def test_resolve_image_content_type_uses_default_when_unknown():
    file_obj = SimpleNamespace(meta={})
    assert resolve_image_content_type(file_obj=file_obj, file_path="/tmp/no-extension") == "image/webp"


def test_resolve_image_content_type_keeps_existing_png_jpeg_guesses():
    file_obj = SimpleNamespace(meta={})
    assert resolve_image_content_type(file_obj=file_obj, file_path="/tmp/a.png") == "image/png"
    assert resolve_image_content_type(file_obj=file_obj, file_path="/tmp/a.jpg") == "image/jpeg"
