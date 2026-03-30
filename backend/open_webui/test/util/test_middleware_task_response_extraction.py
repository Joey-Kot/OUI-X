from open_webui.utils import middleware


def test_extract_task_response_content_from_chat_completions_shape():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "chat content",
                }
            }
        ]
    }
    assert middleware._extract_task_response_content(payload) == "chat content"


def test_extract_task_response_content_from_responses_shape():
    payload = {
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "responses "},
                    {"type": "output_text", "text": "content"},
                ],
            }
        ],
    }
    assert middleware._extract_task_response_content(payload) == "responses content"

