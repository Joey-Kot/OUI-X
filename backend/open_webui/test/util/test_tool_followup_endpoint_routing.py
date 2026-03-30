from open_webui.utils import middleware as middleware_mod


def test_chat_messages_to_responses_input_keeps_tool_followups_and_assistant_tool_calls():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Shanghai"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]

    converted = middleware_mod._chat_messages_to_responses_input(messages)

    assert converted[0] == {"role": "user", "content": "hi"}
    assert converted[1]["type"] == "function_call"
    assert converted[1]["call_id"] == "call_1"
    assert converted[1]["name"] == "get_weather"
    assert '"city": "Shanghai"' in converted[1]["arguments"]
    assert converted[2]["role"] == "assistant"
    assert converted[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "sunny",
    }


def test_chat_messages_to_responses_input_supports_multimodal_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }
    ]

    converted = middleware_mod._chat_messages_to_responses_input(messages)

    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": "https://example.com/cat.png"},
            ],
        }
    ]
