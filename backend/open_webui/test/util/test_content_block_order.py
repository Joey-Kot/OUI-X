from open_webui.utils.middleware import reorder_content_blocks_for_display


def test_reorder_single_text_reasoning_pair():
    blocks = [
        {"type": "text", "content": "Answer"},
        {"type": "reasoning", "content": "Thought"},
    ]

    reordered = reorder_content_blocks_for_display(blocks)
    assert [b["type"] for b in reordered] == ["reasoning", "text"]


def test_reorder_multiple_pairs_in_sequence():
    blocks = [
        {"type": "text", "content": "A1"},
        {"type": "reasoning", "content": "R1"},
        {"type": "text", "content": "A2"},
        {"type": "reasoning", "content": "R2"},
    ]

    reordered = reorder_content_blocks_for_display(blocks)
    assert [b["content"] for b in reordered] == ["R1", "A1", "R2", "A2"]


def test_reorder_does_not_cross_tool_or_ci_boundaries():
    blocks = [
        {"type": "text", "content": "A1"},
        {"type": "tool_calls", "content": []},
        {"type": "reasoning", "content": "R1"},
        {"type": "text", "content": "A2"},
        {"type": "code_interpreter", "content": "print(1)"},
        {"type": "reasoning", "content": "R2"},
    ]

    reordered = reorder_content_blocks_for_display(blocks)
    assert [b["type"] for b in reordered] == [
        "text",
        "tool_calls",
        "reasoning",
        "text",
        "code_interpreter",
        "reasoning",
    ]


def test_reorder_ignores_empty_text_placeholders():
    blocks = [
        {"type": "text", "content": "   "},
        {"type": "reasoning", "content": "R"},
    ]

    reordered = reorder_content_blocks_for_display(blocks)
    assert [b["type"] for b in reordered] == ["text", "reasoning"]
