from open_webui.routers.tasks import _build_refer_context_by_turns


def test_build_refer_context_by_turns_excludes_latest_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]

    refer_context = _build_refer_context_by_turns(messages, turns=2)

    assert "user: u1" in refer_context
    assert "assistant: a1" in refer_context
    assert "user: u2" in refer_context
    assert "assistant: a2" in refer_context
    assert "u3" not in refer_context


def test_build_refer_context_by_turns_zero_returns_empty():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]

    refer_context = _build_refer_context_by_turns(messages, turns=0)

    assert refer_context == ""
