from types import SimpleNamespace

import server


def test_response_text_skips_thinking_blocks():
    resp = SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="text", text="[1,"),
        SimpleNamespace(type="text", text="2]"),
    ])
    assert server._response_text(resp) == "[1,2]"
