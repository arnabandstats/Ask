"""Message rendering helpers."""
from __future__ import annotations

import pytest

from ask.ui import render


class TestUserBubble:
    def test_wraps_in_bubble(self):
        assert render.user_bubble("hello,") == \
            "<div class='ks-user-row'><div class='ks-user'>hello,</div></div>"

    @pytest.mark.parametrize("text,escaped", [
        ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        ("a & b", "a &amp; b"),
        ("say \"hi\"", "say &quot;hi&quot;"),
    ])
    def test_html_is_escaped(self, text, escaped):
        assert escaped in render.user_bubble(text) and "<script>" not in render.user_bubble(text)

    def test_markdown_and_paths_kept_verbatim(self):
        text = r"load C:\Users\me\my_project\**\*.py # [x](y)"
        assert text in render.user_bubble(text)

    def test_line_breaks_preserved_and_trimmed(self):
        assert ">line one\nline two<" in render.user_bubble("  line one\nline two \n")

    def test_empty(self):
        assert "&nbsp;" in render.user_bubble("   ")


class TestCompactCitations:
    def test_single_and_multi_range(self):
        out = render._compact_citations("See [repo:src/a/train.py:L7-8] and [b.py:L3-7,L9-12].")
        assert out == "See  `train.py · L7–8` and  `b.py · L3–7, L9–12`."

    def test_untouched_without_citations(self):
        assert render._compact_citations("plain [link](x)") == "plain [link](x)"
