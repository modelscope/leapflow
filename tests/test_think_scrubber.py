# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for :mod:`leapflow.engine.think_scrubber`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from leapflow.engine.think_scrubber import ScrubberSink, ThinkScrubber


# ============================================================================
# ThinkScrubber unit tests
# ============================================================================


class TestThinkScrubberBasic:
    """Basic tag filtering."""

    def test_no_tags_passthrough(self) -> None:
        s = ThinkScrubber()
        assert s.scrub("hello world") == "hello world"

    def test_simple_think_block_removed(self) -> None:
        s = ThinkScrubber()
        result = s.scrub("before<think>secret</think>after")
        assert result == "beforeafter"

    def test_multiple_think_blocks(self) -> None:
        s = ThinkScrubber()
        result = s.scrub("a<think>x</think>b<think>y</think>c")
        assert result == "abc"

    def test_empty_think_block(self) -> None:
        s = ThinkScrubber()
        assert s.scrub("ok<think></think>done") == "okdone"

    def test_empty_input(self) -> None:
        s = ThinkScrubber()
        assert s.scrub("") == ""

    def test_only_think_block(self) -> None:
        s = ThinkScrubber()
        result = s.scrub("<think>hidden</think>")
        assert result == ""


class TestThinkScrubberCrossChunk:
    """Tag split across multiple chunks."""

    def test_open_tag_split(self) -> None:
        s = ThinkScrubber()
        out = s.scrub("hello<thi")
        out += s.scrub("nk>secret</think>world")
        assert out == "helloworld"

    def test_close_tag_split(self) -> None:
        s = ThinkScrubber()
        out = s.scrub("<think>secret</thi")
        out += s.scrub("nk>visible")
        assert out == "visible"

    def test_open_tag_one_char_at_a_time(self) -> None:
        s = ThinkScrubber()
        out = ""
        for ch in "pre<think>inside</think>post":
            out += s.scrub(ch)
        assert out == "prepost"

    def test_close_tag_one_char_at_a_time(self) -> None:
        s = ThinkScrubber()
        text = "<think>reasoning</think>answer"
        out = ""
        for ch in text:
            out += s.scrub(ch)
        assert out == "answer"

    def test_split_at_every_boundary(self) -> None:
        """Feed the whole string char-by-char."""
        s = ThinkScrubber()
        text = "A<think>B</think>C"
        out = "".join(s.scrub(ch) for ch in text)
        assert out == "AC"


class TestThinkScrubberEdgeCases:
    """Nesting, partial tags, and conservative behavior."""

    def test_nested_tags_outer_wins(self) -> None:
        """Nested <think> inside a think block — outer close wins."""
        s = ThinkScrubber()
        result = s.scrub("<think>a<think>b</think>c</think>d")
        # The first </think> ends the block; "c</think>d" remains.
        # "c" is emitted, then the second </think> is just literal text.
        # Actually: after first </think>, state is NORMAL, so "c" is emitted,
        # then </think> literal — the "<" starts buffering, "/think>" doesn't
        # match <think> prefix, so it's flushed as-is.
        assert result == "c</think>d"

    def test_open_tag_no_close_conservative(self) -> None:
        """Open tag without close — everything after is suppressed."""
        s = ThinkScrubber()
        out = s.scrub("visible<think>hidden forever")
        assert out == "visible"
        # Further chunks are also suppressed.
        assert s.scrub("still hidden") == ""

    def test_open_tag_no_close_flush(self) -> None:
        """flush() in IN_THINK state discards pending buffer."""
        s = ThinkScrubber()
        s.scrub("x<think>y")
        result = s.flush()
        assert result == ""

    def test_flush_normal_partial_tag(self) -> None:
        """flush() in NORMAL with a partial tag buffer emits the buffer."""
        s = ThinkScrubber()
        out = s.scrub("hello<thi")
        assert out == "hello"
        residual = s.flush()
        assert residual == "<thi"

    def test_angle_bracket_not_tag(self) -> None:
        """A '<' that doesn't start <think> is passed through."""
        s = ThinkScrubber()
        assert s.scrub("a < b > c") == "a < b > c"

    def test_partial_tag_then_mismatch(self) -> None:
        """Buffer '<th' then 'x' — flush '<th' + 'x' as safe text."""
        s = ThinkScrubber()
        out = s.scrub("<thx")
        assert out == "<thx"

    def test_html_like_tags_pass_through(self) -> None:
        """Tags that are NOT <think> should pass through."""
        s = ThinkScrubber()
        assert s.scrub("<div>hello</div>") == "<div>hello</div>"

    def test_case_sensitive(self) -> None:
        """<Think> is NOT a match (case-sensitive)."""
        s = ThinkScrubber()
        assert s.scrub("<Think>not hidden</Think>") == "<Think>not hidden</Think>"


class TestThinkScrubberReset:
    """Reset state between turns."""

    def test_reset_clears_state(self) -> None:
        s = ThinkScrubber()
        s.scrub("<think>")
        assert s.scrub("hidden") == ""
        s.reset()
        assert s.scrub("visible again") == "visible again"

    def test_reset_clears_buffer(self) -> None:
        s = ThinkScrubber()
        s.scrub("<thi")
        s.reset()
        # After reset, buffer is empty; new text passes through.
        assert s.scrub("nk>visible") == "nk>visible"


# ============================================================================
# ScrubberSink tests
# ============================================================================


class _FakeInnerSink:
    """Records calls for assertion."""

    def __init__(self) -> None:
        self.chunks: List[str] = []
        self.finals: List[str] = []
        self.thinkings: List[str] = []
        self.tool_starts: List[str] = []
        self.tool_completes: List[str] = []
        self.errors: List[str] = []
        self.closed: bool = False

    @property
    def supports_streaming(self) -> bool:
        return True

    async def emit_chunk(self, chunk: str) -> None:
        self.chunks.append(chunk)

    async def emit_thinking(self, content: str) -> None:
        self.thinkings.append(content)

    async def emit_tool_start(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.tool_starts.append(name)

    async def emit_tool_complete(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.tool_completes.append(name)

    async def emit_error(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.errors.append(content)

    async def emit_final(self, content: str) -> None:
        self.finals.append(content)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture()
def sink_pair() -> tuple[ScrubberSink, _FakeInnerSink]:
    inner = _FakeInnerSink()
    return ScrubberSink(inner), inner


class TestScrubberSinkChunk:
    """emit_chunk scrubbing."""

    @pytest.mark.asyncio
    async def test_clean_chunk_forwarded(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_chunk("hello")
        assert inner.chunks == ["hello"]

    @pytest.mark.asyncio
    async def test_think_chunk_suppressed(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_chunk("<think>hidden</think>")
        assert inner.chunks == []

    @pytest.mark.asyncio
    async def test_mixed_chunk(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_chunk("before<think>x</think>after")
        assert inner.chunks == ["beforeafter"]

    @pytest.mark.asyncio
    async def test_cross_chunk_scrubbing(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_chunk("hi<thi")
        await scrubber.emit_chunk("nk>secret</think>ok")
        # First chunk: "hi" is emitted, "<thi" buffered.
        # Second chunk: completes <think>, scrubs "secret", emits "ok".
        assert "".join(inner.chunks) == "hiok"

    @pytest.mark.asyncio
    async def test_empty_chunk_no_forward(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_chunk("")
        assert inner.chunks == []


class TestScrubberSinkFinal:
    """emit_final scrubbing."""

    @pytest.mark.asyncio
    async def test_final_scrubbed_independently(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_final("answer<think>reason</think> done")
        assert inner.finals == ["answer done"]

    @pytest.mark.asyncio
    async def test_final_clean_passthrough(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_final("just text")
        assert inner.finals == ["just text"]


class TestScrubberSinkPassthrough:
    """Non-scrubbed methods are forwarded unchanged."""

    @pytest.mark.asyncio
    async def test_thinking_passthrough(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_thinking("reasoning content")
        assert inner.thinkings == ["reasoning content"]

    @pytest.mark.asyncio
    async def test_tool_start_passthrough(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_tool_start("search", metadata={"key": "val"})
        assert inner.tool_starts == ["search"]

    @pytest.mark.asyncio
    async def test_tool_complete_passthrough(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_tool_complete("search")
        assert inner.tool_completes == ["search"]

    @pytest.mark.asyncio
    async def test_error_passthrough(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_error("boom", metadata={"severity": "high"})
        assert inner.errors == ["boom"]

    @pytest.mark.asyncio
    async def test_supports_streaming_delegated(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        assert scrubber.supports_streaming is True


class TestScrubberSinkClose:
    """close() flushes residual buffer and closes inner."""

    @pytest.mark.asyncio
    async def test_close_flushes_residual(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_chunk("text<thi")
        await scrubber.close()
        # "<thi" was buffered in NORMAL state, flush emits it.
        assert "".join(inner.chunks) == "text<thi"
        assert inner.closed is True

    @pytest.mark.asyncio
    async def test_close_discards_think_residual(
        self, sink_pair: tuple[ScrubberSink, _FakeInnerSink]
    ) -> None:
        scrubber, inner = sink_pair
        await scrubber.emit_chunk("<think>stuff")
        await scrubber.close()
        # In IN_THINK state, flush discards — nothing extra emitted.
        assert inner.chunks == []
        assert inner.closed is True
