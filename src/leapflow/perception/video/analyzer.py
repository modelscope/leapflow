"""Multi-scale video analysis via VLM.

Three-level progressive analysis:
  L1 (macro)  — full segment + event timeline → workflow actions
  L2 (moment) — targeted queries on flagged timestamps
  L3 (detail) — frame extraction for OCR / UI element identification
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.perception.types import (
    MacroAnalysisResult,
    VideoAction,
    VideoSegment,
)
from leapflow.perception.video.prompts import (
    AnalysisPromptStrategy,
    DashScopeMessageBuilder,
    DefaultAnalysisPrompts,
)
from leapflow.perception.video.segmenter import AnalysisSegment
from leapflow.perception.video.timeline import TimelineReader

logger = logging.getLogger(__name__)


@runtime_checkable
class FrameExtractor(Protocol):
    """帧提取能力的抽象接口。"""

    async def extract(
        self, video_path: str, timestamp_s: float, *, max_size: int = 1024,
    ) -> Optional[bytes]:
        """从视频中提取指定时间戳的帧，返回图片字节数据。"""
        ...


class LLMProvider(Protocol):
    async def achat(self, messages: list, *, stream: bool = True, **kw) -> Any: ...


class VideoAnalyzer:
    """Progressive multi-scale video analysis engine.

    Prompt construction and VLM message formatting are delegated to an
    ``AnalysisPromptStrategy``.  When none is provided, a
    ``DefaultAnalysisPrompts`` backed by ``DashScopeMessageBuilder`` is
    used, preserving full backward compatibility.
    """

    def __init__(
        self,
        vlm: LLMProvider,
        *,
        l2_enabled: bool = True,
        l3_enabled: bool = True,
        max_l2_requests: int = 10,
        max_l3_requests: int = 5,
        l2_time_window_s: float = 2.0,
        frame_extractor: Optional[FrameExtractor] = None,
        prompt_strategy: Optional[AnalysisPromptStrategy] = None,
        url_scheme: str = "file://",
        vlm_max_retries: int = 2,
        vlm_retry_backoff_s: float = 1.0,
    ) -> None:
        self._vlm = vlm
        self._l2_enabled = l2_enabled
        self._l3_enabled = l3_enabled
        self._max_l2_requests = max_l2_requests
        self._max_l3_requests = max_l3_requests
        self._l2_time_window_s = l2_time_window_s
        self._frame_extractor = frame_extractor

        # Prompt / message strategy (OCP: swap via constructor)
        if prompt_strategy is not None:
            self._prompts: AnalysisPromptStrategy = prompt_strategy
        else:
            self._prompts = DefaultAnalysisPrompts(
                message_builder=DashScopeMessageBuilder(url_scheme=url_scheme),
            )

        # Retry configuration (injected, no global dependency)
        self._vlm_max_retries = vlm_max_retries
        self._vlm_retry_backoff_s = vlm_retry_backoff_s

    # ── VLM Retry Helper ──

    async def _call_vlm_with_retry(self, messages: list, *, stream: bool = False) -> Optional[str]:
        """VLM 调用封装，带可配置重试和指数退避。"""
        max_retries = self._vlm_max_retries
        backoff_s = self._vlm_retry_backoff_s
        for attempt in range(max_retries + 1):
            try:
                resp = await self._vlm.achat(messages, stream=stream)
                text = resp.content if hasattr(resp, "content") else str(resp)
                if text and text.strip():
                    return text
                logger.debug("VLM returned empty response on attempt %d", attempt)
            except Exception as exc:
                logger.warning(
                    "VLM call failed (attempt %d/%d): %s",
                    attempt + 1, max_retries + 1, exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(backoff_s * (attempt + 1))
        return None

    async def analyze(
        self,
        segments: List[AnalysisSegment],
        timeline: TimelineReader,
        *,
        goal: str = "",
        progress: Optional[Callable] = None,
    ) -> List[VideoAction]:
        """Run L1→L2→L3 analysis and return time-sorted actions."""
        all_actions: List[VideoAction] = []

        for idx, seg in enumerate(segments):
            if progress:
                progress(f"Analyzing segment {idx + 1}/{len(segments)}")

            l1 = await self._analyze_macro(seg, timeline, goal=goal)
            all_actions.extend(l1.actions)

            if self._l2_enabled and l1.detail_requests:
                l2_actions = await self._analyze_moments(seg, l1, timeline)
                all_actions = _merge_refined(all_actions, l2_actions)

            # L3: 帧提取精确分析（仅当启用且有提取器时）
            if self._l3_enabled and self._frame_extractor and l1.frame_requests:
                l3_actions = await self._analyze_details(seg, l1, timeline)
                all_actions = _merge_refined(all_actions, l3_actions)

        all_actions.sort(key=lambda a: a.start_time)
        return all_actions

    # ── L1: Macro ──
    
    async def _analyze_macro(
        self, seg: AnalysisSegment, timeline: TimelineReader, *, goal: str,
    ) -> MacroAnalysisResult:
        top_apps = ", ".join(
            sorted(seg.app_summary, key=seg.app_summary.get, reverse=True)[:5]
        ) or "unknown"

        base_time = seg.segment.start_time + seg.start_offset
        tl_text = timeline.format_for_prompt(
            base_time,
            end_time=seg.segment.start_time + seg.end_offset,
        )

        messages = self._prompts.build_l1_messages(
            video_path=str(seg.segment.file_path),
            duration=seg.duration,
            top_apps=top_apps,
            timeline_text=tl_text,
            goal=goal or "not specified \u2014 infer from observations",
        )
    
        text = await self._call_vlm_with_retry(messages)
        if text is None:
            logger.warning("video_analyzer.l1_failed segment=%s (all retries exhausted)", seg.segment.segment_id)
            return MacroAnalysisResult()
        return _parse_macro_result(text, seg)

    # ── L2: Moment ──

    async def _analyze_moments(
        self, seg: AnalysisSegment, l1: MacroAnalysisResult, timeline: TimelineReader,
    ) -> List[VideoAction]:  # pragma: no branch
        actions: List[VideoAction] = []
        base_time = seg.segment.start_time + seg.start_offset

        for req in l1.detail_requests[:self._max_l2_requests]:
            ts = float(req.get("timestamp", 0))
            reason = req.get("reason", "")
            messages = self._prompts.build_l2_messages(
                video_path=str(seg.segment.file_path),
                timestamp=ts,
                reason=reason,
                context="",
            )
            text = await self._call_vlm_with_retry(messages)
            if text:
                # L2 解析 VLM JSON 响应，提取动态 confidence
                parsed = _parse_json_response(text)
                if parsed:
                    confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.7))))
                    description = parsed.get("description", text[:200] if text else "")
                    action_name = parsed.get("action", "interaction")
                    app = parsed.get("app", "")
                else:
                    confidence = 0.7
                    description = text.strip()[:500]
                    action_name = "interaction"
                    app = ""

                # L2 绝对时间计算：段基准时间 + 相对时间戳
                window = self._l2_time_window_s
                absolute_start = base_time + max(0.0, ts - window)
                absolute_end = base_time + ts + window
                actions.append(VideoAction(
                    action_name=action_name,
                    description=description[:500],
                    start_time=absolute_start,
                    end_time=absolute_end,
                    confidence=confidence,
                    app=app,
                    analysis_level=2,
                ))
            else:
                logger.debug("video_analyzer.l2_failed ts=%.1f (all retries exhausted)", ts)
        return actions

    # ── L3: Detail ──

    async def _analyze_details(
        self,
        seg: AnalysisSegment,
        l1: MacroAnalysisResult,
        timeline: TimelineReader,
    ) -> List[VideoAction]:
        """L3 微观分析: 从视频提取帧，进行精确内容分析。

        仅在以下场景触发（由 L1 的 frame_requests 驱动）:
        1. 需要精确文本内容（OCR）
        2. 需要 UI 元素标签
        3. 对话框/弹窗内容
        """
        if not self._frame_extractor:
            return []

        actions: List[VideoAction] = []
        base_time = seg.segment.start_time + seg.start_offset
        video_path = str(seg.segment.file_path)

        for req in l1.frame_requests[: self._max_l3_requests]:
            ts = float(req.get("timestamp", 0))
            reason = req.get("reason", "")

            # 1) 帧提取
            try:
                frame_bytes = await self._frame_extractor.extract(video_path, ts)
            except Exception:
                logger.debug("video_analyzer.l3_frame_extract_failed ts=%.1f", ts, exc_info=True)
                continue
            if not frame_bytes:
                logger.debug("video_analyzer.l3_frame_empty ts=%.1f", ts)
                continue

            # 2) 构建 VLM 图片分析请求
            nearby = timeline.format_for_prompt(base_time + ts) if hasattr(timeline, "format_for_prompt") else ""
            messages = self._prompts.build_l3_messages(
                image_data=frame_bytes,
                timestamp=ts,
                reason=reason,
                nearby_events=nearby or "none",
            )

            # 3) VLM 分析 (带重试)
            text = await self._call_vlm_with_retry(messages)
            if text is None:
                logger.debug("video_analyzer.l3_vlm_failed ts=%.1f (all retries exhausted)", ts)
                continue
            parsed = _parse_json_response(text) or {}

            # 4) 构造 VideoAction
            content_type = parsed.get("content_type", "unknown")
            extracted_text = parsed.get("text", "")
            ui_elements = parsed.get("ui_elements", [])
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))

            description_parts = []
            if extracted_text:
                description_parts.append(f"[OCR] {extracted_text}")
            if ui_elements:
                labels = ", ".join(str(e) for e in ui_elements[:10])
                description_parts.append(f"[UI] {labels}")
            if not description_parts:
                description_parts.append(f"[{content_type}] frame analysis at {ts:.1f}s")

            frame_ref = f"l3_frame_{ts:.1f}s"
            actions.append(VideoAction(
                action_name=f"detail_{content_type}",
                description="; ".join(description_parts)[:500],
                start_time=base_time + max(0.0, ts - 0.5),
                end_time=base_time + ts + 0.5,
                confidence=confidence,
                analysis_level=3,
                frame_refs=(frame_ref,),
            ))

        return actions


# ── Helpers ──


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """鲁棒的 JSON 响应解析，支持多种 VLM 输出格式。

    解析策略:
      1. 直接 json.loads
      2. 提取 ```json ... ``` 代码块
      3. 正则匹配第一个 {...} 块
      4. 失败记录原始响应到 debug 日志
    """
    if not text or not text.strip():
        return None

    # 1. 直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. 提取 markdown 代码块
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. 正则 fallback: 第一个 {...} 块
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. 全部失败
    logger.debug("JSON parse failed for VLM response: %s", text[:200])
    return None


def _parse_l3_result(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from L3 VLM response."""
    return _parse_json_response(text) or {}


def _validate_requests(requests: Any) -> List[Dict[str, Any]]:
    """Validate and normalize VLM request list format."""
    if not isinstance(requests, list):
        return []
    return [
        r for r in requests
        if isinstance(r, dict) and "timestamp" in r
    ]


def _parse_macro_result(text: str, seg: AnalysisSegment) -> MacroAnalysisResult:
    """Best-effort JSON extraction from VLM response."""
    data = _parse_json_response(text)
    if data is None:
        return MacroAnalysisResult()

    base = seg.segment.start_time + seg.start_offset
    actions: List[VideoAction] = []
    for step in data.get("steps", []):
        actions.append(VideoAction(
            action_name=step.get("action", "unknown"),
            description=step.get("description", ""),
            start_time=base + float(step.get("start_time", 0)),
            end_time=base + float(step.get("end_time", 0)),
            app=step.get("app", ""),
            goal=step.get("goal", ""),
            confidence=float(step.get("confidence", 0.5)),
            analysis_level=1,
        ))

    return MacroAnalysisResult(
        actions=actions,
        overall_goal=data.get("overall_goal", ""),
        detail_requests=_validate_requests(data.get("detail_requests", [])),
        frame_requests=_validate_requests(data.get("frame_requests", [])),
    )


def _merge_refined(base: List[VideoAction], refined: List[VideoAction]) -> List[VideoAction]:
    """Merge refined (L2/L3) actions into the base list.

    L2 语义：替换重叠的低级别动作。
    L3 语义：丰富（追加描述、合并 frame_refs）重叠的动作，保留原 action_name。
    """
    if not refined:
        return base
    result = list(base)
    for r in refined:
        replaced = False
        for i, b in enumerate(result):
            if b.analysis_level < r.analysis_level and _overlaps(b, r):
                if r.analysis_level == 3:
                    # L3 丰富语义：保留原动作，追加详情与 frame_refs
                    enriched_desc = (
                        f"{b.description} | {r.description}".strip(" |")
                        if b.description else r.description
                    )
                    result[i] = VideoAction(
                        action_name=b.action_name,
                        description=enriched_desc[:500],
                        start_time=b.start_time,
                        end_time=b.end_time,
                        app=b.app,
                        goal=b.goal,
                        confidence=max(b.confidence, r.confidence),
                        analysis_level=max(b.analysis_level, r.analysis_level),
                        corroborating_events=b.corroborating_events,
                        frame_refs=tuple(b.frame_refs) + tuple(r.frame_refs),
                    )
                else:
                    # L2 替换语义
                    result[i] = VideoAction(
                        action_name=r.action_name if r.action_name != "detailed_action" else b.action_name,
                        description=r.description,
                        start_time=b.start_time,
                        end_time=b.end_time,
                        app=r.app or b.app,
                        goal=r.goal or b.goal,
                        confidence=max(b.confidence, r.confidence),
                        analysis_level=r.analysis_level,
                        corroborating_events=b.corroborating_events,
                        frame_refs=tuple(b.frame_refs) + tuple(r.frame_refs),
                    )
                replaced = True
                break
        if not replaced:
            result.append(r)
    return result


def _overlaps(a: VideoAction, b: VideoAction) -> bool:
    return a.start_time < b.end_time and b.start_time < a.end_time
