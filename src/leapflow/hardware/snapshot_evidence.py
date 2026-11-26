# Copyright (c) Alibaba, Inc. and its affiliates.
"""Opt-in, caller-owned retention of an already shared passive snapshot.

No device access, capture loop, or model invocation lives here. The injected
signal port admits metadata through the ordinary runtime boundary; image bytes
remain private session cache content and never reach global event subscribers.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import duckdb

from leapflow.cache.manager import CacheEntry, CacheManager
from leapflow.domain.effect_scope import EffectScope
from leapflow.domain.events import RecordScope, RecordSignalPort
from leapflow.hardware.stream import HardwareEvent, EventKind
from leapflow.layout import workspace_id_for_path
from leapflow.monitor.signal_noise import SignalNoiseConfig, SignalNoiseGate
from leapflow.security.actions import ActionDescriptor, ActionEffect, ActionKind
from leapflow.tools.execution_context import current_tool_context

logger = logging.getLogger(__name__)
_CATEGORY = "snapshot_evidence"
_SOURCE = "hardware_snapshot"


@dataclass(frozen=True)
class SnapshotEvidenceSettings:
    ttl_s: float = 300.0
    quota_bytes: int = 64 * 1024 * 1024
    max_frame_bytes: int = 1024 * 1024
    max_pixels: int = 4096 * 4096
    cleanup_interval_s: float = 5.0
    max_clip_bytes: int = 700 * 1024
    max_clip_s: float = 10.0

    @classmethod
    def from_json(cls, raw: str) -> SnapshotEvidenceSettings:
        values = json.loads(raw or "{}")
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError("hardware.snapshot_evidence contains unknown settings")
        return cls(**values)

    def __post_init__(self) -> None:
        for name, ceiling in (("ttl_s", 300.0), ("cleanup_interval_s", 60.0), ("max_clip_s", 10.0)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
                raise ValueError(f"snapshot evidence {name} must be finite in (0, {ceiling}]")
        for name, ceiling in (("quota_bytes", 64 * 1024 * 1024),
                              ("max_frame_bytes", 1024 * 1024), ("max_pixels", 4096 * 4096),
                              ("max_clip_bytes", 700 * 1024)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"snapshot evidence {name} must be an integer in [1, {ceiling}]")


@dataclass(frozen=True)
class EvidenceReference:
    artifact_id: str
    observation_id: str
    frame: str
    capture_id: str
    camera: str
    observed_at: float
    expires_at: float
    content_hash: str
    mime: str
    size_bytes: int
    sensitive: bool = True
    syncable: bool = False
    kind: str = "image"
    duration_s: float | None = None


@dataclass(frozen=True)
class EvidenceFailure:
    code: str
    message: str
    category: str = "validation"
    source: str = "hardware.snapshot_evidence"
    recoverable: bool = False
    side_effect_state: str = "none"
    next_step: str = "Request a new authorized passive snapshot; never reuse stale evidence."


class EvidenceError(Exception):
    def __init__(self, code: str, message: str, *, category: str = "validation") -> None:
        super().__init__(message)
        next_step = ("Stop this attempt. Do not retry, rephrase, or use another tool without new explicit user consent."
                     if category == "security" else EvidenceFailure.__dataclass_fields__["next_step"].default)
        self.failure = EvidenceFailure(code, message, category, next_step=next_step)


@runtime_checkable
class SessionEvidenceSink(Protocol):
    async def retain_snapshot(
        self, snapshot: Mapping[str, Any], *, workcell: str, started_at: float,
    ) -> EvidenceReference: ...

    async def aclose(self) -> None: ...


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise EvidenceError("evidence_invalid", f"Snapshot lacks bounded {name} identity")
    return value


def _decode_video(video: Any, settings: SnapshotEvidenceSettings) -> tuple[bytes, str, str, float]:
    """Probe bounded embedded clips locally, never fetch a URL or transcode evidence."""
    if not isinstance(video, Mapping) or set(video) - {"mime", "data", "label"}:
        raise EvidenceError("evidence_missing", "Clip did not share admitted video")
    encoded, mime = video.get("data"), video.get("mime")
    if not isinstance(encoded, str) or len(encoded) > 4 * ((settings.max_clip_bytes + 2) // 3):
        raise EvidenceError("evidence_invalid", "Clip exceeds its encoded byte bound")
    try:
        content = base64.b64decode(encoded, validate=True)
        suffix = {"video/mp4": ".mp4", "video/webm": ".webm"}.get(mime, "")
        valid_container = (mime == "video/mp4" and content[4:8] == b"ftyp" or
                           mime == "video/webm" and content.startswith(b"\x1a\x45\xdf\xa3"))
        if not suffix or not valid_container or not 0 < len(content) <= settings.max_clip_bytes:
            raise ValueError("unsupported or mismatched video container")
    except ValueError as exc:
        raise EvidenceError("evidence_invalid", "Clip is not bounded embedded MP4 or WebM") from exc
    try:
        import cv2
    except ImportError as exc:
        raise EvidenceError("evidence_unavailable", "Video validation requires the optional OpenCV dependency") from exc
    with tempfile.TemporaryDirectory(prefix="leap-evidence-") as directory:
        path = Path(directory) / f"clip{suffix}"
        path.write_bytes(content)
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or not math.isfinite(fps) or not 0 < fps <= 32:
                raise ValueError("invalid video rate")
            frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                if (frames > 32 or frame is None or frame.shape[0] * frame.shape[1] > settings.max_pixels
                        or frames / fps > settings.max_clip_s + 0.001):
                    raise ValueError("oversized clip")
            duration = frames / fps
            declared_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if frames < 4 or not math.isfinite(declared_frames) or abs(declared_frames - frames) > 0.5:
                raise ValueError("truncated or too-short clip")
        except (ValueError, cv2.error) as exc:
            raise EvidenceError("evidence_invalid", "Clip must decode completely within frame, pixel and duration bounds") from exc
        finally:
            capture.release()
    return content, mime, suffix, duration


def _decode(image: Any, settings: SnapshotEvidenceSettings) -> tuple[bytes, str, str]:
    """Validate bounded encoded media and decoded dimensions before consent/storage."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise EvidenceError("evidence_unavailable", "Image validation requires the optional Pillow dependency") from exc

    if not isinstance(image, Mapping) or set(image) - {"mime", "data", "label"}:
        raise EvidenceError("evidence_missing", "Snapshot did not share an admitted image")
    encoded = image.get("data")
    if not isinstance(encoded, str) or len(encoded) > 4 * ((settings.max_frame_bytes + 2) // 3):
        raise EvidenceError("evidence_invalid", "Snapshot encoded frame exceeds its byte bound")
    try:
        content = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(content)) as decoded:
            formats = {"JPEG": ("image/jpeg", ".jpg"), "PNG": ("image/png", ".png"),
                       "WEBP": ("image/webp", ".webp")}
            mime, suffix = formats.get(decoded.format, ("", ""))
            if (not mime or image.get("mime") != mime or decoded.width * decoded.height > settings.max_pixels
                    or getattr(decoded, "n_frames", 1) != 1 or not 0 < len(content) <= settings.max_frame_bytes):
                raise ValueError("unsupported, oversized, animated, or mismatched image")
            decoded.verify()
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        raise EvidenceError("evidence_invalid", "Snapshot is not a valid bounded JPEG, PNG, or WebP") from exc
    return content, mime, suffix


class SnapshotEvidenceService:
    """Session-scoped signal admission, retention consent, and bounded storage."""

    def __init__(self, settings: Callable[[], Any], signals: RecordSignalPort) -> None:
        self._settings = settings
        self._signals = signals
        self._scope = EffectScope("hardware-snapshot-evidence")
        self._scope.async_effect(self._close)
        self._sweeper: asyncio.Task[None] | None = None
        self._calls: set[asyncio.Task[Any]] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._retained: dict[str, tuple[CacheManager, CacheEntry]] = {}
        self._closed = False

    async def retain_snapshot(
        self, snapshot: Mapping[str, Any], *, workcell: str, started_at: float,
    ) -> EvidenceReference:
        task = asyncio.current_task()
        if task is not None:
            self._calls.add(task)
        try:
            return await self._retain(snapshot, workcell=workcell, started_at=started_at)
        except TimeoutError as exc:
            raise EvidenceError("evidence_stale", "Snapshot expired before retention completed") from exc
        except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
            logger.warning("Snapshot retention storage or configuration unavailable", exc_info=True)
            raise EvidenceError("evidence_unavailable", "Retention is unavailable; no reference published", category="storage") from exc
        finally:
            if task is not None:
                self._calls.discard(task)

    async def _retain(
        self, snapshot: Mapping[str, Any], *, workcell: str, started_at: float,
    ) -> EvidenceReference:
        context = current_tool_context()
        runtime = self._settings()
        if self._closed or context is None or not context.session_id or not context.task_id:
            raise EvidenceError("evidence_unavailable", "Retention requires an active owning session")
        if not runtime.hardware_share_frames:
            raise EvidenceError("evidence_sharing_disabled", "Enable hardware.share_frames with user approval first")
        if not isinstance(self._signals, RecordSignalPort):
            raise EvidenceError("evidence_unavailable", "Scoped signal admission is unavailable")
        settings = SnapshotEvidenceSettings.from_json(runtime.hardware_snapshot_evidence)
        frame, capture, camera = (_text(snapshot.get(k), k) for k in ("frame", "capture_id", "camera"))
        observed = snapshot.get("observed_at")
        remaining = snapshot.get("valid_for_s")
        if (snapshot.get("validity_timebase") != "node_remaining_duration"
                or type(remaining) not in (int, float) or not math.isfinite(remaining) or remaining <= 0
                or type(observed) not in (int, float) or not math.isfinite(observed)
                or type(started_at) not in (int, float) or not math.isfinite(started_at)
                or started_at > time.monotonic()):
            raise EvidenceError("evidence_stale", "Native remaining-validity evidence is missing or expired")
        # Whole host elapsed time is deliberately conservative across node clocks.
        deadline = started_at + min(settings.ttl_s, remaining)
        kind = "video" if "video" in snapshot else "image"
        if "video" in snapshot and "image" in snapshot:
            raise EvidenceError("evidence_invalid", "Export exactly one evidence kind at a time")
        try:
            decoded = await asyncio.wait_for(
                asyncio.to_thread(_decode_video if kind == "video" else _decode, snapshot.get(kind), settings),
                max(0, deadline - time.monotonic()))
            content, mime, suffix = decoded[:3]
            duration = decoded[3] if kind == "video" else None
        except TimeoutError as exc:
            raise EvidenceError("evidence_stale", "Evidence expired during media validation") from exc
        if deadline <= time.monotonic():
            raise EvidenceError("evidence_stale", "Snapshot expired during transport or validation")
        workspace = workspace_id_for_path(context.workspace_root)
        scope = RecordScope(runtime.profile, workspace, context.session_id)
        cache = await asyncio.wait_for(
            self._preflight(runtime, workspace, context.session_id, len(content), settings.quota_bytes),
            max(0, deadline - time.monotonic()),
        )
        digest = hashlib.sha256(content).hexdigest()
        observation = uuid.uuid4().hex
        event = HardwareEvent(
            kind=EventKind.SNAPSHOT_EVIDENCE, device_id=workcell, channel_id=camera,
            quantity="clip_reference" if kind == "video" else "snapshot_reference",
            detail=f"Caller-owned transient {kind} reference", observed_at=observed,
            value={"observation_id": observation, "frame": frame, "capture_id": capture,
                   "camera": camera, "observed_at": observed, "content_hash": digest,
                   "kind": kind, "duration_s": duration, "sensitive": True, "syncable": False,
                   "media_source": snapshot.get("source", "live"),
                   "run_id": snapshot.get("run_id", ""), "historical": snapshot.get("historical", False)},
        )
        try:
            receipt = await asyncio.wait_for(self._signals.admit_record(
                event.event_type, event.to_payload(), scope=scope,
                dedup_key=hashlib.sha256(json.dumps([workcell, capture, frame, digest]).encode()).hexdigest(),
                noise_gate=SignalNoiseGate(SignalNoiseConfig.from_settings(runtime)),
            ), max(0, deadline - time.monotonic()))
        except TimeoutError as exc:
            raise EvidenceError("evidence_stale", "Snapshot expired during record admission") from exc
        except Exception as exc:  # noqa: BLE001 - a failed signal/privacy gate never admits evidence
            raise EvidenceError("evidence_signal_refused", "Snapshot signal admission failed", category="privacy") from exc
        if not receipt.admitted:
            raise EvidenceError("evidence_signal_refused", f"Snapshot record admission: {receipt.reason}", category="privacy")
        if receipt.scope != scope or receipt.event_type != "hw.snapshot_evidence":
            raise EvidenceError("evidence_signal_refused", "Snapshot admission returned a foreign receipt", category="privacy")
        record = json.loads(receipt.payload_json)
        if record.get("value") != event.value:
            raise EvidenceError("evidence_signal_refused", "Snapshot provenance was removed or changed by admission", category="privacy")
        await self._consent(context, scope, settings, len(content), kind=kind, duration=duration)
        if deadline <= time.monotonic() or self._closed:
            raise EvidenceError("evidence_stale", "Snapshot expired while awaiting retention consent")
        expires = time.time() + (deadline - time.monotonic())
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._expire(), name="snapshot-evidence-expiry")
        entry = await self._persist(cache, content, workspace_id=workspace, session_id=context.session_id,
                                    category=_CATEGORY, source=_SOURCE, suffix=suffix, expires_at=expires,
                                    monotonic_deadline=deadline, max_session_bytes=settings.quota_bytes,
                                    metadata={"mime": mime, "kind": kind, "duration_s": duration,
                                              "observation_id": observation, "record": record})
        return EvidenceReference(entry.entry_id, observation, frame, capture, camera, observed,
                                 expires, digest, mime, len(content), kind=kind, duration_s=duration)

    @staticmethod
    async def _preflight(runtime: Any, workspace: str, session: str, size: int, quota: int) -> CacheManager:
        cache = await asyncio.to_thread(CacheManager, runtime.profile_layout.cache, profile_id=runtime.profile)
        await asyncio.to_thread(cache.cleanup_expired, category=_CATEGORY)
        used = await asyncio.to_thread(cache.list_entries, scope="session", workspace_id=workspace,
                                       session_id=session)
        if sum(entry.size_bytes for entry in used) + size > quota:
            raise EvidenceError("evidence_quota_full", "Session quota is full; no existing artifact is evicted")
        return cache

    async def _consent(self, context: Any, scope: RecordScope, settings: SnapshotEvidenceSettings, size: int,
                       *, kind: str, duration: float | None) -> None:
        evaluate = getattr(context.orchestrator, "evaluate", None)
        if not callable(evaluate):
            raise EvidenceError("evidence_approval_unavailable", "Retention needs an interactive approval route", category="security")
        action = ActionDescriptor(
            kind=ActionKind.FILE_WRITE.value, effect=ActionEffect.WRITE.value,
            summary="Retain passive camera evidence in this session",
            detail=f"Keep one sensitive {f'{duration:g}-second clip' if kind == 'video' else 'snapshot'} "
                   f"for at most {settings.ttl_s:g} seconds, capped by native validity. "
                   + ("The same snapshot may be shown to the primary model under frame-sharing consent. "
                      if kind == "image" else "Clip bytes are not attached to primary chat. ")
                   + "Any specialist-service transfer needs independent destination consent.",
            resource=f"session-evidence:{scope.workspace_id}/{scope.session_id}",
            session_id=scope.session_id, turn_id=context.task_id,
            metadata={"platform": "hardware_snapshot", "sensitivity_category": "cache_sensitive",
                      "bytes": size, "ttl_s": settings.ttl_s, "kind": kind, "duration_s": duration,
                      "syncable": False},
        )
        try:
            decision = await evaluate(action)
        except Exception as exc:  # noqa: BLE001 - a broken approval must fail closed
            logger.warning("Snapshot retention approval failed", exc_info=True)
            raise EvidenceError("evidence_approval_unavailable", "Retention approval failed; no image retained", category="security") from exc
        if not getattr(decision, "approved", False):
            raise EvidenceError("evidence_approval_denied", str(getattr(decision, "denial_message", "") or "Retention denied; do not retry or use another tool"), category="security")

    async def _persist(self, cache: CacheManager, content: bytes, **fields: Any) -> CacheEntry:
        """Cancellation returns immediately; scope owns cleanup of a late disk write."""
        write = asyncio.create_task(asyncio.to_thread(cache.persist_session_content, content, **fields))
        try:
            entry = await asyncio.wait_for(asyncio.shield(write), max(0, fields["monotonic_deadline"] - time.monotonic()))
        except (asyncio.CancelledError, TimeoutError):
            cleanup = asyncio.create_task(self._discard_write(cache, write), name="snapshot-evidence-cancel-cleanup")
            self._cleanup_tasks.add(cleanup)
            cleanup.add_done_callback(self._cleanup_tasks.discard)
            raise
        self._retained[entry.entry_id] = (cache, entry)
        return entry

    async def _discard_write(self, cache: CacheManager, write: asyncio.Task[CacheEntry]) -> None:
        try:
            entry = await write
            await asyncio.to_thread(cache.discard_session_content, entry)
        except (OSError, RuntimeError, ValueError, duckdb.Error):
            logger.warning("Cancelled snapshot write cleanup unavailable; TTL sweep remains active", exc_info=True)

    async def _expire(self) -> None:
        interval = SnapshotEvidenceSettings().cleanup_interval_s
        while not self._closed:
            try:
                runtime = self._settings()
                interval = SnapshotEvidenceSettings.from_json(runtime.hardware_snapshot_evidence).cleanup_interval_s
            except (ValueError, RuntimeError):
                logger.warning("Snapshot expiry settings unavailable; keeping last valid interval", exc_info=True)
            await asyncio.sleep(interval)
            for key, (cache, entry) in tuple(self._retained.items()):
                if entry.metadata["_monotonic_deadline"] > time.monotonic():
                    continue
                try:
                    await asyncio.to_thread(cache.discard_session_content, entry)
                except (OSError, RuntimeError, ValueError, duckdb.Error):
                    logger.warning("Snapshot expiry deletion unavailable; retaining cleanup ownership", exc_info=True)
                else:
                    self._retained.pop(key, None)

    async def _close(self) -> None:
        self._closed = True
        tasks = [task for task in self._calls if task is not asyncio.current_task()]
        if self._sweeper is not None:
            tasks.append(self._sweeper)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        for cache, entry in tuple(self._retained.values()):
            try:
                await asyncio.to_thread(cache.discard_session_content, entry)
            except (OSError, RuntimeError, ValueError, duckdb.Error):
                logger.warning("Snapshot shutdown cleanup unavailable", exc_info=True)
        self._retained.clear()

    async def aclose(self) -> None:
        await self._scope.async_dispose()


async def export_snapshot_result(
    sink: SessionEvidenceSink | None, result: dict[str, Any], *, workcell: str, started_at: float,
    share_frames: bool = False,
) -> dict[str, Any]:
    """Consume volatile bytes; attach the exact retained image only under sharing consent."""
    snapshot = dict(result.get("result") or {})
    result["result"] = {key: value for key, value in snapshot.items() if key not in {"image", "video"}}
    result.pop("_attachments", None)
    if not result.get("ok"):
        return {**result, "evidence": {"status": "missing", "reason": "snapshot_failed"}}
    try:
        if not isinstance(sink, SessionEvidenceSink):
            raise EvidenceError("evidence_unavailable", "Session evidence sink is not bound")
        reference = await sink.retain_snapshot(snapshot, workcell=workcell, started_at=started_at)
    except EvidenceError as exc:
        return {**result, "ok": False, "evidence": {"status": "refused"},
                "error": str(exc), "failure": asdict(exc.failure),
                "failure_code": exc.failure.code, "retryable": False,
                "next_step": exc.failure.next_step}
    if share_frames and reference.kind == "image":
        from leapflow.hardware.skill_results import share_frame
        result["result"] = snapshot
        share_frame(result, consent=True)
    return {**result, "evidence": {"status": "retained", **asdict(reference)},
            "next_step": "This is caller-owned evidence, not a decision. You decide whether to use the shared "
                         "snapshot, consult an advisory service, or gather more context. If consulting, check "
                         "model_describe capacity, label artifacts by viewpoint (scene, wrist, scene_clip), "
                         "provide concrete eligible candidates and put measurements in state. "
                         "Temporal clips can support stability or slip questions; score margins are uncalibrated. "
                         "Runtime requests service consent; do not ask for duplicate consent in dialogue."}
