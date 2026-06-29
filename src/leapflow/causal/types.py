"""Core data model for the Causal Propagation Chain.

Three levels of abstraction:
    CausalEvent  — atomic particle on the propagation chain
    CausalChain  — minimal causal propagation unit (trigger → responses → effects)
    CausalGraph  — session-level snapshot of all propagation chains
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple


class EventType(str, Enum):
    TRIGGER = "trigger"
    RESPONSE = "response"
    EFFECT = "effect"
    NAVIGATION = "navigation"
    BOUNDARY = "boundary"
    NOISE = "noise"


class EventSource(str, Enum):
    SIGNAL = "signal"
    VISUAL = "visual"
    SYSTEM = "system"
    INFERRED = "inferred"


@dataclass
class CausalEvent:
    """Atomic particle on the propagation chain."""

    id: str
    timestamp: float
    event_type: EventType
    source: EventSource
    channel: str
    payload: Dict[str, Any]
    confidence: float = 1.0

    caused_by: Optional[str] = None
    causes: List[str] = field(default_factory=list)
    frame_refs: List[str] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id() -> str:
        return uuid.uuid4().hex[:12]


@dataclass
class FrameRef:
    """Visual evidence reference within a causal chain."""

    frame_id: str
    timestamp: float
    role: str  # "before" / "during" / "after" / "evidence"


@dataclass
class CausalChain:
    """Minimal causal propagation unit: trigger → responses → effects."""

    id: str
    trigger: CausalEvent
    responses: List[CausalEvent] = field(default_factory=list)
    effects: List[CausalEvent] = field(default_factory=list)
    frames: List[FrameRef] = field(default_factory=list)
    time_span: Tuple[float, float] = (0.0, 0.0)
    closed_by: str = ""
    completeness: float = 1.0
    semantic_label: str = ""

    def event_ids(self) -> List[str]:
        ids = [self.trigger.id]
        ids.extend(e.id for e in self.responses)
        ids.extend(e.id for e in self.effects)
        return ids

    def all_events(self) -> List[CausalEvent]:
        return [self.trigger] + self.responses + self.effects

    @property
    def duration(self) -> float:
        return self.time_span[1] - self.time_span[0]


class CausalGraph:
    """Session-level snapshot of all propagation chains. Incrementally built."""

    __slots__ = ("events", "chains", "_by_channel", "_by_chain", "_ring_limit", "metadata")

    def __init__(self, ring_limit: int = 100) -> None:
        self.events: Dict[str, CausalEvent] = {}
        self.chains: List[CausalChain] = []
        self._by_channel: Dict[str, List[str]] = defaultdict(list)
        self._by_chain: Dict[str, str] = {}
        self._ring_limit = ring_limit
        self.metadata: Dict[str, Any] = {}

    # ── Write (append along propagation direction) ──

    def add_event(self, ev: CausalEvent) -> None:
        self.events[ev.id] = ev
        self._by_channel[ev.channel].append(ev.id)

    def add_edge(self, parent_id: str, child_id: str) -> None:
        parent = self.events.get(parent_id)
        child = self.events.get(child_id)
        if parent is None or child is None:
            return
        if child_id not in parent.causes:
            parent.causes.append(child_id)
        child.caused_by = parent_id

    def remove_edge(self, parent_id: str, child_id: str) -> None:
        parent = self.events.get(parent_id)
        child = self.events.get(child_id)
        if parent and child_id in parent.causes:
            parent.causes.remove(child_id)
        if child and child.caused_by == parent_id:
            child.caused_by = None

    def update_edge_confidence(self, parent_id: str, child_id: str, confidence: float) -> None:
        child = self.events.get(child_id)
        if child and child.caused_by == parent_id:
            child.confidence = confidence

    def add_chain(self, chain: CausalChain) -> None:
        self.chains.append(chain)
        for eid in chain.event_ids():
            self._by_chain[eid] = chain.id
        self._enforce_ring_limit()

    # ── Query (traverse along propagation direction) ──

    def get_chain_for_event(self, event_id: str) -> Optional[CausalChain]:
        chain_id = self._by_chain.get(event_id)
        if chain_id is None:
            return None
        for c in self.chains:
            if c.id == chain_id:
                return c
        return None

    def get_connected_component(self, event_id: str) -> Set[str]:
        """BFS from event_id following both caused_by and causes edges."""
        visited: Set[str] = set()
        queue: deque[str] = deque([event_id])
        while queue:
            eid = queue.popleft()
            if eid in visited:
                continue
            visited.add(eid)
            ev = self.events.get(eid)
            if ev is None:
                continue
            if ev.caused_by and ev.caused_by not in visited:
                queue.append(ev.caused_by)
            for child_id in ev.causes:
                if child_id not in visited:
                    queue.append(child_id)
        return visited

    def connected_components(self) -> List[Set[str]]:
        """Find all connected components (propagation islands)."""
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for eid in self.events:
            if eid not in visited:
                comp = self.get_connected_component(eid)
                visited.update(comp)
                components.append(comp)
        return components

    def topological_order(self) -> List[CausalEvent]:
        """Events in propagation order (parents before children)."""
        in_degree: Dict[str, int] = {eid: 0 for eid in self.events}
        for ev in self.events.values():
            for child_id in ev.causes:
                if child_id in in_degree:
                    in_degree[child_id] += 1

        queue: deque[str] = deque(
            eid for eid, deg in in_degree.items() if deg == 0
        )
        result: List[CausalEvent] = []
        while queue:
            eid = queue.popleft()
            ev = self.events[eid]
            result.append(ev)
            for child_id in ev.causes:
                if child_id in in_degree:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        queue.append(child_id)
        return result

    def chains_in_window(self, t0: float, t1: float) -> List[CausalChain]:
        return [
            c for c in self.chains
            if c.time_span[0] <= t1 and c.time_span[1] >= t0
        ]

    def events_by_channel(self, channel: str) -> List[CausalEvent]:
        return [self.events[eid] for eid in self._by_channel.get(channel, [])
                if eid in self.events]

    def iter_edges(self) -> Iterator[Tuple[str, str, float]]:
        """Yield (parent_id, child_id, confidence) for all edges."""
        for ev in self.events.values():
            for child_id in ev.causes:
                child = self.events.get(child_id)
                conf = child.confidence if child else 0.0
                yield (ev.id, child_id, conf)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def chain_count(self) -> int:
        return len(self.chains)

    # ── Serialization ──

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": {eid: _event_to_dict(ev) for eid, ev in self.events.items()},
            "chains": [_chain_to_dict(c) for c in self.chains],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
        graph = cls()
        for eid, ev_data in data.get("events", {}).items():
            graph.add_event(_event_from_dict(ev_data))
        for chain_data in data.get("chains", []):
            graph.add_chain(_chain_from_dict(chain_data, graph.events))
        graph.metadata = data.get("metadata", {})
        return graph

    # ── Internal ──

    def _enforce_ring_limit(self) -> None:
        while len(self.chains) > self._ring_limit:
            oldest = self.chains.pop(0)
            for eid in oldest.event_ids():
                self._by_chain.pop(eid, None)
                self.events.pop(eid, None)
                for ch_ids in self._by_channel.values():
                    if eid in ch_ids:
                        ch_ids.remove(eid)


# ── Serialization helpers ──

def _event_to_dict(ev: CausalEvent) -> Dict[str, Any]:
    return {
        "id": ev.id,
        "timestamp": ev.timestamp,
        "event_type": ev.event_type.value,
        "source": ev.source.value,
        "channel": ev.channel,
        "payload": ev.payload,
        "confidence": ev.confidence,
        "caused_by": ev.caused_by,
        "causes": ev.causes,
        "frame_refs": ev.frame_refs,
        "tags": ev.tags,
    }


def _event_from_dict(data: Dict[str, Any]) -> CausalEvent:
    return CausalEvent(
        id=data["id"],
        timestamp=data["timestamp"],
        event_type=EventType(data["event_type"]),
        source=EventSource(data["source"]),
        channel=data["channel"],
        payload=data.get("payload", {}),
        confidence=data.get("confidence", 1.0),
        caused_by=data.get("caused_by"),
        causes=data.get("causes", []),
        frame_refs=data.get("frame_refs", []),
        tags=data.get("tags", {}),
    )


def _chain_to_dict(chain: CausalChain) -> Dict[str, Any]:
    return {
        "id": chain.id,
        "trigger": _event_to_dict(chain.trigger),
        "responses": [_event_to_dict(e) for e in chain.responses],
        "effects": [_event_to_dict(e) for e in chain.effects],
        "frames": [{"frame_id": f.frame_id, "timestamp": f.timestamp, "role": f.role}
                   for f in chain.frames],
        "time_span": list(chain.time_span),
        "closed_by": chain.closed_by,
        "completeness": chain.completeness,
        "semantic_label": chain.semantic_label,
    }


def _chain_from_dict(data: Dict[str, Any], events: Dict[str, CausalEvent]) -> CausalChain:
    trigger = _event_from_dict(data["trigger"])
    responses = [_event_from_dict(e) for e in data.get("responses", [])]
    effects = [_event_from_dict(e) for e in data.get("effects", [])]
    frames = [FrameRef(f["frame_id"], f["timestamp"], f["role"]) for f in data.get("frames", [])]
    ts = data.get("time_span", [0.0, 0.0])
    return CausalChain(
        id=data["id"],
        trigger=trigger,
        responses=responses,
        effects=effects,
        frames=frames,
        time_span=(ts[0], ts[1]),
        closed_by=data.get("closed_by", ""),
        completeness=data.get("completeness", 1.0),
        semantic_label=data.get("semantic_label", ""),
    )
