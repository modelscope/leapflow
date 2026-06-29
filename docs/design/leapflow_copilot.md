# Workflow Copilot Design Document

## Overview

Workflow Copilot is LeapFlow's context-aware operation prediction engine. It continuously observes the user's operation signal stream and presents next-step suggestions as ghost-hints the moment the user pauses — similar to IDE code completion, but operating across application workflows. Core value: **Transform repetitive operation patterns from "user recall" to "system proactive prompting", reducing operation decision latency.**

## Design Philosophy

| SOLID Principle | Mapping in Copilot |
|:---:|:---|
| **S** — Single Responsibility | Each module does one thing: encoding, prediction, rendering, and feedback are independent |
| **O** — Open/Closed | `PredictorLayer` Protocol allows adding new prediction algorithms without modifying the engine |
| **L** — Liskov Substitution | Any implementation satisfying the Protocol can be hot-swapped directly |
| **I** — Interface Segregation | `Signal`, `HintRenderer`, `SignalChannel` are each minimized |
| **D** — Dependency Inversion | Engine depends on Protocols rather than concrete implementations; Store/LLM are injected |

**Ultimate Goal**: Through Loop γ (execution-as-learning) closed loop, Copilot's prediction accuracy monotonically increases over usage time, ultimately achieving self-evolving operation assistance.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Workflow Copilot                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EventBus ─→ ContextEncoder ─→ SpeculativePipeline              │
│                                      │                          │
│                              ┌───────┴───────┐                  │
│                              │ PredictionEngine                  │
│                              │  ┌──┐┌──┐┌──┐┌──┐               │
│                              │  │L0││L1││L2││L3│                │
│                              │  └──┘└──┘└──┘└──┘               │
│                              └───────┬───────┘                  │
│                                      │                          │
│  IdleDetector ─→ DisplayGate ─→ SuggestionRenderer              │
│                                      │                          │
│                              FeedbackCollector                   │
│                                      │                          │
│                              EvolutionLoop ──→ (write back to Layers) │
│                                                                 │
│  [DegradationPolicy] ←── SystemMetrics                          │
└─────────────────────────────────────────────────────────────────┘
```

| Module | Responsibility |
|:---|:---|
| `ContextEncoder` | Incrementally encodes SystemEvent stream into `ContextState` (O(1)/event) |
| `PredictionEngine` | Cascades all PredictorLayers, aggregates with deduplication + consensus boosting |
| `SpeculativePipeline` | Predicts during operations, three-tier cache ensures zero-latency retrieval on pause |
| `IdleDetector` | Adaptive pause threshold detection with EMA dynamic adjustment |
| `DisplayGate` | Display gating: better to not show than to show with delay |
| `SuggestionRenderer` | Manages suggestion display/dismiss lifecycle |
| `FeedbackCollector` | Tracks user reactions to suggestions, converts to structured feedback signals |
| `EvolutionLoop` | EMA confidence update + feedback broadcast to each Layer |
| `DegradationPolicy` | Resource-aware five-level degradation, ensures foreground operations are never blocked |

## Core Protocols

```python
class Signal(Protocol):
    event_type: str
    timestamp: float
    payload: Dict[str, Any]
    source: str

@dataclass
class ContextState:
    app_bundle: str
    window_title: str
    action_ring: List[str]       # Sliding window
    context_hash: str            # MD5[:16], O(1) index key

@dataclass(frozen=True)
class PredictionCandidate:
    action_description: str
    confidence: float            # [0.0, 1.0]
    source_layer: str            # "L0" | "L1" | "L2" | "L3"
    context_hash: str
    display_delay_ms: int
    is_destructive: bool = False

class PredictorLayer(Protocol):
    layer_id: str
    priority: int                # Lower value = higher priority
    timeout_ms: int
    async def predict(ctx: ContextState) -> List[PredictionCandidate]
    async def on_feedback(signal: FeedbackSignal) -> None
```

## Multi-Layer Prediction Engine

| Layer | Algorithm | Latency Budget | Accuracy Profile | Trigger Condition |
|:---:|:---|:---:|:---|:---|
| **L0** | Context-Hash exact match | 5ms | High precision (driven by historical hit rate) | Every context update |
| **L1** | N-gram Markov transition probability | 10ms | Medium (sequential patterns) | Every context update |
| **L2** | Embedding nearest-neighbor retrieval | 100ms | Medium (semantic similarity) | Async warm-up |
| **L3** | LLM + RAG reasoning | 3000ms | High precision (complex scenarios) | Complexity gate passes |

**Aggregation Strategy**: When multiple layers predict the same action, confidence is fused using an independence assumption:

```
P(combined) = 1 - ∏(1 - Pᵢ)
```

During cascade execution, if any layer produces a result with confidence > 0.9, early termination is triggered (fast-path).

## Speculative Pre-computation

Core idea: **Predict during operations, display on pause.**

```
User action ──→ on_action_observed()
               │
               ├─ sync: L0+L1 → instant cache (< 5ms)
               ├─ async: L2   → warm cache    (< 100ms)
               └─ async: L3   → deep cache    (conditional trigger)
               
User pause ──→ IdleDetector.on_idle()
               │
               └─ get_best() → instant > warm > deep
                               → DisplayGate → show()
```

**Three-Tier Cache**:

| Tier | Data Source | Fill Method | Priority |
|:---:|:---|:---|:---:|
| instant | L0 + L1 | Synchronous (during event handling) | Highest |
| warm | L2 | Async task | Medium |
| deep | L3 | Conditional async task | Lowest |

Cache control: LRU eviction (default 100 slots) + TTL expiration (default 30s).

## Feedback Evolution Loop

```
         ┌──────────────┐
         │  Show Suggestion │
         └──────┬───────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Accept      Ignore      Correct/Reject
 (+1.0)      (-0.1)      (-0.5 / -1.0)
    │           │           │
    └───────────┼───────────┘
                ▼
        EMA Confidence Update
     new = α·reward + (1-α)·old
                │
                ▼
     Broadcast on_feedback → Each Layer performs online learning
```

- L0: Updates accept_count / total_count
- L1: Updates N-gram transition frequency table
- L2/L3: External updates (Embedding index / LLM fine-tune)

## Degradation Strategy

| Level | Trigger Condition | Allowed Layers | Behavior |
|:---:|:---|:---:|:---|
| FULL | Normal | L0 L1 L2 L3 | Full functionality |
| NO_L3 | CPU > 70% | L0 L1 L2 | LLM reasoning disabled |
| NO_L2_L3 | CPU > 90% | L0 L1 | Statistical models only |
| L0_ONLY | Memory > 90% budget | L0 | Exact match only |
| DISABLED | Event queue backlog | ∅ | Copilot silently stops |

Design constraint: Degradation is automatic, observable, and recoverable. In the worst case, Copilot silently stops — **foreground operations are never blocked**.

## Module Structure

```
src/leapflow/copilot/
├── __init__.py          # Public API exports
├── types.py             # All Protocols + data types (zero behavior)
├── config.py            # CopilotConfig — centralized parameter tuning surface
├── context.py           # ContextEncoder + EventBus bridge
├── engine.py            # PredictionEngine — multi-layer cascade scheduling
├── pipeline.py          # SpeculativePipeline — speculative cache
├── idle.py              # IdleDetector — adaptive pause detection
├── renderer.py          # DisplayGate + SuggestionRenderer
├── feedback.py          # FeedbackCollector + EvolutionLoop
├── degradation.py       # DegradationPolicy — five-level degradation
└── predictors/
    ├── l0_hash.py       # O(1) exact match
    ├── l1_markov.py     # N-gram transition probability
    ├── l2_embed.py      # Vector nearest-neighbor retrieval
    └── l3_llm.py        # LLM + RAG reasoning
```

## Extension Guide

### Adding a New Predictor

1. Implement the `PredictorLayer` Protocol (define `layer_id`, `priority`, `timeout_ms`, `predict`, `on_feedback`)
2. Pass the instance when constructing `PredictionEngine`; the engine automatically schedules by priority order
3. If degradation control is needed, register the new layer_id in `DegradationPolicy._LAYER_SETS`

### Adding a New Signal Channel

1. Implement the `SignalChannel` Protocol (`channel_id`, `start`, `stop`, `subscribe`)
2. Bridge to `ContextEncoder` via `CopilotEventSubscriber`
3. Add corresponding event_type incremental encoding logic in `ContextEncoder.on_event`

### Adding a New Renderer

1. Implement the `HintRenderer` Protocol (`show`, `dismiss`, `is_visible`)
2. Inject into `SuggestionRenderer` constructor parameters to replace the display backend
3. Can implement TUI overlay / system notification / GUI floating window or any other form
