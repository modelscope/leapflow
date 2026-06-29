# Workflow Copilot 设计文档

## 概述

Workflow Copilot 是 LeapFlow 的上下文感知操作预测引擎。它持续观察用户操作信号流，在用户停顿瞬间以 ghost-hint 形式展示下一步操作建议——类似 IDE 代码补全，但作用于跨应用工作流。核心价值：**将重复性操作模式从"用户回忆"转变为"系统主动提示"，缩短操作决策延迟。**

## 设计哲学

| SOLID 原则 | 在 Copilot 中的映射 |
|:---:|:---|
| **S** — 单一职责 | 每个模块仅做一件事：编码、预测、渲染、反馈各自独立 |
| **O** — 开闭 | `PredictorLayer` Protocol 允许新增预测算法而不修改引擎 |
| **L** — 里氏替换 | 任何满足 Protocol 的实现可直接热替换 |
| **I** — 接口隔离 | `Signal`、`HintRenderer`、`SignalChannel` 各自最小化 |
| **D** — 依赖倒置 | Engine 依赖 Protocol 而非具体实现；Store/LLM 均为注入 |

**终极目标**：通过 Loop γ（执行即学习）闭环，使 Copilot 的预测准确率随使用时间单调递增，最终实现自主进化的操作辅助。

## 架构总览

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
│                              EvolutionLoop ──→ (回写各 Layer)    │
│                                                                 │
│  [DegradationPolicy] ←── SystemMetrics                          │
└─────────────────────────────────────────────────────────────────┘
```

| 模块 | 职责 |
|:---|:---|
| `ContextEncoder` | 将 SystemEvent 流增量编码为 `ContextState`（O(1)/event） |
| `PredictionEngine` | 级联调度所有 PredictorLayer，聚合去重+共识增强 |
| `SpeculativePipeline` | 操作时即预测，三级缓存保证停顿时零延迟取用 |
| `IdleDetector` | 自适应停顿阈值检测，EMA 动态调节 |
| `DisplayGate` | 展示门控：宁可不展示，不可延迟展示 |
| `SuggestionRenderer` | 管理建议的展示/撤回生命周期 |
| `FeedbackCollector` | 追踪用户对建议的反应，转化为结构化反馈信号 |
| `EvolutionLoop` | EMA 置信度更新 + 反馈广播至各 Layer |
| `DegradationPolicy` | 资源感知五级降级，保障前台操作永不阻塞 |

## 核心协议

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
    action_ring: List[str]       # 滑动窗口
    context_hash: str            # MD5[:16]，O(1) 索引键

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
    priority: int                # 越小优先级越高
    timeout_ms: int
    async def predict(ctx: ContextState) -> List[PredictionCandidate]
    async def on_feedback(signal: FeedbackSignal) -> None
```

## 多层预测引擎

| Layer | 算法 | 延迟预算 | 精度特征 | 触发条件 |
|:---:|:---|:---:|:---|:---|
| **L0** | Context-Hash 精确匹配 | 5ms | 高精度（历史命中率驱动） | 每次上下文更新 |
| **L1** | N-gram Markov 转移概率 | 10ms | 中等（序列模式） | 每次上下文更新 |
| **L2** | Embedding 近邻检索 | 100ms | 中等（语义相似） | 异步预热 |
| **L3** | LLM + RAG 推理 | 3000ms | 高精度（复杂场景） | 复杂度门控通过时 |

**聚合策略**：多层预测同一动作时，采用独立性假设融合置信度：

```
P(combined) = 1 - ∏(1 - Pᵢ)
```

Engine 级联执行时，若某层产出 confidence > 0.9 的结果则提前终止（fast-path）。

## 推测性预计算

核心思想：**操作时即预测，停顿时即展示。**

```
用户操作 ──→ on_action_observed()
               │
               ├─ sync: L0+L1 → instant cache (< 5ms)
               ├─ async: L2   → warm cache    (< 100ms)
               └─ async: L3   → deep cache    (条件触发)
               
用户停顿 ──→ IdleDetector.on_idle()
               │
               └─ get_best() → instant > warm > deep
                               → DisplayGate → show()
```

**三级缓存**：

| 级别 | 数据来源 | 填充方式 | 优先级 |
|:---:|:---|:---|:---:|
| instant | L0 + L1 | 同步（事件处理时） | 最高 |
| warm | L2 | 异步 task | 中 |
| deep | L3 | 条件异步 task | 最低 |

缓存控制：LRU 淘汰（默认 100 slots）+ TTL 过期（默认 30s）。

## 反馈演化闭环

```
         ┌──────────────┐
         │  展示建议     │
         └──────┬───────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Accept      Ignore      Correct/Reject
 (+1.0)      (-0.1)      (-0.5 / -1.0)
    │           │           │
    └───────────┼───────────┘
                ▼
        EMA 置信度更新
     new = α·reward + (1-α)·old
                │
                ▼
     广播 on_feedback → 各 Layer 内部在线学习
```

- L0：更新 accept_count / total_count
- L1：更新 N-gram 转移频率表
- L2/L3：外部更新（Embedding 索引/LLM fine-tune）

## 降级策略

| 级别 | 触发条件 | 允许运行的层 | 行为 |
|:---:|:---|:---:|:---|
| FULL | 正常 | L0 L1 L2 L3 | 全功能 |
| NO_L3 | CPU > 70% | L0 L1 L2 | 禁用 LLM 推理 |
| NO_L2_L3 | CPU > 90% | L0 L1 | 仅统计模型 |
| L0_ONLY | 内存 > 90% budget | L0 | 仅精确匹配 |
| DISABLED | 事件队列积压 | ∅ | Copilot 静默停止 |

设计约束：降级是自动的、可观测的、可恢复的。最差情况 Copilot 静默停止，**永不阻塞前台操作**。

## 模块结构

```
src/leapflow/copilot/
├── __init__.py          # 公共 API 导出
├── types.py             # 所有 Protocol + 数据类型（零行为）
├── config.py            # CopilotConfig — 集中参数调节面
├── context.py           # ContextEncoder + EventBus 桥接
├── engine.py            # PredictionEngine — 多层级联调度
├── pipeline.py          # SpeculativePipeline — 推测缓存
├── idle.py              # IdleDetector — 自适应停顿检测
├── renderer.py          # DisplayGate + SuggestionRenderer
├── feedback.py          # FeedbackCollector + EvolutionLoop
├── degradation.py       # DegradationPolicy — 五级降级
└── predictors/
    ├── l0_hash.py       # O(1) 精确匹配
    ├── l1_markov.py     # N-gram 转移概率
    ├── l2_embed.py      # 向量近邻检索
    └── l3_llm.py        # LLM + RAG 推理
```

## 扩展指南

### 新增 Predictor

1. 实现 `PredictorLayer` Protocol（定义 `layer_id`, `priority`, `timeout_ms`, `predict`, `on_feedback`）
2. 在构造 `PredictionEngine` 时传入实例，引擎自动按 priority 排序调度
3. 如需降级控制，在 `DegradationPolicy._LAYER_SETS` 中注册新 layer_id

### 新增 Signal 通道

1. 实现 `SignalChannel` Protocol（`channel_id`, `start`, `stop`, `subscribe`）
2. 通过 `CopilotEventSubscriber` 桥接至 `ContextEncoder`
3. 在 `ContextEncoder.on_event` 中添加对应 event_type 的增量编码逻辑

### 新增渲染器

1. 实现 `HintRenderer` Protocol（`show`, `dismiss`, `is_visible`）
2. 注入 `SuggestionRenderer` 构造参数即可替换展示后端
3. 可实现 TUI overlay / 系统通知 / GUI 悬浮窗等任意形态
