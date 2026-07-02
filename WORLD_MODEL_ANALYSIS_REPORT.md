# LeapFlow World Model 与自演化机制技术分析报告

## 执行摘要

LeapFlow 的 World Model 不是传统意义的深度学习模型，而是一个**无梯度、预测驱动的学习框架**，通过以下核心机制实现持续自演化：

1. **Context 四层金字塔** (`L1~L4`)：从工作记忆到世界模型，实现逐层知识压缩 (100000:1)
2. **Predict-Execute-Compare-Learn 闭环**：通过预测误差 δ 驱动好奇心和在线学习
3. **OODA 三层循环**：Loop α (演示学习) → Loop β (执行) → Loop γ (演化反馈) 形成自强化系统
4. **信任梯度机制**：STEP→CONFIRM→AUTO 的渐进式自动化，以及双向置信度调整

当前实现已投入生产，但距离完整自主推理能力仍有明显 gap。

---

## 1. World Model 模块详解

### 1.1 模块组成

```
src/leapflow/world_model/
├── prediction.py          # 预测循环核心：Predict→Execute→Compare
├── experience_store.py     # 经验存储与RAG检索 (基于DuckDB)
├── curiosity.py           # 好奇心信号：3层内在动机
├── replay.py              # 离线经验重放与模式发现
├── trajectory_grader.py   # OPD (On-Policy Distillation) 教师评分
├── budget.py              # 学习预算控制 (5个token池)
└── embedding.py           # 语义嵌入 (TFIDF 或 LLM)
```

### 1.2 核心数据结构

#### `Prediction` (预测)
```python
@dataclass(frozen=True)
class Prediction:
    action_description: str       # "点击保存按钮"
    expected_effect: str          # "文件将保存到本地"
    confidence: float             # 0.0~1.0，LLM生成
    reasoning: str = ""           # (可选) 推理说明
```

#### `PredictionOutcome` (预测结果)
```python
@dataclass(frozen=True)
class PredictionOutcome:
    prediction: Prediction
    pre_snapshot: StateSnapshot    # 执行前状态
    post_snapshot: StateSnapshot   # 执行后状态
    actual_effect: str             # 实际观察到的变化
    delta: float                   # 预测误差 δ ∈ [0, 1]
    delta_source: str              # "structural" | "semantic" | "blended"
    timestamp: float
    experience_id: str = ""
```

#### `ExperienceTuple` (经验记录)
```python
@dataclass(frozen=True)
class ExperienceTuple:
    experience_id: str
    action_description: str
    app_context: str               # 应用包ID
    predicted_effect: str
    actual_effect: str
    delta: float                   # 预测误差
    curiosity_score: float         # 后验计算
    pre_state_summary: str
    post_state_summary: str
    timestamp: float
    # OPD 字段 (教师评分)
    advantage: float               # [-1, 1]，来自Loop γ
    is_forking: bool               # 是否为关键决策点
    grade_label: str               # "optimal" | "acceptable" | "suboptimal" | "harmful"
```

---

## 2. Predict-Execute-Compare-Learn 闭环

### 2.1 完整流程 (PredictionLoop)

```
1. 预执行前 (Pre-Execute Phase)
   ├─ 捕获前置状态快照 (StateSnapshot.LIGHT/MEDIUM/HEAVY)
   │  ├─ App bundle ID, 窗口标题
   │  ├─ AX 树摘要 (ax_digest)
   │  └─ 剪贴板内容, 最近事件序列
   │
   ├─ LLM 预测 (with RAG context)
   │  ├─ 调用 ExperienceStore.retrieve_similar()
   │  │  └─ 关键词搜索 + 语义重排
   │  │
   │  ├─ 构建提示：
   │  │  "给定当前状态 (App, 窗口, 最近事件) 和过去经验，
   │  │   预测行为 {action_description} 的效果"
   │  │
   │  └─ 返回 Prediction(effect, confidence)

2. 执行 (Execute Phase)
   └─ 执行用户操作 (async execute_fn)

3. 后执行后 (Post-Execute Phase)
   ├─ 捕获后置状态快照
   │
   ├─ 比较 (Compare Phase)
   │  ├─ 结构化差异 (structural_delta)
   │  │  └─ pre.ax_digest vs post.ax_digest, 应用变化, 剪贴板变化
   │  │
   │  ├─ 如果 structural_delta > semantic_threshold (0.1)
   │  │  ├─ 调用 LLM 语义比较
   │  │  ├─ δ = 0.4 * structural + 0.6 * semantic (加权混合)
   │  │  └─ source = "blended"
   │  │
   │  └─ 否则 δ = structural, source = "structural"
   │
   ├─ 学习 (Store Phase)
   │  └─ ExperienceStore.store() 保存 (action, prediction, actual, δ, ...)
   │
   ├─ 轨迹缓冲 (Trajectory Buffer)
   │  └─ 累积用于 OPD (On-Policy Distillation) 教师评分
   │
   └─ 触发回调 (on_prediction_outcome)
      ├─ 好奇心计算 (CuriositySignal.compute)
      ├─ 注意力调整 (AttentionTuner)
      └─ 主动学习 (ActiveObserver)
```

### 2.2 关键设计决策

**1. 预测置信度的双来源**
- LLM 直接返回 confidence (0.0~1.0)
- 在 L0 Hash Predictor (Copilot) 中，confidence = accept_rate (历史接受率)
- 两者独立，不冲突

**2. 预测误差 δ 的混合计算**
```python
# 结构化 delta (快速路径)
structural_delta = semantic_distance(pre.ax_digest, post.ax_digest)

# 如果需要更高精度，调用 LLM 语义比较 (可选)
if structural_delta > 0.1:  # 有显著变化
    semantic_delta = await llm_compare(prediction, pre, post)
    delta = 0.4 * structural + 0.6 * semantic
else:
    delta = structural
```

**3. 状态快照的分层**
- `SnapshotFidelity.LIGHT`: 仅 ax_digest, 应用信息 (快速)
- `SnapshotFidelity.MEDIUM`: + 窗口标题, 事件摘要
- `SnapshotFidelity.HEAVY`: + 完整视觉内容 (VLM友好)

---

## 3. 好奇心驱动的学习 (Curiosity Signal)

### 3.1 三层内在动机模型

```python
class CuriositySignal:
    """融合 RL 文献中的三个内在动机组件"""
    
    def compute(outcome: PredictionOutcome) -> CuriosityScore:
        ps = prediction_surprise(outcome)    # α: ICM 类比
        ig = information_gain(outcome)       # β: 贝叶斯信息增益
        fn = frequency_novelty(outcome)      # γ: 计数型新颖性
        
        return α*ps + β*ig + γ*fn  # 加权混合
```

#### (1) 预测惊奇度 (Prediction Surprise, α~0.4)
```python
ps = outcome.delta  # 直接使用预测误差作为惊奇信号
```
**含义**：预测错得越远，系统应该越好奇这个情景。

#### (2) 信息增益 (Information Gain, β~0.3)
```python
def information_gain(outcome):
    """贝叶斯信息增益：观察如何减少因果图的不确定性"""
    
    # 获取受影响的事件节点 (app_context 相关)
    affected = [ev for ev in causal_graph.events.values()
                if app_matches(ev)]
    
    # 计算前后熵
    h_before = sum(binary_entropy(ev.confidence) for ev in sample)
    
    # 根据 δ 模拟置信度更新
    p_updated = p + (1.0 - p) * delta * 0.5  # 保守更新
    h_after = sum(binary_entropy(p_updated) for ev in sample)
    
    return (h_before - h_after) / N  # 熵减少量
```
**含义**：观察到的结果如何改变了我们对因果关系的不确定性。

#### (3) 频率新颖性 (Frequency Novelty, γ~0.3)
```python
def frequency_novelty(outcome):
    """计数型新颖性：这个行为模式有多稀有？"""
    
    key = f"{app_context}|{action[:50]}"
    count = frequency_counter[key] += 1
    
    # 60% 来自代理自身的行动计数
    action_novelty = 1.0 / sqrt(count)
    
    # 40% 来自因果图的频道级数据 (live)
    causal_novelty = 1.0 / sqrt(causal_total + 1)
    
    return 0.6 * action_novelty + 0.4 * causal_novelty
```
**含义**：同一个模式重复多次就不再新奇，应该减少探索。

### 3.2 成熟度自动平衡

```python
def maturity_stage():
    """根据积累的数据量自动调整权重"""
    total_experiences = store.count()
    event_count = len(causal_graph.events)
    
    if event_count < 100 and total_experiences < 20:
        return "early"       # (α,β,γ) = (0.2, 0.3, 0.5)  # 强调探索
    elif event_count < 500 and total_experiences < 100:
        return "middle"      # (α,β,γ) = (0.4, 0.4, 0.2)
    else:
        return "mature"      # (α,β,γ) = (0.6, 0.3, 0.1)  # 强调利用
```

**学习曲线**：
- 早期：频率新颖性主导，鼓励多样化探索
- 中期：平衡预测惊奇和信息增益
- 成熟期：信息增益和预测惊奇主导，减少重复

### 3.3 OPD (On-Policy Distillation) 调制

```python
def compute_with_trajectory_context(outcome, advantage):
    """用教师评分调制好奇心"""
    
    base_curiosity = compute(outcome)
    
    # advantage ∈ [-1, 1]
    # 负数 (失败): 扩大好奇心 → 鼓励在坏状态下更多探索
    # 正数 (成功): 减小好奇心 → 已理解的状态可减少探索
    
    modifier = 1.0 - modulation_strength * advantage
    adjusted = base_curiosity.total * modifier
    
    return clamp(adjusted, 0.1, 2.0)
```

---

## 4. 离线经验重放 (ExperienceReplayEngine)

### 4.1 两种重放策略

#### 策略 1: 高预测误差反思 (Reflection)
```python
async def replay_session():
    # 收集 δ > 0.5 的经验 (高错误)
    high_delta = store.retrieve_high_delta(delta_min=0.5, limit=5)
    
    # LLM 反思这些失败
    insights = await reflect_batch(high_delta, focus="prediction_errors")
    # → "为什么预测失败了？有什么因果规则可以从中提取？"
```

#### 策略 2: 跨应用模式发现 (Transfer)
```python
# 收集跨应用的高误差经验
cross_app = collect_cross_app_experiences()

# LLM 提取迁移规则
# "在 Finder 中有效的文件操作是否也适用于 VS Code？"
insights = await reflect_batch(cross_app, focus="transfer_rules")
```

### 4.2 自蒸馏 (Self-Distillation)

```python
async def self_distill():
    """从教师评分的经验中提取启发式规则"""
    
    # 收集有 grade_label 的经验 (Loop γ 已评分)
    graded = collect_graded_experiences(sort_by="abs(advantage)")
    
    # LLM 生成三类规则
    """
    1. 启发式规则: 高advantage经验 → 可复用的最佳实践
    2. 修正规则: 低advantage经验 → 什么样的操作会失败
    3. 分叉洞察: is_forking=True → 关键决策点，需要多个选项
    """
    
    distilled_rules = await llm.extract_rules(graded)
    
    # 例子
    # {
    #   "type": "heuristic",
    #   "description": "移动大于100MB的文件前需要检查磁盘空间",
    #   "confidence": 0.85,
    #   "actionable": true
    # }
```

### 4.3 回归检测

```python
def detect_regression(recent_outcomes, window=5):
    """检测预测准确度是否下降"""
    
    recent_deltas = [o.delta for o in recent_outcomes[-5:]]
    recent_mean = mean(recent_deltas)
    
    all_exps = store.retrieve_high_delta(delta_min=0.0, limit=200)
    hist_mean = mean([e.delta for e in all_exps])
    
    # 如果最近比历史差 > 15%，触发警报
    return recent_mean > hist_mean + 0.15
```

---

## 5. On-Policy Distillation (OPD) 教师评分

### 5.1 教师角色 (Full Hindsight)

```python
class TrajectoryGrader:
    """LLM 担当教师，有完整事后信息"""
    
    async def grade_trajectory(trajectory, goal):
        """
        输入: 完整轨迹 + 用户目标
        输出: 每步的 (advantage, is_forking, grade_label)
        """
        
        # 构造提示
        prompt = f"""
        目标: {goal}
        轨迹步骤:
        Step 1: action={...}, predicted={...}, actual={...}, delta={...}
        Step 2: ...
        
        对每一步评分:
        - advantage ∈ [-1, 1]: 这一步比"平均"好还是差？
        - is_forking: 这是关键决策点吗？
        - grade_label: optimal | acceptable | suboptimal | harmful
        """
        
        # LLM 评分所有步骤 (一次调用!)
        grades = await llm.score_trajectory(prompt)
        
        # 写回 ExperienceStore
        for exp_id, grade in zip(trajectory, grades):
            store.update_advantage(exp_id, grade.advantage, ...)
```

### 5.2 学生 vs 教师的非对称性

| 角色 | 可见信息 | 目标 |
|------|---------|------|
| **学生** (PredictionLoop) | 仅当前状态 | 预测下一步 |
| **教师** (TrajectoryGrader) | 完整轨迹 + 最终结果 | 回溯评估每一步的质量 |

**这正是人类学习的方式**：
- 做中学（当下可见信息少）
- 事后反思（全景视角，深度学习）

---

## 6. 因果推理与World Model的协作

### 6.1 因果推理三层架构

```
src/leapflow/causal/
├── Tier 1: Rule-based (确定性, 置信度 ≥ 0.9)
├── Tier 2: Heuristic (概率, 置信度 0.5~0.9)
└── Tier 3: VLM (高保真, 异步)
```

#### Tier 1: 规则推理
```yaml
rules:
  - name: click_to_visual
    parent_channel: click
    child_channel: visual_change
    time_delta_max: 0.5s
    confidence: 0.95
```
**用途**：快速确定性因果关系，零成本。

#### Tier 2: 启发式推理
```python
def heuristic_score(parent, child):
    """概率因果评分"""
    # 基于
    # - 时间接近度
    # - 空间接近度 (点击位置vs变化区域)
    # - 频道可靠性 (EMA学习)
    # - 语义类似性 ("copy"→"clipboard_change")
    
    return 0.1*temporal + 0.2*spatial + 0.4*reliability + 0.3*semantic
```

#### Tier 3: VLM 验证
```python
async def vlm_verify(parent, child, frames):
    """调用视觉语言模型进行高保真因果验证"""
    # 输入: 事件前后的视频帧 + 事件描述
    # 输出: 因果关系置信度 + 自然语言解释
    
    prompt = f"""
    用户点击了按钮 (帧{parent.frame_id})
    之后屏幕变化如下 (帧{child.frame_id})
    这个点击导致了这个变化吗？
    """
```

### 6.2 World Model 与因果图的双向反馈

```
PredictionLoop                      CausalGraph
   ↓                                   ↓
预测 action 的效果                 因果推理 event 关系
   ↓                                   ↓
   └─→ 如果 δ 很高？ ←────────────→ 是否有遗漏的因果边？
       ├─ 可能是因果图不完整
       ├─ CausalGraph 应该加入新边
       └─ 下次预测可利用新边
```

**例**：
- 预测失败：用户点击"保存"→ "应该保存文件" → 但实际没有保存
- 好奇心激发→ 经验重放
- LLM 反思：可能有其他前置条件 (文件需要修改)
- 因果图添加新规则：`file_modified → file_changed → save_effective`

---

## 7. Memory 层级系统

### 7.1 L1-L4 与 Memory Provider 的映射

```
Context Pyramid (4层)          Memory Providers (3层)
──────────────────           ──────────────────
L1 Working (O(1))     ──→     Working Memory
   当前状态                    (事件驱动, 实时)
   
L2 Episodic (100:1)   ──→     Episodic Memory
   Session历史                (DuckDB, 衰减评分)
   
L3 Semantic (1000:1)  ──→     Semantic Memory
   技能库/经验                (持久, 交叉领域)
   
L4 World Model        ──→     (隐含在L3中)
   (100000:1)                 通过 ExperienceStore
   压缩知识                    + 因果图
```

### 7.2 Memory Manager 的跨层搜索

```python
async def search_cross_domain(query, time_window=300s):
    """发现跨模态信息关联"""
    
    # 1. 在所有层中搜索
    all_entries = await search_all_providers(query)
    
    # 2. 分组 (时间聚类)
    clusters = group_by_timestamp(all_entries, window=300)
    
    # 3. 跨域增强 (同一时间窗口的不同domain)
    for entry in unique_entries:
        cross_domain_count = count_different_domains_in_window(entry)
        entry.score *= (1.0 + cross_domain_count * 0.2)  # 最多2倍提升
    
    return sorted_by_score(unique_entries)
```

**应用**：
- 用户在 Finder 中复制文件 (FS domain)
- 同时向 Slack 粘贴 (clipboard domain)
- 系统识别跨域关联 → "file-to-message" 工作流

---

## 8. Copilot L0-L3 预测器 (Workflow 层)

### 8.1 四层预测器架构

```
ContextState (5ms)                    (50ms)              (10ms)        (3000ms)
      ↓                                 ↓                   ↓              ↓
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Context Hash                  Semantic Distance     N-gram Seq Prob.    LLM  │
  │ (L0)                          (L2)                  (L1)                (L3) │
  │                                                                              │
  │ O(1) lookup        Embedding + cosine   Markov transition   RAG + reasoning │
  │ "exact match"      sim (TFIDF/LLM)      probabilities       complexity gate  │
  └─────────────────────────────────────────────────────────────────────────────┘
   exact_match           semantic             temporal              reasoning
   (quick)              (medium)              (medium)              (deep)
```

#### L0: 精确哈希匹配 (5ms 预算)
```python
class L0HashPredictor:
    """O(1) 上下文哈希查找"""
    
    async def predict(context: ContextState):
        # context.context_hash = hash(app, window_title, clipboard, ...)
        hits = await store.query_by_hash(context.context_hash)
        
        # 只返回 accept_rate > 30% 的候选
        return [
            PredictionCandidate(
                action=hit.action,
                confidence=min(hit.accept_rate, 0.99),
                source_layer="L0"
            )
            for hit in hits if hit.accept_rate > 0.3
        ]
    
    async def observe(context, action):
        """无监督学习：记录 context→action 映射"""
        await store.record_observation(context.context_hash, action, accepted=True)
```

#### L1: 马尔可夫序列 (10ms)
```python
class L1MarkovPredictor:
    """N-gram 转移概率"""
    
    def __init__(ngram_n=3, top_k=5, min_prob=0.1):
        self._transitions = {}  # context_key → action → count
    
    async def predict(context):
        key = "→".join(context.action_ring[-3:])  # 最近3个行动
        if key not in transitions:
            return []
        
        probs = {
            action: count / total
            for action, count in transitions[key].items()
        }
        
        return [
            PredictionCandidate(action, confidence=prob)
            for action, prob in sorted_desc(probs)
            if prob >= 0.1
        ][:5]
    
    async def on_feedback(signal):
        """在线学习：接受 → 更新转移矩阵"""
        if signal.feedback_type in (ACCEPT, CORRECT):
            actual = signal.actual_action
            self._transitions[key][actual] += 1
```

#### L2: 语义嵌入 (50ms)
```python
# 实现较复杂，核心思想：
# context embedding + action embedding 的相似度
# 基于 L1 搜索结果重排

def _semantic_rerank(action_desc, exps):
    query_vec = embedder.embed(action_desc)
    
    scored = [
        (cosine_similarity(query_vec, embedder.embed(exp.content)), exp)
        for exp in exps
    ]
    
    return sorted_by_score(scored)
```

#### L3: LLM 推理 (3000ms)
```python
class L3LLMPredictor:
    """深度推理：处理 L0-L2 无法处理的复杂上下文"""
    
    async def predict(context):
        # 复杂度门限：防止不必要的昂贵调用
        complexity = (unique_apps / 3.0) + (len(action_ring) / 10.0)
        if complexity < threshold:
            return []  # 让L0-L2处理
        
        # RAG 增强
        rag_hits = await rag_provider.retrieve(
            f"{context.app_bundle} {' '.join(context.action_ring[-5:])}"
        )
        
        # 构造提示
        prompt = f"""
        Current app: {context.app_bundle}
        Recent actions: {' → '.join(context.action_ring[-5:])}
        
        Similar past experiences:
        {format_rag_hits(rag_hits)}
        
        Predict the most likely next action(s).
        Return JSON: [{{"action": "...", "confidence": 0.8, "reasoning": "..."}}]
        """
        
        response = await llm.complete(prompt)
        return parse_json_candidates(response)
```

### 8.2 Copilot 反馈循环 (EvolutionLoop)

```python
class EvolutionLoop:
    """将用户反馈转化为预测器权重更新"""
    
    reward_map = {
        ACCEPT: +1.0,
        IGNORE: -0.1,
        CORRECT: -0.5,
        EXPLICIT_REJECT: -1.0
    }
    
    async def process_feedback(signal: FeedbackSignal):
        # 1. 计算奖励
        reward = reward_map[signal.feedback_type]
        
        # 2. 更新 EMA 置信度
        ctx_hash = signal.candidate.context_hash
        self._confidence_scores[ctx_hash] = (
            0.9 * old_conf + 0.1 * reward
        )
        
        # 3. 广播给所有预测器
        for layer in self._layers:
            await layer.on_feedback(signal)
        
        # 4. 在线学习 (L0-L1 更新存储, L3无操作)
```

---

## 9. OODA 三层循环与自演化

### 9.1 Loop α：演示学习 (分钟~天)

```
Demonstrate → Record (零侵入)
    ↓
    Observe: 捕获完整轨迹 + 标注疑似噪声
    ↓
    Orient: 6层去噪管线
    ├─ L1: DenoisePass (undo, 幂等)
    ├─ L2: GroupingPass (连续事件合并)
    ├─ L3: PatternPass (模式识别)
    ├─ L4: CausalAnalysis (因果链提取)
    ├─ L5: CrossModalVerify (视觉+结构一致性)
    └─ L6: ConsensusDistill (多轨迹共识)
    ↓
    Decide: 新技能 vs 已知技能 vs 需要LLM优化
    ↓
    Act: 代码生成 + 安全验证 + 技能库注册
```

**压缩比**：50 raw steps → 6 semantic actions → 4 causal steps (12:1 抽象)

### 9.2 Loop β：执行 (秒~分钟) [虚线表示当前未充分实现的功能]

```
User Request
    ↓
    Observe: 意图分类 + 技能触发匹配
    ├─ 关键字快速路径 (0延迟)
    └─ LLM 分类 (高准确但慢)
    ↓
    Orient: 风险评估
    ├─ 技能成熟度 (版本数, 置信度)
    ├─ 执行历史 (最近成功率)
    ├─ 销毁性评估 (文件操作?)
    └─ 环境风险 (其他应用状态?)
    ↓
    Decide: [当前简化实现]
    ├─ 新技能 → STEP (每步审批)
    ├─ 发展中 → CONFIRM (整体审批)
    └─ 成熟 → AUTO (无审批)
    ↓
    Act: 执行绑定到 VSI 端口
```

**信任梯度**：逐次执行验证 → 能力验证 → 自动化

### 9.3 Loop γ：演化 (持续, 自动)

```
Skill Execution (记录完整轨迹)
    ↓
    Observe: 通过同一事件总线捕获 (统一观察)
    ↓
    Orient: LCS 对齐比较
    ├─ 执行轨迹 vs 演示轨迹
    └─ 计算 Levenshtein 距离或相似度
    ↓
    Decide: 分类结果
    ├─ 改进 (δ↓): 新轨迹更干净
    │   → 与演示合并 (多轨迹共识 v2)
    │
    ├─ 不变 (δ≈): 置信度++ (更多证据)
    │
    └─ 退化 (δ↑): 执行偏离演示
        → confidence-- (降低信任)
        → confirmation_level++ (STEP降级)
        → 警报 (用户审查)
    ↓
    Act: 更新技能版本 + 置信度 + 确认级别
         (这改变了 Loop β 在下次执行时的决策)
```

**双向反馈**：
- 成功 → 置信度上升, 自动化升级
- 失败 → 置信度下降, 自动化降级 (回到 STEP)

### 9.4 三环相互促进

```
               信任梯度 (confidence, confirmation_level)
                        ↑        ↓
┌──────────────────┐   ┌─────────────────┐   ┌──────────────┐
│ Loop α           │──→│ Loop β          │──→│ Loop γ       │
│ Acquisition      │   │ Execution       │   │ Evolution    │
│ (slow, deep)     │   │ (fast, intuitive)  │ (silent,auto)│
└──────────────────┘   └─────────────────┘   └──────────────┘
   ▲                           ▲                   │
   │ 新技能注册                │ 置信度变化          │
   │                          │                   │
   └──────────────────────────┴───────────────────┘
           多轨迹共识增强 (通过Loop γ执行)
```

---

## 10. 知识压缩与表示

### 10.1 Context 金字塔的压缩比

| 层级 | 内容 | 压缩比 | 延迟 | 职责 |
|------|------|--------|------|------|
| **L1 Working** | 当前快照 | 10:1 | O(1) | 实时反应 |
| **L2 Episodic** | Session历史 | 100:1 | <100ms | 最近模式 |
| **L3 Semantic** | 技能库 + 经验 | 1000:1 | <50ms | 跨session泛化 |
| **L4 World Model** | 因果图 + 规则 | 100000:1 | <1ms | 推理基础 |

### 10.2 知识表示方式

#### 显式表示
```python
# 因果规则 (确定性)
CausalRule(
    name="file_save",
    parent_channel="keyboard",
    parent_type=KEYSTROKE,
    parent_payload_match={"keys": "Cmd+S"},
    child_channel="file",
    confidence=0.95
)

# 启发式规则 (概率)
{
    "type": "heuristic",
    "description": "在Chrome中按Cmd+T打开新标签",
    "confidence": 0.88,
    "actionable": True
}
```

#### 隐式表示
```python
# L0 哈希表
{
    "hash_xyz": {
        "copy_file": (accept=95, total=100),  # accept_rate=0.95
        "move_file": (accept=5, total=100),
    }
}

# L1 马尔可夫
{
    "open_finder→navigate→select": {
        "copy": 45,
        "move": 30,
        "delete": 5,
    }
}
```

### 10.3 意图保真压缩

```
Raw Events (50):
  click(234, 456)
  keystroke("test")
  clipboard.set("content")
  file.create("/path/file.txt")
  ...

Semantic Actions (6):
  open_text_editor
  type_content
  copy_to_clipboard
  create_file
  ...

Causal Steps (4):
  create_file_with_content
  copy_to_clipboard
  [implicit: other steps not in causal chain]
```

**核心原理**：
- 原始事件 = 瞬时, 与平台相关
- 语义动作 = 意图清晰, 跨应用可转移
- 因果步骤 = 逻辑最小集, 不可约

---

## 11. 当前实现的成熟度与Gap

### 11.1 已实现 (生产就绪)

✅ **World Model 核心循环**
- Predict-Execute-Compare 完整链路
- 预测误差计算 (结构化 + 语义混合)
- 经验存储和 RAG 检索

✅ **好奇心驱动学习**
- 三层内在动机 (ps, ig, fn)
- 成熟度自适应权重
- 好奇心与 OPD 调制

✅ **离线经验重放**
- 高错误经验反思
- 跨应用模式发现
- 自蒸馏规则提取

✅ **OPD 教师评分**
- 事后轨迹评分 (advantage, forking)
- 反写 ExperienceStore
- 形成闭环

✅ **Copilot L0-L3 预测器**
- L0 精确哈希 (已启用)
- L1 马尔可夫 (已启用)
- L2 语义嵌入 (框架完成)
- L3 LLM 推理 (启用但较慢)

✅ **因果推理三层**
- Tier 1 规则 (YAML 定义)
- Tier 2 启发式评分
- Tier 3 VLM 验证 (框架)

### 11.2 部分实现 (需要增强)

⚠️ **Loop γ (演化循环)**
- 当前：置信度线性 EMA 更新
- 缺失：
  - LCS 轨迹差异计算 (框架存在, 需集成)
  - 回归检测与自动降级 (检测有, 自动应用不完整)
  - 多轨迹共识 v2 (演示学习有, 执行反馈集成不完整)

⚠️ **Context Pyramid L4**
- 当前：因果图作为 L4 的代理
- 缺失：
  - 显式的知识压缩算法
  - 规则库的自动更新
  - 跨平台规则泛化

⚠️ **信任梯度双向机制**
- 当前：STEP→CONFIRM→AUTO 渐进 (UI层)
- 缺失：
  - 自动从 AUTO 降级到 CONFIRM/STEP
  - 降级触发条件的精细调整
  - 用户透明度反馈

### 11.3 未实现 (设计存在, 代码缺失)

❌ **完整 Loop γ 集成**
- 演化循环需与所有组件深度集成
- 当前实现为"沙箱"状态

❌ **从"模仿学习"到"自主推理"的过渡**
- 当前：技能基于演示学习
- 目标：技能从执行反馈中自我优化, 最终能独立推理新场景
- Gap：需要 Chain-of-Thought 推理、计划验证、假设检验

❌ **多源 Hub 集成下的自演化**
- 当前：单机技能库
- 目标：技能从多个源学习, 形成统一世界模型
- Gap：跨源因果规则融合、版本控制

❌ **长周期任务的演化**
- 当前：单次执行的反馈
- 目标：跨多天/周的任务学习
- Gap：长期因果图维护、季节性模式识别

---

## 12. 能力边界分析

### 12.1 从"模仿学习"到"自主推理"的能力阶梯

```
┌─────────────────────────────────────────────────────────────────┐
│ 能力阶梯                                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ L5 (未实现)  │ 独立推理新场景 (无示例)                         │
│             │ ├─ 假设检验 (我能在这个新上下文中...吗?)       │
│             │ ├─ 计划验证 (这个计划是否安全?)               │
│             │ └─ 自我矫正 (执行失败后立即调整)              │
│             │                                                │
│ L4 (部分)   │ 零样本泛化 (概念转移)                           │
│             │ ├─ "在A应用中有效" → "在B应用中可能有效"      │
│             │ ├─ 因果规则迁移                              │
│             │ └─ 由 ExperienceReplayEngine 支持             │
│             │                                                │
│ L3 (已实现) │ 技能微调 (少样本学习)                          │
│             │ ├─ 多轨迹共识 (2-3次演示)                     │
│             │ ├─ 执行反馈学习 (Loop γ)                      │
│             │ └─ OPD 教师评分                               │
│             │                                                │
│ L2 (已实现) │ 模式匹配 (已知场景)                            │
│             │ ├─ L0-L3 预测器                               │
│             │ ├─ 马尔可夫转移                               │
│             │ └─ 语义相似度                                │
│             │                                                │
│ L1 (已实现) │ 精确匹配 (完全相同的上下文)                    │
│             │ ├─ L0 哈希查找                                │
│             │ ├─ Context hash ≡                            │
│             │ └─ O(1) 延迟                                  │
│             │                                                │
└─────────────────────────────────────────────────────────────────┘

↑ 推理能力 | 数据效率 (样本数) ↓
```

### 12.2 当前能力上限

**已可靠工作**：
1. 精确匹配 (L1-L2)：相同上下文 → 历史行动
2. 少样本泛化 (L3)：1-2 次演示 → 可迁移技能
3. 执行反馈学习 (γ)：多次执行 → 技能精化

**脆弱区域**：
1. ⚠️ 跨应用概念转移：规则是否能从 Finder 迁移到 VS Code？
2. ⚠️ 长程规划：5+ 步的任务中，中间失败时的自我恢复
3. ⚠️ 异常处理：从未见过的错误状态中恢复
4. ⚠️ 假设检验：在执行前评估"这会成功吗？"

**缺失的能力**：
1. ❌ 从头推理：完全陌生的任务
2. ❌ 符号规划：组合多个技能完成复杂目标
3. ❌ 反事实推理："如果我选择不同的路径会怎样？"
4. ❌ 目标分解：大目标 → 子目标自动分解

---

## 13. 架构设计的本质

### 13.1 为什么 World Model 是"不可训练"的？

LeapFlow 故意避免了梯度下降。原因：

1. **无闭包形式**
   - 桌面自动化的状态空间无界
   - 无法建立标准的监督学习目标函数

2. **实时反馈的价值**
   - 用户执行产生的反馈即是训练信号
   - 与 RL 中的 reward 相同 (但无需复杂的价值函数估计)

3. **解释性**
   - LCS 对齐、因果规则、启发式都是可审查的
   - 梯度更新是黑盒

4. **计算效率**
   - 无反向传播
   - 离线重放 (experience replay) 比在线微调便宜

### 13.2 与 RL/IL 的对比

| 维度 | LeapFlow | RL | IL (Imitation Learning) |
|------|----------|----|----|
| **学习信号** | 预测误差 δ + 用户反馈 | Reward r(s,a) | 演示轨迹 |
| **优化器** | 无梯度 (EMA, 启发式) | 策略梯度 | BC 或 GAIL |
| **数据效率** | 中等 (1-2 演示) | 低 (需大量交互) | 高 (少数演示) |
| **解释性** | 高 (因果规则、LCS) | 低 (黑盒策略) | 中等 (行为克隆) |
| **实时性** | 高 (ms级决策) | 中等 (推理延迟) | 高 (前向通过) |
| **适配开环** | ✓ (无需奖励模型) | ❌ (需reward) | ✓ |

---

## 14. 关键文件清单

### 核心World Model
- `/Users/jason/work/github/leapflow/src/leapflow/world_model/prediction.py` - PredictionLoop (Predict-Compare-Learn)
- `/Users/jason/work/github/leapflow/src/leapflow/world_model/experience_store.py` - 经验存储与RAG
- `/Users/jason/work/github/leapflow/src/leapflow/world_model/curiosity.py` - 好奇心信号
- `/Users/jason/work/github/leapflow/src/leapflow/world_model/replay.py` - 经验重放
- `/Users/jason/work/github/leapflow/src/leapflow/world_model/trajectory_grader.py` - OPD教师评分
- `/Users/jason/work/github/leapflow/src/leapflow/world_model/budget.py` - 学习预算

### Copilot 预测
- `/Users/jason/work/github/leapflow/src/leapflow/copilot/predictors/l0_hash.py` - L0精确哈希
- `/Users/jason/work/github/leapflow/src/leapflow/copilot/predictors/l1_markov.py` - L1马尔可夫
- `/Users/jason/work/github/leapflow/src/leapflow/copilot/predictors/l3_llm.py` - L3 LLM推理
- `/Users/jason/work/github/leapflow/src/leapflow/copilot/feedback.py` - EvolutionLoop

### 因果推理
- `/Users/jason/work/github/leapflow/src/leapflow/causal/inference.py` - 三层推理引擎
- `/Users/jason/work/github/leapflow/src/leapflow/causal/components.py` - 前端组件 (EventDenoiser等)
- `/Users/jason/work/github/leapflow/src/leapflow/causal/channel.py` - 通道注册表

### Memory系统
- `/Users/jason/work/github/leapflow/src/leapflow/memory/manager.py` - 统一内存管理器
- `/Users/jason/work/github/leapflow/src/leapflow/memory/providers/` - 多个provider实现

### OODA框架文档
- `/Users/jason/work/github/leapflow/docs/design/ooda_framework.md` - 官方设计文档

---

## 15. 总结与建议

### 15.1 核心创新

1. **无梯度学习**
   - 预测误差作为学习信号
   - LCS 对齐进行轨迹比较
   - 极大简化了在开环环境中的学习

2. **多层预测金字塔**
   - L0-L3 从精确到推理
   - 复杂度门限避免冗余
   - 在线和离线学习并行

3. **闭环自演化**
   - Loop γ 从执行反馈持续学习
   - 信任梯度反映实际能力
   - 无需显式重训练

4. **因果感知世界模型**
   - 规则 + 启发式 + VLM 三层
   - 与预测循环深度耦合
   - 支持模式发现和知识转移

### 15.2 实现的成熟度评估

| 组件 | 成熟度 | 备注 |
|------|--------|------|
| PredictionLoop | 90% | 核心链路完整, 细节可优化 |
| ExperienceStore | 85% | RAG工作良好, 语义重排需改进 |
| CuriositySignal | 95% | 设计完整, 权重可微调 |
| ExperienceReplay | 80% | 两个策略完整, 洞察应用不完整 |
| TrajectoryGrader | 75% | 基础功能完整, 级联应用不完整 |
| Copilot L0-L3 | 85% | L0-L1启用, L2-L3框架完整但集成有限 |
| CausalGraph | 70% | Tier1-2完整, Tier3框架存在 |
| Loop γ 集成 | 50% | 核心逻辑存在, 端到端闭环不完整 |

### 15.3 建议的改进方向

**短期 (1-2周)**
1. 完成 Loop γ 与所有组件的深度集成 (特别是自动降级机制)
2. 改进语义距离计算 (当前结构化差异较粗糙)
3. 增强回归检测的鲁棒性

**中期 (1-2月)**
1. 实现完整的多轨迹共识 v2 (用于 Loop γ 演化)
2. 从经验重放的洞察自动更新因果图
3. 实现跨应用规则迁移 (当前实验性)

**长期 (3-6月)**
1. 实现 L5 能力：独立推理 + 假设检验
2. 支持长周期任务学习 (当前基于单次执行)
3. 多源 Hub 集成下的统一世界模型

### 15.4 关键指标

**当前应该监控的指标**：
- 预测准确度 (δ < 0.3 的比例)
- 好奇心分布 (各阶段的平均好奇度)
- 技能成熟度分布 (STEP vs CONFIRM vs AUTO 的比例)
- 经验重放洞察的采纳率
- Loop γ 回归检测触发频率

---

## 附录：术语速查表

| 术语 | 定义 | 使用场景 |
|------|------|---------|
| δ (Delta) | 预测误差, ∈ [0, 1] | 预测环、好奇心驱动 |
| Context Hash | 当前状态的哈希值 (app+窗口+剪贴板) | L0预测 |
| acceptance_rate | 历史接受率 = accept_count / total_count | L0置信度 |
| advantage | OPD教师给出的评分, ∈ [-1, 1] | 技能升降级 |
| is_forking | 是否为关键决策点 | 自蒸馏重点 |
| Curiosity Score | 三层内在动机的加权和 | 探索导向 |
| SNR (信噪比) | Context Attention的保持指标 | 质量评估 |
| compression_ratio | 原始步骤 / 抽象步骤 | Orient质量 |
| LCS (Longest Common Subsequence) | 最长公共子序列 | 多轨迹共识 |
| OPD | On-Policy Distillation | 教师评分框架 |
| VLM | Vision Language Model | 跨模态验证 |
| RAG | Retrieval-Augmented Generation | L3增强 |

