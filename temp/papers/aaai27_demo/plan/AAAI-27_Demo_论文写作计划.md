# AAAI-27 Demo 论文写作计划

> 主题：从**自我进化 Harness**视角，结合 LeapFlow 的 **LLM world model + leapspace + 自我进化机制**，并以面向真实世界信号的 **OODA loop** 为辅线，写一篇 AAAI-27 Demonstrations Program 短论文（demo paper）。
>
> 本计划为写作蓝图，不是论文正文；正文按第 4 节大纲分节撰写。
>
> 约束来源：`../official/AAAI-27_Demonstration_Program_官方信息与投稿要求.md`（下称 official 文档，其中 [S1]=官方 Call，[S2]=Author Kit，[S3]=OpenReview 站点）。
> 技术事实来源：`/Users/jason/work/github/leapflow` 代码库只读核查（文件路径见第 13 节证据台账）。

---

## 0. 官方硬约束回执（写作前必须锁定）

以下每条都直接决定写作与排版，均取自 official 文档，不得违反：

| 约束 | 取值 | 对写作的含义 |
|---|---|---|
| 正文篇幅 | two-page short paper | 正文严格控制在 **2 页**，超出即被退回。 |
| 参考文献 | one page of references only | references 单独占 **1 页**，且该页只能放参考文献。 |
| 排版 | AAAI two-column style（`aaai2027.sty` / `aaai2027.bst`） | 用 [S2] 的 `AnonymousSubmission2027.tex` 起稿；不得改 page layout。 |
| 视频 | video up to 5 minutes（可用 slides 替代，但 video 权重更高） | 必须产出 ≤5 分钟 demo 视频，作为 supplementary materials 传 OpenReview。 |
| 视频开头 | highly recommend 加 30s–1min overview（非强制） | 视频前 30–60s 放系统总览。 |
| 内容要求 | 呈现 technical details + 讨论 related work + 描述 significance + previously unpublished | 四要素在 2 页内都要出现，缺一不可。 |
| 新颖性 | 必须是 new ideas，非 mainstream products/services 已有 | 论文主张要落在“治理化、可解释、可回滚的信号驱动进化”这一差异点。 |
| 盲审 | single-blind 或 double-blind 二选一 | 见第 3 节匿名策略；默认建议 double-blind。 |
| reproducibility checklist | Demo track 不要求 | 不写、不传 checklist。 |
| 现场义务 | 至少一位 author 现场 in-person 演示；有 live demo 时段 | 论文与视频都要体现“可现场跑”，不是纯录播概念。 |
| 提交入口/截止 | [S3] OpenReview；实际底线取较早者 | [S1] 名义 Sep 18, 2026 11:59 PM AOE (UTC-12)；但 [S3] 系统强制 **Sep 18, 2026 11:59 UTC**（约早 24h），以较早的 [S3] 时间为提交底线，见第 12 节。 |
| Author Kit 范围 | 仅取 two-column style / anonymous 参考 | [S2]（含 `CameraReady2027.tex`）是面向 accepted-paper publication 的通用发表指南；其 source 上传、文件命名、copyright form、page charge 等**不作为 Demo 初投义务**，除非 [S1] 明确要求。 |
| 附录/页数边界 | Content Appendices 计入 page limits | 附录不为 Demo 增加页；全部非参考文献内容须落在 2 页内，额外一页只放 references。 |
| backup mode | official 措辞为 `It is expected`（非 `must`） | 仍应准备不依赖 special arrangement 的 backup，但定位为“预期”而非强制。 |

---

## 1. 论文定位与核心主张

### 1.1 一句话定位（demo thesis）

> **展示一个“自我进化 Harness”：它把真实 headless structural signal、条件性 signal-mode archive 与受控 counterfactual evidence 组织为可审计的 capability decision——cold-path world model 提供 teacher/student 与 four-value action space，OODA/PCD 解释执行深度，leapspace 提供 ground-truth structural envelope，而 off-by-default 的 governed pipeline 将 capability hypothesis 与实际 mutation authority 分离。**

### 1.2 差异化卖点（对应 [S1] 的 new ideas 要求）

主流 agent 产品的能力集要么是硬编码、要么是无治理地自动写代码。本 demo 的新点在于把“进化”做成一条**可治理、可审计、可回滚**的管线，并显式区分两层：

- **“agent 知道什么”**（world model 的 grading + distillation）：始终运行、不写代码、不改能力集、零额外风险；
- **“agent 能做什么”**（acquire→proposal→…→verified 的能力写入）：off-by-default、逐门放行、每步留痕。

这一“knows vs can-do”分离，加上对 no-op/reject/reuse/unexercised boundary 的 causal evidence 可视化，是可现场演示的系统贡献；真实 install/approval/daemon long-horizon 仍按第 2 节的证据边界处理。

### 1.3 贡献点（demo paper 版，写进 Introduction 末尾，3 条以内）

- **C1**：一个可观测、可干预的 `signal → OODA → world model → governed decision` Harness：真实 headless structural signal 与受控 CE-X 反事实都能进入同一证据叙事；不把它表述为 GUI-agent end-to-end execution。
- **C2**：cold-path teacher/student world model 的 information-context 不对称与 four-value action space（absorb/rebind/acquire/escalate），分别服务知识蒸馏与 capability hypothesis；在 controlled ablation 完成前，不以现有 N=3/confounded 样本宣称可泛化诊断准确率。
- **C3**：把 capability evolution 的 policy/lifecycle/approval contracts 与可审计 evidence boundary 展示为治理对象；CE-X 保留 no-op/reject/reuse/acquire-hypothesis，但真实 install、approval、daemon long-horizon trust/rollback 只在已有同 run artifact 时宣称。

---

## 2. 忠实性边界（本计划最重要的一节：可宣称 vs 不可宣称）

demo paper 必须 previously unpublished 且不得 overclaim。以下边界基于代码核查，务必在正文/视频中严格遵守。

### 2.1 可以作为“可运行/可现场演示”宣称

- world model 的 **grading + distillation 始终运行**（不受 `evolution_enabled` 控制）：对每个 session 打分、把学到的环境知识蒸馏进下一轮 context、并推荐优先使用哪个已装 provider。证据：`config.py` 注释、`world_model/trajectory_grader.py`、`storage/distilled_knowledge_store.py`。
- **four-value 裁决**（absorb/rebind/acquire/escalate）与 `rebind`/`acquire` 的替代者判定逻辑（`build_alternatives_provider` + `requires_environment_affordances`）。证据：`domain/adaptation_verdict.py`、`learning/degradation_sink.py`。
- **OODA loop + 弹性预算 + PCD**：`_run_agent_loop()`、`IterationBudget`（fixed/elastic）、`DisclosurePlanner`（CORE/EXPANDED/FULL）。证据：`engine/engine.py`、`engine/budget.py`、`engine/context_disclosure.py`、`engine/agent_loop.py`。
- **治理管线的 Protocol 可达性与状态机**：`EvolutionProposalView`/`EvolutionLifecycleStore`/`OutcomeStore`、`AdaptiveEvolutionPolicy`、`LifecycleGovernor`、`PluginTrustLedger`；acquisition lifecycle 状态 `PENDING→GENERATED→APPROVED→INSTALLED→PROBATION→VERIFIED→REJECTED/FAILED/QUARANTINED`。证据：`plugins/evolution_contracts.py`、`plugins/adaptive_policy.py`、`plugins/lifecycle_governor.py`、`learning/plugin_trust.py`。
- **可现场展示的入口**：CLI（`leap`）、TUI、leapd daemon；`leap config` 查看 `evolution_enabled` 等开关；`plugin_list` 的 live `capability_report`；`self_management` 工具插件的 `plugin_propose`/`plugin_generate`/`plugin_install`。证据：`cli/`、`daemon/`、`plugins/tool_plugins/self_management.py`。
- **leapspace signal 模式**可运行：sandbox 内 `BaseLeapApp` 原子写 `state.json` / `events.jsonl`，`LeapAppHarness` 录制 reference stimulus 并运行 `expect()` 判定 PASS/FAIL。证据：`src/leapspace/app_space/harness.py`、`.../apps/_base.py`、`.../signal.py`。这不是 agent 在 sandbox 内求解任务。
- **真实 headless structural seam**可运行：两个 offscreen PyQt6 `BaseLeapApp` 版本写出 role-aware `elements`；`LeapSpaceHostRpc` 以 `state.json` 满足 HostRpc 的结构查询；实验侧 `LeapSpaceEnvironmentSource` 将真实 rename 送入 shipped `CapabilityObservationService`，获得 opt-in `interface_drift` observation 和 `CapabilityRequirement(origin=environment_probe)`，并有 benign negative control/default refusal。证据：`src/leapspace/app_space/host_rpc.py`、`temp/leapspace_exp/evo-02/leapexp2/leapspace_source.py`、对应 tests。

### 2.2 必须明确标注为“off-by-default / 实验性 / 未实现”，不可当既成能力宣称

- **`evolution_enabled` 默认 False**：`acquire` 裁决→排队 proposal 的分支默认关闭；即便开启，generation/validation/approval/sandbox/trust 仍是其前方多道门。→ 正文措辞用“governed, opt-in pipeline”，**不要**写“系统会自主写代码上线”。
- **learning 回路需显式配置**：`accepted_evidence_kinds` / `active_signal_sources` 为空时评分-获取回路处于惰性；现有 skills 皆 builtin。→ 演示自进化时须说明这是“operator opt-in”的实验配置。
- **leapspace e2e 模式 = NotImplementedError**（`app_space/e2e.py` 为空）：signal mode 只录制 reference stimulus 与 ground-truth verdict，不能让 agent 在 sandbox 里“求解”。→ 演示 leapspace 时不得暗示 e2e 闭环。
- **headless structural seam 无 framebuffer**：`LeapSpaceHostRpc.screen.capture_frame` 诚实返回空；真实 widget rename→requirement 不等同于 Linux/KVM/AT-SPI 的 L2 GUI/OS/pixel 证据。→ 作为独立 lane，不升级为 EXP-7 已完成。
- **C2 的 acquire 不等于真实 mutation**：canonical runner 明示未执行 code generation、approval prompt、real install、daemon、RecoveryCoordinator 与真实 LLM diagnosis。→ 画面必须显示 `not_exercised`，不可展示为 plugin 已上线。
- **S1–S4（持久化方向、事件驱动重入、在线校准、常驻无限循环/resident agent）** 多为设计或默认关闭（`agent.reentry_enabled` 默认关闭）。→ 归入 “future work / roadmap”，不进 demo 主线。
- `temp/leapspace_exp` 的 headless environment source 已是受测实验 seam，但仍是 experiment/reference layer；正文应描述其真实 widget→observation 事实，同时注明它尚非 e2e/daemon runtime 证据。

### 2.3 写作纪律

- 动词分级：始终运行的用“does / runs”；开关后的用“can, when enabled / opt-in”；未实现的用“is designed to / planned”。
- 每条能力主张尽量对应视频里一个真实可见的画面（第 6、7 节）。

---

## 3. 标题、作者与匿名策略

### 3.1 候选标题（保持术语原文，二选一或微调）

- 主推：**“LeapFlow: A Self-Evolving Harness that Turns Real-World Signals into Governed Capability Evolution”**
- 备选：**“From Signals to Governed Evolution: Demonstrating a World-Model-Driven Self-Evolving Agent Harness”**

要点：标题需含 demo/system 气质，并同时出现 self-evolving / Harness / world model / governed 等关键词。

### 3.2 匿名策略（对应 official 第 7 节）

- **默认建议 double-blind**：用 [S2] `AnonymousSubmission2027.tex`（`\usepackage[submission]{aaai2027}`），作者写 “Anonymous Submission”、清空 affiliations、提交前用 metadata-cleaning 工具清 PDF metadata、首页不放 copyright footer。
- 视频与代码链接也要匿名化（去掉含真实机构/个人的仓库 URL、账号、水印）；如需公开仓库，改用匿名镜像或 anonymized 链接。
- 若最终选 single-blind，则作者信息正常出现，但仍需保证内容一致。

---

## 4. 逐节写作大纲与篇幅预算（2 页正文）

> 目标：2 页两栏。建议总正文 ~1300–1600 词 + 1 张主图 +（可选）1 张小图/小表。references 另占第 3 页。以下“预算”为占版比例的经验值，撰写时以不超 2 页为硬红线。

### 4.1 Title + Abstract（~8%）
- Abstract 100–130 词：点出 gap（固定能力集/无治理自写代码）→ 我们 demo 什么（Harness + world model + governed evolution + OODA/leapspace）→ 现场观众能看到什么 → 差异点（knows vs can-do 的治理化分离）。

### 4.2 §1 Introduction & Motivation（~22%）
- 真实世界信号驱动的 agent 面临的问题：环境会漂移，能力集要么僵化、要么被不受控地自动改写。
- 提出 demo 的核心命题（1.1）与 3 条贡献（1.3）。
- 明确“这是一个可现场交互的系统 demo”，并预告 §3 现场脚本。

### 4.3 §2 System Overview（~30%，配主图 Figure 1）
分 4 个小段，每段 2–4 句，对应架构 4 大件：
- **(a) OODA loop 与自适应深度**：Observe（意图分类/技能触发）→ Orient（PCD 分层定向，CORE/EXPANDED/FULL）→ Decide（LLM 选工具/回复）→ Act（执行+观察）；`IterationBudget` 按 difficulty 弹性调节迭代上限。
- **(b) leapspace 环境侧信号**：`BaseLeapApp` 原子写 `state.json`/`events.jsonl` 与 role-aware `elements`；`LeapSpaceHostRpc` 将这些 ground truth 暴露为结构化 HostRpc facets。signal mode 可录制 reference stimulus 并运行 `expect()`；独立的 offscreen real-widget seam 证明 rename→`interface_drift`→`environment_probe` requirement，且保留 benign/default-off negative controls。
- **(c) LLM world model（teacher/student, cold path）**：teacher 见全轨迹、student 只见当前态（information-context 不对称，源自 On-Policy Distillation 的 teacher-as-reward-model 思路）；`grade_and_propose()` 单次 LLM 调用同时产出 grades 与 four-value verdicts；`absorb/rebind` 走知识蒸馏与 provider 优选，`acquire` 才是能力获取的 hypothesis，`escalate` 上抛。
- **(d) Governed self-evolution pipeline**：acquire→proposal→generate→validate（syntax/structure/import/protocol）→compatibility→approval→install→sandbox smoke→register(DRAFT)→behavior tests→probation→trust accrual→verify；由 `AdaptiveEvolutionPolicy` 读结构化 (trust/risk/status/autonomy) 决策，`LifecycleGovernor` 记录转移与 trust ledger。**强调 off-by-default 与逐门放行。**

### 4.4 §3 Demonstration（~28%，可配 Figure 2 或时间线小图）
- 现场剧本（见第 6 节）压缩为 3 个 scene：①真实 headless widget rename/benign control→opt-in requirement；②CE-X 的 none/reject/reuse/acquire-hypothesis 与 teacher/student/OODA 边界；③完整时才出现的 signal archive 与 read-only evidence console。
- 明确“观众看到的界面”：headless seam record、`records.jsonl`/`summary.json`、bundle console、条件性的 signal archive、以及可选 live `causal_trace` lens；不把 `plugin_list` 或 `plugin_install` 的普通入口当作该 experiment 已执行的证据。
- 一句话点出可交互性：观众可选择 rename/benign fixture、是否 opt-in evidence kind 和 C2 arm，观察 observation、requirement、resolution 与明确的 `not_exercised` 边界；不在现场改写 capability。

### 4.5 §4 Significance, Related Work & Limitations（~12%）
- Significance：对 AI 社区的意义——把 self-evolution 从“黑箱自改写”变成“可解释、可回滚、可审计”的治理对象；knows/can-do 分离降低风险。
- Related work（2–4 句，密集引用，见第 8 节）：train-free / on-policy distillation、LLM agents 与 tool learning、self-improving/self-modifying agents、agent governance & safety、world models for agents。
- Limitations（1–2 句，诚实）：acquire 通道 opt-in；CE-X 未执行真实 install/approval/daemon/independent oracle；leapspace e2e 未实现，headless seam 无 pixels；teacher live diagnosis 仍是 N=3/confounded；resident/reentry 未默认启用——与第 2.2 节一致。

### 4.6 References（第 3 页，仅参考文献）
- 用 `\bibliography` + `aaai2027.bst`；控制在能放满但不溢出 1 页的数量（约 12–20 条）。

---

## 5. 图表计划（适配两栏，PDFLaTeX 只接受 .pdf/.png/.jpg）

- **Figure 1（主图，跨栏或单栏均可）— 系统架构与信号流**：从 leapspace/真实交互的 signals → OODA loop（含 PCD 分层与 elastic budget）→ world model（teacher/student + four-value verdict）→ 分叉：`absorb/rebind`（distilled knowledge / provider 优选，always-on）与 `acquire`（governed pipeline，gated）。图上用不同底色区分 always-on vs off-by-default。**这是必备图**，承担 §2 主要信息量。
- **Figure 2（可选，小图/时间线）— governed evolution lifecycle**：画 acquisition lifecycle 状态机（PENDING→…→VERIFIED，以及 REJECTED/FAILED/QUARANTINED 分支），标出 approval 门与 trust/probation 节点。若 2 页排不下则并入 Figure 1 或省略。
- **不使用**：截图堆叠、type-3 字体、.gif/.eps；图内文字用矢量 pdf，保证 300dpi 以上或矢量。
- 图注（caption）承担部分说明，减轻正文字数压力。

---

## 6. Demo 现场脚本（storyboard，供 §3 与视频共用）

3 幕均以可核验 artifact 为前提；TUI/直接 `plugin_install` 不再是主线，避免用未闭合的 e2e/approval 代替证据：

- **Scene A — 真实 headless structural signal（~1.25 min）**
  - 运行 write-once headless seam：真实 offscreen PyQt6 widgets 写 `elements`，`send_button → dispatch_button` 经 `LeapSpaceEnvironmentSource` 产生 `interface_drift` 与 `environment_probe` requirement。
  - 同屏展示 benign negative control 和 default classifier refusal，强调 observation ingress 不授予 mutation authority；明确无 pixels/GUI-agent e2e。

- **Scene B — CE-X 受控四臂与 world-model/OODA 边界（~2 min）**
  - 运行/回放 `run_c2.py` 的 `records.jsonl`/`summary.json`，对称展示 none、rejected、reuse 和 acquire hypothesis；C2 的 live catalog resolution、profile isolation、authorisation 和 handler-reported verifier 是可见事实。
  - world model/PCD 仅展示架构或同 run 记录的 information boundary 与 four-value action space；不把 N=3/confounded 样本说成 accuracy result。

- **Scene C — 条件性 signal archive 与 read-only evidence console（~1 min）**
  - 有完整 manifest 时播放 reference stimulus + `expect()` PASS、state/event 与 media hashes；无 archive 时 console 显示 `not_supplied`，不以代理画面替代。
  - `aaai_demo render` 的 bundle console 展示 claims、unexercised approval/install/daemon/independent oracle；`causal_trace` 仅在 live framework-evolution finding 存在时作为只读补充。

- **收尾（~0.5 min）**：回到 thesis——knows/can-do 分离、负结果可见、每种证据层级不互相升级。

> 备用/backup mode（official 第 10 节；[S1] 措辞为 `It is expected`，非 `must`）：若现场 LLM/网络不可用，运行 deterministic headless seam + C2 + 已封存 bundle；若已审核 signal archive 存在，使用 `backup-reel` 拼接它。不得为了 backup 伪造 GUI/e2e/approval 成功。

---

## 7. 视频计划（≤5 分钟）

| 时间 | 内容 | 对应 |
|---|---|---|
| 0:00–0:35 | overview：一句话 thesis + evidence-level architecture（对应 [S1] 建议的 30–60s overview） | §Abstract/§1 |
| 0:35–1:25 | Scene A：headless PyQt rename、negative control、opt-in observation admission | §2(b), §3① |
| 1:25–2:10 | teacher/student + OODA/PCD 的结构边界，不报告未校准 rate | §2(a,c) |
| 2:10–3:55 | Scene B：CE-X 四臂与 `not_exercised` 证据边界 | §2(d), §3② |
| 3:55–4:35 | Scene C：完整时才播放 signal archive；否则显示 `not_supplied` | §2(b), §3③ |
| 4:35–5:00 | 收尾、limitations、`run_c2 → bundle → render` 复现路径 | §4 |

- 主视频严格 ≤5 分钟；`aaai_demo.render` 的 EDL 必须是 300 秒。录屏用真实 terminal output、bundle console 和已审核 archive；关键处持续显示 evidence level 与 run ID。
- double-blind：视频去除机构水印、真实账号、可识别路径；导出前清 metadata。
- 交付格式：常见 mp4/H.264；作为 supplementary materials 上传 OpenReview。

---

## 8. Related work 引用清单（供撰写与 .bib 组织，按主题）

按主题各选代表作，密集但精炼（最终 12–20 条，放第 3 页）：

- **On-Policy Distillation / train-free learning**：teacher-as-reward-model、hindsight 轨迹打分（对应 `trajectory_grader.py` 的思想来源）。
- **LLM agents & tool learning**：ReAct、Toolformer、以及 tool-use/agent survey。
- **Self-improving / self-modifying agents**：Voyager（skill library 自增长）、自反思类（Reflexion）、自动化 agent 构建。
- **Agent governance / safety / approval**：human-in-the-loop 审批、能力沙箱与隔离、progressive autonomy/trust 的相关工作。
- **World models for agents**：world-model-based planning/adaptation 的代表作。
- **Environment / benchmark & CUA sandbox**：computer-use agent、GUI/desktop 交互环境（对应 leapspace 定位）。

> 写作时每个主题 1–3 句带过，用 `\citep`/`\citet`；避免逐篇展开（篇幅不允许）。真实 bib 条目在起稿时据实补全，勿编造引用。

---

## 9. 术语与命名规范（保持原文，不翻译）

正文中以下术语**一律保留英文原样**（首次出现可加一句中文/英文释义，其后直接用原词）：

- 架构/机制：`Harness`、`world model`、`teacher/student`、`OODA loop`、`Observe/Orient/Decide/Act`、`Progressive Context Disclosure (PCD)`、`leapspace`、`self-evolution`、`capability`、`plugin`。
- 裁决/生命周期：`absorb / rebind / acquire / escalate`、`EvolutionIntent`、`CapabilityRequirement`、`AdaptiveEvolutionPolicy`、`LifecycleGovernor`、`trust / probation / quarantine`、lifecycle 状态 `PENDING/GENERATED/APPROVED/INSTALLED/PROBATION/VERIFIED/REJECTED/FAILED/QUARANTINED`。
- 开关/配置：`evolution_enabled`、`accepted_evidence_kinds`、`selection_policy`、`distilled_knowledge_ttl_s`。
- 入口：`leapd`、`TUI`、`leap config`、`plugin_list`、`capability_report`、`plugin_propose/generate/install`。

统一大小写与拼写（如 `LeapFlow`、`leapspace`、`OODA`），全篇一致。

---

## 10. 排版与提交 checklist（AAAI Kit + OpenReview）

起稿与交付逐项核对（对应 official 第 6/7/11 节）：

- [ ] 用 `AnonymousSubmission2027.tex` 起稿，加载 `\usepackage[submission]{aaai2027}`；引用 `aaai2027.bst`。
- [ ] 正文 ≤2 页；references 单独 1 页且只放参考文献。
- [ ] 不改 page layout（禁 `\columnsep`/`\textwidth`/geometry 等）；不打印页码；正文 10pt Times、无正文着色。
- [ ] 图为 .pdf/.png/.jpg，无 type-3 字体，不侵入 margin。
- [ ] double-blind：无作者名/机构、清 PDF metadata、首页无 copyright footer；视频/链接同步匿名。
- [ ] 不含 reproducibility checklist（Demo track 不要求）。
- [ ] 视频 ≤5min，含 30–60s overview，作为 supplementary 上传；（可选）代码作为 supplementary。
- [ ] 按官方“expected”预期具备 backup mode（预录 signal trace + cassette 回放）；[S1] 用词为 `It is expected`、非 `must`。
- [ ] Author Kit 范围自限：不把 [S2] 中仅面向 accepted-paper publication 的事项（LaTeX source 上传、文件命名、copyright form、page charge）当作 Demo 初投义务，除非 [S1] 明确要求。
- [ ] 页数边界自检：Content Appendices 计入 page limits、不为 Demo 增加附录页；全部非参考文献内容落在 2 页内，额外一页只放 references。
- [ ] OpenReview 账号就绪，经 [S3] 提交；以较早的 [S3] 系统强制时间 **Sep 18, 2026 11:59 UTC** 为实际底线（[S1] 名义为 Sep 18 11:59 PM AOE / UTC-12，约晚 24h），务必按较早者完成 final submission。
- [ ] 全篇“动词分级”自检（第 2.3 节），无 overclaim。

---

## 11. 建议的 plan 目录产出物

本次先产出本写作计划；后续在同目录（`../plan/`）可逐步补齐：

- `outline.md`（可选）：把第 4 节大纲拆成逐段 bullet，供直接填字。
- `figures/`（可选）：Figure 1/2 的草图与最终矢量图源。
- `related_work.bib`（可选）：第 8 节引用的真实 bib 条目。
- `video_script.md`（可选）：第 7 节脚本逐镜头细化 + 旁白稿。

（正文 `.tex` 建议放到独立的 `../submission/` 目录，与 plan 分离。）

---

## 12. 任务分解与时间线（实际提交底线 [S3] Sep 18, 2026 11:59 UTC / notify Nov 6 / camera-ready Nov 20）

> 截止时间跨来源核对：[S1] 名义 Sep 18, 2026 11:59 PM AOE (UTC-12)（≈Sep 19 11:59 UTC），[S3] OpenReview 系统强制 Sep 18, 2026 11:59 UTC（约早 24h）。本计划一律以较早的 [S3] 时间为底线。

| 阶段 | 产出 | 建议完成点 |
|---|---|---|
| T0 计划确认 | 本文件 + 与合作者确认 thesis/边界 | 立即 |
| T1 证据固化 | 复核第 2 节可宣称/不可宣称清单，逐条对齐代码 | +2 天 |
| T2 主图 | Figure 1 定稿（架构+信号流+always-on/gated 分色） | +4 天 |
| T3 初稿 | §1–§4 正文（控 2 页）+ references 草表 | +8 天 |
| T4 demo 环境 | 现场剧本三幕可跑 + backup（cassette）验证 | +10 天 |
| T5 视频 | 录屏 + 剪辑 ≤5min + overview + 匿名化 | +13 天 |
| T6 匿名与排版终审 | double-blind 自检 + Kit 合规 checklist 全绿 | 截止前 3 天 |
| T7 提交 | OpenReview 上传 paper + 视频/补充材料 | [S3] Sep 18, 2026 11:59 UTC 前（较 [S1] 名义 AOE 早约 24h，取较早者） |

（若录用：Nov 20 前完成 camera-ready，并按 official 第 10 节与 Chairs 敲定现场展台/特殊需求。）

---

## 13. 证据台账（技术主张 → 代码路径）

供撰写与自检时溯源，避免 overclaim：

- world model / teacher-student：`src/leapflow/world_model/trajectory_grader.py`、`src/leapflow/learning/world_model_driver.py`、`src/leapflow/storage/distilled_knowledge_store.py`
- four-value / 替代者判定：`src/leapflow/domain/adaptation_verdict.py`、`src/leapflow/domain/evolution_intent.py`、`src/leapflow/learning/degradation_sink.py`（`build_alternatives_provider` / `requires_environment_affordances`）
- OODA / 预算 / PCD：`src/leapflow/engine/engine.py`（`_run_agent_loop`）、`src/leapflow/engine/budget.py`（`IterationBudget`）、`src/leapflow/engine/context_disclosure.py`（`DisclosurePlanner`）、`src/leapflow/engine/agent_loop.py`
- 治理管线：`src/leapflow/plugins/evolution_contracts.py`、`src/leapflow/plugins/adaptive_policy.py`、`src/leapflow/plugins/lifecycle_governor.py`、`src/leapflow/learning/plugin_trust.py`、`src/leapflow/learning/capability_gap_detector.py`、`src/leapflow/learning/plugin_generator.py`
- 开关/配置：`src/leapflow/config.py`（`evolution_enabled` 默认 False、`accepted_evidence_kinds`、`distilled_knowledge_*`、`selection_policy`）
- 入口/演示：`src/leapflow/cli/`、`src/leapflow/daemon/`、`src/leapflow/plugins/tool_plugins/self_management.py`
- leapspace：`src/leapspace/app_space/harness.py`（signal 可用 / e2e `NotImplementedError`）、`src/leapspace/app_space/apps/_base.py`（role-aware elements）、`src/leapspace/app_space/host_rpc.py`（headless structural HostRpc）、`src/leapspace/app_space/signal.py`、`src/leapspace/app_space/e2e.py`（空）
- headless environment seam：`temp/leapspace_exp/evo-02/leapexp2/leapspace_source.py`、`tests/test_environment_source_l1.py`、`tests/test_environment_source_live.py`；它们证明 widget→observation/requirement，不替代 e2e。
- Demo evidence tooling：`temp/papers/aaai27_demo/aaai_demo/{headless_seam,evidence,render,cli}.py`、`demo_fixtures/`、`reproduction.md`。
- 诚实局限性素材：`temp/leapspace_exp/` 的实验计划与 reports，尤其是 L1/L2/L3 划分、real-LLM N=3/confounded 与 C2 substitutions。
