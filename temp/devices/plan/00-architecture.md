# 00 — 目标架构：硬件 Fabric（节点织体）

> 长线终极形态。不采用任何短线替代（MCP 桥、SSH 轮询、逐设备插件）。本文是
> `02-lhp-wire-protocol-v1.md`（协议）与 `01-execution-plan.md`（实施）的依据。

## 一句话定义

把所有硬件接入形态统一为 **一个"节点（Fabric Node）"抽象**：节点是自治的硬件
运行时单元，节点之间通过 **LHP 线协议** 经 **可插拔 transport 绑定**通信；高速
闭环永远留在节点本地；MCP 降级为第三方互操作适配器；治理与演化全局统一。

## 目标架构总览

```
┌─ Operator 侧（Mac / 任意主机）───────────────────────────────┐
│ System-2: AgentEngine / 审批 / 会话 / 世界模型 / 演化治理       │
│ FabricManager（由 FleetManager 演进）:                         │
│   节点目录 · 心跳 · 重连 · 拓扑 · 路由 · 信任聚合 · 审计合并     │
│ lhp transport（本仓内置）→ 同一 HardwareRegistry               │
│ （下游零改动：采样 / 审批 / LeapBoard / 证据 / 演化）            │
└──────────────┬───────────────────────────────────────────────┘
               │ LHP v1（绑定：unix socket / TCP+TLS / WS / ssh-stdio）
┌──────────────┴─ Node 运行时（leapd edge profile / leap hw serve）┐
│ HardwareRegistry + admission（本仓同一套）                      │
│ Sampler / Stream / PreviewBroker（P3 起节点侧启用）              │
│ ControlHierarchy (S2→S1.5→S1) / HighFrequencyControlBus（本地）  │
│ InferenceStrategy（vla_local / position_control）· LocalSafety   │
│ 本地 EventBus · ReadingStore 断链缓冲（P3+）                      │
│ 可选：本地 TUI / IM Gateway / 本地 LLM（独立模式）                │
└──────────────┬───────────────────────────────────────────────┘
               │ 既有 provider / transport 插件表（不改）
           设备（RealSense / GPIO / 电机 / 仪器 …）
```

核心闭环：**设备声明住在硬件所在的节点上**；主机通过 FabricManager 把节点设备
物化进同一个 `HardwareRegistry`（`TransportRef(kind="lhp")`），此后所有既有链路
（采样、审批、信任、LeapBoard）对本地/远程设备完全同构。

## 核心决策（含否决方案与边界）

### D1. LHP 线协议取代 MCP 成为 fabric 数据面；MCP 降级为互操作适配器
- 理由（MCP 作为 fabric 的五个结构性缺陷）：
  1) 现有 `McpManager` 仅支持 stdio command 型服务器；2) 纯请求/响应，无推送、
  无背压、无序号语义；3) 无二进制流（`FrameTransport` 无法承载）；4) 工具名全
  局命名空间（first-wins），多节点互相挤压；5) `halt` 走普通工具调用，无法表达
  最高优先级。
- 被否决：① 改造 MCP 成 fabric（成本≈新协议且语义方向不对）；② 主机侧逐设备
  YAML + SSH/SSH-stdio 轮询（每样本往返、无推送、无身份/授权语义）；③ 两侧双
  agent 直连协同（违反"会话身份属于创建它的客户端"）。
- 边界：MCP transport（`transports/mcp.py`）继续保留，仅用于对接第三方 MCP
  服务器；`fleet.py` 中"no new RPC protocol"的旧注记随本决策正式废除（P1 已在
  文档中修订）。

### D2. 节点 = 可部署的运行时单元；三种部署模式是它的三种摆位
- embedded（同进程，现状不变）/ local（同机独立进程，unix socket）/ remote（网
  络，TCP）。同一份 leapd + `leap hw serve`，不做第二个产品。
- 主机不需要为远端设备写任何 YAML —— 声明在节点侧（D5）。

### D3. 信号分类路由（规范表）

| 信号类别 | 频率/量级 | 通道 | 保证 | 现有载体 |
|---|---|---|---|---|
| 控制指令 | 人/agent 级 | LHP `write`/`write_batch` | 审批 + 幂等键 + readback + `side_effect_state` | 审批链 / `WriteOutcome` |
| **安全 halt** | 事件级 | 节点本地优先 + fabric 广播取消 | 永不依赖网络/审批延迟；节点每周期 SafetyPolicy 兜底 | `ControlBus` / `hw_estop` / `hw_fleet_halt_all` |
| 遥测标量 | 0.05–50 Hz | 节点采样（P3 订阅推送，可降采样） | 序号丢样检测、双端 store | `stream.py` / `reading_store` |
| **高频闭环** | 100–1000 Hz | **永不跨 fabric**（节点内） | 主机只发目标（SubtaskGoal），节点自分解 | `ControlBus` + `realtime` + `ControlHierarchy` |
| 帧/媒体 | ≤30 fps | 二进制通道（P3）/ Foxglove / ROS2 | 背压 + 带宽档位 | `FrameTransport` / `preview` |
| 事件/告警 | 离散 | LHP 推送（P3）+ 健康探测 | 节点离线 → 设备 degraded | `health_monitor` / `fleet` |
| 环境/演化证据 | 快照/delta/outcome | LHP evolution 消息（P4） | provenance 带 node_id | `EnvironmentObservation` / `evolution` |

### D4. 本地控制主权与会话侧审批主权
急停、安全策略、闭环、断链缓冲、证据本地存储永远在节点。审批与隐私同意属于
**发起动作的 session/control-plane owner**：双端部署时通常是 host `leapd`；完整 LeapFlow
单独部署在端侧时可以是端侧 `leapd` + 本地 TUI/web/IM channel。`leap hw serve` 仅是
hardware data plane，不是 agent 或人机审批面。

审批是政策门，`SafetyPolicy` 是物理真值，双层防御、互不替代。节点侧
`LhpNodeServer` 在每次 write 前独立执行 `registry.check_safety_policy`。当前 LHP v1 的
shared token 只认证 trusted control-plane client，不证明某次人类同意；跨独立信任域的
远端写必须升级为 `10-human-approval-deployment.md` 定义的单次签名 execution permit，
由节点验签后仍继续执行本地安全检查。急停永不等待审批或网络往返。

### D5. 接入规则：协议一次、设备免费
- 新设备 = **节点上的 YAML**（主机零工作）；
- 新协议 = 1 个 transport 插件（entry-point 安装，`leapflow.hardware.transports`）；
- 新发现源 = 1 个 provider 插件；
- 新节点 = `LEAPFLOW_FLEET_NODES`（或未来 settings）一条 + （P4）配对；
- 新本地绑定 = 1 个 LHP binding（non-goal 若无名：**不预建占位**）。

### D6. 演化闭环接上 fabric（复用现有半成品）
`PhysicalEnvironmentSource`（`hardware/environment_source.py`，已实现但
`plugin.py` 中 `env_source = None` 未接线）泛化为 FabricEnvironmentSource：
snapshot=全节点设备；delta=节点/设备出现、消失、affordance 变化（typed delta）；
outcome=`EvidenceStore` 裁决 → `CapabilityRequirement` → **live resolution**
（Resolution Before Acquisition）→ no-op / binding 更新 / 受治理提案 →
插件安装 → 信任晋级（P4）。

### D7. 人机界面即信号源
TUI（已有）、IM Gateway（feishu/dingtalk/telegram/webhook/api_server 已就绪）、
语音（P5，新 `SignalSource` 插件 + STT）、teleop（`robot/teleop.py` 已有）——
全部走插件机制，fabric 不做特例。独立模式（模式三）即"节点自带 agent + 本地
人机外设"，反向联邦可选（fabric 天然对称）。

## 与现有代码的映射（复用清单）

| 架构要素 | 现有模块 | 本方案动作 |
|---|---|---|
| 设备声明 SSOT | `hardware/context.py`（hc.v0/v1、`to_dict`/`from_mapping` 完整） | 直接作为 LHP `list` 载荷 |
| 六方法 + side protocol | `hardware/transport.py`（`BatchTransport`/`FrameTransport`） | `LhpTransport` 实现六方法 + Batch |
| 传输工厂表 | `hardware/transports/__init__.py`（entry-point 可扩展） | 增加 `lhp` 行 |
| 发现契约 | `registry.py`（`load`/`reconcile`/`_admit`） | 增加外部源注册 API（节点设备显式登记，保持发现被动本地） |
| 节点管理半成品 | `hardware/fleet.py`（心跳/降级/路由已写，探测/发现是 MCP 全局桩） | P1 改为 LHP；P2 完成化 |
| 高频与分层控制 | `control_bus.py` / `realtime.py` / `control_hierarchy.py` | 节点本地，不动 |
| 语义工具 | `capability_router.py`（affordance → 工具） | P2 起跨 fabric 路由 |
| 演化探针 | `environment_source.py`（PhysicalEnvironmentSource） | P4 接线 + 泛化 |
| 协作边界 | `gateway/`（IM 信号边界） | 模式三的人机入口之一 |

## 反目标（明确不做）

1. 不做"每设备一插件"；2. 不做"发现即连接"（节点显式登记才是有权威的发现；
   opt-in mDNS 仅便利，永不是权威）；3. 不做跨网实时闭环；4. 不做 JSON 高速流；
5. 不在核心加厂商特例（新协议=新 transport 插件）；6. 不搞第二注册表（节点设备
进同一个 `HardwareRegistry`）；7. 不把 MCP 当 fabric（仅互操作）；8. 不以"扫得
更狠"解决感知（发现契约保持被动本地）。
