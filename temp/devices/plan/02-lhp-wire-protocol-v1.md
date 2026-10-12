# 02 — LHP 线协议 v1（LeapFlow Hardware Protocol）

> 状态：**冻结（v1）**。实现与 `hardware/lhp/` 一一对应；改协议 = 改本文件 +
> `LHP_PROTOCOL_VERSION` 递增 + `protocol_mismatch` 兼容测试。

## 1. 定位与分层

LHP 是**硬件域协议**，不是通用 RPC：它只承载"节点 ↔ 设备"的六方法语义
（`open/close/read/write/probe/halt` + 批量）。控制面（本版）为 **JSON Lines**；
流面（P3：订阅推送、二进制帧）预留 `event` 消息族，本版不实现。

```
TransportRef(kind="lhp", config)
        │  build_transport
        ▼
   LhpTransport ──── LhpClient ────┬── DirectLhpClient（进程内绑定 / 同信任域）
        (host)                    └── SocketLhpClient（tcp:// | unix://，P4 加 TLS）
                                          │  JSON Lines
                                          ▼
                                   LhpNodeServer（node）
                                          │  registry.read/write/probe/halt
                                          ▼
                                   HardwareRegistry（node 本地）
```

## 2. 传输绑定（binding）

| 绑定 | 冒号形式 | 用途 | 信任模型 |
|---|---|---|---|
| direct | `{"server": <LhpNodeServer>}`（config 内对象） | 进程内（测试 / 嵌入式） | 同进程 = 同信任域，免鉴权 |
| tcp | `tcp://host:port` | 远程节点（Jetson） | v1：hello 共享令牌；P4：mTLS |
| unix | `unix:///path.sock` | 同机独立进程 | 文件权限即边界；hello 令牌仍校验 |
| 预留 | ws / ssh-stdio | 浏览器 / 调试兼容 | 后续版本 |

## 3. 帧格式

一行一个 JSON 对象，UTF-8，`\n` 结尾。控制消息上限 **1 MiB/行**（超限断开）。
`ensure_ascii=False`；浮点原样（不转字符串）。

### 3.1 请求

```json
{"id": 1, "method": "read", "params": {"device_id": "host", "channel_id": "cpu.load1"}}
```

### 3.2 响应（成功）

```json
{"id": 1, "ok": true, "result": {"device_id": "host", "channel_id": "cpu.load1", "value": 1.2, "quantity": "load", "unit": "", "observed_at": 1759900000.123, "sequence": 42, "quality": "ok"}}
```

### 3.3 响应（错误）

```json
{"id": 1, "ok": false, "error": {"code": "unknown_channel", "message": "unknown channel 'x'"}}
```

**错误响应语义 = "无法尝试"（could not attempt）。** 已到达设备的失败必须走
`result` 中的 `WriteOutcome`（携带 `side_effect_state`）——错误响应里不存在
"可能已生效"的表达，与 `TransportError` 契约一致。

## 4. 方法族（v1）

| method | params | result | 说明 |
|---|---|---|---|
| `hello` | `{protocol, token, client}` | `{protocol, node_id, device_count, declarations?, capabilities}` | **socket 绑定必须先发**；令牌校验；版本协商；`declarations` 是导出声明集的稳定指纹，旧节点可不提供 |
| `list` | `{}` | `{node_id, devices: [HardwareContext.to_dict()…]}` | 完整声明（主机据此构建本地 context）；**`transport.config` 置空、仅保留 `kind`**（节点私有绑定——串口路径/驱动凭据/链式 lhp 令牌——不过线；主机本就以自己的 `lhp` 绑定替换）；**`control_bindings` 不导出**（未解析的同类绑定原文，RV2 R2-7） |
| `read` | `{device_id, channel_id}` | `Reading.to_dict` | |
| `read_batch` | `{device_id, channel_ids: []}` | `{device_id, readings: [], observed_at}` | 原子批量（`BatchTransport`） |
| `write` | `{device_id, channel_id, value}` | `WriteOutcome.to_dict`（ok 可 false） | 节点先跑 `check_safety_policy`（纵深防御） |
| `write_batch` | `{device_id, commands: [[ch, v]…]}` | `{ok, outcomes: [], side_effect_state}` | 最坏情况合并（UNKNOWN/PARTIAL 优先） |
| `probe` | `{device_id}` | `TransportStatus.to_dict` | **不在 I/O 锁内**：可回答"某读卡住时它还活着吗" |
| `halt` | `{device_id}` | `TransportStatus.to_dict` | **不在 I/O 锁内**：物理急停不得排队 |

`capabilities` = 除 `hello` 外的全部方法名（本版固定集合，为 P3 增删留位）。`declarations` 对 wire-safe `list.devices` 的规范 JSON 计算 SHA-256 短指纹；心跳发现指纹变化时，Host 重新注册该节点并由 registry reconcile 差量替换声明。这样节点在一次心跳内重启、没有经历 offline 状态时，Host 也不会长期保留旧 envelope 或遗漏新设备。
`SocketLhpClient.connect()` 与 `DirectLhpClient.connect()` 均返回该 hello 结果全文（RV2 统一），
主机侧据 `capabilities` 做特性探测（P3 `subscribe`/帧方法上线后的前向兼容门）。

## 5. 错误码

| code | 含义 | 主机侧映射 |
|---|---|---|
| `bad_request` | 缺字段 / 未知方法 / 非法 JSON | `TransportError` |
| `unauthorized` | hello 缺失或令牌不符（随后断开） | 连接建立失败 → `connected=False` |
| `protocol_mismatch` | 协议版本不符 | 连接建立失败 |
| `unknown_device` | 节点无此设备 | `TransportError` |
| `unknown_channel` | 节点无此通道 | `TransportError` |
| `transport_error` / 具体 failure_code | 节点侧 `TransportError` 透传（含 failure_code） | `TransportError(failure_code=原码)` |
| `internal_error` | 节点处理异常（**写路径**可能已触达设备） | 写 → `SIDE_EFFECT_UNKNOWN`；读 → `TransportError` |
| `timeout`（客户端产生） | 请求超时（响应可能丢失） | 写 → `SIDE_EFFECT_UNKNOWN`；读/probe → 失败状态 |
| `connection_closed`（客户端产生） | 连接断开/EOF | 同上 |

## 6. 写路径三分类（主机侧 `LhpTransport.write` 的唯一裁决表）

| 收到 | 判定 | 返回 |
|---|---|---|
| `ok:true` 结果 | 明确成功 | `WriteOutcome(ok=True, side_effect_state=原样)` |
| `ok:false` 结果（含 `safety_policy_denied`） | 节点明确裁决 | `WriteOutcome` 原样（`NONE` 必然可证） |
| 错误码 `bad_request`/`unknown_*`/`transport_error` | 未触达设备 | 抛 `TransportError` |
| 错误码 `internal_error` / `timeout` / `connection_closed` | **效果未知** | `WriteOutcome(ok=False, side_effect_state=UNKNOWN)` |

## 7. 时基约定（与 `Reading` 文档一致）

- wire 只传 **wall-clock `observed_at`**（`Reading.to_dict` 本就不含 `monotonic_at`）；
- 主机收到时**本地重打** `monotonic_at = time.monotonic()`——跨机 monotonic 不可比；
- 跨节点相关性要求节点 NTP；R1 回归记录两端 `timedatectl` 作为证据。
- `sequence` 原样透传：主机据此检测丢样，不重编号。
- `sequence` 是**节点进程内计数**：节点重启即复位（R1.9 实测 89→1）。主机判定陈旧/新鲜一律以
  `observed_at` 为准；sequence 回跳视为"节点会话断层"信号（可检测、不可视为丢样），
  不做跨进程持久化承诺（P3 订阅引入 `session_epoch` 后可显式区分）。

## 8. 设备身份与命名空间

- 节点设备的 `device_id` 是**节点本地的**（如 `host`）；主机注册时命名空间化为
  `<node_id>__<device_id>`（`[a-z0-9_]` 合法），`TransportRef.config` 携带
  `remote_device_id` 指回节点本地 id；
- 读值本地化：`LhpTransport` 把返回的 `Reading.device_id` 改写为主机侧 id；
- 冲突规则：本地声明先胜（`load()` 合并顺序：providers → 外部源，`_admit`
  duplicate 拒绝后者）。

## 9. 鉴权与会话（v1）

- socket 绑定：连接建立后**首个消息必须是 `hello`**；令牌用
  `hmac.compare_digest` 常量时间比较；失败 → `unauthorized` + 断开；
- `direct` 绑定：同进程信任域，`handle_request` 不做令牌门（`hello` 仍校验版本）；
- 单连接请求并发：服务端每请求一个 task（信号量限流），响应写串行化；**halt/
  probe 与普通请求并发执行，互不排队**。v1 不做请求级优先级（halt 免锁已够用；
  P3 视需要加）。
- **v1 信任边界**：shared token 仅认证 control-plane client；`write`/`write_batch` 不携带
  per-action human approval proof。故 v1 远端写只适用于同一 trusted operator domain；
  暴露到独立信任域前必须按 `10-human-approval-deployment.md` 升级为可协商的 LHP v2
  signed execution permit。节点本地 `SafetyPolicy` 仍不可被 permit 覆盖。

## 10. 版本与演进规则

- `LHP_PROTOCOL_VERSION = 1`；`hello` 双向携带；不符 → `protocol_mismatch`；
- 新增方法 = 迭代版本并更新 `capabilities`（客户端不认的方法视为 `bad_request`）；
- 消息族扩展（P3）：`{"type":"event", ...}` 推送帧——**非请求/响应**，无 `id`；
  订阅以 `subscribe` 方法进入，本版未实现，禁止提前占用字段。
