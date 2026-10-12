# 05 — Deep Review 记录（P1 改动集）

> 状态：**retained review evidence**，不是当前执行计划；后续工作以 `01`、`07`、`10` 为准。
>
> 方式：CodeReview subagent 对未提交变更集全量审查（`temp/` 排除，属规划资料非代码）。
> 结论：**0 blocker / 0 major / 5 minor + 1 nit**；6 项全部已修复并附回归测试；
> 修复后硬件相关套件 902 例全绿，全量单测 5477 通过，改动集 ruff 全部通过（基线例外见文末）。

## 审查范围

- **新增**：`src/leapflow/hardware/lhp/`（`protocol.py` 202 行 / `server.py` 595 行 /
  `client.py` 362 行 / `runtime.py` 187 行 / `__main__.py` / `__init__.py`）、
  `src/leapflow/hardware/transports/lhp.py`（392 行）、`tests/test_lhp_fabric.py`（1154 行）。
- **修改**：`fleet.py`（+492）、`plugin.py`、`registry.py`、`transport.py`、
  `transports/__init__.py`、`cli/cli.py`、`cli/commands/hardware.py`、`cli/context.py`（P1.12 注册点）、
  `src/benchmarks/native/fleet_isolation.py`，以及测试基建（`hardware_stubs.py`、`leapd.py`、
  `conftest.py`、`test_architecture_contracts.py`、`test_cli_hardware.py`、
  `test_hardware_transport_contract.py`、`test_journey_harness.py`）。

## 发现与修复（按严重度）

| # | 级别 | 现象 | 修复 | 回归测试 |
|---|---|---|---|---|
| 1 | minor | `SocketLhpClient.close()` 未与 `connect()` 串行化：close 落在 connect 进行中时，`_open_transport` 随后写入的 reader/writer 不会被收掉 → 活连接泄漏 | `close()` 置 `_closed` 后在 `_connect_lock` 内执行 teardown；connect 在 `_open_transport()` 后重查 `_closed`，已关闭则立即 teardown 并抛 `LhpRemoteError` | `test_close_during_connect_cannot_leave_a_live_connection` |
| 2 | minor | 超限帧路径直接从 `_read_loop` 退出，不关闭传输：服务端连接悬挂至超时才释放 | `_read_loop` 自身退出路径的 finally 中关闭 writer（reader/writer/task 一并置空） | `test_client_closes_the_connection_when_a_frame_exceeds_the_limit` |
| 3 | minor | **批量写路径缺少回读身份校验**：单写路径两端都校验 readback channel 与请求一致（node `_verify_reading_identity`；host 比较 channel），`write_batch` 未校验 → 错误通道的证据可冒充成功 | node 侧：`_handle_write_batch` 对每个 outcome 调 `_verify_reading_identity(readback, device_id, commands[index][0])`；host 侧：`LhpTransport.write_batch` 比较 `readback.channel_id`，不一致降级为 `WriteOutcome(ok=False, side_effect_state=UNKNOWN, failure_code="lhp_result_mismatch")` | `test_a_node_mislabelling_batch_evidence_cannot_pass_as_success`（host）、`test_a_batch_of_mislabelled_evidence_degrades_to_unknown`（node 端到端） |
| 4 | minor | `register_nodes_safely` 逐节点串行注册：每个不可达节点完整消耗一次连接超时，叠加在 daemon 就绪之前（多节点时启动阻塞 ~15s/节点） | `asyncio.gather(..., return_exceptions=True)` 并发注册，保持输入顺序，单节点失败仍被包含 | `test_register_nodes_safely_runs_concurrently_and_contains_failures`（asyncio.Event 全启动门：串行实现必超时） |
| 5 | minor | `bind_runtime(registry_changed=True)` 重绑定后旧 `FleetManager` 心跳任务成孤儿：持续对着旧 registry 发送心跳 | `_stale_fleet_managers` 暂存旧实例；`start_fleet()` 前与 `_teardown()` 时经 `_close_stale_fleet_managers()` 回收 | `test_rebind_closes_the_previous_fleet_manager_before_reregistering` |
| 6 | nit | `configured_nodes_from_env` 对 JSON `null` 调 `str()` 产生字面量 `"None"` 作为 token → hello 以"错误密钥"失败，认证报错难以排查 | 所有键 `str(config.get(...) or "")`（node_id / display_name / endpoint / token），null 一律视为未设置 | `test_configured_nodes_from_env_parses_and_skips_invalid`（新增 `"nulls"` 条目断言 `token == ""`、`display_name == ""`） |

## 子代理确认正确的关键路径（抽查通过，未改动）

- hello-first 鉴权门：首帧必须为 hello 且 `hmac.compare_digest` 比对 token，未通过即关闭；
- §6 写失败三分类端到端：validation → raise；明确无效应 → `NONE` 可安全重试；不确定 → `UNKNOWN`（`ERR_INTERNAL_ERROR`/`ERR_TIMEOUT`/`ERR_CONNECTION_CLOSED` 均归 UNKNOWN）；
- 读值 `device_id` 本地化与 identity round-trip（node 侧声明 → host 侧 `<node>__<device>` 命名空间 → 回路由）；
- 外部源生命周期：`register_external_contexts` → `reconcile()`（本地声明先胜）→ `unregister_external_source` 对称退出；
- P1.12 启动注册：daemon 侧失败包含 + 幂等（重复 `start_fleet()` 无副作用），CLI 读面不依赖 daemon。

## 测试证据（2026-10-12，本机 `.venv`）

| 套件 | 结果 |
|---|---|
| `test_lhp_fabric.py` + `test_cli_hardware.py` + `test_hardware_transport_contract.py` | **202 passed, 2 skipped** |
| 硬件/插件/架构宽口径（`test_hardware_*.py`、`test_plugin_*.py`、`test_inert_wiring_audit.py`、`test_architecture_contracts.py`、`test_effect_*.py`） | **902 passed, 2 skipped** |
| 6 项修复的回归测试（按名筛选，实到 7 例含 1 例既有单写孪生测试） | **7 passed** |
| `ruff check`（CI 同口径；本改动集全部 12 个文件） | **All checks passed**（顺带清除 2 处：`client.py` F841 为本轮新码；`fleet.py` F401 系 HEAD 既有，一行清除） |
| 全量单测 `-m "not e2e"`（并行） | **5477 passed, 27 skipped, 1 xfailed**；12 failed + 2 errors 均为基线环境问题（见下） |

## 既有基线问题（非本轮引入，不影响 P1 验收）

- `test_dsh_compatibility.py` ×12：环境无 Node.js ≥22（`DshRuntimeUnavailable`），与硬件改动无交集；
- `test_robot_arm_transport.py` / `test_physical_skill_e2e.py` 收集错误 ×2：`.venv` 未装 `numpy`（环境缺依赖），与硬件改动无交集；
- `test_tool_capability_declaration.py`：并行全量时偶发单例污染（本轮全量跑未复现；单独跑 8 passed）；
- journey 截图 cassette 漂移（`0da5bc5` 起）与 reading_store 保留期 flake（墙钟敏感，独立跑 3/3 过）：均属既有记录，另行跟；
- 仓库级 `ruff check` 剩余 44 处报错全部位于本轮未触碰文件（`robot/*`、`test_realtime_control.py` 等），与改动集无关，另行跟。

## 结论

P1 改动集（LHP v1 核心 + host 接线 + 端侧运行时 + P1.12 启动注册）通过 deep review：
无阻塞级问题；6 项轻微问题全部修复并有新增回归测试锁定。代码层面**已具备进入 R1 真机回归的条件**，
唯一剩余前置为端侧部署审批（见 `03` §3 审批清单 A1/A3/A4/A5 与 `04` Q3/Q4）。

---

## RV2 — 第二轮全面复审（R1 之后，2026-10-12）

> 方式：CodeReview subagent 复审全量未提交改动 + 结合 R1 真机日志人工交叉核对。
> 结论：**6 major + 3 优化**，全部已修复并附回归测试；J-RV2 真机门通过。

| # | 级别 | 现象 | 修复 | 回归测试 |
|---|---|---|---|---|
| R2-1 | major | `configured_nodes_from_env` 跳过非法条目时 `%r` 打印**原始配置**，含共享令牌 → 令牌进 daemon 日志 | 仅打印条目序号 + `node_id`/`endpoint`；非对象条目也给出告警 | `test_skipped_fleet_entries_never_log_the_shared_secret` |
| R2-2 | major | **同一插件实例**重复注入同一 registry 被误判为"换 registry"：增量 runtime rebind 会把已启动的 FleetManager 换成无人启动的新 manager，并遗留旧心跳 | `registry_changed` 改为身份比较 `new is not self._registry` | `test_same_registry_rebind_keeps_the_started_fleet` |
| R2-3 | major（安全） | 节点 `list` 导出 `HardwareContext.to_dict()` **含 `transport.config`**：节点私有绑定（串口路径、驱动凭据、MCP token、链式 lhp 令牌）过线泄漏给主机 | `_wire_context()`：`transport.config` 置空，仅留 `kind` 作南向诊断；02 规格 §4 同步 | `test_list_exports_declarations_without_node_private_bindings` |
| R2-4 | 优化 | `SocketLhpClient.connect()` 只返回 `{protocol,node_id}`，与 `DirectLhpClient` 的 hello 全文不对称；`capabilities` 无法用于特性探测 | 缓存并返回 hello 全文（副本） | `test_socket_and_direct_clients_return_the_same_hello_payload` |
| R2-5 | 优化 | 心跳串行探测：N 个离线节点各耗一个探测超时，拖慢在线节点监督；degraded / 恢复转换无日志（06 观察项 3） | `_heartbeat_sweep()` 并发探测；online→degraded / →online 恢复均打 INFO | `test_heartbeat_sweep_logs_outage_and_recovery`（并发峰值=2 + 恢复日志） |
| R2-6 | 优化（结构） | `fleet.py` 1053 行（HEAD 已 844，超 800 软限） | 拆出 `fleet_config.py`（env 解析/启动注册，P2 设置双轨的落点）与 `fleet_tools.py`（LLM 工具面；`hw_fleet_status` 健康探测改并发）；`fleet.py` 剩 ~850 行并在模块 docstring 写明不再细拆的理由 | 既有 fleet/plugin/CLI 用例全绿 |
| R2-7 | major（安全，ROS2/HCP 复核时补出） | R2-3 的同类旁路：`HardwareContext.to_dict()` 还带 **`control_bindings`**（未解析的绑定原文：MCP url/token、ROS2 图名、总线地址），`list` 仍会把它送过线 | `_wire_context()` 同时剔除 `control_bindings`（主机只能经节点到达设备，不需要） | `test_list_exports_declarations_without_node_private_bindings`（扩展断言） |
| R2-8 | major（治理，HCP 复核时补出） | `_register_trust_configs()` 只在 runtime 初次 bind 时运行；fleet context 是稍后由 `start_fleet()` 才注入 registry，故远端 `TrustConfig` 未进入 `HardwareTrustGate`，静默回退全局默认 | fleet 注册完成后再次幂等注册当前 contexts；P2 重握手增删改时改为事务式同步 | `test_start_fleet_registers_env_nodes_and_skips_unreachable`（携带远端 CANDIDATE 配置并断言 gate 收到） |
| R2-9 | major（J-RV2 真机门发现） | 真正的 scoped reload 会创建**新插件实例**；旧实现先读取/发布 `fresh_plugin.tools`，再注入依赖。`HardwareContextPlugin.tools` 在 registry 注入前为空，故 reload 虽返回 active，却丢失全部 `hw_*`；新实例另建未启动 manager，旧心跳失去所有权 | scoped reload 改为 state adoption → 仅向 fresh plugin 预注入相关依赖 → 注册/发布工具；hardware plugin 转移 live registry/fleet/heartbeat/health 所有权并重建 handler；不再全局重绑所有插件 | `test_reload_prepares_dependency_generated_tools_before_publication` + `test_fresh_hardware_plugin_adopts_live_fleet_on_reload`；J-RV2 二次 reload 后工具/10 devices/心跳均保持 |

另：02 规格 §7 补充"`sequence` 为节点进程内计数，重启复位；陈旧判定以 `observed_at` 为准"。

**RV2 最终测试证据**：plugin reload + fabric + governance + transport contract + CLI + architecture/inert wiring
**401 passed, 2 skipped**；changed-file ruff 全部通过。早期宽口径与全量证据仍为 **829 passed, 2 skipped**、
全量 `-m "not e2e"` **5480 passed, 27 skipped, 1 xfailed**；其 14 个基线失败分类不变（见上文）。

**J-RV2 真机门：✅ 通过**。第一次真实 reload 暴露 R2-9；修复、重部署后第二次 reload 保留
3 个 fleet 工具与 10 个 registry devices，且未出现第二次 manager 初始化。停节点后 10.1 s degraded、
30.3 s offline；重启后出现 `back online`。节点最终已停止。完整证据与令牌轮换记录见 `08-j-rv2-results.md`。
