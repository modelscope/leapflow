# R1 真机回归执行结果（Jetson Orin NX ↔ macOS 主机）

> 状态：**retained real-board evidence**；不是当前执行 runbook。
>
> 执行日：2026-10-12。结论：**R1.1–R1.10 全部通过**。端侧操作逐项经用户审批
> （A1 上传 / A3 装入板上 conda env `leap` / A4 前台启动 / A5 停-启-终停）。
> 原始证据：`../jetson/r1-evidence/`（9 个文件，见文末对照表）。

## 环境与工件（回归时点）

- 主机：macOS，leapd pid=46440（conda env `leap`），`LEAPFLOW_FLEET_NODES` 指向
  `tcp://192.168.55.1:8767`（token 见 `../jetson/.lhp-token`，不入档明文）。
- 边缘：`yahboom`（hostname），`~/miniconda3/envs/leap/bin/leap hw serve
  --node-id yahboom --host 0.0.0.0 --port 8767`（前台、SSH 会话 2 次拉起）。
- 节点声明 9 台设备：camera_0–5（RealSense depth）、host、microphone_0/1。
  **回归全程未读取任何相机/麦克风通道**（避免触碰 RealSense/8765 管线）。

## 逐步结果

| 步 | 结果 | 关键证据 |
|---|---|---|
| R1.1 | ✅ | leapd 日志 12:00:31：`FleetManager initialized with 1 configured node(s)` → `Fleet: registered node 'yahboom' (tcp://192.168.55.1:8767) with 9 device(s)`；节点侧 `lhp_node_ready` JSON（protocol=1, token_required=true） |
| R1.2 | ✅ | `leap hw list`：本地 `host` + `yahboom__host` + 6 相机 + 2 麦克风（名字空间化）；全部 `writable=0`，与节点声明一致 |
| R1.3 | ✅ | 板载参照：load 0.02 / nproc 8 / df 115G free 49%；5×+3× `cpu.load1` 读=0.009，sequence 17→19→23 单调；`disk.root.free_bytes`=122,432,323,584（≈114 GiB，与 df 相符）；时钟：`observed_at`=1791777706.5238264 ∈ [Mac 1791777705.170, 1791777706.530] → 端到端差 **< 1.4 s**（< 2 s 要求） |
| R1.4 | ✅ | `hw describe yahboom__host` 正常（4 通道：cpu.count / cpu.load1 / disk.root.free_bytes / disk.root.used_percent）；`hw status` → `connected=true`；写请求被 **`channel_not_writable`** 拒（主机层，先于审批——设计如此） |
| R1.5 | ✅（适配） | 本队无任何可写通道（Q4 范围）：写无法"走审批链"，但零效应被**双层证明**——主机层直接拒（零网络调用），节点层拒（见 R1.6）；见 `r1_write_guard.json` **6/6 checks passed** |
| R1.6 | ✅ | 绕过主机 gate、把 `LhpTransport` 直插节点写只读通道：`ok=false, side_effect_state=none, failure_code=not_writable`（节点 `_write_precheck` 纵深防御生效）。host 设备未声明 envelope/safety（只读），超限构造 N/A——`not_writable` 在其之前即拒绝 |
| R1.7 | ✅ | `leap hw estop yahboom__host` → `halt_supported=false, halted=false, detail="a host resource has no emergency stop"`（能力声明诚实） |
| R1.8 | ✅ | A5 停节点：`nc -vz → Connection refused`；远端读优雅失败（`node 'yahboom' not registered: … unreachable: [Errno 61]` + `unknown_device`）；**本地 `host.cpu.load1` 读不受影响**（value=2.962, ok=true）；leapd 12:08:02 `WARNING Fleet: node 'yahboom' offline (30.1s since last heartbeat)`（`node_timeout_s=30`） |
| R1.9 | ✅ | 重启后 `lhp_node_ready`（同 9 设备）；`nc 8767` 通；连续读 sequence **1→2→3**（停前基线 89，进程内计数复位=可检测断层）；50 s 观察窗无新 offline 告警（心跳恢复）；`hw status` → `connected=true` |
| R1.10 | ✅ | `pytest tests/test_hardware_*.py tests/test_lhp_fabric.py` → **706 passed, 2 skipped in 9.79 s** |
| 收尾 | ✅ | A5 终停（用户批准"结束后停止"）：`NODE_STOPPED`、8767 refused、SSH 会话退出；板上恢复测试前状态（仅保留 `~/leapflow-src` 与 conda env 内的 editable 安装，均为已批准工件） |

## 观察与说明（如实记录，非缺陷）

1. **sequence 是节点进程内计数**：重启后 89→1 回跳。设计上跨进程连续性不做持久化断言，
   判定"新鲜/陈旧"应以 `observed_at`（墙钟，随读数过线）为准；sequence 仅用于同进程内
   丢样检测（`RingBuffer.record`）。R1.9 的"无重复、可检测断层"即按此语义验收。
2. **12:04:10 出现第二次 `FleetManager initialized`**：紧随 `desktop_semantic plugin
   activated`，属 deferred 插件重组路径；其后心跳监督被证实仍活跃（12:08:02 offline 告警）。
   **RV2/J-RV2 已完整定位并修复**：同实例 rebind 的 identity 判定是第一层（R2-2）；真实 reload
   还会创建 fresh plugin，并曾在依赖注入前发布空工具目录、丢失 live manager 所有权（R2-9）。
   J-RV2 修复后二次 reload 保留 fleet 工具、10 个 devices 与原心跳，见 `08-j-rv2-results.md`。
3. **degraded 中间态无日志**（当时代码仅 online→offline 打 WARNING）；离线证据即
   30.1 s 告警本身。**RV2 已补**：degraded 与恢复（back online）转换均打 INFO（`05` R2-5）。
4. **Mac 侧 daemon 仍保留（已离线的）yahboom 注册**：`LEAPFLOW_FLEET_NODES` 未移除，
   避免 `daemon restart` 打断当时在连的 TUI 会话。清理步骤见 03 §5 回滚（移除 env →
   重启 daemon）；节点离线期间心跳每 10 s 探测一次失败端点，属预期监督行为。
5. `expect` 在沙箱内无法分配 PTY（`system has no more ptys`），端侧执行均在沙箱外完成。

## 复跑指引（最小命令集）

```bash
# 0) 前置：Mac 侧 env（token 见 temp/devices/jetson/.lhp-token）
export LEAPFLOW_FLEET_NODES='[{"node_id":"yahboom","endpoint":"tcp://192.168.55.1:8767","token":"<TOKEN>"}]'

# 1) 启动节点（在仓库根；沙箱外运行）
./temp/devices/jetson/jssh.exp 'LEAPFLOW_LHP_TOKEN="$(cat ~/.leapflow/.lhp-token)" ~/miniconda3/envs/leap/bin/leap hw serve --node-id yahboom --host 0.0.0.0 --port 8767'

# 2) 读平面验收
.venv/bin/leap hw list --json
.venv/bin/leap hw read yahboom__host cpu.load1 --json
.venv/bin/leap hw status yahboom__host --json

# 3) 写防护 + 端侧防线（6/6 期望）
.venv/bin/python temp/devices/jetson/r1_write_guard_check.py

# 4) 停/重启生命周期
./temp/devices/jetson/jssh.exp "pkill -f '[l]eap hw serve'"
#    （观察 leapd 日志 30 s 出现 offline WARNING；本地设备读不受影响）
#    重启：重复第 1 步
```

## 证据文件对照（`../jetson/r1-evidence/`）

| 文件 | 覆盖 |
|---|---|
| `r1_write_guard.json` | R1.4/R1.5/R1.6：6/6 checks（主机层拒绝 + 节点层拒绝 + 身份探测） |
| `r1_estop_remote.json` | R1.7：halt_supported=false |
| `r1_read_prestop.json` | R1.9 基线：停前 sequence=89 |
| `r1_offline_remote.txt` | R1.8：nc refused + 远端读优雅失败 |
| `r1_offline_local.txt` | R1.8：本地读不受影响（ok=true, value=2.962） |
| `r1_offline_daemon_log.txt` | R1.8：12:08:02 offline WARNING（30.1 s） |
| `r1_restart_reads.txt` | R1.9：nc 通 + sequence 1→2→3 |
| `r1_restart_daemon_log.txt` | R1.9：观察窗内 offline 告警计数保持 1（心跳恢复） |
| `r1_restart_status.json` | R1.9：device recovered, connected=true |
