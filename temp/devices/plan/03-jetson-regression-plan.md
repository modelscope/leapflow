# 03 — Jetson 真机回归计划（R1）

> 状态：**已完成的历史 runbook**。结果以 `06-r1-results.md` 为准；未来 gate 使用
> `01-execution-plan.md` 与 `10-human-approval-deployment.md`，不要原样复制旧命令。
>
> 目标：验证 **Mac（operator, leapd）↔ Jetson（edge node, `leap hw serve`）** 的
> LHP fabric 端到端：注册 → 设备出现（`/board`）→ 遥测读取 → 写入（审批 + 安全
> 策略）→ estop → 注销。远程设备使用节点自身的 `host` provider（Jetson 的
> CPU/温度/磁盘），不依赖 RealSense。

## 0. 已确认事实（来源：`temp/devices/jetson/README.md` + `REALSENSE.md`）

| 项 | 值 |
|---|---|
| 板卡 | Jetson Orin NX（Yahboom 套件；型号 "Engineering Reference Developer Kit Super"；JetPack 6.2 / L4T R36.4.3 / Ubuntu 22.04） |
| 接入 | USB gadget 网络：`ssh jetson@192.168.55.1`；Mac `en7: 192.168.55.100` |
| 链路 | 板↔Mac 实测 42.4 MB/s；板外网经 Mac NAT（`com.local.jetson-nat` 持久化） |
| 已有外设 | RealSense D435IF（ROS2 Humble + foxglove_bridge :8765，USB3 已通） |
| SSH 助手 | `jssh.exp`（含密码，目录被 git 忽略）；`rs.sh`（相机控制） |
| 磁盘 | NVMe 已扩至 ~233 GB（余 115 GB，2026-10-10 复核） |

## 1. 只读清点（2026-10-10 已完成；无需审批）

| 检查 | 实测结果 | 对方案的影响 |
|---|---|---|
| `python3 --version` | 3.10.12（JetPack 自带系统解释器） | 不能用系统 Python 装 leapflow（要求 ≥3.11） |
| `uv` | **已存在**：`~/.local/bin/uv` 0.8.11；非交互 shell 的 PATH 不含它（`which uv` 会误报缺失） | **A2 取消** |
| uv 管 Python 3.11 | **已安装** 3.11.6（`~/.local/share/uv/python/…`，2025-08-15，早于本轮） | 备用资产；A3 已改用 conda 方案（见下行） |
| conda env `leap` | **已存在**：`~/miniconda3/envs/leap`（Python 3.13.16 / pip 26.2.1；2026-10-12 复核） | **R1 安装目标**（用户指定）；无需新建 venv/下载解释器 |
| `timedatectl` | NTP synchronized=yes（Asia/Shanghai） | 时基可比，R1.3 时钟差验收可行 |
| `hostname` | `yahboom`（满足 `[a-z0-9_]+`） | node_id 取 `yahboom`；设备命名空间 `yahboom__host` |
| `df -h /` | 233 GB 总 / 余 115 GB | 部署空间充裕 |
| `nproc` / 内存 | 8 核 / RAM 15.6 GB（清点时 ~9.7 GB free） | 节点运行时资源画像正常 |
| 端口 `8765` / `8767` | 清点时均空闲 | 8767 可用；Foxglove 8765 无冲突 |
| `systemctl --user` | 可用 | 常驻形态（P2）可走 systemd user；R1 仍手工前台 |

> 与旧记录的差异（uv 已存在、Python 3.11 就绪）已同步回填
> `temp/devices/jetson/README.md`，并在 `04-open-questions.md` 关闭 G1/G2/Q1、取消 A2。

## 2. 部署形态（R1 最小闭环）

```text
Jetson:  ~/miniconda3/envs/leap/bin/pip install -e ~/leapflow-src   # 板上既有 conda env leap（Python 3.13.16）
         └─ LEAPFLOW_LHP_TOKEN="$(cat ~/.leapflow/.lhp-token)" \
              ~/miniconda3/envs/leap/bin/leap hw serve \
              --node-id yahboom --host 0.0.0.0 --port 8767   # 前台运行；token 不进 argv/output
Mac:     LEAPFLOW_FLEET_NODES='[{"node_id":"yahboom",
             "display_name":"Yahboom Jetson Orin NX",
             "endpoint":"tcp://192.168.55.1:8767","token":"<T>"}]'
         leap daemon restart → /board rescan
```

- 启动时注册（P1.12）：leapd 启动即自动注册 `LEAPFLOW_FLEET_NODES` 中的节点（无需手动 RPC）；
  `leap hw list/read` 的进程内读面亦会自行注册（心跳仅 daemon 持有）；
- 节点 runtime 启动成功后 stdout 打印一行 `lhp_node_ready` JSON（endpoint / 设备数），
  作为 R1.1 的节点侧证据留存；
- 不做 systemd/开机自启（R1 手工启动；常驻形态 P2 再定）；
- 不改板卡网络/内核/驱动；不触碰 RealSense 进程（8765 不受影响）；
- token 用一次性随机串，存于 mode `0600` 文件并通过 `LEAPFLOW_LHP_TOKEN` 加载；
  不进入 argv、终端输出、日志或版本库。此处按 J-RV2 令牌事件后的最终规范回填。

## 3. 端侧操作审批清单（**逐项单独请求，得到批准前不执行**）

| # | 操作 | 命令（预览） | 影响 | 风险级 |
|---|---|---|---|---|
| A1 | 上传 leapflow 源码到板卡 | ✅ **已批准并执行**（2026-10-12；验证 24 MB 落盘）：`temp/devices/jetson/jrsync.exp '~/leapflow-src/' src pyproject.toml README.md`（仓库根执行；expect 包 rsync 应答密码；不含 .git——setuptools attr 版本无需） | — | 低（仅写入用户目录） |
| A3 | 安装 leapflow（**用户指示改用板上既有 conda env `leap`**；uv venv 方案取消） | ✅ **已批准**：`~/miniconda3/envs/leap/bin/pip install --progress-bar off -e ~/leapflow-src` | 依赖装入既有 conda env（经 NAT 取 PyPI） | 中（包安装） |
| A4 | 启动节点服务（**已批准：前台运行**） | `LEAPFLOW_LHP_TOKEN="$(cat ~/.leapflow/.lhp-token)" ~/miniconda3/envs/leap/bin/leap hw serve --node-id yahboom --host 0.0.0.0 --port 8767` | **开放 TCP 8767**（局域网 192.168.55.0/24）；token 不进 argv/output | 中（网络服务） |
| A5 | 停止节点服务（A4 配对回滚，**已批准：停→启→终停**） | `pkill -f '[l]eap hw serve'` | ✅ 已执行：停止后 30.1s 主机标记 offline；端口即释放；R1 结束后终停 | 低 |

> 审批与执行记录（2026-10-12 用户确认）：A1 ✅ 已执行；A3 ✅ 改 conda 方式；A4 ✅ 前台；
> Q2 关闭（端口 8767）；Q3 关闭（前台优先）；Q4 关闭（写测试仅只读 + deny_write 验证）。
> A5 ✅（同轮追加确认）：批准「停→启」用于 R1.8/R1.9；最终状态按用户选择「结束后停止」
> 完成终停（8767 已释放）。
> A2（安装 uv + Python 3.11）已因 §1 清点结果取消——板上原已存在且非本轮安装。

Mac 侧（无需端侧审批）：环境变量 + `leap daemon restart` + `/board rescan` +
`leap hw list/read/estop`。

## 4. 回归步骤与验收（R1 验收线）

| 步 | 操作 | 验收标准 | 证据 |
|---|---|---|---|
| R1.1 | A1–A4 完成；Mac 配置节点并重启 leapd | 日志出现 `Fleet: registered node 'yahboom' (…) with N device(s)` | daemon 日志 + 节点 `lhp_node_ready` 行 |
| R1.2 | `/board rescan` / `leap hw list` | 出现 `yahboom__host`（或节点导出的其它设备），Read/Write 能力与节点声明一致 | `leap hw list --json` |
| R1.3 | `leap hw read yahboom__host cpu.<…>` ×5 | 值与 Jetson 本机 `top/uptime` 量级一致；sequence 递增；`observed_at` 与两端时钟差 < 2s | 两端命令输出 |
| R1.4 | `leap hw describe yahboom__host` 后 `leap hw status` | describe 流程正常；未 verify 的写被 deny_write（预期行为，如设计） | CLI 输出 |
| R1.5 | 写测试（若节点设备存在可写通道）：`hw_write` 走主机审批链 | 审批提示出现；批准后节点侧生效（读回验证）；未批准时零效应 | 审批记录 + 读回 |
| R1.6 | 节点侧安全策略验证：构造超限值写入（若声明 envelope/safety） | 主机收到 `ok=false, side_effect_state=none`，证据可查（纵深防御生效） | `leap hw` JSON |
| R1.7 | `hw_estop`（对 `yahboom__host`） | 返回 `halt_supported=False`（host 设备无急停——能力声明诚实） | CLI 输出 |
| R1.8 | A5 停节点 → 等 2×心跳 | 拓扑显示 node offline/degraded；`yahboom__host` 读失败但**主机本身设备不受影响**（隔离验收） | topology JSON |
| R1.9 | 重启节点 → 重新注册 | 设备恢复在线；sequence 从节点侧继续（无重复、可检测断层） | 两次读输出 |
| R1.10 | 回归既有套件（Mac 侧） | `pytest tests/test_hardware_*.py tests/test_lhp_fabric.py` 全绿 | pytest 摘要 |

### 4.1 执行结果（2026-10-12）：R1.1–R1.10 全部通过 ✅

| 步 | 结果摘要 |
|---|---|
| R1.1 ✅ | daemon 日志 `Fleet: registered node 'yahboom' (…) with 9 device(s)`；节点 `lhp_node_ready` |
| R1.2 ✅ | `hw list`：`yahboom__host` + 6 相机 + 2 麦克风，全部 writable=0 |
| R1.3 ✅ | load 0.009（板载参照 0.02 量级）；seq 17→19→23 单调；时钟差 <1.4s（<2s 要求） |
| R1.4 ✅ | describe 4 通道正常；写被 `channel_not_writable` 拒（主机层，先于审批） |
| R1.5 ✅* | 全队无可写通道（Q4）：零效应双层证明（主机层直拒 + 节点层直拒） |
| R1.6 ✅ | 绕 host gate 直写节点 → `ok=false, side_effect_state=none, failure_code=not_writable` |
| R1.7 ✅ | estop → `halt_supported=false`（"a host resource has no emergency stop"） |
| R1.8 ✅ | 停：nc refused + 远端读优雅失败 + 本地读 ok（2.962）+ 12:08:02 offline WARNING（30.1s） |
| R1.9 ✅ | 重启：seq 1→2→3（停前 89，断层可检测）；观察窗无新 offline 告警；connected=true |
| R1.10 ✅ | `pytest tests/test_hardware_*.py tests/test_lhp_fabric.py` → 706 passed, 2 skipped（9.79s） |

*R1.5 按 Q4 范围适配（全队只读，无可写通道可走审批链）。完整证据、观察项与复跑
指引见 `06-r1-results.md`；原始 JSON/日志在 `../jetson/r1-evidence/`（9 文件）。

## 5. 回滚（每步可逆）

- 节点进程：`pkill -f '[l]eap hw serve'`（A5），端口即释放；
- Mac 侧配置：移除 `LEAPFLOW_FLEET_NODES` 后 `leap daemon restart`，registry 不再含外部源；
- 板上文件：本轮新增 `~/leapflow-src`，可整体删除（**删除操作同样需要审批**）；
  另以 editable 方式将 leapflow 装入既有 conda env `leap`（还原：`~/miniconda3/envs/leap/bin/pip uninstall leapflow`；
  env 本身为板上既有资产，不删除）；`~/.local/bin/uv` 与 uv 的 Python 3.11 为板上原有资产，
  不删除、不改动；不做任何系统级改动，无需回滚系统；
- 全程不触碰 RealSense/ROS/Foxglove（8765）与 NAT 配置。

## 6. 回归期间禁止事项

- 不执行 reboot/shutdown/内核/驱动/apt 升级类命令；
- 不占用/重启 8765（Foxglove）与 RealSense 节点；
- 不修改 `/etc` 下任何文件；不安装 systemd 服务；
- 一切端侧操作以本文件第 3 节清单为准，逐项审批。
