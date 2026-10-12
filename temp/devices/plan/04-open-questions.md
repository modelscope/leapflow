# 04 — 已关闭问题与决策记录

> 状态：**closed evidence**。当前无待用户拍板项；新问题进入对应 active plan，
> 本文件只保留 R1 决策因果链。
>
> 原规则：影响方案的差异先记录在此；能在本仓/真机只读核实的，自行核实并更新；
> 无法自行决定的（需要拍板）在对话中向用户单独提出。

## 已关闭的用户拍板项（R1 历史前提）

| # | 问题 | 选项 | 倾向 |
|---|---|---|---|
| Q1 | ~~安装 uv + Python 3.11~~ **已关闭**：只读清点确认两者均已存在（2025-08-15 安装，非本轮），A2 取消 | — | — |
| Q2 | ~~节点服务端口用 8767~~ **已关闭**：随 A4 批准确认 8767（避开 Foxglove 8765） | — | — |
| Q3 | ~~前台运行 vs `setsid` 后台~~ **已关闭**：用户选择**前台**（退出 SSH 即停，可控优先） | — | — |
| Q4 | ~~远端写测试范围~~ **已关闭**：用户选择**仅只读 + deny_write 验证**（不为 host provider 配置可写通道） | — | — |

## 自行核实清单（只读；2026-10-10 已完成，结果回填 `03` §1 与 `temp/devices/jetson/README.md`）

- 结论：系统 Python 3.10.12；**uv 0.8.11 与 Python 3.11.6 已存在**（A2 取消）；
  NTP 已同步；hostname `yahboom`；磁盘余 115 GB；8765/8767 空闲；`systemctl --user` 可用。

## 已知信息缺口与处置

| # | 缺口 | 处置 |
|---|---|---|
| G1 | ~~板上 Python 版本未记录~~ | **已解决**：系统 3.10.12；uv 已备 3.11.6 → A2 取消；README 已回填 |
| G2 | ~~板卡 hostname 未记录~~ | **已解决**：`yahboom`（node_id 直接取用）；README 已回填 |
| G3 | `fleet.py` 旧的 MCP 表述与现方案不一致 | 已随 P1 代码修订 docstring；README 中相关描述 P2 统一 |
| G4 | `test_dashboard_view.py` 等是否消费 `FleetTopology.to_dict` 的 `mcp_endpoint` 字段 | 已核对：保留 `endpoint` 并同步输出 `mcp_endpoint` 兼容键，测试保持绿色为准 |
| G5 | 板上是否已有可写通道设备（host provider 只读） | 预期无；R1.5 以 deny_write/审批路径验证为主 |

## 决策记录（随进展追加）

- D-2026-10-10-1：LHP 取代 MCP 成为 fabric 数据面；MCP 降级为互操作适配器（见 `00` D1）。
- D-2026-10-10-2：节点设备进入主机 registry 走"外部源注册"而非 provider 扫描——发现契约保持被动本地（见 `02` §8 与 `registry.register_external_contexts` 文档）。
- D-2026-10-10-3：P1 不做订阅/帧/连接池（防蔓延）；每 transport 独立连接，P3 随订阅一并引入多路复用。
- D-2026-10-10-4：启动时注册前移为 P1.12——R1.1/R1.2/R1.8 验收要求进程真实消费 `LEAPFLOW_FLEET_NODES`，而原 P1 中 `_pending_fleet_nodes` 无消费方、`register_node`/`start_heartbeat` 无生产调用方；最小闭环（env 解析 + 安全注册 + 心跳；注册点 = daemon ctx 与 CLI 读面）先行，settings 双层 / 失败退避 / 重注册生命周期仍归 Phase 2。
- D-2026-10-10-5：板上上传集仅 `src/` + `pyproject.toml` + `README.md`——版本走 setuptools `attr = "leapflow.version.__version__"`，不依赖 `.git`；rsync 经 `jrsync.exp`（本机无 sshpass）。
- D-2026-10-12-1：端侧安装方式修正——用户指示使用板上**既有 conda env `leap`**（Python 3.13.16 / pip 26.2.1）执行 `pip install -e ~/leapflow-src`，**取消 uv venv 方案**（`~/leapflow-venv` 不再创建）；uv 0.8.11 与 Python 3.11.6 保持为板上原有资产不动；A4 服务命令相应改为 `~/miniconda3/envs/leap/bin/leap hw serve …`（A1 已于 2026-10-12 执行，24 MB 落盘验证通过）。
