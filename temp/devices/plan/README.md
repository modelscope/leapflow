# LeapFlow 硬件 Fabric（节点织体）重构计划

> 状态目录：本文档集是长线重构方案的唯一执行清单。所有阶段任务、验收标准、
> 真机回归步骤与端侧操作审批门均以此为准。

## 文档索引

| 文件 | 内容 |
|---|---|
| `README.md` | 本文件：索引 + 执行状态看板 |
| `00-architecture.md` | 目标架构总览、核心决策（D1–D7）、信号分类规范表、反目标 |
| `01-execution-plan.md` | Completed baseline + Stage A–G and cross-cutting Stage I execution order, gates and acceptance |
| `02-lhp-wire-protocol-v1.md` | LHP 线协议 v1 规格（消息族 / 错误码 / 时基 / 安全 / 版本规则） |
| `03-jetson-regression-plan.md` | 真机回归方案：前置清点、runbook、端侧操作审批清单、回滚 |
| `04-open-questions.md` | 已关闭的 R1 问题与决策因果记录（历史证据） |
| `05-review-notes.md` | Deep review 记录：发现/修复/回归测试/测试证据/基线问题 |
| `06-r1-results.md` | R1 真机回归结果：逐步证据、观察项、复跑指引、证据文件对照 |
| `07-ros2-hcp-integration.md` | ROS2/HCP 覆盖审计、Jetson Python 边界、sidecar 架构、实施包与 LLM 配置复用 |
| `08-j-rv2-results.md` | RV2 真机门：真实 reload 缺陷、修复、工具/设备保持、失联恢复与令牌轮换 |
| `09-plan-audit-2026-10-12.md` | 全量计划/代码审计：完成度、缺口、过期内容处置与当前测试证据 |
| `10-human-approval-deployment.md` | 端侧单独部署与 host+edge 双端部署的人类审批架构、实施与回归门 |

Active execution authority: `README`, `00`, `01`, `02`, `07`, `09`, and `10`.
Completed/closed evidence retained for causality: `03`, `04`, `05`, `06`, and `08`.

## 执行状态看板（随进度更新）

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | Contract freeze (ADR, signal taxonomy and LHP specification) | ✅ Complete |
| P1 | LHP v1 + HCP adapter + registry/fleet wiring + edge runtime | ✅ Complete; R1 passed |
| RV1 | First deep review | ✅ Complete; 6 findings fixed |
| RV2 | Post-R1 review and optimization | ✅ Complete; 6 major + 3 optimizations fixed |
| A / J-RV2 | Deploy reviewed delta; hot-reload and outage/recovery board gate | ✅ Passed; see `08-j-rv2-results.md` |
| B | HCP v1 specification + direct ROS2 hardening/actions/batch/tests | ▶️ Approved; next implementation stage |
| C | Fabric lifecycle/settings/reconciliation/trust/degradation | ▶️ Approved; may run after/alongside B |
| D | Jetson ROS2 Python-3.10 sidecar + Python-3.13 backend | ⏳ Approved; depends on B+C |
| E | Typed telemetry subscription + Foxglove preview federation | ⏳ Approved; depends on D |
| F | Identity, credentials, audit and environment evolution | ⏳ Approved; follows E |
| G | Robot/peripheral convergence and safety completion | ⏳ Approved; follows F |
| I | Human-interaction broker, edge approval channels and signed remote execution permits | ▶️ Selected cross-cutting target; implementation pending; see `10` |
| R1 | Mac ↔ Jetson real-board regression R1.1–R1.10 | ✅ Complete; edge node stopped |

## 本轮（P1）执行范围 —— 已完成 / 待完成

- [x] `hardware/lhp/`：`protocol.py` + `server.py` + `client.py` + `__main__.py`
- [x] `hardware/transports/lhp.py` + 工厂表注册
- [x] `HardwareRegistry` 外部源注册 API（节点设备进同一注册表）
- [x] `FleetManager`：LHP 探测/发现 + 命名空间化注册/注销
- [x] `leap hw serve`（端侧节点运行时）+ `python -m leapflow.hardware.lhp`
- [x] 测试：conformance case（含 Direct 绑定全编解码路径）+ TCP/Unix 集成测试 + fleet 端到端
- [x] 本地全量相关测试通过（202 + 902 + 5477 passed；基线例外见 `05`）
- [x] Deep review records (`05-review-notes.md`: RV1 + RV2; all findings fixed with regression tests)
- [x] ROS2/HCP coverage audit and restructured remaining plan (`07`, Stage A–G)
- [x] User confirmation of the remaining plan and Stage A approvals
- [x] J-RV2 real-board gate (including R2-9 found/fixed/reverified; `08-j-rv2-results.md`)
- [x] Full plan/current-code audit and document classification (`09-plan-audit-2026-10-12.md`; focused suites: 307 passed)
- [x] Edge-only and dual-deployment approval architecture target selected (`10-human-approval-deployment.md`; implementation Stage I remains pending)
- [x] 端侧部署审批请求（R1 前置）—— **全部获批**（A1 ✅ 已执行；A3 ✅ 改板上 conda env `leap`；A4 ✅ 前台；A5 ✅ 停-启-终停均已批准执行；Q2/Q3/Q4 均已关闭）
- [x] R1 真机回归：R1.1–R1.10 全部通过（证据 `06-r1-results.md` + `../jetson/r1-evidence/` 9 个文件）
- [x] 端侧收尾：A5 终停（用户批准「结束后停止」），8767 已释放，板上恢复测试前状态
