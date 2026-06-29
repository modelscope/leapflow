# LeapFlow

**从真实操作中学习与进化的桌面自动化框架**

LeapFlow 观察你与电脑的日常交互，自动将操作演示蒸馏为可复用的参数化技能。技能随每次执行持续演化——越用越聪明。

## 核心特性

- **零侵入录制** — 后台观察，不干扰正常操作
- **六层噪声过滤** — 从 50 步噪声录制中蒸馏出 4-5 步干净技能
- **渐进式信任** — 新技能需逐步确认，成熟技能自动执行（STEP → CONFIRM → AUTO）
- **Video-First 感知** — 连续视频录制 + VLM 多尺度分析，精准还原用户意图
- **Workflow Copilot** — 基于世界模型的下一步预测与主动建议

## 快速开始

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | 必需 |
| [uv](https://github.com/astral-sh/uv) | latest | 包管理器 |
| macOS | 14+ | 原生 Host（可选，`--mock-host` 可绕过） |

### 安装

```bash
git clone https://github.com/modelscope/leapflow.git
cd leapflow
make setup
```

`make setup` 自动完成：创建虚拟环境、安装依赖、生成 `.env` 配置文件。

### 配置

编辑 `.env`，设置 LLM API Key（唯一必填项）：

```bash
LEAPFLOW_LLM_API_KEY=sk-your-key-here
# 默认使用 DashScope (qwen3.7-plus)，支持任意 OpenAI 兼容接口
# LEAPFLOW_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LEAPFLOW_LLM_MODEL=qwen3.7-plus
```

### 启动

```bash
# Mock 模式（任意平台，无需 Swift Host）
uv run leap --mock-host

# 完整模式（macOS，需先启动 Host）
make host          # Terminal 1: 启动原生 Host
uv run leap        # Terminal 2: 进入交互式 REPL
```

## 使用方式

### 命令概览

```bash
leap                    # 交互式 REPL（默认）
leap "你的问题"          # 单轮对话
leap learn              # 录制操作演示
leap run "整理 PDF"     # 触发技能执行
leap skills list        # 查看已学技能
leap host start/stop    # 管理 OS Host 服务
```

### 交互模式

进入 REPL 后可使用以下命令：

```
learn start [目标]    — 开始录制
learn stop           — 停止录制并蒸馏
annotate <文本>      — 标注当前步骤
skip [n]             — 标记噪声步骤
run <触发词>          — 执行技能
skills list/show     — 管理技能
help                 — 查看所有命令
exit                 — 退出
```

### 典型流程：录制 → 蒸馏 → 执行

```bash
# 1. 录制演示
> learn start 整理下载文件夹里的 PDF

# [正常操作，LeapFlow 后台观察...]

# 2. 停止录制（自动蒸馏）
> learn stop
# → 新技能 "整理 PDF 文件" 已就绪 (confidence: 72%)

# 3. 下次直接触发
> run 整理我的 PDF
# → 执行 "整理 PDF 文件"，完成。
```

## OS Host 服务（macOS）

OS Host 提供原生系统感知能力（AXTree、屏幕录制、文件监控）：

```bash
leap host setup      # 构建 + 安装 + 注册开机自启
leap host start      # 启动
leap host stop       # 停止
leap host status     # 查看状态
```

首次运行需在 **系统设置 → 隐私与安全** 中授予 Accessibility 和 Screen Recording 权限。

## 项目结构

```
leapflow/
├── src/leapflow/           # Python Brain
│   ├── cli/                  # CLI 入口 (leap 命令)
│   ├── copilot/              # Workflow Copilot (下一步预测)
│   ├── domain/               # 领域类型定义
│   ├── engine/               # 会话编排 & ReAct 引擎
│   ├── recording/            # 实时录制
│   ├── perception/           # 视频感知 & VLM 分析
│   ├── analysis/             # 离线分析管线
│   ├── learning/             # 技能蒸馏 & 代码生成
│   ├── skills/               # 技能运行时 & 注册表
│   ├── platform/             # 平台适配层 (RPC Bridge)
│   ├── memory/               # 三级事件驱动记忆
│   ├── world_model/          # 世界模型 & 预测编码
│   ├── causal/               # 因果推理引擎
│   ├── signal_fusion/        # 多模态信号融合
│   └── llm/                  # LLM Provider 抽象
├── os_host/                # 原生 Host（跨平台）
│   ├── darwin/               # macOS 实现 (Swift)
│   ├── linux/                # Linux (planned)
│   └── windows/              # Windows (planned)
├── tests/                  # 测试套件
├── Makefile                # 构建快捷命令
└── pyproject.toml          # 项目配置
```

## 开发

```bash
make setup            # 初始化环境
make test             # 运行测试 (pytest)
make lint             # 代码检查 (ruff)
make host             # 构建并运行 Swift Host (debug)
make brain ARGS='--mock-host --prompt "hello"'  # 运行 Brain
```

### 常用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LEAPFLOW_LLM_API_KEY` | — | LLM API Key（必填） |
| `LEAPFLOW_LLM_BASE_URL` | DashScope | OpenAI 兼容端点 |
| `LEAPFLOW_LLM_MODEL` | `qwen3.7-plus` | 模型名 |
| `LEAPFLOW_MOCK_HOST` | `0` | `1` 启用 Mock 模式 |
| `LEAPFLOW_RECORDING_MODE` | `video` | 录制模式：video / default / vision_only |
| `LEAPFLOW_LOG_LEVEL` | `INFO` | 日志级别 |

完整配置参见 [`.env.example`](.env.example)。

## License

Apache 2.0 — see [LICENSE](LICENSE).
