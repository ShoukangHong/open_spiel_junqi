# OpenSpiel 开发会话纪要

> 日期：2025-05-20 ~ 2026-06-07

## 1. 总体目标与当前目标

### 总体目标
基于 OpenSpiel 框架，使用 C++ 构建高性能四国军棋（Junqi / Luzhanqi）游戏，并用深度强化学习（AlphaZero 架构）训练 AI。
总体阶段：
1. 使用Othello跑通整个流程
2. 搭建linux下的训练环境，支持云端训练。
3. 使用中国象棋验证训练对于类似游戏的有效性。
4. 创建游戏环境，需要抽象封装，支持简易版军棋游戏（如二国军棋、更小的棋盘、更少棋子等），四国军棋，明棋、暗棋的训练。支持gui交互
5. 训练明棋ai：
    - 使用简易版军棋游戏验证训练有效性，探索最优方法（预算500）
    - 使用正式版四国军棋进行训练。（预算2500）
6. 训练暗棋ai：
    - 使用简易版军棋游戏验证训练有效性，探索最优方法（预算2000）
    - 使用正式版四国军棋进行训练。（预算5000）
7. 视训练情况，考虑推广，获取投资。加强训练。出售产品等。

### 当前进展
1. 已完成
2. 已完成，othello云端训练可达到专业级别水平，训练吞吐量、超参数等已有比较充分的优化。
3. 正准备开始，需要注意中国象棋有与军棋类似的一些特点，要在中国象棋的训练中解决这些问题：
 - 对局长度很长，game有重复局面和棋的限制。
 - 容易出现消极比赛导致和棋的局面，模型停止进步。
 - action space复杂，不像围棋和othello一样action很有限。
 - 棋子类型多样，通道数多。

目前首先需要：
1. 优化代码架构，支持尽可能复用othello中已有的训练流程。
2. 实现训练的运行，模型能够学会下棋。
3. 解决上述中国象棋的特点导致的问题。
4. 寻找最优超参数，为军棋训练做准备。

### 军棋与 Othello 的关键差异
| 属性 | Othello | 军棋 |
|------|---------|------|
| 信息 | 完全信息 | **不完全信息**（暗棋） |
| 随机性 | 确定性 | 可能有布阵随机性 |
| 需要的 MCTS | 普通 MCTS | **IS-MCTS**（Information Set MCTS） |
| 相关算法 | AlphaZero | AlphaZero + CFR / Deep CFR |

---

## 2. Windows 环境安装 OpenSpiel（源码编译）

### 前置条件
- **Visual Studio 2022 Community**（免费），安装时勾选 "Desktop development with C++"
- **CMake** >= 3.17
- **Git**
- **Python** 3.11+（64-bit）

### 步骤

#### 2.1 克隆仓库与依赖
```powershell
cd C:\Users\shouk\Github
git clone https://github.com/deepmind/open_spiel.git open_spiel_junqi
cd open_spiel_junqi

# 用 SSH 克隆依赖（国内网络 HTTPS 慢）
git clone --single-branch --depth 1 git@github.com:pybind/pybind11.git pybind11
git clone --single-branch --depth 1 git@github.com:pybind/pybind11_json.git open_spiel\pybind11_json
git clone --single-branch --depth 1 git@github.com:abseil/abseil-cpp.git open_spiel\abseil-cpp
git clone --single-branch --depth 1 git@github.com:nlohmann/json.git open_spiel\json
git clone git@github.com:pybind/pybind11_abseil.git open_spiel\pybind11_abseil
git clone -b develop --single-branch --depth 1 git@github.com:jblespiau/dds.git open_spiel\games\bridge\double_dummy_solver
```

#### 2.2 编译 C++ 项目（含 Python 绑定）
```powershell
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -DOPEN_SPIEL_BUILD_WITH_PYTHON=ON ../open_spiel
cmake --build . --config Release -j
cd ..
Copy-Item -Path build\python\Release\pyspiel.pyd -Destination pyspiel.pyd -Force
```

#### 2.3 创建 venv 并安装
```powershell
cd C:\Users\shouk\Github\open_spiel_junqi
python -m venv venv
.\venv\Scripts\activate

# 国内镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装（改动 C++ 后 rebuild 即可生效）
pip install -e .
```

#### 2.4 验证
```powershell
python -c "import pyspiel; print(pyspiel.load_game('tic_tac_toe'))"
python -c "import pyspiel; print(pyspiel.load_game('othello'))"
```

### 遇到的坑
1. **MinGW GCC 6.3.0 不支持 C++20** → 换用 Visual Studio 2022
2. **`git submodule update` 无反应** → 该项目不使用 submodule，依赖需手动 git clone
3. **Python 架构不匹配警告** → 不影响，确认安装 64-bit Python
4. **`spawn.py` 第 28 行 `fork` 在 Windows 上报错** → 修改 `except RuntimeError` 为 `except (RuntimeError, ValueError)`
5. **orbax-checkpoint 安装失败（路径过长）** → 启用 Windows 长路径支持（注册表 `LongPathsEnabled=1`）
6. **`pyspiel` 不是 pip 包** → 通过 `pip install -e .` 从本地 C++ 编译产物安装

---

## 3. AlphaZero 训练方法

### 3.1 安装 JAX 生态
```powershell
pip install jax jaxlib flax optax chex orbax-checkpoint
```

### 3.2 AlphaZero 文件结构
```
open_spiel/python/algorithms/alpha_zero/
├── alpha_zero.py      # 主训练循环（多进程：actors/learner/evaluators）
├── model_nnx.py       # Flax NNX 神经网络（MLP/Conv2D/ResNet）
├── model_linen.py     # Flax Linen 备选后端
├── evaluator.py       # 神经网络 → MCTS Evaluator 接口
├── replay_buffer.py   # JAX replay buffer
└── utils.py           # 数据结构（TrainInput, Losses, API selector）

open_spiel/python/examples/
└── alpha_zero.py      # 示例入口，所有默认参数在这里
```

### 3.3 核心参数说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `--game` | 游戏名 | `othello` |
| `--nn_model` | 网络类型 | `mlp` / `conv2d` / `resnet` |
| `--nn_width` | 网络宽度（通道数/隐藏单元） | 32-256 |
| `--nn_depth` | 网络深度（残差块/层数） | 2-20 |
| `--max_simulations` | MCTS 每步模拟次数 | 32-800 |
| `--actors` | 并行自博弈进程数 | 2-16（建议 ≤ CPU 核心的 75%） |
| `--evaluators` | 评估进程数 | 1-2 |
| `--replay_buffer_size` | buffer 容量（state 数） | 2048-500000 |
| `--replay_buffer_reuse` | 每个 state 被学习次数 | 2-4 |
| `--train_batch_size` | 每次梯度更新的 batch | 128-256 |
| `--learning_rate` | 学习率 | 0.001 |
| `--weight_decay` | L2 正则化 | 1e-4 |
| `--max_steps` | 总学习轮数 | 50-500 |
| `--checkpoint_freq` | 每 N 步存一次 checkpoint | 25-50 |
| `--temperature` | MCTS 动作选择温度 | 1（探索） |
| `--temperature_drop` | 第 N 步后贪婪选择 | 2-10 |
| `--uct_c` | UCT 探索常数 | 1.41 |
| `--policy_epsilon` | Dirichlet 噪声权重 | 0.25 |
| `--policy_alpha` | Dirichlet 浓度参数 | 1.0 |
| `--path` | checkpoint 保存目录 | 建议指定固定路径 |
| `--nn_api` | Flax 后端 | `nnx`（新版） |

### 3.4 训练命令示例

**快速验证（~5 分钟）**：
```powershell
python open_spiel/python/examples/alpha_zero.py --game=othello --nn_model=mlp --nn_width=32 --nn_depth=2 --max_simulations=8 --actors=2 --evaluators=0 --replay_buffer_size=512 --quiet=False
```

**CPU 4 小时训练**：
```powershell
python open_spiel/python/examples/alpha_zero.py \
  --game=othello --nn_model=resnet --nn_width=32 --nn_depth=5 \
  --max_simulations=128 --actors=6 --evaluators=1 \
  --replay_buffer_size=16384 --replay_buffer_reuse=2 \
  --temperature=2 --temperature_drop=10 \
  --max_steps=100 --checkpoint_freq=25 \
  --path=C:\Users\shouk\othello_train \
  --nn_api=nnx
```

### 3.5 训练循环流程

```
actors (N 进程)                    learner (主进程)
─────────────                      ────────────────
持续自博弈                          每轮：
  MCTS 搜索(每步)                    1. collect_trajectories()
  → 产生 (state, policy, value)        从 actors 的 queue 收集 states
  → 推入 replay buffer                 凑够 learn_rate 个新 state 后返回
                                     2. learn()
                                       从 buffer 采样 batch 训练
                                       遍历 buffer 一轮
                                     3. 广播新权重给 actors → 继续
```

**关键：buffer 是 FIFO ring buffer，新数据覆盖旧数据。learner 永远等 actors 凑够数据才开始训练。**

### 3.6 如何解读训练指标

| 指标 | 健康信号 | 警告信号 |
|------|---------|---------|
| **value loss** | 持续下降 → 逼近 0.1 以下 | 横盘不动 → MCTS 太弱或模型太浅 |
| **policy loss** | 缓慢下降（下限约 1.5） | 剧烈波动或发散 |
| **L2 loss** | 稳定 | 暴涨（过拟合） |
| **states/s/actor** | 稳定 | 剧烈下降（进程抢占 CPU） |

---

## 4. 重要经验与陷阱

### 4.1 Othello 的 ObservationTensor 是相对视角！
`state.observation_tensor()` 在 Othello 中返回**当前玩家视角**（channel 1=我的棋，channel 2=对方的棋）。`apply_action` 后 current_player 翻转，observation 的 channel 含义随之反转。

**需要绝对视角时**，使用 `state.observation_tensor(0)`（固定 player 0 视角）。

这与 TicTacToe 不同——TicTacToe 的 ObservationTensor 忽略 player 参数，始终返回绝对棋盘。

### 4.2 CPU 训练的时间估算
- 你的 i7-12800HX（16 核 24 线程）上，resnet(width=32, depth=6) + max_sim=64 + actors=6 ≈ **1.8 min/step**
- 宽度翻倍 → NN 推理约 4 倍慢
- MCTS 模拟翻倍 → 每步约 2-3 倍慢
- 预估要诚实，注意 CPU 上 NN 推理是瓶颈

### 4.4 当前训练结果分析
- 训练的模型实力很弱（不会占角），原因：
- cpu训练效率太低，一晚上只训练了1000盘。

### 4.5 后续上 Linux GPU 的注意事项
- 同一份代码不需改动，JAX 自动检测 CUDA
- GPU 上可以大幅提高 max_simulations和nn_width
- actors 可以更多（NN 推理走 GPU，actors 主要等 I/O）

### 4.6 Windows 多进程
- Windows 只支持 `spawn`（不支持 `fork`）
- `spawn` 启动略慢但完全可用，多核调度正常
- 需注意 `spawn.py` 的 bug 修复

### 4.7 PyTorch vs JAX 决策
- **当前选择：JAX**。OpenSpiel 的 AlphaZero 实现是 JAX/Flax 专属
- 移植到 PyTorch 约需 1500-2000 行代码，不值得
- JAX 在 Windows CPU 上可用，Linux GPU 上直接起飞
- 如果后续做军棋需要 PyTorch 算法，现有 `python/pytorch/` 下有 DQN、PPO、Deep CFR 等实现

- 5/22更新：由于对高性能训练的需要，打算自建batch mcts和使用pytorch batch eval的高性能的训练框架。

---

## 5. 相关文件索引

| 文件 | 用途 |
|------|------|
| `open_spiel/games/othello/othello.{h,cc}` | Othello C++ 实现 |
| `open_spiel/python/examples/alpha_zero.py` | AlphaZero 训练入口 |
| `open_spiel/python/algorithms/alpha_zero/alpha_zero.py` | AlphaZero 训练核心 |
| `open_spiel/python/algorithms/alpha_zero/model_nnx.py` | Flax NNX 神经网络 |
| `open_spiel/python/algorithms/alpha_zero/evaluator.py` | NN → MCTS Evaluator |
| `open_spiel/python/algorithms/mcts.py` | MCTS 实现（框架无关） |
| `open_spiel/python/utils/spawn.py` | 多进程封装（已修 Windows bug） |
| `open_spiel/python/pytorch/` | PyTorch 算法集 |
| `docs/windows.md` | Windows 安装文档 |
| `docs/developer_guide.md` | 添加新游戏的开发指南 |
| `game_ui/othello_ui.py` | Othello 人机对战 UI（pygame） |
| `junqi_dev/session_notes.md` | 本纪要 |
