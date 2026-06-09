  Chinese Chess 接入计划

  核心分析

  Othello 与 Xiangqi 对比

  - **Obs shape**  
    - Othello: [4, 8, 8]  
    - Xiangqi: [15, 10, 9]

  - **Actions**  
    - Othello: 65 (flat)  
    - Xiangqi: 8100 (90×90 from-to)

  - **Max length**  
    - Othello: ~60  
    - Xiangqi: 500

  - **Draw**  
    - Othello: 罕见  
    - Xiangqi: 常见（500步上限）

  - **重复规则**  
    - Othello: 无  
    - Xiangqi: OpenSpiel 未实现

  - **Chance nodes**  
    - Othello: 无  
    - Xiangqi: 无

    可直接复用的模块（纯 game-agnostic）：batch_mcts/ 全部、core/replay_buffer.py、core/policy.py、core/weak_move.py、core/checkpoint.py、core/game_logger.py、core/base_config.py

    需要 refactor 的模块（当前有 Othello 硬编码）：
    1. train/core/model_builder.py — 硬编码 OthelloResNet
    2. train/batch_mcts/shared_evaluator.py — 硬编码 build_othello_model
    3. train/core/replay_buffer.py — 从 Othello 文件 import TrainInput
    4. train/train_othello.py — 主循环逻辑通用，但 import 全是 Othello
    5. train/eval_match.py — 硬编码 "othello"

    ---
    Phase 1: 重构共享代码（解耦 Othello）

    1a. 抽取共享类型 — train/core/types.py
    - 把 TrainInput、Losses 从 train/model/othello_resnet.py 移到共享位置
    - replay_buffer.py 和训练循环都只依赖这两个类型

    1b. 重构 model_builder.py — 改为注册机制
    - build_model(game, cfg) 根据 game.get_type().short_name 自动选择网络
    - 或更简单：接受一个 build_net_fn 参数，各游戏自己提供

    1c. 重构 shared_evaluator.py
    - _run_server 接受 build_model_fn 参数，不再硬编码 build_othello_model
    - InferenceServer.register_model 已支持传入 state_dict + 维度，这部分已经是通用的

    1d. 抽取共享训练循环
    - 把 train_othello.py 中 game-agnostic 的主循环逻辑抽到 train/core/train_loop.py
    - 暴露 run_training(game_name, build_model_fn, play_game_fn, SymmetryClass, TrainConfigClass) 接口
    - train_othello.py 变成一行调用 + Othello 参数

    ---
    Phase 2: 创建 Xiangqi 游戏模块

    2a. train/model/xiangqi_resnet.py
    - XiangqiResNet(nn.Module) — 输入 [15, 10, 9]，输出 policy logits [8100] + value [3]
    - 复用 Othello 的 ConvBlock / ResBlock 结构
    - 与 OthelloResNet 共享 torso + head 设计模式
    - Model wrapper 直接复用（它不关心网络内部结构）
    - 需要考虑：8100 维 policy head 带来 2×10×9×8100 ≈ 1.46M 的 FC 参数。如果过于庞大，可以用因式分解 policy（from-head + to-head），但先以 flat 8100 起步

    2b. train/games/xiangqi/config.py
    - XiangqiTrainConfig(BaseTrainConfig) — 加 xiangqi 特有参数（如弱着步数上限等）

    2c. train/games/xiangqi/play.py
    - play_game(game, mcts_black, mcts_white, config, rng, ...)
    - 大部分逻辑和 Othello 的 play_game 相同，可以直接复用 _should_prune、_setup_weak_moves、try_weak_move 等
    - 可能需要调整的是温度策略和剪枝阈值（因为棋盘更大、对局更长）

    ---
    Phase 3: 创建训练入口

    3a. train/train_xiangqi.py
    - 调用 Phase 1d 的 run_training()，传入 xiangqi 的 model / play / symmetry / config

    ---
    建议执行顺序

    1. Phase 1a — 抽取 TrainInput/Losses 到 train/core/types.py（影响面小、先行）
    2. Phase 1b + 1c — 重构 model_builder 和 shared_evaluator（依赖 1a）
    3. Phase 2a — 创建 XiangqiResNet + 测试（可以独立进行）
    4. Phase 2b + 2c — 创建 config 和 play_game
    5. Phase 1d + Phase 3 — 抽取共享训练循环 + 创建 train_xiangqi.py

    ---
    我的判断

    - 改动集中在「把 Othello 特定代码从共享模块中剥离」，不涉及 BatchMCTS、replay buffer、weak move 等核心算法的修改
    - 8100 维 policy 先 flat 起步是合理的（Othello 65 维已经验证了 mask + softmax 机制工作正常）
    - 需要写测试的主要是 XiangqiResNet 的 shape 正确性和 play_game 的基本流程

    ---
    这个计划你看有什么需要调整的地方吗？比如：
    - 对 policy head 的 8100 flat vs factored 有什么偏好？
    - 重构的粒度是否合适？（太激进？太保守？）
    - 是否要先让 Othello 训练流程保持稳定不动，纯增量式加 xiangqi？
