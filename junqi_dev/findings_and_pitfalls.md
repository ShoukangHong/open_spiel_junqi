# 发现、修复及遗留问题

## 关键 Bug 修复

### 1. MCTS solver 从未执行（致命）
`_backprop` 用 `list.pop()` 遍历 path，清空了列表。Phase 3 中的 solver 循环
`for node in reversed(path_nodes)` 拿到的是空列表，MCTS-Solver 从未运行。
terminal 结果无法向上传播，已知必胜/必败分支不被标记。

**影响**：Othello 纵深大不明显；TicTacToe fork 局面 `proven_children=0/6`。
**修复**：改为索引遍历，不动原列表。

### 2. PUCT `float("inf")` 捷径（中等）
`puct_with_virtual` 对未探索节点返回 inf。PUCT 公式天生不需要 inf：
未探索时 N=0 → `c × P × sqrt(parent_N) / 1`。inf 让 max() 无视 prior，
policy network 的引导被稀释。

**修复**：移除 inf 捷径。

### 3. Checkpoint 并发读写崩溃（高影响）
`torch.save` 非原子操作。learner 每步保存 `checkpoint--999.pt`，
actor 同时检测文件更新就加载。读到半完成文件 → CUDA context 损坏。

**现象**：step 42 时报 `CUDA error: unknown error`，位于 `torch.relu`。
**修复**：先写 `.tmp`，再 `os.replace` 原子重命名。

### 4. 训练数据 value 视角与观察不一致（中等）
observation tensor 原来只有 3 通道（空/黑/白），同一个棋盘轮到黑白时
输入完全相同，但 value target 相反。模型被迫拟合矛盾目标。

**修复**：增加第 4 通道 player-to-move indicator。

## 设计改进

### 5. Solve-aware policy
全解局面的 policy 应绝对确定：最优 outcome 子节点平分概率，其他为 0。
非全解但有 proven 信息时，用 reward/penalty 模型补正：
`weight = exp(α × diff × √N)`，对称处理高估和低估。

### 6. Weak-move 探索 + rare case 挖掘
自对弈主动走出弱着法，发现模型盲区。
- 开局势预 roll 步数（几何分布，cap=3）
- 比较走弱着后 NN 估值 vs 当前 MCTS 估值
- 用胜率相对下降比例 `|p_before-p_after|/max(p_before,p_after)` 判定
- 三岔路：rare（fork 稀有局）/ weak（接受）/ weak_final（接受后不再尝试）
- rare case 主对局保 MCTS 着法，fork 新游戏从弱着后出发

### 7. Per-actor 日志
每个 actor 独立写 `actor_{i}.log`，每 100 局采样完整对局记录（moves + MCTS stats），
其余一行摘要。主 log 每 step 输出 weak stats。

## 发现的问题与教训

### 8. 线程交互：MCTS `_propagate_solved` 治标不治本
在发现 bug 1 之前，我们加了一个全树递归扫描来补齐 solver。这是 hack。
根因是 `_backprop` 的 pop 清空 path。

### 9. Batch size 影响探索广度
batch_size=64 时第一批所有线程撞 root，被去重为 1 次 expand。
TicTacToe 只有 9 个合法着法，这个浪费非常明显。Othello 分支因子大，不明显。

### 10. Temperature τ=0.03 是近似 argmax
10% 的 visit 领先被放大为 96% vs 4%。policy target 基本退化为 one-hot。

### 11. MCTS 评估赛确定性导致比分镜像
MCTS + `best_child()` 是确定性的。两个接近的模型对弈时，每局走法完全相同，
交替先手导致比分精确 50/50。**解决方案**：温度采样替代贪心选子。

## 遗留问题

- **value 学习慢**：observation 有 player-to-move 通道后，value target 仍用
  `returns[0]`（黑棋视角）。模型被迫自己学翻转，增加了学习负担。
  改 `returns[cur_player]` 需同步改 MCTS Phase 3 的视角处理，
  旧训练 + 旧 Phase 3 凑巧自洽，暂不修改。

- **weak move 数据利用**：rare tag 已打入 buffer，训练时未区分采样/加权。
  当前仅用于统计。

- **epsilon greedy / softmax 探索**：`step_with_policy` 返回 one-hot policy，
  eval_match 改温度采样后才解决"镜像比分"。

- **计算几何分布的手雷**：`P(N=k)=p^{k-1}(1-p)`, `E[N]=1/(1-p)`。
  p=0.5 期望 2 次，但分布始终从 1 开始，`p/(1-p)` 是不一样的公式。
