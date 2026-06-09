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

### 14. Value 输出的视角约定（WDL vs 标量）

MCTS 回传时用 `returns = [p0_value, p1_value]` 形式的零和向量，通过 `returns[decision_player]` 取对应玩家回报。关键约束：`returns[0]` 必须是 p0 视角的值，`returns[1]` 必须是 p1 视角的值。

**标量模型**输出的是 p0（黑棋）视角的单值 v，eval 返回 `[v, -v]` 天然满足约束。

**WDL 模型**输出的是**当前走子方视角**的 `[w, d, l]`。MCTS 将其转为标量 `Q = (w-l) × max_utility`。但 Q 是谁的视角取决于叶节点的玩家：叶节点是 p0 时 Q 是 p0 视角，叶节点是 p1 时 Q 是 p1 视角。此时 `[Q, -Q]` 作为零和向量是错误的——当叶节点是 p1 时，`returns[1]` 应该是 `+Q`（p1 自己的估值），而非 `-Q`。

**修复原则**：WDL 模式下根据叶节点玩家拼装 returns——`returns[leaf_player]=Q, returns[opponent]=-Q`。标量模式保持 `[v, -v]` 不变。

**教训**：value 的输出视角是全系统的隐含契约。改动训练 target（从 `returns[0]` 改为 `returns[cur_player]`）时，MCTS 的视角处理必须同步修改。测试中所有 WDL evaluator 也必须按当前走子方视角生成 WDL 值，不能返回常量。

### 15. Policy target 存原始分布，不 sharpen

训练存储的是 MCTS 原始访问比例分布（如 `{a:0.67, b:0.18, c:0.09}`），不做温度 sharpen。温度只在走子选择时生效——drop 前 `τ=1` 按比例采样，drop 后直接选 `best_child()`。之前存入 buffer 的 policy 是经过 `** (1/τ)` sharpen 后的近 one-hot，导致模型学不到搜索的不确定性，eval 时走法缺乏多样性。

### 16. Symmetry 变换方向一致性

D4 变换是用 `np.rot90`（默认逆时针，CCW）做空间变换，action 映射必须用相同的旋转方向。曾出现过 action 映射用 CW 公式 `(r,c)→(c,7-r)` 但 obs 变换用 CCW 的情况，导致增强后的 obs 和 policy 不匹配。测试必须用**不对称棋子**验证所有 8 种变换的方向一致性——用对称棋子（如两黑子）无法检测方向错误。

### 17. 模型构建统一入口

所有需要加载 checkpoint 建模型的地方（训练、eval_match、游戏 UI）统一走 `train/core/model_builder.py` 的 `build_othello_model()`。避免各处手写 `OthelloResNet(...)` 导致参数遗漏（尤其是 `num_value_classes` 和 `value_classes`）。

### 18. Shared evaluator 预编码：4.7× 吞吐

多 actor + 共享 GPU server 架构下的两大瓶颈：

1. **GPU forward**（主瓶颈）：avg_fwd=11.5ms，占总耗时 ~45%
2. **Server 侧其他开销**：per-state 的 `observation_tensor()` / `legal_actions_mask()` C++ 调用、`hasattr` 类型检查、Python list 构造——每秒数千 states 时累积为主开销

**方案**：actor 进程在提交队列前预编码 `obs`/`mask` 为 numpy 数组，server 端零 C++ 调用只做纯数组操作（`np.concatenate → forward → np.where(mask)` scatter）。

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| avg_fwd | 11.5ms | 4.3ms（-63%） |
| states/s | 3526 | 16375（4.7×） |
| avg_batch | 82→137 | 超 `_MAX_BATCH` 饱负荷 |

**教训**：GPU server 作为核心吞吐瓶颈，必须保持热路径上只有矩阵运算。所有博弈状态到数组的转换应下推到 actor 端完成。

### 19. 温度采样被 argmax 覆盖（严重）

`play_game` 中温度 drop 后的动作选择逻辑有 bug（`train/games/othello/play.py:97-98`）：

```python
# 第 90 行正确计算了温度 sharpen 后的分布
sel_probs = policy ** (1.0 / tau) / sum(...)

# 但第 97-98 行完全忽略了 sel_probs
elif after_drop:
    action = mcts_action   # ← 永远是 best_child（argmax）
```

温度只在 drop 前（`τ=1`）生效，drop 后（`τ=0.2`）永远是 argmax。导致 buffer 里存的 policy target 全是 one-hot，模型学不到搜索的不确定性，策略极度确定。eval 时走法完全缺乏多样性。

**修复**：删除 `elif after_drop` 分支，drop 前后统一走 `rng.choice(len(sel_probs), p=sel_probs)`。

**教训**：温度 drop 前后的采样路径必须一致——drop 只改变 tau 值，不改变选择逻辑。

### 20. eval_match 并行模式受 GIL 限制

`run_match_parallel` 原本用 `threading.Thread` 跑 MCTS 搜索，但 MCTS 树遍历全是纯 Python 操作，全程持有 GIL。10 个 actor 实际串行执行，server `avg_batch=8.4`（等于 MCTS batch_size），GPU 利用率 40%。

**修复**：改为 `multiprocessing.Process`，每个 actor 独立进程绕过 GIL。

### 21. eval_match `register_actor` 时序错误

`run_match_parallel` 在 `server.start()` 之后才注册 actor queue。Windows `spawn` 模式下子进程复制了空的 `_result_qs`，后续注册的 actor queue 子进程不可见。

**修复**：所有 actor 在 `server.start()` 前注册完毕。

### 22. Rare fork 对局未跳过温度 drop（灾难性）

Rare case 触发后会 fork 一个新对局从弱着状态出发，目的是评估该分支的后果。但 fork 对局的 `move_num` 从 0 开始计数，前 `temperature_drop` 步（如 8 步）使用 τ=1 随机走子，导致 fork 对局无法干净评估弱着后果——τ=1 的随机性完全淹没弱着的影响。

另外，临时 drop 前加 Dirichlet 噪声也有问题。Othello 开局极其敏感，第 6 步的一步错棋就可能导致让角必输。temp drop 前加噪声会让 MCTS 无谓偏离最优走法。

**修复**：
- Fork 对局全程使用 post-drop τ（`config.temperature`），走法确定
- 主对局 temp_drop 前 `mcts_search(add_noise=False)`，禁用 Dirichlet 噪声
- Fork 对局噪声保留（影响不大）

**教训**：温度系统和 fork 机制有隐含耦合。`move_num` 在 fork 对局中从 0 开始，但 fork 对局从残局出发不需要 warm-up 阶段的随机探索。

### 23. 剪枝条件仅依赖 Q 值，忽略 solver outcome

剪枝触发条件只用 `root.q_value >= threshold`，但 solver 已证明必胜时 `root.outcome = [1, -1]` 而 `q_value` 可能因搜索早停远低于 1.0。proven_win 应直接触发剪枝。

**修复**：条件改为 `proven_win or Q >= threshold`。骰子仅在首次达标时掷一次（`pruned["checked"]`），不中后不再重试。

### 24. Rare 触发后耗尽弱着配额

Rare 分支设置 `weak_count = weak_max + 1`，导致该局后续弱着点全部跳过。阈值极低（0.02）时第一发几乎必中 rare → 每局只有一个 rare 进账。

**修复**：rare 改为 `weak_count += 1`，仅 `weak_final` 耗尽配额。

### 25. Policy target 直接用 MCTS visit 分布，忽略 action advantage

MCTS 搜索可能把优势动作埋没在低 visit 中（Q=0.8 但 N=10 的动作只拿到 10% target）。模型学到的是"搜索偏好"而非"真实价值"。

**修复**：新增 advantage mixing — `π'(a) ∝ (1-α)·π_mcts + α·softmax(A/T)`。只作用于 unsolved 状态。α=0.5, T=0.2。封装在 `train/core/policy.py`。

### 26. SQLite buffer 文件体积过大

BLOB 存原始字节，Othello obs 含大量 0，未压缩下 ~7-10× 于 .npz。

**修复**：`_pack`/`_unpack` 加 zlib 压缩，14× 压缩比。旧未压缩 DB 可读（fallback），提供 `_compress_db` 转换脚本。

### 27. 评估参考集随训练不同步

训练早期只有少量 checkpoint，固定间隔里程碑选不到对手。后期里程碑不覆盖全部训练历史。

**修复**：删 `eval_milestone_interval`，改为等分当前训练步数取最近 checkpoint。1 个 best + (N-1) 个等分点。`select_eval_references` 封装。

---

## 测试运行

```bash
# 激活 venv 后
python -m pytest tests/ -v
```