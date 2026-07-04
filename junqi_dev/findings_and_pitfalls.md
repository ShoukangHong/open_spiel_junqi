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

### 28. `compute_solved_policy` 置信度用 root_visits 而非 child_visits

原公式对所有 child 用 `sqrt(child_N)` 算置信度。但 solver 停止子节点探索后，已证明的子节点访问量不再增长，只有根节点继续累积。一个小-N 的 proven win 用 `sqrt(child_N)` 只有弱置信度，容易被高-N 的 unproven 子节点压过。

**修复**：proven child 改用 `sqrt(root_N)`（根节点总访问量），unproven 保持 `sqrt(child_N)`。同时 α 从 2.0 提升到 5.0，增强 proven 节点的区分力。

### 29. 新模型 `value_fc2` 均匀初始化导致 draw bias

某些随机种子下，`nn.init.xavier_uniform_` 初始化的 value 头 logits 几乎相等 → softmax 后 W≈0, D≈0.99, L≈0。MCTS 开局时 D 值极高，导致所有着法 Q≈0，搜索无方向。

**修复**：`nn.init.zeros_(m.weight)` 使 logits 全零 → softmax 均匀 [0.33, 0.33, 0.33] → 正确反映开局不确定性。

### 30. Windows 共享内存 `buf[slice] = bytes` 不支持

Python 3.11 Windows 上，`SharedMemory.buf` 返回的 `memoryview` 不支持 `buf[offset:offset+n] = bytes_obj` 的切片赋值（Linux 可以）。`np.ndarray(buffer=shm.buf, offset=off)` 的零拷贝视图在 Windows 上也不稳定。

**修复**：`_write_bytes` fallback 用 `np.frombuffer(data, dtype=np.uint8)` 中转，再通过 `np.ndarray` 写入 buffer。`_has_direct_view()` 在非 Windows 平台才启用 `np.copyto` 零拷贝路径。

### 31. 消息处理循环重复 append 导致结果队列残留

合并 `shared_evaluator.py` 时旧版消息处理代码未被完全删除，每条推理消息被 append 到 `pending` 两次。服务器返回两份结果：`_recv` 取走第一份（正确），第二份残留队列中被下一次 `_recv` 提前取走，拿到过期的前一步推理结果。

**现象**：`test_weight_update_changes_output` 权重更新后推理结果不变。

**修复**：删除重复的旧代码块。

**教训**：合并冲突解决后必须逐行对比 diff，确认删除了所有旧逻辑。自动化测试的权重更新场景能检测这类问题。

### 32. Inference server 纯 busy-polling 浪费 CPU

原主循环用 `get_nowait()` + `sleep(0.001)` / `sleep(0.0005)` 轮询，空闲时每秒空转 1000 次，凑 batch 时每秒 2000 次。CPU 被无效轮询占据。

**修复**：空闲时 `get(timeout=0.1)` 阻塞等待，有消息才唤醒；凑 batch 时 `get(timeout=0.002)` 短暂阻塞。提取 `_handle_msg()` 内部函数消除 drain / 等待 / 处理三处重复代码。预分配 `obs_buf = np.empty((max_batch, obs_dim))` 消除每次 batch 的 `np.concatenate` malloc/memcpy。

### 35. Policy head 3×3 conv 输出层初始化不当导致 prior 极度集中（严重）

将 policy head 从 `Conv1×1` 扩展为 `3×3→ReLU→3×3→ReLU→1×1` 后，3×3 conv 用 `kaiming_normal`（fan_out 模式下 std≈0.08），1×1 也是 `kaiming_normal`。forward 通过两层 ReLU+3×3（每层放大~1×）再经 1×1（再放大），最终 logits std≈31。softmax 后 8100 个动作中一个占 99.9% 概率，其余全 0。

MCTS 展开后每批 10 个状态全部走同一个动作，batch 内去重后只剩 1 个叶子，600 次 sim 仅探索 60 个动作——完全丧失搜索广度。

**修复**：`policy_conv3`（最后一层 1×1）用 `uniform(-1e-3, 1e-3)` 初始化。3×3 conv 保持 `kaiming_normal(relu)`。forward 输出 logits 在 ±0.24 窄区间，softmax 后 prior 接近均匀。

**教训**：多卷积层 policy head 的输出尺度需要显式控制。最后一层 1×1 是信息瓶颈——其权重范围直接决定 logits 方差。3×3 + ReLU 负责特征提取（可放开），1×1 负责压缩到 logits（必须掐死）。

### 36. Probe 批量推理超过 SHM buffer 限制

`_speculative_probe` 的 level 0 将所有 root children 和 root state 合并为一个大 batch 推理，deep walk 的 leaves 也是整批发送。当 branching factor 大时（象棋 ~40-80 合法着法），单 batch 超过 SHM buffer 的 `mcts_batch_size × num_actors` 容量，server 返回空结果导致 `np.exp(pl - pl.max())` 在 zero-size array 上崩溃。

**修复**：所有 probe 中的 `batch_inference_raw` 调用按 `self.config.batch_size` 分片发送。

**教训**：所有发送到 SHM server 的 batch 必须 ≤ `batch_size`。树内展开（probe、solver）的批量评估很容易忘掉这个约束——它们处理的集合可能是所有 root children（~80）而非 MCTS 路径（~10）。

---

### 33. Policy head 的 BN + ReLU 破坏 NN prior（致命）

Xiangqi / Othello ResNet 的 policy head 原为 `Conv1x1 → BN → ReLU`。OpenSpiel AlphaZero 的标准做法是 `Conv1x1 → Flatten → FC(relu) → FC(None)`——policy logits 的**最后一层无 BN 无激活**。

BN + ReLU 会带来三个问题：
1. **训练时 batch statistics 耦合**：同一 batch 内样本的 logit 相互依赖，policy CE loss 的梯度被 batch statistics 的噪声污染，policy head 学不懂（pkl tail 长期不降）
2. **ReLU 限制 logits ≥ 0**：模型无法表达"这步绝不该走"，劣着和优着的 logit 差被压缩在 [0, ~2] 的窄区间，softmax 区分力差
3. **推理时 BN 用 running stats**：去掉 BN 后加载旧 checkpoint 的 `policy_conv` 权重是在 BN 存在时训练的，输出尺度完全失控——`policy_conv` 训练时输出 σ≈1 供 BN 归一化，去掉 BN 后 raw conv 输出可达 ±几十到几百，softmax 后 44 个走法中一个 NN=0.988 其他全 0，MCTS 完全丧失探索引导

**修复**：删 `self.policy_bn` 和 `F.relu`，改 `forward` 为 `policy_logits = self.policy_conv(x).reshape(batch, -1)`（Xiangqi）和 `p = self.policy_conv(x).reshape(batch, -1); policy_logits = self.policy_fc(p)`（Othello）。旧 checkpoint 用 `strict=False` 加载，optimizer state 因参数不匹配被重置。

**教训**：Policy head 的最后一层必须保留线性输出（可正可负）。BN 在 head 中的应用违背 DeepMind/Leela 的架构惯例——这些网络都在 trunk 用 BN，head 的最后一层不归一化不激活。

### 34. MCTS 失势时未访问节点 Q=0 导致访问分散（严重）

MCTS 的 PUCT 公式对未访问节点（`explore_count==0`）使用 `Q=0`，配合 PUCT 的探索项一起参与 max 选择。在失势局面下，已访问节点的 Q 收敛为负数（如 -0.5），而未访问节点 Q=0 相对更有吸引力——即使其 prior 极低，PUCT 探索项仍会让它被选中。结果是**访问在众多低 prior 动作间均匀分散**，每个动作的 Q 统计噪声极大，MCTS 策略变为近均匀分布，质量不如 NN prior。

**修复**：FPU（First Play Urgency）。未访问节点的 Q 不再固定为 0，改为继承父节点 Q 并依 prior 惩罚：

```
Q_unvisited = Q_parent - λ × (p_max - p) / p_max
```

- 最高 prior 节点继承 Q_parent，不受罚
- 低 prior 节点受 λ 权重惩罚，在失势时 Q 更负，自然被 PUCT 避开
- λ=0.2 时效果显著：搜索聚焦于高 prior 动作

**影响范围**：`Node.puct_with_virtual` 新增 `q_parent`/`fpu_lambda`/`prior_max` 参数，`MCTSConfig` 新增 `fpu_lambda` 字段，`_select` 在每节点计算 `p_max` 和 `q_parent` 传入。

---

## 测试运行

```bash
# 激活 venv 后
python -m pytest tests/ -v
```

未解决报错：
[06-17 22:48:43]     [eval] dir=/hy-tmp/cloud_b  ckpts_found=17  refs=[160, 60, 120]
  [eval] loaded cloud_b:180  params=3032895
[inf-srv GPU0] CPU affinity: core 127 (of 128 available)
Process actor-gpu0-48:
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/root/open_spiel_junqi/train/core/train_loop.py", line 146, in actor_process
    states_info, returns, rare_games, wstats = play_game_fn(
                                               ^^^^^^^^^^^^^
  File "/root/open_spiel_junqi/train/core/play.py", line 148, in play_game
    root = mcts.mcts_search(state)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/open_spiel_junqi/train/batch_mcts/mcts.py", line 681, in mcts_search
    values_arr, priors_list = self.evaluator.batch_inference_raw(
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/open_spiel_junqi/train/batch_mcts/shared_evaluator.py", line 71, in batch_inference_raw
    return self._process_batch(data, mask_b)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/open_spiel_junqi/train/batch_mcts/shared_evaluator.py", line 105, in _process_batch
    pl = np.exp(pl - pl.max())
                     ^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/_core/_methods.py", line 45, in _amax
    return umr_maximum(a, axis, None, out, keepdims, initial, where)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation maximum which has no identity


在10万数据，1e-3学习率下3000个256的batch的学习结果： 
Done. 3000 batches in 342s (0.1s/batch) 
P-KL: 0.2553 → 0.2319 (-9.2%) 
V-KL: 0.1498 → 0.1006 (-32.8%)
这个是整体3e-4学习率下的效果： 
Done. 3000 batches in 341s (0.1s/batch) 
P-KL: 0.2562 → 0.2321 (-9.4%) 
V-KL: 0.1497 → 0.1011 (-32.5%)

---

## 2026-07 会话新增

### 37. DB 归档文件字符串排序在 10M 处出错（致命）
`_all_db_paths` 用 `sorted(archives, reverse=True)` 对文件名排序。
`buffer_9000000.db` 和 `buffer_10000000.db` 的字符串比较 `"9" > "1"`，
10M 文件排在 9M 之后，绝对 ID 映射全错。采样窗口指向完全错误的时间段。

**影响**：模型在学 1000 万行之前的旧策略，loss 周期性跳变，ELO 断崖下跌。
**修复**：改用 `key=lambda p: int(re.search(r'_(\d+)\.db$', p).group(1))` 数值排序。
**教训**：任何依赖文件名字符串排序的代码必须用数值排序或加测试覆盖跨位数场景。

### 38. `_rotate_db` suffix 冲突导致静默覆盖归档（致命）
旧公式 `suffix = _total - max_db_rows`。续训时若旧归档已被手动删除，
`_sync_total` 重新计算 `_total` → suffix 可能等于已被删除归档的旧命名 →
`os.rename` 在 Linux 上静默覆盖同名文件。15M 行数据瞬间消失。

**影响**：训练分布断层，loss 跳变。
**修复**：`suffix = _total`（单调递增，永久唯一）。
**教训**：文件命名必须保证单调唯一；旋转时检查目标是否存在。

### 39. `_clamp_draw` 未清零 `total_reward`（中等）
Solver 证明某节点为和棋时 `_clamp_draw` 只设 `draw_reward = explore_count`，
但 `total_reward` 保持原值不变。证明前 MCTS 探索过的"优势"分支 → Q 残留 > 0 →
`node.q_value` 显示必胜而非和棋。

**影响**：MCTS 搜索中已证明的和棋节点仍被 PUCT 当成优势选。
**修复**：`_clamp_draw` 同步设 `total_reward = 0.0`。

### 40. no-capture 边界最后一手将军被误判为将死（严重）
Xiangqi `DoApplyAction` 先递增 `moves_since_capture_`，再检查
`IsInCheck(current_player_)` → `LegalActions()`。`LegalActions()` 第一行
`if (IsTerminal()) return {};`——由于 `moves_since_capture_` 已超限，
`IsTerminal()` 返回 true，`LegalActions()` 返回空 → 被误认为将死。

**影响**：40 步无吃子终局前的最后一手将军被判为获胜，和棋变必胜。
**修复**：checkmate 检测加上 `!IsTerminal()` guard。

### 41. `WouldLeaveInCheck` 禁止吃将（严重）
`WouldLeaveInCheck` 在临时移动后检查自己的将是否仍被攻击。
但吃掉对方老将后，攻击方（对方的车/炮等）仍在原位置，`IsInCheck` 返回 true →
吃将被判非法。

**影响**：被将军时无法用吃将解围，`LegalActions` 在将死局面返回空，
但实际存在解围着法。
**修复**：检测到 `captured.type == kGeneral` 时直接返回合法。

### 42. Opening book 中 `weak_enabled` 被关闭
旧代码 `weak_enabled = allow_weak and init_state is None`。
使用 opening book 时 `init_state is not None` → 弱着被完全关闭。

**影响**：50% 的对局没有弱着探索。
**修复**：改为 `weak_enabled = allow_weak`（只有 fork 传 False）。

### 43. OpeningBook 启动时加载全部 state 到内存
旧实现 `self._states.append(game.deserialize_state(raw))`。
xiangqi state 含完整走子历史，每个开销大。开局库 >500 局 → 内存爆炸。

**修复**：只存序列化字符串，`sample()` 时按需 `deserialize_state`。

### 44. 系统安装的旧 pyspiel.so 被优先加载（严重）

项目根目录有最新编译的 `pyspiel.so`（obso=17 通道），
但 `/usr/local/lib/python3.11/dist-packages/pyspiel.so` 是旧版（15 通道）。
Python import 优先级 `dist-packages` > 当前目录。

当从项目根目录执行 `python -c "import pyspiel"` 时，cwd 优先 → 加载新版（17 通道）。
当 pytest 从 `tests/` 子目录运行时，cwd 不包含 `.so` → fallback 到 dist-packages（15 通道）。

**影响**：训练正常（train_*.py 从项目根目录启动），但测试在 import 路径上拿到
旧 .so，observation_tensor_shape 与模型期望不匹配。手动验证和自动测试结果矛盾。

**修复**：删掉 `/usr/local/lib/python3.11/dist-packages/pyspiel.so`，
或将新版 `.so` 复制/软链接到 dist-packages。

**教训**：编译型 Python 模块（.so/.pyd）可能存在多份副本。排查 shape 不匹配问题时，
首先检查 `pyspiel.__file__` 而不是 `pyspiel.load_game().observation_tensor_shape()`——
后者在同一个模块里内部一致，不会暴露加载了错误副本的问题。