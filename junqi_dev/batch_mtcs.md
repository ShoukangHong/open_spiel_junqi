● Batch MCTS 架构设计

  文件结构

  train/
  ├── batch_mcts/
  │   ├── __init__.py
  │   ├── config.py          # MCTSConfig — 所有超参数集中管理
  │   ├── node.py            # Node — MCTS 树节点（内存优化）
  │   ├── mcts.py            # BatchMCTS — virtual loss + leaf batching 核心算法
  │   └── evaluator.py       # BatchEvaluator 抽象 + 通用 Stub evaluators
  ├── model/
  │   ├── __init__.py
  │   └── othello_resnet.py  # PyTorch ResNet + ModelEvaluator 适配器

  Virtual Loss 算法流程

  单轮 batch (处理 K 个 simulation):

  Phase 1: 收集 K 个叶子（虚拟线程接力遍历）
    for k in 1..K:
      node = root
      path = [root]
      while node.is_expanded and not node.is_terminal:
        action = argmax( Q(s,a) + c_puct * P(s,a) * √N_parent / (1 + N(s,a) + VL(s,a)) )
        node = node.children[action]
        node.virtual_loss += 1      ← 施加虚拟损失，影响下一个线程
        path.append(node)
      leaves.append(node); paths.append(path)

  Phase 2: Batch NN 推理
    values, policies = evaluator.evaluate_batch([leaf.state for leaf in leaves])

  Phase 3: Expand + Backup
    for path, value, policy in zip(paths, values, policies):
      leaf = path[-1]
      if not leaf.is_terminal: expand(leaf, policy)
      for node in reversed(path):
        node.virtual_loss -= 1
        node.visit_count += 1
        node.total_value += value
        value = -value               ← 每层翻转视角

  root policy: π(a) ∝ N(root.children[a])^(1/τ)

  Virtual Loss 只在 exploration 项生效

  PUCT = Q(s,a) + c_puct × P(s,a) × √(Σ(N+V)) / (1 + N + V)

          ↑ Q 不含 V（保持价值估计干净）
                        ↑ 分母含 V（推开后续线程）
                              ↑ 分子含 V（略微提升整体探索）

  关键设计决策

  ┌──────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
  │                     决策                     │                                 理由                                  │
  ├──────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Virtual loss 只影响 visit count，不污染 Q 值 │ Q 保持对真实 return 的无偏估计                                        │
  ├──────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ 同步模拟（非真并行）                         │ Python GIL + Windows 不支持 fork，interleaved 模拟即可达到 batch 效果 │
  ├──────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ Node 存 state clone                          │ Othello 状态小（8×8），clone 便宜；军棋阶段可改为 action sequence     │
  └──────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘