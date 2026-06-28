# Value Target 架构梳理

> 更新日期：2026-06-28

## 1. 数据流概览

```
play_game (play.py)
  │
  ├─ states_info[i] = (obs, mask, policy, cur_player, tag, q, dr)
  │                    ↑ q, dr 来自 _stable_qdr(root)
  │
  ├─ extra_surprise[]  = (obs, ..., tag, q, dr, kl_sum, step_idx)
  │                    ↑ root 级 surprise，存 step_idx = len(states_info)-1
  │                    ↑ child 级 surprise，无 step_idx（len=8）
  │
  └─ states_info.extend(copies)
     └─ 原始状态在前 i=0..game_length-1，copies 在末尾 i≥game_length

train_loop 处理:
  │
  ├─ game_length = 第一个含 "surprise" tag 的索引（即原始对局长度）
  │
  ├─ 每个 states_info[i]:
  │   ├─ "child_" in tag  → val = _mcts_wdl(q, dr)        # 纯 MCTS
  │   └─ 其他            → step_i = item[8] or i           # surprise 取存的索引
  │                       → alpha = compute_alpha(step_i, game_length, offset, td)
  │                       → val = alpha·outcome_WDL + (1-α)·mcts_WDL
  │
  └─ val 为 3 维 WDL [w, d, l]，作为模型 value head 的 cross-entropy target
```

## 2. `compute_alpha` — 混合权重

```python
def compute_alpha(step_index, game_length, offset, temperature_drop):
    if step_index == game_length - 1:      # 终端状态
        return 1.0                          # → 100% game outcome
    denom = max(offset + game_length - 1 - temperature_drop, 1)
    step_pos = offset + step_index          # 绝对步数
    if step_pos < temperature_drop:         # temp drop 之前
        return 0.0                          # → 100% MCTS WDL
    return min((step_pos - temperature_drop) / denom, 1.0)
```

**关键变量**：
- `offset` = `rare_at` = `len(init_state.history())`（fork 起始偏移）
- `step_index` = 对局内相对步数（0-based，不含 init_state 历史）
- `step_pos` = `offset + step_index` = 全局绝对步数

**alpha 曲线**：temp_drop 前为 0 → temp_drop 到终局线性爬升 → 终局精确为 1.0。

**修复过的 bug**：旧版 denom 未减 `temperature_drop`，导致 alpha 永远 < 1.0。

## 3. `_stable_qdr` — MCTS 的 Q 和 draw_rate

```python
def _stable_qdr(node):
    if node.outcome is not None:            # solver 已证明
        s = node.state
        p = s.current_player() if s and not s.is_terminal() else node.player
        q = node.outcome[p]
        return q, 1.0 if q == 0 else 0.0   # 和棋时 draw=1，否则 0

    valid = [c for c in node.children if c.explore_count > 1]  # 排除噪声
    if valid:
        q = sum(c.total_reward for c in valid) / sum(c.explore_count)
        dr = sum(c.draw_reward for c in valid) / sum(c.explore_count)
    else:
        q = node.q_value
        dr = node.draw_rate
    return q, dr
```

**视角约定**：
- 非终端 outcome：用 `state.current_player()`（非 `node.player`，因为 child 的 `node.player` = 父状态的 current_player）
- 终端 outcome：`state.current_player()` 返回 TERMINAL(-4)，退回 `node.player`

## 4. `_mixed_target` — 混合 WDL 目标

```python
mcts_wdl = _wdl_from_qdr(q_value, draw_rate)    # MCTS 的 [w,d,l]
game_wdl = _outcome_wdl(game_ret)               # 终局的 [w,d,l] one-hot
mixed_wdl = alpha * game_wdl + (1-alpha) * mcts_wdl
```

- `_wdl_from_qdr`: 从 Q 和 draw_rate 反推 WDL。公式 `w=(Q+1-d)/2, l=(1-Q-d)/2`，对合法 MCTS 值（|Q|≤1-d）精确无 clamp
- `_outcome_wdl`: 终局标量 → one-hot WDL

## 5. Surprise 的 value target 处理

| 类型 | tag | step_index 来源 | value target |
|------|-----|----------------|-------------|
| root surprise | "surprise" / "super_surprise" | `item[8]` (存储的 `len(states_info)-1`) | `_mixed_target`（与原始状态相同 alpha）|
| child surprise | "child_surprise" / "child_super_surprise" | 无 | `_mcts_wdl` 纯 MCTS（不混 outcome）|

**修复过的 bug**：surprise copy 在 `states_info` 末尾（index ≥ game_length），旧代码用 loop index `i` 直接算 alpha → 始终 ≥ game_length → alpha 被 clamp 到 1.0 → 全是 100% outcome。

## 6. 视角一致性

| 组件 | 视角 | 依据 |
|------|------|------|
| `root.q_value` | `state.current_player()` | backprop 用 `returns[player]` |
| `root.nn_q` | `state.current_player()` | NN 输出 WDL then Q=w-l |
| `c.nn_q`（probe）| `c.state.current_player()` | NN 评估 child state |
| `_stable_qdr(c)` | `c.state.current_player()` | 从 c.children 聚合（children.player = c.state.current_player()），或 outcome 用 state.current_player() |

**修复过的 bug**：
- `_stable_qdr` 的 outcome 分支曾用 `node.player`（= 父状态 player），改为 `state.current_player()`
- `compute_solved_policy` 的 child 调用曾用 `c.player`，改为 `c.state.current_player()`

## 7. `draw_reward` 回传

```
backprop:
  node.total_reward += target          # 终局回报（赢+1/输-1/和0）
  node.draw_reward += draw_prob        # 和棋概率
  node.explore_count += 1

draw_prob 来源:
  - 终端叶子     → 1.0 if draw else 0.0          （实际结果）
  - 非终端叶子   → NN 的 d 值                     （预测值）
  - Probe 路径   → NN 的 d 值 或 终端实际结果     （修复过漏传）
```

**修复过的 bug**：probe 的 `_backprop` 调用（surprise 终止 / 终端 / 深层叶子）曾漏传 `draw_prob`，默认为 0.0。

## 8. 风险点

- `_stable_qdr` 和 `compute_solved_policy` 的视角在 child 节点和 root 节点的 `player` 含义不同。改代码时必须确认目标玩家是谁
- `extra_surprise` 元组长度不一致（root=9, child=8），index 操作要检查 `len(item)`
- `game_length` 通过扫描 "surprise" tag 截断，依赖 tag 命名约定
- terminal 状态的 `current_player()` 返回 -4，不能直接当数组索引
