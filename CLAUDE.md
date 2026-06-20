# 项目工作备忘录（AI Agent 用）

> 本项目是单人开发，使用 git 分支 `exp`，Windows 本地开发 + Linux 云端训练。

---

## 1. 修改前必须说清楚

- **先解释你打算做什么**，得到确认后再改代码。不要假设用户同意你的方向
- 涉及架构或设计决策时，列出多个方案及权衡，让用户选择
- 不确定的功能不要猜——直接问，在明确了用户需求后再实现。

## 2. 必须写测试

- 新增函数必须封装（可独立测试），然后在 `tests/` 下写测试
- 测试目录结构镜像 `train/` 结构：`tests/core/` ↔ `train/core/`，`tests/games/othello/` ↔ `train/games/othello/`
- 概率性逻辑（如弱着 Boltzmann 选择）用多轮统计测试，而非单次确定性断言
- 修改行为时同步更新受影响的测试
- 跑全量测试输入：`source venv/Scripts/activate && python -m pytest tests/ -q`

## 3. 修改配置要全面

- `base_config.py` 加参数 → config JSON 文件也要加
- 测试中的假 config（如 `_TestCfg`、`_WeakCfg`）也要补新字段
- 废弃的配置字段同步从 JSON 删掉

## 5. 记录到 findings_and_pitfalls.md

- 遇到重要 bug / 设计教训单独编号条目
- 说明现象、根因、修复、教训

## 6. 代码风格

- 不写装饰性注释（函数名已说明 WHAT，只注释 WHY和HOW）
- 不要为了"未来可能需要"而抽象
- 代码要清晰容易读，将复杂逻辑封装起来避免函数过长。
