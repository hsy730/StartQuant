---
标题: GFlowNetMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/gflownet_mining_service.py]
---

## 是什么

`backend/services/gflownet_mining_service.py` 中用策略网络逐步构建公式的因子挖掘服务。

## 关键事实

- 依赖 `torch`，未安装时构造即抛出 `ImportError`。
- 默认 `n_trajectories=200`、`n_iterations=50`、`hidden_dim=128`、`learning_rate=1e-3`。
- 默认 `max_expression_depth=5`、`temperature=1.0`、`reward_scale=10.0`、`buffer_size=1000`。
- 策略网络为带可学习 `log_z` 的多层 MLP，按 [[GFlowNet表达式状态机]] 的合法动作采样。
- 每轮依次采样轨迹、评估适应度、更新 Hall-of-Fame、经验回放采样最多 32 条、计算轨迹平衡损失并反传。
- 梯度裁剪上限为 1.0，Hall-of-Fame 容量上限为 20。
- 返回最优因子列表、逐代最优与平均适应度、策略损失历史。
- 候选去重与惩罚规则见 [[表达式相似度去重与惩罚]]。
