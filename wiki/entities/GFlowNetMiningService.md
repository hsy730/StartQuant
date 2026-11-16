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
- 策略网络为带可学习归一化项的多层网络，按合法动作状态机采样表达式。
- 每轮依次完成轨迹采样、适应度评估、精英池更新与轨迹平衡损失反传。
- 设有梯度裁剪、精英池容量与经验回放容量上限。
- 返回最优因子列表、逐代适应度曲线与策略损失历史；去重惩罚见 [[表达式相似度去重与惩罚]]。
