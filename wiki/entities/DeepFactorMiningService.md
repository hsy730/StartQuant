---
标题: DeepFactorMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/deep_factor_mining_service.py]
---

## 是什么

`backend/services/deep_factor_mining_service.py` 中基于 Transformer 的深度隐式因子挖掘服务。

## 关键事实

- 硬依赖 torch，未安装时构造即抛出导入错误。
- 与 [[GeneticFactorMiningService]]、[[PySRFactorMiningService]] 接口一致，可互换或并行调用。
- 流程为预计算基础因子、构建滑动窗口、训练模型、提取隐因子、逐因子验证并持久化模型。
- 每次运行前先清理上一轮资源并重置取消标志，避免显存泄漏与旧状态干扰。
- 复杂度以模型参数量衡量；模型经 [[ModelRegistry]] 持久化，保存失败只记警告，设计见 [[隐因子序列建模]]。
