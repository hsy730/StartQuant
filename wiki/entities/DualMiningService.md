---
标题: DualMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/dual_mining_service.py]
---

## 是什么

`backend/services/dual_mining_service.py` 第 58 行的统一挖掘调度器，用一个入口在五种算法之间切换并转发股票池与进度回调。

## 关键事实

- `SUPPORTED_ALGORITHMS` 包含 `genetic`、`pysr`、`tree_prescreen`、`gflownet`、`deep_implicit` 五种模式。
- 构造参数按算法分组保存为 `_gp_params`、`_pysr_params`、`_tree_prescreen_params`、`_gflownet_params`、`_deep_factor_params`。
- `mine_factors()` 按 `algorithm` 分派；未知算法记录告警并回退到 `genetic`。
- 每个结果附加 `source` 字段标识来源算法；可选依赖缺失时返回 `success=False`、`best_factors=[]` 与缺失说明。
- `set_stock_pool()` 先缓存股票代码与起止日期，子服务创建后再通过 `_pass_stock_pool_to_service()` 传递。
- 进度回调统一追加 `algorithm` 关键字；树模型预筛选中非数值的阶段名会被过滤，避免污染数值进度。
- 子服务类型与共享能力来自 [[BaseMiningService]]，工厂注册与取用见 [[MiningAlgorithmRegistry]]。
