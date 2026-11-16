---
标题: BaseMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/base_mining_service.py]
---

## 是什么

`backend/services/base_mining_service.py` 第 110 行的挖掘服务抽象基类，为遗传规划、GFlowNet、PySR、树模型预筛选提供共享基础设施，子类只需实现 `mine_factors()`。

## 关键事实

- 继承 `ABC`，抽象方法为 `mine_factors()`；子类通过 `_service_name` 自定义日志前缀。
- 构造默认值：`return_column="return"`、`max_eval_stocks=50`、`fitness_objective="ic_mean"`、`cv_folds=0`、`naming_pattern="factor_{i}"`、`max_base_factors=30`。
- 基础因子数超过 `max_base_factors` 时随机抽样；预计算使用线程池，线程数为因子数与 CPU 核数的较小值。
- 模块级基础因子 LRU 缓存容量 256，键为数据指纹与因子代码，读取时返回副本；提供 `clear_base_factor_cache()` 与 `get_base_factor_cache_stats()`。
- `set_stock_pool()` 取多股票数据后做防御性拷贝，并用 `close.pct_change().shift(-1)` 生成收益列；评估样本最多保留 `max_eval_stocks` 只。
- 进度回调签名为 `callback(iteration, total_iterations, best_fitness, avg_fitness)`；`request_cancel()` 只置取消标志，当前迭代结束后停止。
- 先验常量：IC 均值 0.03、IC 标准差 0.02、IR 均值 0.5、IR 标准差 0.3；除零与 IR 计算统一走 [[安全除法]]。
- `release_memory()` 置空数据、收益与因子缓存，调用后实例不可复用，由任务收尾阶段调用。
- 算法实例的统一创建与注册见 [[MiningAlgorithmRegistry]]。
