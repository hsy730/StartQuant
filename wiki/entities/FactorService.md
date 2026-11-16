---
标题: FactorService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_service.py]
---

## 是什么

`backend/services/factor_service.py` 中的因子服务单例，负责预置因子加载、因子增删改查、代码校验与按股票计算。

## 关键事实

- 模块级实例 `factor_service`，内部持有一个 [[FactorCalculator]]。
- `load_preset_factors()` 读取 `config/factors.yaml`，缺失时写入默认预置因子，默认集合包含 Alpha101 因子。
- 预置因子以 `source="preset"` 写入，按名称去重，已存在则跳过。
- `get_factor_stats()` 返回预置数、用户数、总数、股票缓存数与 akshare 健康状态；健康检查带 5 秒超时。
- `create_factor()` 对挖掘因子做门控：必须关联 generated_factors 记录且 `is_valid` 为真，通过后标记已保存。
- 同名活跃因子拒绝重复创建；同名但已软删除的因子会先硬删除旧记录再新建。
- `validate_factor_code()` 用 100 行线性构造的测试数据实际执行一次计算，全 NaN 或含无穷大判为不通过。
- `calculate_factors_for_stock()` 先查 factors 表，缺失的因子回退到 generated_factors 表达式，可选滚动标准化与时间特征，最后把无穷大统一替换为 NaN。
- `calculate_factors_for_stocks()` 用线程池并行，默认 `max_workers=10`，单票失败只记录告警并跳过。
- 版本留痕由 [[FactorVersionService]] 承担，版本语义见 [[版本快照与回滚]]。
