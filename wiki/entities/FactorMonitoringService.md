---
标题: FactorMonitoringService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_monitoring_service.py]
---

## 是什么

`backend/services/factor_monitoring_service.py` 中的因子时序动态监测服务。

## 关键事实

- 入口为 `monitor_dynamics`，输出滚动带、状态转移矩阵、结构断点与周期性四部分。
- 滚动带默认 `window=20`、`std_multiplier=2.0`，上下轨为滚动均值加减倍数标准差。
- 状态转移矩阵按分位状态统计迁移概率，口径见 [[因子暴露转移矩阵]]。
- 结构断点用 60 日滚动均值变化检测，周期分析用傅里叶功率谱，见 [[滚动均值断点与傅里叶周期]]。
- 周期分析要求至少 50 个数据点，且只取功率谱前 5 个主周期。
- 代表样本通过内部 `_find_longest_stock` 选取，委托公共工具实现。
