---
标题: DualMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/dual_mining_service.py]
---

## 是什么

`backend/services/dual_mining_service.py` 中的统一挖掘调度器，用一个入口在多种算法之间切换并转发股票池与进度回调。

## 关键事实

- 维护受支持算法名单与「算法 → 参数组」映射，构造时按算法分组保存参数。
- 派发时按算法名路由，遇到未知算法记录告警并回退到默认算法。
- 每个结果附带来源算法标识；可选依赖缺失时返回失败标记与缺失说明而不抛异常。
- 股票池先暂存，待子服务创建后再传递；进度回调统一追加算法标识，并过滤非数值的阶段名。
- 子服务共享能力来自 [[BaseMiningService]]，工厂注册与取用见 [[MiningAlgorithmRegistry]]。
