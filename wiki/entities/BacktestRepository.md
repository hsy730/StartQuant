---
标题: BacktestRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/backtest_repository.py]
---

## 是什么

`backend/repositories/backtest_repository.py` 中的回测结果与交易记录数据访问类。

## 关键事实

- `save_result` 接收结果字典，缺失时策略名取「未命名策略」、初始资金取 1000000、交易次数取 0。
- `get_history(limit=20, offset=0)` 按创建时间倒序分页。
- `delete_by_id` 先删交易记录再删回测结果，异常时回滚并返回 `False`，见 [[先子后父删除关联记录]]。
- `save_trade_records` 逐条添加后一次提交，返回保存条数。
- `get_trade_records` 按交易日期升序返回。
- `get_statistics` 返回总数、平均累计收益与平均夏普，空值在聚合时跳过。
- 写库前的 JSON 字段转换见 [[JSON字段入库前转字典]]。
