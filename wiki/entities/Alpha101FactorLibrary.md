---
标题: Alpha101FactorLibrary
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/alpha101_factors.py]
---

## 是什么

`backend/services/alpha101_factors.py` 中的 Alpha101 因子库，以纯数据方式提供 101 号因子中选录的表达式集合。

## 关键事实

- 唯一对外入口 `get_alpha101_factors()`，返回"分类名 → 因子列表"的字典。
- 五个分类键：`Alpha101-动量反转`、`Alpha101-量价关系`、`Alpha101-波动率`、`Alpha101-趋势强度`、`Alpha101-综合信号`。
- 每个因子为字典，字段固定 `name`、`code`、`description`，例如 `alpha001` 的表达式 `TSRANK(SIGNEDPOWER(IF(RETURNS(close) < 0, STD(RETURNS(close), 20), close), 2), 5) - 0.5`。
- 表达式为 TA-Lib 风格 DSL，运算依赖执行上下文中的 `np`。
- 除零保护统一用 `.replace(0, np.nan)`，文件头注明与项目规范一致。
- 组织规则见 [[Alpha101表达式因子库]]，表达式安全校验见 [[公式白名单沙箱]]。
