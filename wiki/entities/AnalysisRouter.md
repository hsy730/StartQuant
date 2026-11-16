---
标题: AnalysisRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/analysis.py]
---

## 是什么

`backend/api/routers/analysis.py` 中的因子分析路由模块，把计算、IC/IR、稳定性、暴露、归因与报表等分析能力暴露为 HTTP 端点。

## 关键事实

- 全部为 POST 端点，请求统一围绕因子名、股票列表与日期区间三要素。
- 因子代码由内部解析函数从 factors 表或 generated_factors 表取回，未命中返回 404。
- 计算与分析一律委托对应服务，路由层只做参数整理与响应清洗。
- 端点挂载于 [[FactorFlowAPI应用]]。
