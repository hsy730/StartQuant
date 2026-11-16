---
标题: FactorSummaryService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_summary_service.py]
---

## 是什么

`backend/services/factor_summary_service.py` 第 15 行的因子摘要服务，把基础统计、IC 汇总与稳定性汇总压缩成一份带质量分与评级的摘要。

## 关键事实

- 主入口 `generate_factor_summary(factor_name, factor_data, ic_analysis, stability_analysis=None)` 返回因子名、基础统计、IC 摘要、稳定性摘要、质量分与评级。
- 基础统计逐列给出样本数、均值、标准差、最小值、最大值、中位数与缺失数。
- IC 摘要从 `ic_analysis["ic_stats"]` 的中文字段 `IC均值`、`IC标准差`、`IR`、`IC>0占比`、`IC绝对值均值` 取值，数值统一经 `safe_float` 转换。
- 稳定性摘要按分布、时间序列、滚动三部分提取，时间序列部分保留平稳性布尔值与 p 值，p 值缺省按 1.0 处理。
- 质量分按 IC 40、IR 30、稳定性 20、IC 为正比例 10 计算，规则见 [[因子质量分级评分]]。
- `generate_multi_factor_summary` 汇总多个因子后按质量分降序给出排名列表。
- `generate_report_text` 输出纯文本报告，供前端或导出直接使用。
- 质量分只取 IC 摘要中的第一个因子，多因子场景需逐个生成摘要再汇总。
