---
标题: VisualizationService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/visualization_service.py]
---

## 是什么

`backend/services/visualization_service.py` 第 18 行的可视化服务，把因子分析结果渲染成 Plotly 图表对象，供前端直接展示。

## 关键事实

- 所有方法返回 `plotly.graph_objects.Figure`，不在服务内落盘图片，由调用方决定渲染或导出。
- `plot_factor_decay(factor_returns, factor_name="因子", n_periods=20)` 以累计收益曲线加 1.0 基准线呈现衰减，图高 500，模板 `plotly_white`。
- `plot_factor_correlation_network(correlation_matrix, threshold=0.5, width=800, height=600)` 只画相关系数绝对值超过阈值的边，networkx 缺失时自动回退为热图。
- 其余方法：`plot_factor_radar`（`max_score` 默认 1.0）、`plot_ic_distribution`、`plot_ic_time_series`（滚动窗口默认 20）、`plot_layered_returns`、`plot_turnover_analysis`、`plot_multi_factor_comparison`。
- 输入为纯数据结构（Series、字典、矩阵），服务本身不依赖数据库与缓存，便于复用与单测。
- 相关性矩阵可直接取 [[FactorCorrelationService]] 的输出，衰减数据可取自 [[StatisticsService]]。
- 分层收益图接收 `Q1` 到 `Q5` 的字典，与 [[横截面分层与单调性]] 的输出结构一致。
- 网络图仅用于展示，阈值筛选不改变原始相关性矩阵。
