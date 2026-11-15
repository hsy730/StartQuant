---
标题: FactorOrchestrator
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_orchestrator_service.py]
---

## 是什么

`backend/services/factor_orchestrator_service.py` 中的因子编排器，把表达式因子送进一条固定顺序的验证流水线并输出结论与报告。

## 关键事实

- 类定义在该文件第 78 行，构造参数为 `OrchestratorConfig`；`PipelineStatus` 枚举与 `PipelineStageResult` 定义在第 29、42 行。
- `OrchestratorConfig` 默认开启阶段：未来函数检测、IC 分析、Alphalens、分组回测、Tear Sheet；`enable_shap_analysis=False` 默认关闭。
- 失败与告警配置：`fail_fast_on_bias=True`、`ic_threshold=0.02`。
- 输出控制：`include_raw_data=False`、`include_intermediate_results=True`。
- 主要入口 `validate(expression, stock_codes, start_date, end_date)`，另有 `batch_validate` 批量执行。
- 阶段实现为 `_stage_compute_factor`、`_stage_lookahead_detection`、`_stage_ic_analysis`、`_stage_alphalens_analysis`、`_stage_quantile_backtest`、`_stage_tear_sheet`、`_stage_shap_analysis`。
- 结果含 `status`（PASSED / REJECTED / PARTIAL）、`report_markdown` 与结构化报告；流水线规则见 [[一键验证流水线]]。
