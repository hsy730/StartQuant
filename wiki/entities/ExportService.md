---
标题: ExportService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/export_service.py]
---

## 是什么

`backend/services/export_service.py` 中的导出服务，把回测与对比结果写成多 Sheet 的 Excel 文件。

## 关键事实

- 类定义在该文件第 14 行，依赖 `pandas.ExcelWriter` 与 `openpyxl` 引擎。
- `export_backtest_to_excel(backtest_result, output_path, metrics=None, strategy_name="策略")` 依次写入 5 个 Sheet：摘要、净值曲线、持仓历史、交易记录、收益率分析。
- `_write_summary_sheet` 输出策略名称、导出时间（`%Y-%m-%d %H:%M:%S`）、回测参数（含 initial_capital）与指标。
- `export_comparison_to_excel(comparison_result, output_path)` 负责多方案对比的单独导出。
- 输出路径由调用方传入，服务不自行决定落盘位置。
- Sheet 结构约定见 [[多Sheet导出约定]]，回测数据来源见 [[VectorBTBacktestService]]。
