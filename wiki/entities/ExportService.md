---
标题: ExportService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/export_service.py]
---

## 是什么

`backend/services/export_service.py` 中的导出服务，把回测与多策略对比结果写成多 Sheet 的 Excel 文件。

## 关键事实

- 基于 pandas 的 ExcelWriter 与 openpyxl 引擎实现，输出路径由调用方传入，服务不自行决定落盘位置。
- 回测导出把结果拆成摘要、净值曲线、持仓历史、交易记录与收益率分析几个 Sheet。
- 摘要页包含策略名称、导出时间、回测参数与指标。
- 另提供多方案对比结果的独立导出入口。
- Sheet 结构约定见 [[多Sheet导出约定]]，回测数据来源见 [[VectorBTBacktestService]]。
