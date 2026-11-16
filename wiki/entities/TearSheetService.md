---
标题: TearSheetService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/tear_sheet_service.py]
---

## 是什么

`backend/services/tear_sheet_service.py` 第 60 行的因子全貌报告服务，把分组收益、累计收益、换手与 IC/IR 汇总成一份带评分与建议的结构化报告。

## 关键事实

- 配置类 `TearSheetConfig` 的六个开关默认全开：分组收益、累计收益、换手率、Bootstrap、专业解读，另含评分配置。
- 评分权重固定为：IC 强度 25、IR 质量 20、单调性 20、稳定性 15、换手效率 10、显著性 10，合计 100。
- 主入口 `create_full_tear_sheet(factor_data, factor_name, price_column="close", return_series=None)` 返回 `{"success": ..., "tear_sheet": ...}`，报告含 `metadata`、`summary`、`sections`、`interpretation`、`recommendations`。
- 每个板块独立 `try/except`，单板块失败只记入 `warnings` 并计入完成率，整份报告不中断。
- 分组收益、累计收益、换手率三项委托 [[FactorReturnAnalysisService]]；IC/IR 摘要委托 [[AlphalensAnalysisService]]，按日频优先取一个周期。
- 评分公式：IC 强度 = `min(IR × 15 + (IC 为正比例 − 0.5) × 40, 100)`，IR 质量 = `min(IR × 25, 100)`，单调性 = 单调比例 × 100，显著性 = 基础分（显著 50 / 不显著 20）+ `min(多空利差 × 1000, 50)`，稳定性与换手效率取 0–100 截断。
- 等级划分：80 以上 A、65 以上 B、50 以上 C、35 以上 D，其余 F，属 [[评分归一与截断]] 的多维加权应用。
- 解读中出现 IR 与常量 `IR_PASS_THRESHOLD = 0.5`、换手与 `TURNOVER_THRESHOLD = 0.5` 的比较。
- 报告完成率以 4 个板块为分母，`sections_completed` 记录实际成功板块名。
