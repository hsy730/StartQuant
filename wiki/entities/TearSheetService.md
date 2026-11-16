---
标题: TearSheetService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/tear_sheet_service.py]
---

## 是什么

`backend/services/tear_sheet_service.py` 中的因子全貌报告服务，把分组收益、累计收益、换手与 IC/IR 汇总成带评分与建议的结构化报告。

## 关键事实

- 报告由元数据、摘要、各板块内容、专业解读与建议五部分组成，配置类控制各板块开关与评分规则。
- 各板块独立容错，单板块失败只记入告警并计入完成率，不中断整份报告。
- 分组收益、累计收益与换手委托 [[FactorReturnAnalysisService]]，IC/IR 摘要委托 [[AlphalensAnalysisService]]。
- 评分由 IC 强度、IR 质量、单调性、稳定性、换手效率与显著性六维加权成百分制，再映射为等级，属 [[评分归一与截断]] 的多维加权应用。
- 解读中的通过与否以 [[全局常量模块]] 的 IR 与换手阈值为准。
