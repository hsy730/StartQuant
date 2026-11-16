---
标题: SmartPreprocessingDetector
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/smart_preprocessing_detector.py]
---

## 是什么

`backend/services/smart_preprocessing_detector.py` 中的智能预处理参数检测器，依据数据特征推荐预处理配置并给出置信度与理由。

## 关键事实

- 由两个数据描述对象构成：输入侧的数据特征快照与输出侧的推荐结果。
- 流程分两步：先刻画数据规模、板块、市值分布、因子分布形态、行业集中度与异常值情况，再据此产出配置、置信度、理由与告警。
- 板块由股票代码识别，板块类型决定去极值强度等参数的取值范围。
- 另提供面向展示的配置摘要文本，供前端智能模式使用；推荐结果最终交给 [[FactorPreprocessingPipeline]] 执行。
- 遵循 [[智能默认与手动覆盖]]，推荐只给默认值，不剥夺手动调整能力。
