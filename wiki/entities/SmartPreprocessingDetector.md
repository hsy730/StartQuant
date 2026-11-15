---
标题: SmartPreprocessingDetector
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/smart_preprocessing_detector.py]
---

## 是什么

`backend/services/smart_preprocessing_detector.py` 中的智能预处理参数检测器，依据数据特征推荐预处理配置并给出置信度。

## 关键事实

- 类定义在该文件第 66 行，数据特征由 `DataCharacteristics` 描述，推荐结果由 `PreprocessingRecommendation` 描述。
- `analyze_data(...)` 计算股票数、日期数、总样本、板块、市值分布、因子偏度与峰度、行业数量与最小行业样本量、异常值比例、是否肥尾、波动率是否时变。
- `recommend_config(...)` 输出 `config_dict`、`confidence`、`reasoning`、`data_characteristics`、`warnings`。
- `_detect_market_board` 按股票代码识别板块，板块决定去极值 n_sigma 的取值范围。
- `get_config_summary` 生成人类可读的配置摘要，供前端"智能模式"展示。
- 设计原则见 [[智能默认与手动覆盖]]，推荐结果最终交给 [[FactorPreprocessingPipeline]] 执行。
