---
标题: SmartSlippageDetector
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/smart_slippage_detector.py]
---

## 是什么

`backend/services/smart_slippage_detector.py` 中的智能滑点检测器，按市场特征给出滑点档位建议与成本敏感性分析。

## 关键事实

- 输入支持市场数据表或按股票代码组织的数据字典，输出市场特征与推荐结果。
- 市场特征包含板块分布、流动性等级与换手情况。
- 推荐支持 `conservative`、`aggressive` 与默认三档偏好。
- 未显式提供换手时默认年化换手 12 倍。
- 敏感性测试档位与分级规则见 [[滑点成本敏感性分级]]。
- `get_recommendation_summary` 生成人类可读的推荐报告。
- 加价分层规则见 [[滑点自适应推荐]]。
