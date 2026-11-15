---
标题: PreprocessingConfigAPI
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/preprocessing_api.py]
---

## 是什么

`backend/api/routers/preprocessing_api.py` 是预处理参数推荐的 REST 路由，提供推荐、预设列表与配置校验三个端点。

## 关键事实

- `PreprocessingConfigRequest` 必填 `stock_codes`、`factor_names`、`start_date`、`end_date`，另有 `mode`（默认 `smart`）与可选的 `user_config`。
- `POST /recommend` 返回 `SmartRecommendationResponse`，结构为 `{success, data}`；内部先用 `factor_service.calculate_factors_for_stocks(..., rolling_window=None)` 取原始因子，再交给 `smart_detector` 出推荐。
- `GET /presets` 由 `_get_available_presets()` 返回可选预设模板。
- `POST /validate` 接收配置字典并做合法性校验。
- 端点之间互不隐式调用，推荐结果由调用方决定是否采用，规则见 [[配置覆盖优先级]]。
- 推荐算法本体见 [[SmartPreprocessingDetector]]，前端消费方见 [[PreprocessingConfigPanel]]。
