---
标题: PreprocessingConfigAPI
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/preprocessing_api.py]
---

## 是什么

`backend/api/routers/preprocessing_api.py` 中的预处理参数推荐 REST 路由，提供推荐、预设列表与配置校验三类能力。

## 关键事实

- 请求模型必填股票列表、因子名与日期区间，并可选模式与用户自定义配置。
- 推荐流程先取原始因子值，再交给智能检测器出推荐，响应统一包裹成功标记与数据。
- 预设列表由内部函数给出可选模板，校验端点只做配置合法性判断。
- 端点之间互不隐式调用，推荐结果由调用方决定是否采用，规则见 [[智能默认与手动覆盖]]。
- 推荐算法本体见 [[SmartPreprocessingDetector]]，前端消费方见 [[PreprocessingConfigPanel]]。
