---
标题: ApiClient
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/services/api.ts, frontend/react-antd/src/App.tsx]
---

## 是什么

`frontend/react-antd/src/services/api.ts` 导出的 axios 实例与方法集合，是前端页面访问后端的唯一通道。

## 关键事实

- 实例 `baseURL` 为 `/api`，默认超时 300000 毫秒，请求头 `Content-Type: application/json`。
- 响应拦截器直接返回 `response.data`，因此页面拿到的是后端统一信封，而非 axios 响应对象。
- 定义 `ApiResponse<T>` 类型，字段为 `success`、可选 `data`、可选 `message`，并允许附加字段。
- 错误拦截器按 400/401/403/404/500 映射中文提示，其余状态码回退为 `请求失败 (状态码)`。
- 方法覆盖七类：因子管理、因子分析、数据管理、组合分析、策略回测、数据预处理、因子挖掘。
- 因子分析覆盖 IC、因子值计算、暴露、有效性、归因、动态监测、相关性、前视偏差、分组收益、累计收益与 Tear Sheet。
- 挖掘相关方法包含提交、状态、结果、取消、活跃任务、历史列表、历史详情与删除，对应 [[MiningTaskRouter]]。
- 超时按接口耗时分级设置，具体分档见 [[按耗时分级设置请求超时]]。
- 请求拦截器预留 token 注入位置，当前未实现鉴权。
- 后端对接入口为 [[FactorFlowAPI应用]]。
