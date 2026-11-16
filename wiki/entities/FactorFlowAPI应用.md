---
标题: FactorFlowAPI应用
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/main.py, backend/api/routers/__init__.py]
---

## 是什么

`backend/api/main.py` 中创建的 FastAPI 应用实例，是 FactorHub 后端唯一 HTTP 入口，聚合八个路由模块。

## 关键事实

- 应用标题 `FactorFlow API`，版本 `1.0.0`，文档路径 `/docs` 与 `/redoc`，默认响应类为 JSONResponse。
- 生命周期启动时依次执行日志初始化、`init_db()`、加载预置因子、将中断挖掘任务标记为 aborted；关闭时调用 `shutdown_all_mining_tasks()`。
- 路由前缀固定：`/api/factors`、`/api/analysis`、`/api/mining`、`/api/portfolio`、`/api/backtest`、`/api/data`；编排器前缀为 `/api`；预处理路由自带 `/api/preprocessing`。
- CORS 允许 `localhost` 与 `127.0.0.1` 的 5173、3000、3001 端口，允许携带凭证与全部方法与请求头。
- 自定义 NumpyJSONEncoder 处理 numpy 整数、浮点与数组，无穷与 NaN 一律转为 null。
- 生产模式下挂载 `frontend/react-antd/dist/assets`，并对非 `/api`、`/docs`、`/redoc`、`/openapi.json` 的 404 请求回退 index.html。
- 全局异常处理器返回 500 与 `{success: false, message, detail}`，与 [[统一响应信封与错误映射]] 一致。
- 额外提供 `/api` 根信息与 `/health` 健康检查。
- 与 [[FactorHub]] 是同一平台的后端入口，前端经 [[ApiClient]] 访问。
