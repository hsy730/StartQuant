---
标题: FactorHub
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [README.md, .trae/rules/project_rules.md, start_api.py, start_all.py]
---

## 是什么

FactorHub（FactorFlow）是一个本地部署的量化因子研究平台，覆盖因子管理、智能挖掘、多维分析、回测验证与组合优化。

## 关键事实

- 前后端分离：后端 Python 加 FastAPI、SQLAlchemy 与 SQLite，前端 React 加 Ant Design 与 TypeScript。
- 计算栈以 pandas、numpy、scipy、empyrical、vectorbt 为主。
- 后端按路由、服务、模型、仓储、工具、策略分层，依赖方向单向向下。
- 提供仅启动后端与一键启动前后端两种脚本，日志落本地文件并按大小轮转。
- 核心计算统一走公共入口，如 [[FactorPreprocessingPipeline]]，避免各服务各自实现。
