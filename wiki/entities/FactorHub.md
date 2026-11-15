---
标题: FactorHub
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [README.md, .trae/rules/project_rules.md, start_api.py, start_all.py]
---

## 是什么

FactorHub（FactorFlow）是本地量化因子研究平台，覆盖因子管理、智能挖掘、多维分析、回测验证与组合优化。

## 关键事实

- 技术栈：后端 Python 3.12 + FastAPI + SQLAlchemy + SQLite；计算依赖 pandas、numpy、empyrical、scipy、vectorbt。
- 前端位于 `frontend/react-antd`，使用 React 18 + Ant Design 5 + TypeScript + Vite。
- 启动方式：`python start_api.py` 仅启动后端（默认 8000 端口）；`python start_all.py` 同时启动前后端并自动打开浏览器。
- 后端按层划分：`backend/api/routers`、`backend/services`、`backend/models`、`backend/repositories`、`backend/utils`、`backend/strategies`。
- 日志写入 `logs/factorhub.log`，单文件 10MB 轮转、保留 5 份备份。
- 核心计算走统一入口：[[FactorPreprocessingPipeline]]、[[WeightOptimizer]]、[[安全除法]]。
