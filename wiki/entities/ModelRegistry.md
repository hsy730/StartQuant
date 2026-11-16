---
标题: ModelRegistry
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/model_registry.py]
---

## 是什么

`backend/services/model_registry.py` 第 123 行的模型版本与生命周期管理中心，默认以文件系统存储任意可序列化的机器学习模型。

## 关键事实

- 支持五种框架：`xgboost`、`lightgbm`、`pytorch`、`sklearn`、`generic`；四种阶段：`development`、`staging`、`production`、`archived`。
- 默认存储目录为项目下 `models/registry`；初始化时从磁盘扫描 `*_metadata.json` 重建内存索引。
- `save()` 每次生成新版本，`model_id` 格式为 `{model_name}_v{version}`，同时写入 `v{n}.bin` 与 `v{n}_metadata.json`。
- 元数据记录框架、版本、阶段、标签、训练参数、特征列、指标、文件大小与 SHA256 校验和。
- 序列化按框架分派：XGBoost/LightGBM 用 `save_model`，sklearn 用 joblib，PyTorch 存 `state_dict`，通用模型用 pickle。
- `promote()` 提升到 `production` 时，同名其他生产模型自动降为 `staging`。
- 对外方法含 `save`、`load`、`list_models`、`get_version_history`、`promote`、`compare_models`、`get_statistics`、`delete`。
- 模块底部导出全局实例 `model_registry`，被 [[StockRankerService]] 用于模型持久化，流程位置见 [[排序模型训练与预测链路]]。
