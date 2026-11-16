---
标题: StockRankerService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/stock_ranker_service.py]
---

## 是什么

`backend/services/stock_ranker_service.py` 第 159 行的机器学习选股服务，用 XGBoost 排序模型训练、预测、回测并解释多因子排序结果。

## 关键事实

- 初始化时若未安装 xgboost 直接抛 `ImportError`；模型状态枚举为 `training`、`ready`、`deprecated`、`failed` 四种。
- `RankTrainingConfig` 默认：目标 `rank:ndcg`、学习率 0.05、最大深度 6、200 棵树、早停 20 轮、评估 `ndcg@5`、NDCG 截断 5、成对方式 `mean`、L1/L2 正则 0.1/1.0。
- `train(...)` 要求日期列与标签列存在（标签默认 `forward_return_5d`），标签缺失先剔除，清洗后少于 100 条即报错。
- 特征列默认取全部数值列，排除日期、标签、`stock_code`、`asset`、`index`；无可用数值特征时报错。
- 训练前默认执行因子预处理与未来函数检测，仅当风险等级为 `high` 或 `critical` 时记入告警，检测器为 [[LookaheadBiasDetector]]。
- 验证集比例为后 20%，切分点自动对齐日期组边界；缺失值用训练集中位数填充并写入元数据，训练与预测共用 `_fill_missing_features`，符合 [[训练期统计量防泄漏]]。
- 内存中最多保留 10 个模型，由线程安全的 `_LRUModelCache` 管理，未命中时从模型注册中心或文件回退加载。
- `predict(model_id, features, feature_cols=None, top_n=50)` 校验特征列一致后打分，按日期分组给出 `rank_position`，输出均值、标准差、极值、变异系数等预测质量指标。
- `predict_and_backtest(...)` 逐期预测、按 Top-N 构建权重并交给回测引擎，默认按月调仓。
- `explain_model(model_id, feature_sample=None, max_display=20)` 输出 gain 与 weight 两种特征重要性，传入样本时附加 SHAP 摘要。
- 训练完成后模型通过 `_save_model` 持久化，元数据含特征列、特征中位数与训练指标，供预测阶段复原口径。
