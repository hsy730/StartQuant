---
标题: FactorPrimitives
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_primitives.py]
---

## 是什么

`backend/services/factor_primitives.py` 中为 DEAP 遗传规划提供受保护算子、原语集工厂与表达式工具的工具模块。

## 关键事实

- 受保护算子包括 `safe_div`（分母近零置 NaN）、`safe_log`（非正置 NaN）、`safe_sqrt`（负数置 NaN）与横截面百分位排名 `pct_rank`。
- 基础原语集 9 个：`add`/`sub`/`mul`/`div`/`neg`/`abs`/`log`/`sqrt`/`rank`。
- 扩展原语集 +16：`ts_mean_5/10/20`、`ts_std_5/10/20`、`ts_delay_1/5`、`ts_delta_1/5`、`ts_corr_5/10/20`、`max`/`min`、`sigmoid`/`tanh`。
- 窗口相关系数使用 Spearman 而非 Pearson，与项目 IC 口径一致。
- 每个窗口算子都有 `_masked` 版本，接收可交易性掩码并按 `min_valid_ratio`（默认 0.6）放宽最小有效点。
- `create_pset` 三态构造：注入真实掩码、启用掩码但无掩码数据、传统无掩码；后两者分别打 info 与 warning 日志。
- `create_pset_numpy` 提供等价的 ndarray 版原语，消除 pandas 索引对齐开销。
- 表达式工具包括 `tree_to_expression`、`zobrist_hash`、`count_duplicate_subtrees`、`replace_duplicate_subtrees`、`compute_weighted_complexity`。
- 简化工具 `simplify_gp_expression` 与 `simplify_math_expression` 依赖 SymPy，未安装时原样返回输入。
- 用于 [[GeneticFactorMiningService]] 的个体构建与评估，并被 [[GFlowNetMiningService]] 复用于相似度计算。
- `ts_corr_np` 使用滑动窗口视图加向量化排名求 Pearson，注释标注相对 pandas 版本约 60 倍加速。
