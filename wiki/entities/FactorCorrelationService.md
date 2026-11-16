---
标题: FactorCorrelationService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_correlation_service.py, backend/constants.py]
---

## 是什么

`backend/services/factor_correlation_service.py` 第 34 行的因子相关性分析服务，按八个步骤输出横截面相关性、时序相关性、滚动稳定性、显著性与 VIF。

## 关键事实

- `analyze(factor_panel, factor_cols, config=None)` 要求面板为 `date × asset` 的 MultiIndex，入口先 `copy()`，遵守不就地修改入参的约定。
- 默认配置：`rolling_window=120`、`rolling_step=20`、`use_knn=True`、`knn_neighbors=5`、`winsorize_method="mad"`、`n_sigma=3.0`、`max_missing_ratio=0.3`。
- 执行顺序固定：数据质量 → 预处理 → 横截面相关 → 时序相关 → 滚动稳定性 → 显著性检验 → VIF → 解读 → 建议；滚动稳定性要求日期数大于 `rolling_window × 2`。
- 预处理先检测各因子频率并统一对齐，缺失率不超过 0.3 的列用 KNN 插补，并校验插补前后均值变化是否超过 10%、标准差变化是否超过 15%。
- 去极值用 MAD 法，`scale="normal"` 等价于乘 1.4826；MAD 小于 `1e-10` 的近似常数列跳过。
- 相关性口径规则见 [[相关性口径分离]]，VIF 分档规则见 [[多重共线性VIF筛查]]，高相关阈值取自 `HIGH_CORRELATION_THRESHOLD=0.7`。
- 显著性检验对每日 Pearson 相关系数分别做 Fisher z 变换后再求均值与标准误，避免 `arctanh(mean(r))` 与 `mean(arctanh(r))` 混用。
- 另有 `calculate_rms_correlation` 输出 RMS 相关与分散化评分；`analyze_mixed_type` 在安装了 `phik` 时可处理混合类型列。
