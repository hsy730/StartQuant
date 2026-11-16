---
标题: FactorCalculator
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_service.py]
---

## 是什么

`backend/services/factor_service.py` 中的因子计算器，把因子表达式或因子定义转换成时间序列。

## 关键事实

- 计算前先做 AST 安全校验：只允许白名单节点类型，禁止访问以下划线开头的属性与危险名称。
- import 仅允许 pandas 与 numpy，禁止调用 exec、eval、compile、open、`__import__`、input。
- 内置 TALib 函数十四项：EMA、RSI、MACD、ADX、CCI、ATR、BBANDS、OBV、STOCH、STOCHRSI、WILLR、KAMA、ROC、MOM；SMA 由麦语言函数版本覆盖以支持命名参数。
- 提供麦语言兼容函数集合，覆盖 REF、HHV、LLV、CROSS、BARSLAST、TSRANK 等。
- 核心方法：`calculate(df, factor_code)` 返回单列序列，`calculate_multiple(df, factors)` 批量计算。
- `rolling_standardize(df, window=252)` 做滚动标准化，`add_time_features(df)` 追加时间特征。
- 与 [[FactorService]] 组合使用，安全校验理念见 [[公式白名单沙箱]]，除零保护走 [[安全除法]]。
