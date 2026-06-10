# FactorHub 代码坏味道防范规范

> 基于 2026-06-07 代码审查中发现的系统性坏味道，提炼为团队编码规范。
> 每条规范都来自真实 Bug，违反即埋雷。

---

## 规范1：安全除法 — 禁止裸除法，统一使用 `safe_divide`

### 问题根因

项目中存在 14 处各自发明的除零保护模式，行为不一致：

```python
# 模式A：+1e-10 hack（产生极大IR值如5e8）
ir = ic_mean / (ic_std + 1e-10)

# 模式B：手动阈值判断（阈值不统一，有的1e-10有的0）
ir = ic_mean / ic_std if abs(ic_std) > 1e-10 and not np.isnan(ic_std) else 0.0

# 模式C：浮点噪声盲区（pd.Series([0.05]*20).std() ≈ 7e-18 ≠ 0）
ir = ic_mean / ic_std if ic_std != 0 else 0.0

# 模式D：完全无保护
ir = ic_mean / ic_std
```

### 规范

```python
from backend.utils.safe_math import safe_divide, safe_ir

# ✅ 正确：使用统一工具函数
ir = safe_ir(ic_mean, ic_std, default=0.0)       # IR计算
weight = safe_divide(numerator, denominator, default=0.0)  # 通用除法
cv = safe_divide(std, mean, default=None)          # 不可计算时返回None

# ❌ 错误：裸除法
ir = ic_mean / ic_std
ratio = a / b if b != 0 else 0.0
ratio = a / (b + 1e-10)  # hack
```

### 强制检查项

- [ ] 所有 `/` 运算符是否已替换为 `safe_divide`？
- [ ] 所有 IR 计算是否使用 `safe_ir`？
- [ ] 是否存在 `+ 1e-10` 或 `+ 1e-8` hack？
- [ ] `default` 参数是否符合规则6（不可计算→None，回退值→0.0）？

---

## 规范2：单一真相源 — 风险指标必须通过统一入口

### 问题根因

同一指标在不同模块有不同实现，行为漂移：

| 指标 | risk_metrics | base_strategy(旧) | vectorbt(旧) |
|------|-------------|-------------------|--------------|
| 空数据 | None | 0.0 | 0.0 |
| 回撤符号 | 负值 | — | 正值 |
| VaR/CVaR | empyrical | — | 手动分位数 |
| 波动率 | empyrical | — | std*sqrt(252) |

### 规范

```python
from backend.services.risk_metrics import calculate_risk_metrics, _empty_metrics

# ✅ 正确：所有风险指标通过统一入口
result = calculate_risk_metrics(returns, risk_free_rate=0.03)

# ❌ 错误：独立调用empyrical
import empyrical
sharpe = empyrical.sharpe_ratio(returns)  # 绕过统一入口

# ❌ 错误：手动计算
volatility = returns.std() * np.sqrt(252)  # 边界条件未处理
```

### 权重计算统一入口

```python
from backend.services.weight_optimizer_service import WeightOptimizer

# ✅ 正确：通过WeightOptimizer统一入口
optimizer = WeightOptimizer()
result = optimizer.calculate_weights(stock_data, factor_names, method="ic_weight")

# ❌ 错误：在API路由中内联实现权重计算
weights = {}
for factor in factor_names:
    ic = stock_data[factor].corr(returns)
    weights[factor] = abs(ic)
# ... 100行权重计算代码
```

### 强制检查项

- [ ] 新增的风险指标计算是否通过 `risk_metrics.py` 统一入口？
- [ ] 新增的权重计算是否通过 `WeightOptimizer` 统一入口？
- [ ] 是否存在绕过统一入口直接调用 empyrical/scipy 的代码？
- [ ] 空值/不可计算值的返回是否统一为 None（规则6）？

---

## 规范3：输入数据不可变 — 服务层入口必须 `.copy()`

### 问题根因

Python 引用语义导致传入的 DataFrame 被就地修改，调用方数据被意外污染：

```python
# 调用方
stock_data = data_service.get_stock_data("600036")
result = backtest_service.run(stock_data, factor_names)
# stock_data 现在被添加了 factor 列！下次调用数据已被污染
```

### 规范

```python
# ✅ 正确：服务层入口先复制
def analyze_effectiveness(self, factor_data, factor_names):
    factor_data = {k: v.copy() for k, v in factor_data.items()}
    # ... 后续操作安全

# ✅ 正确：DataFrame参数先复制
def cross_sectional_backtest(self, df, ...):
    df = df.copy()
    # ... 后续操作安全

# ❌ 错误：直接修改传入数据
def process(self, factor_data):
    for code, df in factor_data.items():
        df["future_return"] = ...  # 调用方数据被污染！
```

### 强制检查项

- [ ] 服务层方法是否在入口处对 DataFrame/dict 参数做了 `.copy()`？
- [ ] API 路由层是否对 `data_service` 返回的数据做了 `.copy()` 后再修改？
- [ ] 是否存在 `df["new_col"] = ...` 直接修改传入参数的代码？

---

## 规范4：日志纪律 — 禁止 print() 和静默异常

### 问题根因

- `print()` 在生产环境无法控制日志级别，无法被日志收集系统捕获
- `except Exception: pass` 吞掉异常，问题被隐藏，排查时无迹可寻

### 规范

```python
import logging
logger = logging.getLogger(__name__)

# ✅ 正确：使用logger，可控制级别
logger.info(f"开始因子分析: {len(stock_codes)}只股票")
logger.warning(f"IC计算失败: {e}")
logger.debug(f"权重归一化: sum={weight_sum:.4f}")

# ❌ 错误：使用print
print(f"Warning: {e}")
print(f"Processing {code}...")

# ✅ 正确：异常必须记录
try:
    result = compute_ic(factor, returns)
except Exception as e:
    logger.warning(f"IC计算失败: {e}")
    return default_result

# ❌ 错误：静默吞掉异常
try:
    result = compute_ic(factor, returns)
except Exception:
    pass  # 问题被隐藏！
```

### 日志级别选择指南

| 级别 | 使用场景 | 示例 |
|------|---------|------|
| `logger.error` | 影响核心功能的错误 | 数据库连接失败、API超时 |
| `logger.warning` | 可恢复的异常/降级 | IC计算失败回退等权、数据不足 |
| `logger.info` | 关键业务节点 | 开始分析、完成预处理、推荐配置 |
| `logger.debug` | 调试信息 | 中间计算结果、权重值 |

### 强制检查项

- [ ] 是否存在 `print(` 调用？（搜索 `print(` 排除测试文件）
- [ ] 是否存在 `except.*pass` 模式？（搜索 `except.*:\s*pass`）
- [ ] 模块是否定义了 `logger = logging.getLogger(__name__)`？
- [ ] 异常处理中是否有 `logger.warning` 或 `logger.error`？

---

## 规范5：代码复用 — 禁止复制粘贴，提取公共服务

### 问题根因

相同逻辑在多处重复实现，修改时容易遗漏：

| 重复代码 | 出现次数 | 后果 |
|---------|---------|------|
| 风险指标计算 | 3处 | 空值处理不一致 |
| 权重计算逻辑 | 2处(290行) | 方法支持不一致 |
| "找最长股票"遍历 | 4处 | 修改需同步4处 |

### 规范

```python
# ✅ 正确：提取公共方法
class FactorMonitoringService:
    def _find_longest_stock(self, factor_data, factor_name):
        """找到因子数据最长的股票"""
        longest_code = max(factor_data.keys(), key=lambda k: len(factor_data[k]))
        return longest_code, factor_data[longest_code]

    def _calculate_rolling_bands(self, ...):
        code, df = self._find_longest_stock(factor_data, factor_name)
        # ...

# ❌ 错误：复制粘贴
def _calculate_rolling_bands(self, ...):
    longest_code = None
    max_len = 0
    for code, df in factor_data.items():  # 第1次
        if len(df) > max_len:
            longest_code = code
            max_len = len(df)

def _calculate_transition_matrix(self, ...):
    longest_code = None
    max_len = 0
    for code, df in factor_data.items():  # 第2次，完全相同
        if len(df) > max_len:
            longest_code = code
            max_len = len(df)
```

### 提取判断标准

```
相同逻辑出现 ≥ 2 次？
├── 是 → 提取为公共方法/服务类
└── 否 → 逻辑相似但细节不同？
    ├── 是 → 提取为参数化方法
    └── 否 → 保持独立
```

### 强制检查项

- [ ] 新增的量化计算是否已有公共服务？（查 risk_metrics、WeightOptimizer、safe_math）
- [ ] 相同逻辑是否出现 ≥ 2 次？
- [ ] 新增服务方法是否与已有方法功能重叠？

---

## 规范6：禁止全局副作用 — warnings/配置必须局部化

### 问题根因

模块级 `warnings.filterwarnings("ignore", ...)` 是全局操作，影响所有线程：

```python
# 模块级（错误）— 影响整个进程
warnings.filterwarnings("ignore", ".*divide by zero.*")
warnings.filterwarnings("ignore", category=FutureWarning)
```

这会隐藏其他模块的真正 Bug，且在线程环境中不安全。

### 规范

```python
# ✅ 正确：使用上下文管理器局部抑制
from contextlib import contextmanager

@contextmanager
def _suppress_scipy_warnings():
    """局部抑制scipy兼容性警告（不影响其他模块）"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        yield

def t_test_ic(self, ic_series):
    with _suppress_scipy_warnings():
        result = scipy.stats.ttest_1samp(ic_series.dropna(), 0)
    return result

# ✅ 正确：单次调用局部抑制
def process_group(self, df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = df.groupby(...).apply(...)
    return result

# ❌ 错误：模块级全局抑制
warnings.filterwarnings("ignore", category=FutureWarning)  # 影响所有线程！
```

### 强制检查项

- [ ] 是否存在模块级 `warnings.filterwarnings`？
- [ ] 是否存在模块级 `warnings.simplefilter`？
- [ ] warnings 抑制是否使用 `with warnings.catch_warnings()` 包裹？
- [ ] 是否有其他模块级全局状态修改（如 `np.seterr`、`pd.set_option`）？

---

## 规范7：金融计算语义正确性 — 统计量定义必须符合业界标准

> 基于 2026-06-10 全面代码审查中发现的系统性金融计算语义错误，提炼为规范。
> 这些 Bug 不涉及语法或运行时崩溃，而是计算结果的语义与业界标准不一致，
> 比运行时崩溃更危险——因为结果"看起来合理"但实际错误。

### 规则7.1：IC 必须使用横截面 Spearman 秩相关，禁止池化 Pearson

**来源**：`enhanced_analysis_service.py` 池化 Spearman 相关、`weight_optimizer_service.py` IR 加权使用 Pearson、`ic_calculator.py` 滚动 Spearman 用全局排名、`factor_validation_service.py` IC验证使用 Pearson、`weight_optimizer_service.py` IC加权使用池化 Spearman

```python
# ❌ 错误：池化所有股票数据计算单个相关系数
all_factor = pd.concat([df["factor"] for df in factor_data.values()])
all_return = pd.concat([df["return"] for df in factor_data.values()])
ic = all_factor.corr(all_return, method="spearman")  # 违反独立性假设

# ❌ 错误：滚动 Spearman 用全局排名代替逐窗口排名
factor_rank = factor.rank()  # 全局排名
returns_rank = returns.rank()
ic = factor_rank.rolling(20).corr(returns_rank)  # 不等于逐窗口Spearman

# ❌ 错误：IR 加权用 Pearson 代替 Spearman
ic_series = factor.rolling(20).corr(returns)  # 这是 Pearson！

# ❌ 错误：IC验证使用 Pearson（factor_validation_service.py）
ic = aligned_data["factor"].corr(aligned_data["return"])  # 默认Pearson！

# ❌ 错误：IC加权使用池化 Spearman（weight_optimizer_service.py）
ic, _ = spearmanr(aligned_data['factor'], aligned_data['returns'])  # 池化！违反独立性

# ❌ 错误：IR验证使用滚动 Pearson IC（factor_validation_service.py）
rolling_ic = factor.rolling(20).corr(returns)  # Pearson IC → IR基于错误值

# ✅ 正确：横截面 IC — 每个时间截面上计算 Spearman 相关
ic_list = []
for date, group in data.groupby("date"):
    ic, _ = spearmanr(group["factor"], group["return"])
    ic_list.append(ic)
mean_ic = np.mean(ic_list)

# ✅ 正确：滚动 Spearman — 每个窗口内独立排名
def rolling_spearman(x, y_series):
    y_aligned = y_series.loc[x.index]
    valid = x.notna() & y_aligned.notna()
    if valid.sum() < 5:
        return np.nan
    return spearmanr(x[valid], y_aligned[valid])[0]

ic_series = factor.rolling(20).apply(
    lambda x: rolling_spearman(x, returns), raw=False
)
```

**强制检查项**：
- [ ] IC 计算是否为横截面（per-date）而非池化？
- [ ] Spearman IC 是否在每个窗口内独立排名？
- [ ] IR 加权是否使用 Spearman 而非 Pearson？
- [ ] IC 显著性检验是否基于横截面 IC 序列的 t 检验？

### 规则7.2：换手率必须基于横截面分位数变化，禁止用时序滚动排名

**来源**：`factor_validation_service.py` 使用 `rolling(252).rank(pct=True)` 计算换手率

```python
# ❌ 错误：时序滚动百分位排名 ≠ 换手率
factor_rank = factor_values.rolling(252).rank(pct=True)  # 时序排名
turnover = factor_rank.diff().abs().mean()  # 这不是换手率！

# ✅ 正确：横截面分位数换手率
n_bins = 5
factor_ranks = factor_values.rank(pct=True)
factor_bins = pd.cut(factor_ranks, bins=n_bins, labels=False)
rank_change = (factor_bins != factor_bins.shift(1)).astype(float)
turnover = rank_change.mean()  # 分位数变化的比例
```

**判断标准**：换手率衡量的是"因子排名在相邻期间发生变化的程度"，必须在横截面上定义分位数桶。

### 规则7.3：Fisher z 变换必须方法一致，禁止混用理论值和经验值

**来源**：`factor_correlation_service.py` 混用方法A的z值和方法B的标准误

```python
# ❌ 错误：z_val 来自 arctanh(avg_corr)（方法A），z_se 来自 std(daily_z)/sqrt(n)（方法B）
z_val = np.arctanh(avg_corr)           # 方法A：对平均相关做z变换
z_se = np.std(daily_z, ddof=1) / np.sqrt(n)  # 方法B：每日z值的标准误
p_value = 2 * (1 - norm.cdf(abs(z_val) / z_se))  # 混用！

# ✅ 正确：方法B一致 — z_val 也用每日z值的均值
z_val = np.mean(daily_z)               # 方法B：每日z值的均值
z_se = np.std(daily_z, ddof=1) / np.sqrt(n)  # 方法B：标准误
p_value = 2 * (1 - norm.cdf(abs(z_val) / z_se))
```

### 规则7.4：spearmanr 不处理 NaN，调用前必须清理数据

**来源**：`analysis_service.py` 传入含 NaN 的数据给 `spearmanr`，结果全为 NaN

```python
# ❌ 错误：spearmanr 默认 nan_policy='propagate'，含 NaN 返回 NaN
ic = spearmanr(factor_masked, return_masked)[0]  # 含NaN → 返回NaN

# ✅ 正确：先对齐并清理 NaN
valid = factor.notna() & returns.notna()
if valid.sum() >= min_periods:
    ic = spearmanr(factor[valid], returns[valid])[0]
else:
    ic = np.nan
```

**注意**：pandas 的 `.rolling().corr()` 内部会自动跳过 NaN 对，但 `scipy.stats.spearmanr` 不会。两者行为不一致，是常见的 Bug 来源。

### 规则7.5：组合方差可能因浮点精度为负，sqrt 前必须截断

**来源**：`portfolio_analysis_service.py` 协方差矩阵接近奇异时方差为微小负数

```python
# ❌ 错误：浮点精度导致方差略小于0
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
volatility = np.sqrt(portfolio_variance)  # 负数 → NaN

# ✅ 正确：截断到非负
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
volatility = np.sqrt(max(0.0, portfolio_variance))
```

### 规则7.6：零标准差阈值必须统一使用 `< 1e-10`，禁止 `== 0`

**来源**：`risk_metrics.py` 中 `calculate_sharpe` 用 `== 0`，`calculate_risk_metrics` 用 `< 1e-10`

```python
# ❌ 错误：== 0 无法捕获浮点噪声
if np.std(returns) == 0:  # pd.Series([0.05]*20).std() ≈ 7e-18 ≠ 0
    return None

# ✅ 正确：使用阈值
if np.std(returns) < 1e-10:  # 捕获近零浮点值
    return None
```

**根因**：浮点运算中，"几乎恒定"的序列标准差可能为 1e-17 而非精确的 0。`== 0` 漏检后，Sharpe 等指标会产生 1e8 级别的极端值。

### 规则7.7：因子收益率缺失值禁止填充为 0.0

**来源**：`portfolio_analysis_service.py` 用 `fillna(0.0)` 处理因子收益率

```python
# ❌ 错误：NaN → 0.0 扭曲优化结果
factor_returns = factor_returns.fillna(0.0)  # 缺失=无收益？不，缺失=无观测

# ✅ 正确：前向填充保持连续性，再删除剩余 NaN
factor_returns = factor_returns.fillna(method='ffill').dropna()
# 或直接删除
factor_returns = factor_returns.dropna()
```

**原则**：0.0 收益率 = "该因子当日无收益"，NaN = "该因子当日无观测"。两者含义完全不同，填充 0.0 会人为拉低均值并扭曲协方差。

### 规则7.8：衰减率/敏感度计算中，分母为零时 default 必须与语义一致

**来源**：`comprehensive_scoring_service.py` 中 `return_decay` 用 `default=0.0`，`sensitivity_ratio` 用 `default=float('inf')`

```python
# ❌ 错误：年化收益为0时衰减率=0%，实际应为100%或无穷
return_decay = safe_divide(annual_cost, annual_return, default=0.0) * 100

# ✅ 正确：年化收益为0时，成本完全吞噬收益，衰减为无穷
return_decay = safe_divide(annual_cost, annual_return, default=float('inf')) * 100
```

**原则**：`safe_divide` 的 `default` 参数必须反映业务语义——"不可计算"和"无穷大"是不同的概念。

### 规则7.9：MultiIndex 场景下 groupby+apply 禁止 droplevel，必须用 transform

**来源**：`return_calculator.py` 用 `groupby(level=1).apply(...).droplevel(0)` 丢失资产维度

```python
# ❌ 错误：droplevel 丢失资产维度，产生重复日期索引
prices.groupby(level=1).apply(
    lambda s: s.pct_change(period).shift(-period)
).droplevel(0)  # 丢失 asset 层！

# ✅ 正确：transform 保持原始索引结构
prices.groupby(level=1).transform(
    lambda s: s.pct_change(period).shift(-period)
)
```

**原则**：`groupby().apply()` 可能改变索引结构，`groupby().transform()` 保证输出与输入索引一致。

### 规则7.10：IC 标准差为零时 IR 不可计算，必须返回 None 而非 0.0

**来源**：`factor_validation_service.py` 中 IC 恒正但 IR=0.0，导致好因子被错误拒绝；`analysis_service.py` 中 `safe_ir(ic_mean, ic_std, default=0.0)` 导致同样问题

```python
# ❌ 错误：IC 恒正（如所有窗口IC=0.05）但 IR=0.0
if ic_std == 0:
    ir = 0.0  # 好因子被拒绝！

# ❌ 错误：safe_ir 使用 default=0.0（analysis_service.py）
ir = safe_ir(ic_mean, ic_std, default=0.0)  # IC_std≈0时返回0.0，好因子被拒！

# ✅ 正确：IR 不可计算时返回 None
if ic_std < 1e-10:
    ir = None  # 不可计算，不是0

# ✅ 正确：safe_ir 使用 default=None
ir = safe_ir(ic_mean, ic_std, default=None)  # 不可计算→None，让下游判断
```

**原则**：IR = IC_mean / IC_std。当 std→0 且 mean≠0 时，IR→∞（因子极其稳定），而非 IR=0（因子无效）。返回 None 让下游逻辑自行判断。

### 规则7.11：滚动 IR 计算必须使用 safe_divide(default=None)，禁止 default=np.nan

**来源**：`analysis_service.py` 中 `_calculate_rolling_ir` 使用 `safe_divide(rolling_mean, rolling_std, default=np.nan)`

```python
# ❌ 错误：default=np.nan，与项目规范不一致
ir = safe_divide(rolling_mean, rolling_std, default=np.nan)
# NaN 在数值运算中传播，可能导致下游静默出错

# ✅ 正确：default=None（在pandas Series中表现为NaN，但语义明确）
ir = safe_divide(rolling_mean, rolling_std, default=None)
# None 在 Series 中存储为 NaN，但通过 sanitize_dict 序列化时转为 JSON null

# ✅ 正确：IC_mean和IC_std都接近0时，IR=0是合理的（无信号无波动）
both_near_zero = (rolling_mean.abs() < 1e-10) & (rolling_std.abs() < 1e-10)
ir = ir.mask(both_near_zero, 0.0)
```

**原则**：`default=None` 是项目规范（规则6）的统一标准，表示"不可计算"。`default=np.nan` 虽然在 Series 中表现相同，但语义不明确，且与 `sanitize_dict` 的 NaN→None 转换逻辑不一致。

### 规则7.12：IC 验证和 IR 验证必须使用 Spearman，禁止 Pearson

**来源**：`factor_validation_service.py` 中 `_validate_ic` 使用 Pearson、`_validate_ir` 使用滚动 Pearson IC

```python
# ❌ 错误：IC验证使用 Pearson（对非线性单调关系和异常值不敏感）
ic = aligned_data["factor"].corr(aligned_data["return"])  # 默认Pearson

# ❌ 错误：IR验证使用滚动 Pearson IC
rolling_ic = factor.rolling(20).corr(returns)  # Pearson IC → IR基于错误值

# ✅ 正确：IC验证使用 Spearman
from scipy.stats import spearmanr
ic, _ = spearmanr(aligned_data["factor"], aligned_data["return"])

# ✅ 正确：IR验证使用滚动 Spearman IC
def _rolling_spearman_ic(x):
    y_aligned = returns.loc[x.index]
    valid = x.notna() & y_aligned.notna()
    if valid.sum() < min_periods:
        return np.nan
    return spearmanr(x[valid], y_aligned[valid])[0]

rolling_ic = factor.rolling(20).apply(_rolling_spearman_ic, raw=False)
```

**原则**：因子验证是因子质量把关的第一道防线。Pearson 对非线性单调关系不敏感，可能遗漏有效的因子信号。所有 IC/IR 验证必须使用 Spearman 秩相关，与业界标准一致。

### 规则7.13：IC 加权必须使用横截面 IC，禁止池化 Spearman

**来源**：`weight_optimizer_service.py` 中 `_ic_weight` 使用 `spearmanr(aligned_data['factor'], aligned_data['returns'])` 池化计算

```python
# ❌ 错误：池化 Spearman — 将所有时间点数据混合计算单个相关系数
ic, _ = spearmanr(aligned_data['factor'], aligned_data['returns'])
# 违反独立性假设：同一日不同股票观测值并不独立

# ✅ 正确：MultiIndex 数据按日期截面计算IC后取均值
if isinstance(aligned_data.index, pd.MultiIndex):
    daily_ics = []
    for date, group in aligned_data.groupby(level=0):
        if len(group) >= 5:
            ic_val, _ = spearmanr(group['factor'], group['returns'])
            if not np.isnan(ic_val):
                daily_ics.append(ic_val)
    ic = float(np.mean(daily_ics))

# ✅ 正确：单股票时序使用滚动Spearman IC均值
from backend.utils.ic_calculator import calculate_rolling_ic
rolling_ic = calculate_rolling_ic(factor, returns, window=20, method='spearman')
ic = float(rolling_ic.dropna().mean())
```

**原则**：IC 加权直接影响多因子组合的权重分配。池化相关违反独立性假设，可能产生虚假的高/低 IC 值，导致权重偏离真实因子预测能力。

### 规则7.14：IC 计算器默认方法必须是 Spearman，禁止 Pearson

**来源**：`ic_calculator.py` 中 `calculate_ic` 和 `calculate_rolling_ic` 默认 `method="pearson"`

```python
# ❌ 错误：默认 Pearson，调用方省略 method 参数时静默使用错误方法
def calculate_ic(factor, returns, method: str = "pearson", ...):
    ...

# ✅ 正确：默认 Spearman，与项目规范和业界标准一致
def calculate_ic(factor, returns, method: str = "spearman", ...):
    ...
```

**原则**：IC 计算是因子分析的基础设施。默认值决定了大多数调用方的行为，必须与项目规范（规则7.1/7.12）一致。任何需要 Pearson IC 的场景应显式指定 `method="pearson"`。

### 规则7.15：IC 标准差为零但均值非零时，t 统计量应为无穷大，禁止设为 0

**来源**：`analysis_service.py`、`alphalens_analysis_service.py` 中 ic_std≈0 时 t_stat=0, p_value=1.0

```python
# ❌ 错误：IC 恒正（如所有截面IC=0.05）但 t_stat=0, p_value=1.0 → 因子被判为不显著
if ic_std <= 0:
    t_stat = 0.0
    p_value = 1.0

# ✅ 正确：IC_std≈0 且 IC_mean≠0 → 因子极其稳定，高度显著
if ic_std < 1e-10:
    if abs(ic_mean) > 1e-10:
        t_stat = float('inf')
        p_value = 0.0
    else:
        t_stat = 0.0
        p_value = 1.0
```

**原则**：t = mean / (std / sqrt(n))。当 std→0 且 mean≠0 时，t→∞（因子信号极其稳定）。设为 0 恰好得出相反结论——因子无效。

### 规则7.16：Welch's t 检验公式禁止混入交叉项

**来源**：`factor_return_analysis_service.py` 中多空价差 t 统计量公式错误

```python
# ❌ 错误：分母包含交叉项 std_top²/n_bottom + std_bottom²/n_top
spread_std = np.sqrt(std_top**2 + std_bottom**2)
t_stat = spread / (spread_std * np.sqrt(1/n_top + 1/n_bottom))
# 展开 = spread / sqrt(std_top²/n_top + std_top²/n_bottom + std_bottom²/n_top + std_bottom²/n_bottom)

# ✅ 正确：Welch's t 检验
se = np.sqrt(std_top**2 / n_top + std_bottom**2 / n_bottom)
t_stat = spread / se if se > 0 else 0.0
```

**原则**：Welch's t 检验的标准误只包含各组方差除以各自样本量，不含交叉项。交叉项使分母系统性偏大，导致 t 统计量偏小，检验过于保守（假阴性）。

### 规则7.17：因子归因的高/低暴露收益必须横截面计算，禁止池化分位数

**来源**：`factor_attribution_service.py` 中高/低暴露组收益使用全局分位数阈值

```python
# ❌ 错误：池化所有股票-日期观测值，用全局分位数阈值分组
all_factor_values = pd.Series(...)  # 所有股票所有日期
high_threshold = all_factor_values.quantile(0.7)
high_return = returns[all_factor_values >= high_threshold].mean()

# ✅ 正确：每个日期独立计算横截面分位数阈值
daily_high_returns = []
for date, group in data.groupby("date"):
    if len(group) < 5:
        continue
    high_threshold = group["factor"].quantile(0.7)
    high_ret = group.loc[group["factor"] >= high_threshold, "return"].mean()
    daily_high_returns.append(high_ret)
high_return = np.mean(daily_high_returns)
```

**原则**：与 IC 计算同理（规则7.1），池化分组违反横截面独立性假设。一只持续高因子值的股票会主导高暴露组，使"高暴露收益"退化为该股票的平均收益。

### 规则7.18：R² 在因变量方差为零时不可计算，禁止返回 1.0

**来源**：`factor_attribution_service.py` 中 ss_tot=0 时 R²=1.0

```python
# ❌ 错误：ss_tot=0 → safe_divide(ss_res, ss_tot, default=0.0) → R²=1.0-0.0=1.0
r_squared = 1.0 - safe_divide(float(ss_res), float(ss_tot), default=0.0)

# ✅ 正确：因变量恒定时 R² 无定义
if ss_tot < 1e-10:
    r_squared = None
else:
    r_squared = 1.0 - safe_divide(float(ss_res), float(ss_tot), default=None)
```

**原则**：R²=1.0 意味着"模型完美解释数据"，但当因变量方差为零时，R² 无定义。返回 1.0 误导用户认为模型拟合完美。

### 规则7.19：pyportfolioopt 输入必须尺度无关，禁止直接对原始因子值 diff()

**来源**：`weight_optimizer_service.py` 中 `_max_sharpe`/`_min_variance`/`_risk_parity` 对原始因子值 diff()

```python
# ❌ 错误：原始因子值 diff() 不具尺度不变性
factor_returns = factor_df[factor_names].diff().dropna()
# 因子A范围[0,1000] → diff()量级~10，因子B范围[-3,3] → diff()量级~0.1
# 优化被因子A主导

# ✅ 正确：标准化后再 diff()
factor_standardized = factor_df[factor_names].apply(
    lambda x: (x - x.mean()) / x.std() if x.std() > 1e-10 else x - x.mean()
)
factor_returns = factor_standardized.diff().dropna()
```

**原则**：pyportfolioopt 的均值-方差优化对输入尺度敏感。不同因子的绝对值范围可能差几个数量级，直接 diff() 会使优化被高量级因子主导，与因子的实际预测能力无关。

### 规则7.20：累计收益必须复利计算，禁止存储期间收益并标记为"累计"

**来源**：`factor_return_analysis_service.py` 中 `group_cumreturns` 存储期间收益但标签为"累计"

```python
# ❌ 错误：存储期间收益但变量名和键名暗示累计
group_cumreturns[f"Q{q+1}"].append(group_returns.get(q, 0.0))  # 这是期间收益！

# ✅ 正确：复利累计
group_cumulative = {q: 1.0 for q in range(n_quantiles)}
for q in range(n_quantiles):
    group_cumulative[q] *= (1 + group_returns.get(q, 0.0))
    group_cumreturns[f"Q{q+1}"].append(group_cumulative[q] - 1)
```

**原则**：期间收益和累计收益是两个完全不同的概念。期间收益可直接加减，累计收益必须复利。前端图表如果按累计收益绘制，使用期间收益会产生错误的曲线。

### 规则7.21：多期收益率的 Sharpe 年化必须调整频率，禁止直接 sqrt(252)

**来源**：`factor_return_analysis_service.py` 中 forward_period>1 时 Sharpe 年化错误

```python
# ❌ 错误：5日收益率的 Sharpe 直接乘 sqrt(252)，高估约 2.24 倍
sharpe = calculate_sharpe(returns_5day)  # 内部 * sqrt(252)

# ✅ 正确：先将多期收益率转为日频等价
daily_equivalent = (1 + period_returns) ** (1 / forward_period) - 1
sharpe = calculate_sharpe(daily_equivalent)
```

**原则**：Sharpe 年化因子 = sqrt(252) 仅适用于日频收益率。5日收益率的波动率已是5日尺度，直接年化会高估 Sharpe 约 sqrt(forward_period) 倍。

### 规则7.22：Duplicate DatetimeIndex 下 .loc 会膨胀样本量，必须用 .iloc

**来源**：`enhanced_analysis_service.py` 中合并多股票 DataFrame 后 .loc 查找膨胀

```python
# ❌ 错误：合并后的 DatetimeIndex 有重复日期，.loc 返回所有匹配行
for date, group_idx in date_series.groupby(date_series):
    factor_group = factor_values.loc[group_idx.index]  # 3只股票×3次匹配=9行

# ✅ 正确：用整数索引避免重复问题
factor_values_reset = factor_values.reset_index(drop=True)
return_values_reset = return_values.reset_index(drop=True)
date_series_reset = date_series.reset_index(drop=True)
for date, group_idx in date_series_reset.groupby(date_series_reset):
    factor_group = factor_values_reset.iloc[group_idx.index]
    return_group = return_values_reset.iloc[group_idx.index]
```

**原则**：`pd.concat([df1, df2])` 不加 `ignore_index=True` 时，DatetimeIndex 会包含重复值。`.loc[duplicate_dates]` 返回所有匹配行，导致样本量膨胀 N 倍（N=股票数）。

### 规则7.23：评分函数必须有上下界截断，禁止负分

**来源**：`comprehensive_scoring_service.py` 中 Sharpe 评分无下限截断

```python
# ❌ 错误：负 Sharpe 产生负分，拖累总分
sharpe_score = min(sharpe_ratio / 2.0 * 100, 100)  # sharpe=-1 → score=-50

# ✅ 正确：截断到 [0, 100]
sharpe_score = max(min(sharpe_ratio / 2.0 * 100, 100), 0)
```

**原则**：评分系统的每个维度分数应在 [0, 100] 范围内。负分会使总分低于预期范围，误导用户对整体质量的判断。

### 规则7.24：API 方法签名必须与实际参数匹配，禁止静默失败

**来源**：`stock_ranker_service.py` 中 `process_single_factor` 调用传入不存在的 `factor_name` 参数

```python
# ❌ 错误：factor_name 不在方法签名中，TypeError 被 except 吞掉，预处理静默失败
result = pipeline.process_single_factor(df[feat], factor_name=feat)

# ✅ 正确：匹配实际签名，正确解包返回值
processed_series, stats = pipeline.process_single_factor(df[feat])
df[feat] = processed_series.values
```

**原则**：Python 的 `**kwargs` 和 broad `except Exception` 组合会导致参数错误被静默吞掉。调用外部 API 时必须核对方法签名，返回值类型也必须匹配（单个值 vs 元组）。

### 规则7.25：前视偏差检测器自身必须使用 Spearman IC，禁止 Pearson

**来源**：`lookahead_bias_detector.py` 中多个检测方法使用 Pearson IC

```python
# ❌ 错误：前视偏差检测用 Pearson，对非线性单调关系不敏感
ic = aligned["f"].corr(aligned["r"])  # Pearson
rolling_ic = aligned["f"].rolling(20).corr(aligned["r"])  # Pearson

# ✅ 正确：使用 Spearman 检测前视偏差
from scipy.stats import spearmanr
ic, _ = spearmanr(aligned["f"], aligned["r"])
# 滚动 Spearman
rolling_ic = factor.rolling(20).apply(
    lambda x: spearmanr(x, returns.loc[x.index].dropna())[0]
    if (x.notna() & returns.loc[x.index].notna()).sum() >= 5 else np.nan,
    raw=False
)
```

**原则**：前视偏差可能表现为非线性单调关系（如因子值排名与未来收益排名高度一致但非线性相关）。Pearson 对此不敏感，可能漏检。检测器自身必须使用比被检测对象更严格的标准。

### 规则7.26：`or 0.0` 模式将 None 转为 0.0，违反规则6

**来源**：`factor_attribution_service.py` 中 `calculate_sharpe(...) or 0.0`

```python
# ❌ 错误：or 0.0 将 None（不可计算）转为 0.0（零值）
"sharpe": calculate_sharpe(returns) or 0.0,
"volatility": calculate_volatility(returns) or 0.0,

# ✅ 正确：让 None 传播，由 sanitize_dict 统一处理
"sharpe": calculate_sharpe(returns),
"volatility": calculate_volatility(returns),
```

**原则**：`None or 0.0` = `0.0`，`0.0 or 0.0` = `0.0`。这两种情况无法区分。正确做法是让 None 传播到序列化层，由 `sanitize_dict` 统一转为 JSON null。

### 规则7.27：零标准差时必须返回可计算指标，禁止全部返回 None

**来源**：`risk_metrics.py` 中 `np.std(returns) < 1e-10` 时返回 `_empty_metrics()`，导致 `total_return`、`annual_return`、`win_rate` 等不依赖标准差的指标也被设为 None

```python
# ❌ 错误：零标准差时所有指标都返回 None，包括不依赖标准差的指标
if np.std(returns_arr) < 1e-10:
    return _empty_metrics()  # total_return=None, annual_return=None, win_rate=None

# ✅ 正确：只将依赖波动率的指标设为 None，可计算的指标正常返回
if np.std(returns_arr) < 1e-10:
    return {
        "total_return": float(empyrical.cum_returns_final(returns_arr)),
        "annual_return": float(empyrical.annual_return(returns_arr)),
        "volatility": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "max_drawdown": None,
        "calmar_ratio": None,
        "win_rate": float((returns_arr > 0).mean()),
        "var_95": None,
        "cvar_95": None,
    }
```

**原则**：`total_return`、`annual_return`、`win_rate` 不依赖标准差，即使波动率为零也可计算。全部返回 None 会使稳定盈利策略看起来毫无收益。

### 规则7.28：Bootstrap 重采样必须保持横截面结构，禁止逐行独立重采样

**来源**：`factor_return_analysis_service.py` 中 `_bootstrap_quantile_returns` 使用 `df.sample(replace=True)` 逐行重采样

```python
# ❌ 错误：逐行独立重采样破坏横截面结构
sample_df = df.sample(n=len(df), replace=True)
# 某些日期被过度代表，某些日期缺失，不再构成有效横截面

# ✅ 正确：按日期聚类重采样，同一日期所有股票观测值保持在一起
unique_dates = df.index.unique()
sampled_dates = np.random.choice(unique_dates, size=len(unique_dates), replace=True)
sample_df = pd.concat([df.loc[d] for d in sampled_dates])
```

**原则**：横截面数据分析中，同一日期的不同股票观测值不独立。Bootstrap 必须以日期为聚类单位，否则重采样后的数据不再代表任何有效的横截面，置信区间统计无效。

### 规则7.29：Alpha/Beta 分解必须通过 `calculate_relative_metrics` 统一入口

**来源**：`factor_attribution_service.py` 中手动用 `np.cov` 计算 alpha/beta，绕过统一入口

```python
# ❌ 错误：手动计算 alpha/beta，绕过 empyrical 和统一入口
cov_matrix = np.cov(y, X.flatten())
beta = safe_divide(cov_matrix[0, 1], cov_matrix[1, 1], default=None)
alpha = y.mean() - beta * X.mean()

# ✅ 正确：通过 risk_metrics.py 统一入口，底层委托 empyrical
from backend.services.risk_metrics import calculate_relative_metrics
relative = calculate_relative_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.03)
alpha_annual = relative.get("alpha")
beta = relative.get("beta")
```

**原则**：与规则2一致，风险/收益指标必须通过统一入口。手动计算 alpha/beta 与 empyrical 的年化方式不一致，且边界条件处理不同。

### 规则7.30：滚动 Spearman IC 必须使用 `ic_calculator.calculate_rolling_ic`，禁止各服务自行实现

**来源**：`lookahead_bias_detector.py`、`analysis_service.py` 等多处自行实现滚动 Spearman IC

```python
# ❌ 错误：每个服务自行实现滚动 Spearman
def _rolling_spearman(x):
    y_aligned = returns.loc[x.index]
    valid = x.notna() & y_aligned.notna()
    if valid.sum() < 5:
        return np.nan
    return spearmanr(x[valid], y_aligned[valid])[0]

rolling_ic = factor.rolling(20).apply(_rolling_spearman, raw=False)

# ✅ 正确：使用统一工具函数
from backend.utils.ic_calculator import calculate_rolling_ic
rolling_ic = calculate_rolling_ic(factor, returns, window=20, method='spearman')
```

**原则**：与规则5一致，相同逻辑出现 ≥ 2 次必须提取为公共方法。`ic_calculator.calculate_rolling_ic` 已处理 NaN 清理、最小样本数等边界条件，各服务不应重复实现。

### 规则7.31：API 路由层 IC 计算必须与服务层一致，使用 Spearman

**来源**：`analysis.py`、`portfolio.py` 中 API 路由直接使用 `.corr()` 和 `.rolling().corr()` 计算 Pearson IC

```python
# ❌ 错误：API路由层绕过服务层，直接用 Pearson IC
ic = factor_series.rolling(20).corr(future_returns)  # Pearson
portfolio_ic = aligned_factor.corr(aligned_returns)   # Pearson

# ✅ 正确：使用统一的 IC 计算工具
from backend.utils.ic_calculator import calculate_rolling_ic
ic_series = calculate_rolling_ic(factor, returns, window=20, method='spearman')

from scipy.stats import spearmanr
portfolio_ic, _ = spearmanr(aligned_factor, aligned_returns)
```

**原则**：服务层的 IC 计算规范必须同样适用于 API 路由层。API 路由中任何内联的 IC 计算都应使用 `ic_calculator` 或 `spearmanr`，而非 `.corr()`（默认 Pearson）。

### 规则7.32：年化收益必须使用几何复利，禁止算术平均复利

**来源**：`factor_attribution_service.py` 中 `(1 + daily_mean) ** 252 - 1` 系统性高估收益

```python
# ❌ 错误：算术平均复利，由 Jensen 不等式系统性高估
annual_return = (1 + returns.mean()) ** 252 - 1
# 日收益交替+2%/-2%：mean=0%→年化0%，实际年化-4.8%

# ✅ 正确：几何复利（empyrical 标准实现）
annual_return = float(empyrical.annual_return(returns, period='daily'))
```

**原则**：`E[(1+r)^252] > (1+E[r])^252`（Jensen 不等式）。算术平均复利系统性高估年化收益，高估幅度随波动率增大而增大。必须使用 empyrical 的几何复利实现。

### 规则7.33：IC 加权必须使用股票收益率，禁止使用因子收益率均值

**来源**：`portfolio_analysis_service.py` 中 IC 加权传入 `factor_returns.mean(axis=1)` 作为收益率

```python
# ❌ 错误：IC 衡量因子与因子均值的相关性，无预测意义
combined_returns = factor_returns.mean(axis=1)
result = optimizer.calculate_weights(stock_data, factor_names, method="ic_weight", returns=combined_returns)

# ✅ 正确：IC 衡量因子对股票收益率的预测能力
result = optimizer.calculate_weights(stock_data, factor_names, method="ic_weight", returns=stock_returns)
# 当缺少 stock_returns 时，回退到等权并记录警告
```

**原则**：IC 的定义是因子值与未来收益率的相关性。因子收益率的均值不是股票收益率，用其计算 IC 衡量的是"因子与所有因子平均的相关性"，无预测意义。

### 规则7.34：`safe_divide` 在标准误（se）小于 min_threshold 时会吞掉有效的 t 统计量

**来源**：`returns.py` 中 `safe_divide(mean_ic, se, default=0.0)` 当 se < 1e-10 时返回 0.0

```python
# ❌ 错误：se < 1e-10 时 safe_divide 返回 default=0.0，吞掉极大 t 统计量
se = std_ic / np.sqrt(n)  # std_ic=1.5e-10, n=100 → se=1.5e-11
t_statistic = safe_divide(mean_ic, se, default=0.0)  # 返回 0.0！

# ✅ 正确：se 保证为正（std_ic >= 1e-10 且 n >= 2），直接除法
t_statistic = float(mean_ic) / float(se)
```

**原则**：`safe_divide` 的 `min_threshold=1e-10` 是为"分母接近零表示不可计算"设计的，但标准误 `se = std/sqrt(n)` 可以合法地小于 1e-10（当 std 刚过阈值且 n 较大时）。此时 t 统计量极大（因子高度显著），不应被吞掉。

### 规则7.35：变异系数（CV）分母必须使用 `abs(mean)`，禁止直接用 `mean`

**来源**：`factor_stability_service.py` 中 CV 计算产生负值

```python
# ❌ 错误：mean 为负时 CV 为负，语义无意义
cv = safe_divide(std, mean, default=None)  # mean=-0.03, std=0.02 → cv=-0.667

# ✅ 正确：CV 衡量相对离散度，分母取绝对值
cv = safe_divide(std, abs(mean), default=None)  # cv=0.667
```

**原则**：CV = std/|mean| 衡量的是"标准差相对于均值的比例"，与均值符号无关。负均值不意味着低离散度，负 CV 无语义。

---

## 代码审查 Checklist

提交 PR 时，reviewer 必须确认以下各项：

### 除法安全
- [ ] 所有 `/` 运算符是否使用了 `safe_divide`？
- [ ] 所有 IR 计算是否使用了 `safe_ir`？
- [ ] 是否存在 `+ 1e-10` hack？
- [ ] `default` 参数是否符合语义（不可计算→None，无穷→float('inf')）？

### 统一入口
- [ ] 风险指标是否通过 `risk_metrics.py` 统一入口？
- [ ] 权重计算是否通过 `WeightOptimizer` 统一入口？
- [ ] 空值/不可计算值是否返回 None（规则6）？

### 数据安全
- [ ] 服务层方法是否对传入参数做了 `.copy()`？
- [ ] 是否存在直接修改传入 DataFrame 的代码？
- [ ] 缓存返回的数据是否做了 `.copy()`？

### 日志规范
- [ ] 是否存在 `print()` 调用？
- [ ] 是否存在 `except.*pass` 模式？
- [ ] 异常处理是否有日志记录？

### 代码复用
- [ ] 相同逻辑是否出现 ≥ 2 次？
- [ ] 新增计算是否复用了已有公共服务？

### 全局副作用
- [ ] 是否存在模块级 `warnings.filterwarnings`？
- [ ] warnings 抑制是否局部化？

### 金融计算语义
- [ ] IC 计算是否为横截面 Spearman，而非池化 Pearson？
- [ ] 滚动 Spearman 是否在每个窗口内独立排名？
- [ ] 换手率是否基于横截面分位数变化？
- [ ] Fisher z 变换的方法是否一致（理论值 vs 经验值不混用）？
- [ ] `spearmanr` 调用前是否已清理 NaN？
- [ ] 组合方差 sqrt 前是否截断到非负？
- [ ] 零标准差检查是否统一使用 `< 1e-10`？
- [ ] 因子收益率缺失值是否禁止填充为 0.0？
- [ ] MultiIndex 场景是否使用 `transform` 而非 `apply+droplevel`？
- [ ] IR 不可计算时是否返回 None 而非 0.0？
- [ ] IC 验证是否使用 Spearman 而非 Pearson？（规则7.12）
- [ ] IR 验证是否使用滚动 Spearman IC 而非 Pearson？（规则7.12）
- [ ] IC 加权是否使用横截面 IC 而非池化 Spearman？（规则7.13）
- [ ] `safe_ir` 的 `default` 是否为 None 而非 0.0？（规则7.10）
- [ ] 滚动 IR 是否使用 `safe_divide(default=None)` 而非 `default=np.nan`？（规则7.11）
- [ ] IC 计算器默认方法是否为 Spearman？（规则7.14）
- [ ] IC_std≈0 且 IC_mean≠0 时 t_stat 是否为 inf？（规则7.15）
- [ ] Welch's t 检验是否不含交叉项？（规则7.16）
- [ ] 因子归因暴露收益是否横截面计算？（规则7.17）
- [ ] R² 在 ss_tot=0 时是否返回 None？（规则7.18）
- [ ] pyportfolioopt 输入是否尺度无关？（规则7.19）
- [ ] 累计收益是否复利计算？（规则7.20）
- [ ] 多期收益率 Sharpe 是否调整年化频率？（规则7.21）
- [ ] Duplicate DatetimeIndex 下是否用 .iloc？（规则7.22）
- [ ] 评分函数是否有 [0, 100] 截断？（规则7.23）
- [ ] API 调用签名是否与实际匹配？（规则7.24）
- [ ] 前视偏差检测器是否使用 Spearman IC？（规则7.25）
- [ ] 是否存在 `or 0.0` 将 None 转为 0.0？（规则7.26）
- [ ] 零标准差时是否仍返回可计算指标（total_return, annual_return, win_rate）？（规则7.27）
- [ ] Bootstrap 重采样是否保持横截面结构（按日期聚类）？（规则7.28）
- [ ] Alpha/Beta 分解是否通过 `calculate_relative_metrics` 统一入口？（规则7.29）
- [ ] 滚动 Spearman IC 是否使用 `ic_calculator.calculate_rolling_ic`？（规则7.30）
- [ ] API路由层是否也使用 Spearman IC？（规则7.31）
- [ ] 年化收益是否使用几何复利而非算术平均？（规则7.32）
- [ ] IC加权是否使用股票收益率而非因子收益率均值？（规则7.33）
- [ ] `safe_divide` 在 se < min_threshold 时是否吞掉了有效 t 统计量？（规则7.34）
- [ ] CV 计算分母是否使用 `abs(mean)`？（规则7.35）

---

**最后更新**: 2026-06-10
**维护者**: FactorHub Core Team
**适用版本**: v1.0.0+
