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

## 代码审查 Checklist

提交 PR 时，reviewer 必须确认以下各项：

### 除法安全
- [ ] 所有 `/` 运算符是否使用了 `safe_divide`？
- [ ] 所有 IR 计算是否使用了 `safe_ir`？
- [ ] 是否存在 `+ 1e-10` hack？

### 统一入口
- [ ] 风险指标是否通过 `risk_metrics.py` 统一入口？
- [ ] 权重计算是否通过 `WeightOptimizer` 统一入口？
- [ ] 空值/不可计算值是否返回 None（规则6）？

### 数据安全
- [ ] 服务层方法是否对传入参数做了 `.copy()`？
- [ ] 是否存在直接修改传入 DataFrame 的代码？

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

---

**最后更新**: 2026-06-07
**维护者**: FactorHub Core Team
**适用版本**: v1.0.0+
