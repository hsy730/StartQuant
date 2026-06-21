"""
DEAP GP primitives for factor mining.

Defines protected operators and the PrimitiveSet used by the genetic
programming engine to build factor expressions with guaranteed syntactic
correctness.

Mask-First Design (v2.0):
==========================
所有时间序列窗口算子都支持可选的tradable_mask参数。
当提供mask时，自动过滤不可交易日（涨跌停/停牌），避免价格污染。

核心原理：
- 涨跌停价格不可交易，但会污染rolling/corr等窗口计算
- Mask-First: 在数据加载阶段构建mask，让所有算子接收并传递
- 解决IC虚高18%、夏普虚高0.44的问题

使用方式：
    # 无mask（向后兼容，但会有污染警告）
    result = ts_mean(price_series, 20)

    # 有mask（推荐，纯净计算）
    result = ts_mean_masked(price_series, 20, mask=tradable_mask)
"""

import numpy as np
import pandas as pd
import logging
import random
from deap import gp
from typing import Optional
from functools import partial
from scipy.stats import spearmanr
from scipy.special import expit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protected (safe) operators – avoid division-by-zero / log-of-negative
# ---------------------------------------------------------------------------


def safe_div(a, b):
    """x / y with y=0 or near-zero mapped to NaN."""
    if hasattr(b, "replace"):
        b = b.replace(0, np.nan)
        b = b.where(b.abs() >= 1e-10, np.nan)
    else:
        b = np.where(np.abs(b) < 1e-10, np.nan, b)
    return a / b


def safe_log(a):
    """log(x) with x <= 0 mapped to NaN."""
    if hasattr(a, "mask"):
        a = a.mask(a <= 0, np.nan)
    else:
        a = np.where(a > 0, a, np.nan)
    return np.log(a)


def safe_sqrt(a):
    """sqrt(x) with x < 0 mapped to NaN."""
    if hasattr(a, "mask"):
        a = a.mask(a < 0, np.nan)
    else:
        a = np.where(a >= 0, a, np.nan)
    return np.sqrt(a)


def pct_rank(a):
    """Cross-sectional percentile rank."""
    return a.rank(pct=True)


# ---------------------------------------------------------------------------
# Time-series window operators - Original versions (backward compatible)
# ⚠️ 这些版本不过滤涨跌停日，仅用于向后兼容
# ---------------------------------------------------------------------------


def ts_mean(a, n=5):
    """Rolling mean over *n* periods.

    ⚠️ Warning: This version does NOT filter limit-up/down days.
       For A-share market, please use ts_mean_masked() instead.
    """
    logger.debug(
        "ts_mean() called without tradable_mask - results may be contaminated by limit prices"
    )
    return a.rolling(window=int(n), min_periods=1).mean()


def ts_std(a, n=5):
    """Rolling std over *n* periods.

    ⚠️ Warning: This version does NOT filter limit-up/down days.
    """
    logger.debug("ts_std() called without tradable_mask - results may be contaminated")
    return a.rolling(window=int(n), min_periods=1).std()


def ts_delay(a, n=1):
    """Lag *a* by *n* periods (REF in 麦语言)."""
    return a.shift(int(n))


def ts_delta(a, n=1):
    """Difference: a - a.shift(n)."""
    return a - a.shift(int(n))


def _rolling_spearman(x, y_series, min_periods):
    """Rolling Spearman rank correlation helper."""
    y_aligned = y_series.loc[x.index]
    valid = x.notna() & y_aligned.notna()
    if valid.sum() < min_periods:
        return np.nan
    return spearmanr(x[valid], y_aligned[valid])[0]


def ts_corr(a, b, n=5):
    """Rolling Spearman rank correlation between *a* and *b* over *n* periods.

    Uses Spearman rank correlation instead of Pearson to capture nonlinear
    monotonic relationships, consistent with project IC calculation standards.

    ⚠️ Warning: This version does NOT filter limit-up/down days.
       IC may be inflated by ~18% in A-share market!
    """
    logger.debug("ts_corr() called without tradable_mask - IC may be inflated by 18%")
    window = int(n)
    min_periods = max(2, int(window * 0.6))
    return a.rolling(window=window, min_periods=min_periods).apply(
        lambda x: _rolling_spearman(x, b, min_periods), raw=False
    )


# ---------------------------------------------------------------------------
# Time-series window operators - Mask-First versions (RECOMMENDED ✅)
# 这些版本接受tradable_mask参数，自动过滤不可交易日
# ---------------------------------------------------------------------------


def ts_mean_masked(
    a: pd.Series,
    n: int = 5,
    mask: Optional[pd.Series] = None,
    min_valid_ratio: float = 0.6,
) -> pd.Series:
    """
    带掩码的滚动平均 - Mask-First设计

    将不可交易日（涨停/跌停/停牌）设为NaN，让rolling自动忽略，
    确保移动平均不被异常价格拉高或压低。

    Args:
        a: 价格/因子序列
        n: 窗口大小
        mask: 可交易性掩码（True=可交易，False=不可交易）
        min_valid_ratio: 窗口内最小有效数据比例（默认60%）

    Returns:
        滚动平均序列（不可交易日为NaN）
    """
    if mask is None:
        logger.debug("ts_mean_masked(): 未提供mask，退化为普通ts_mean()")
        return ts_mean(a, n)

    # 应用掩码：将不可交易日设为NaN
    a_masked = a.where(mask)

    # 计算滚动平均，要求至少60%的数据点有效
    window = int(n)
    min_periods = max(1, int(window * min_valid_ratio))

    result = a_masked.rolling(window=window, min_periods=min_periods).mean()

    return result


def ts_std_masked(
    a: pd.Series,
    n: int = 5,
    mask: Optional[pd.Series] = None,
    min_valid_ratio: float = 0.6,
) -> pd.Series:
    """
    带掩码的滚动标准差 - Mask-First设计

    排除涨跌停日的波动率计算，避免被压缩或放大。

    Args:
        a: 价格/因子序列
        n: 窗口大小
        mask: 可交易性掩码
        min_valid_ratio: 最小有效数据比例

    Returns:
        滚动标准差序列
    """
    if mask is None:
        logger.debug("ts_std_masked(): 未提供mask，退化为普通ts_std()")
        return ts_std(a, n)

    a_masked = a.where(mask)
    window = int(n)
    min_periods = max(2, int(window * min_valid_ratio))

    return a_masked.rolling(window=window, min_periods=min_periods).std()


def ts_corr_masked(
    a: pd.Series,
    b: pd.Series,
    n: int = 5,
    mask: Optional[pd.Series] = None,
    min_valid_ratio: float = 0.6,
) -> pd.Series:
    """
    带掩码的滚动相关系数 - **最关键的改进！**

    这是解决IC虚高18%问题的核心方法。
    因子值与收益率的相关系数必须排除不可交易日的虚假信号。

    Args:
        a: 因子值序列
        b: 收益率或其他序列
        n: 窗口大小
        mask: 可交易性掩码（双方都要应用）
        min_valid_ratio: 最小有效数据比例

    Returns:
        滚动相关系数序列（范围[-1, 1]）
    """
    if mask is None:
        logger.debug("ts_corr_masked(): 未提供mask，IC可能虚高18%")
        return ts_corr(a, b, n)

    # 双方都应用掩码
    a_masked = a.where(mask)
    b_masked = b.where(mask)

    window = int(n)
    min_periods = max(2, int(window * min_valid_ratio))

    result = a_masked.rolling(window=window, min_periods=min_periods).apply(
        lambda x: _rolling_spearman(x, b_masked, min_periods), raw=False
    )

    return result


def _pair_max(a, b):
    """Element-wise max of two Series."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return a.combine(b, max, fill_value=np.nan)
    return np.maximum(a, b)


def _pair_min(a, b):
    """Element-wise min of two Series."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return b.combine(a, min, fill_value=np.nan)
    return np.minimum(a, b)


def _sigmoid(a):
    """Sigmoid activation (clamps output to [0, 1] range).

    使用 scipy.special.expit（C实现），比手工 np.exp 快 2-3x，
    且自动处理 ±inf 边界，无需手动 clip。
    """
    return expit(a)


def _tanh(a):
    """Hyperbolic tangent activation (output in [-1, 1] range)."""
    return np.tanh(np.clip(a, -500, 500))


# ---------------------------------------------------------------------------
# PrimitiveSet factory
# ---------------------------------------------------------------------------


def create_pset(
    n_factors: int,
    extended: bool = True,
    use_masked: bool = True,
    tradable_mask: Optional[pd.Series] = None,
) -> gp.PrimitiveSet:
    """Build a DEAP ``PrimitiveSet`` for factor expressions.

    Args:
        n_factors: 基础因子数量
        extended: 是否包含扩展算子（时间序列窗口等）
        use_masked: 是否使用Mask-First版本的算子（默认True）
        tradable_mask: 可交易性掩码（True=可交易，False=涨跌停/停牌）。
            仅在 use_masked=True 时生效。传入后算子自动过滤不可交易日，
            消除IC虚高问题且不再产生警告。

    Terminals
    ---------
    ``factor_0`` … ``factor_{n_factors-1}`` – placeholders that receive the
    pre-computed :class:`pd.Series` of the corresponding base factor.

    Primitives (base set, 9)
    ------------------------
    =========  ========  ============================================
    Name       Arity     Description
    =========  ========  ============================================
    add        2         element-wise addition
    sub        2         element-wise subtraction
    mul        2         element-wise multiplication
    div        2         safe division (0 → NaN)
    neg        1         element-wise negation
    abs        1         absolute value
    log        1         safe natural log
    sqrt       1         safe square root
    rank       1         cross-sectional percentile rank
    =========  ========  ============================================

    Extended primitives (Phase 7, +16 when extended=True)
    -----------------------------------------------------
    When use_masked=True (RECOMMENDED for A-share market):
        All time-series operators will use Mask-First versions that filter
        limit-up/down/suspended days to avoid price contamination.

    =============  ========  =========================================
    Name           Arity     Description
    =============  ========  =========================================
    ts_mean_5      1         5-period rolling mean (masked if enabled)
    ts_mean_10     1         10-period rolling mean
    ts_mean_20     1         20-period rolling mean
    ts_std_5       1         5-period rolling std (masked if enabled)
    ts_std_10      1         10-period rolling std
    ts_std_20      1         20-period rolling std
    ts_delay_1     1         1-period lag
    ts_delay_5     1         5-period lag
    ts_delta_1     1         1-period difference
    ts_delta_5     1         5-period difference
    ts_corr_5      2         5-period rolling correlation (masked!)
    ts_corr_10     2         10-period rolling correlation (masked!)
    ts_corr_20     2         20-period rolling correlation (masked!)
    max            2         element-wise max
    min            2         element-wise min
    sigmoid        1         sigmoid activation
    tanh           1         hyperbolic tangent activation
    =============  ========  =========================================

    All operators accept and return :class:`pd.Series`.
    """
    pset = gp.PrimitiveSet("MAIN", n_factors)

    # ---- Base primitives (9) ----
    # binary
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(safe_div, 2, name="div")

    # unary
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(np.abs, 1, name="abs")
    pset.addPrimitive(safe_log, 1, name="log")
    pset.addPrimitive(safe_sqrt, 1, name="sqrt")
    pset.addPrimitive(pct_rank, 1, name="rank")

    # ---- Extended primitives (Phase 7, +16) ----
    if extended:
        if use_masked and tradable_mask is not None:
            # ✅ Mask-First版本 + 实际mask注入（最佳实践）
            ts_mean_fn = partial(ts_mean_masked, n=5, mask=tradable_mask)
            ts_mean_10_fn = partial(ts_mean_masked, n=10, mask=tradable_mask)
            ts_mean_20_fn = partial(ts_mean_masked, n=20, mask=tradable_mask)

            ts_std_fn = partial(ts_std_masked, n=5, mask=tradable_mask)
            ts_std_10_fn = partial(ts_std_masked, n=10, mask=tradable_mask)
            ts_std_20_fn = partial(ts_std_masked, n=20, mask=tradable_mask)

            ts_corr_5_fn = partial(ts_corr_masked, n=5, mask=tradable_mask)
            ts_corr_10_fn = partial(ts_corr_masked, n=10, mask=tradable_mask)
            ts_corr_20_fn = partial(ts_corr_masked, n=20, mask=tradable_mask)

            logger.info(
                "✅ PrimitiveSet: 使用Mask-First版本算子（已注入tradable_mask，过滤涨跌停）"
            )
        elif use_masked:
            # ⚠️ Mask-First版本但无mask（退化，不再spam警告）
            def ts_mean_fn(a):
                return ts_mean_masked(a, 5)

            def ts_mean_10_fn(a):
                return ts_mean_masked(a, 10)

            def ts_mean_20_fn(a):
                return ts_mean_masked(a, 20)

            def ts_std_fn(a):
                return ts_std_masked(a, 5)

            def ts_std_10_fn(a):
                return ts_std_masked(a, 10)

            def ts_std_20_fn(a):
                return ts_std_masked(a, 20)

            def ts_corr_5_fn(a, b):
                return ts_corr_masked(a, b, 5)

            def ts_corr_10_fn(a, b):
                return ts_corr_masked(a, b, 10)

            def ts_corr_20_fn(a, b):
                return ts_corr_masked(a, b, 20)

            logger.info(
                "PrimitiveSet: 使用Mask-First版本算子（无mask，退化为普通版本）"
            )
        else:
            # ❌ 传统版本（不过滤）
            def ts_mean_fn(a):
                return ts_mean(a, 5)

            def ts_mean_10_fn(a):
                return ts_mean(a, 10)

            def ts_mean_20_fn(a):
                return ts_mean(a, 20)

            def ts_std_fn(a):
                return ts_std(a, 5)

            def ts_std_10_fn(a):
                return ts_std(a, 10)

            def ts_std_20_fn(a):
                return ts_std(a, 20)

            def ts_corr_5_fn(a, b):
                return ts_corr(a, b, 5)

            def ts_corr_10_fn(a, b):
                return ts_corr(a, b, 10)

            def ts_corr_20_fn(a, b):
                return ts_corr(a, b, 20)

            logger.warning(
                "⚠️ PrimitiveSet: 使用传统版本算子（未过滤涨跌停，IC可能虚高）"
            )

        # Time-series window operations (unary, fixed window)
        pset.addPrimitive(ts_mean_fn, 1, name="ts_mean_5")
        pset.addPrimitive(ts_mean_10_fn, 1, name="ts_mean_10")
        pset.addPrimitive(ts_mean_20_fn, 1, name="ts_mean_20")
        pset.addPrimitive(ts_std_fn, 1, name="ts_std_5")
        pset.addPrimitive(ts_std_10_fn, 1, name="ts_std_10")
        pset.addPrimitive(ts_std_20_fn, 1, name="ts_std_20")
        pset.addPrimitive(lambda a: ts_delay(a, 1), 1, name="ts_delay_1")
        pset.addPrimitive(lambda a: ts_delay(a, 5), 1, name="ts_delay_5")
        pset.addPrimitive(lambda a: ts_delta(a, 1), 1, name="ts_delta_1")
        pset.addPrimitive(lambda a: ts_delta(a, 5), 1, name="ts_delta_5")

        # Time-series correlation (binary, fixed window)
        pset.addPrimitive(ts_corr_5_fn, 2, name="ts_corr_5")
        pset.addPrimitive(ts_corr_10_fn, 2, name="ts_corr_10")
        pset.addPrimitive(ts_corr_20_fn, 2, name="ts_corr_20")

        # Pairwise operations
        pset.addPrimitive(_pair_max, 2, name="max")
        pset.addPrimitive(_pair_min, 2, name="min")

        # Activation functions
        pset.addPrimitive(_sigmoid, 1, name="sigmoid")
        pset.addPrimitive(_tanh, 1, name="tanh")

    # rename ARG0…ARG{N-1} → factor_0…factor_{N-1}
    renames = {f"ARG{i}": f"factor_{i}" for i in range(n_factors)}
    pset.renameArguments(**renames)

    return pset


# ---------------------------------------------------------------------------
# Tree → expression helpers
# ---------------------------------------------------------------------------


def tree_to_expression(tree, base_factor_codes: dict) -> str:
    """Convert a *PrimitiveTree* to a human-readable expression.

    Parameters
    ----------
    tree : gp.PrimitiveTree
        The evolved tree.
    base_factor_codes : dict
        Mapping ``{"factor_0": "RSI(close,14)", …}`` used to replace
        placeholder terminal names with real factor codes.

    Returns
    -------
    str
    """
    expr = str(tree)
    # replace longest keys first to avoid partial-match collisions
    # 不加额外括号：GP 前缀表达式中 factor_N 总是作为函数参数出现（在括号内），
    # 逗号已分隔参数，无需再用括号包裹。避免 sqrt((SCALE(...))) 双括号。
    for var_name in sorted(base_factor_codes, key=len, reverse=True):
        code = base_factor_codes[var_name]
        expr = expr.replace(var_name, code)
    return expr


def tree_to_placeholder_expr(tree) -> str:
    """Return the placeholder expression (e.g. ``add(factor_0, factor_1)``)."""
    return str(tree)


def compile_tree(tree, pset: gp.PrimitiveSet):
    """Compile a tree into a callable.

    The returned function accepts keyword arguments ``factor_0`` … ``factor_N``
    (each a :class:`pd.Series`) and returns a :class:`pd.Series`.
    """
    return gp.compile(tree, pset)


# ---------------------------------------------------------------------------
# Numpy-optimized primitives for GP evaluation path
# 消除 pandas 索引对齐开销，加速约 3-5x
# ---------------------------------------------------------------------------


def safe_div_np(a, b):
    """Numpy版安全除法 — a/b 都是 ndarray，分母近零映射为 NaN。"""
    b = np.where(np.abs(b) < 1e-10, np.nan, b)
    return a / b


def safe_log_np(a):
    """Numpy版安全对数 — 非正数映射为 NaN。"""
    return np.log(np.where(a > 0, a, np.nan))


def safe_sqrt_np(a):
    """Numpy版安全平方根 — 负数映射为 NaN。"""
    return np.sqrt(np.where(a >= 0, a, np.nan))


def pct_rank_np(a):
    """Numpy版百分位排名 — 等价于 pd.Series(a).rank(pct=True).values。"""
    valid = ~np.isnan(a)
    result = np.full_like(a, np.nan, dtype=float)
    if valid.sum() > 0:
        order = np.empty(valid.sum(), dtype=int)
        order[np.argsort(a[valid])] = np.arange(valid.sum())
        result[valid] = (order + 1.0) / valid.sum()
    return result


def ts_mean_np(a, n=5):
    """Numpy版滚动平均 — 使用 cumsum 实现高效滑动窗口。"""
    n = int(n)
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < n:
        return out
    cumsum = np.nancumsum(a)
    # 前n-1个位置用部分窗口
    for i in range(min(n - 1, len(a))):
        out[i] = cumsum[i] / (i + 1)
    # n及之后用完整窗口
    cumsum_padded = np.concatenate([[0], cumsum])
    out[n - 1 :] = (cumsum_padded[n:] - cumsum_padded[:-n]) / n
    return out


def ts_std_np(a, n=5):
    """Numpy版滚动标准差 — 使用 E[X²]-E[X]² 方法。"""
    n = int(n)
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < n:
        return out
    mean_a = ts_mean_np(a, n)
    mean_a2 = ts_mean_np(a * a, n)
    var = mean_a2 - mean_a * mean_a
    # 浮点精度可能导致微小负值，截断到0
    var = np.maximum(var, 0.0)
    out = np.sqrt(var)
    return out


def ts_delay_np(a, n=1):
    """Numpy版延迟 — 等价于 pd.Series.shift(n)。"""
    n = int(n)
    out = np.empty_like(a)
    out[:n] = np.nan
    out[n:] = a[:-n]
    return out


def ts_delta_np(a, n=1):
    """Numpy版差分 — 等价于 a - a.shift(n)。"""
    n = int(n)
    out = np.empty_like(a)
    out[:n] = np.nan
    out[n:] = a[n:] - a[:-n]
    return out


def ts_corr_np(a, b, n=5):
    """纯 Numpy 滚动 Spearman 相关 — 零 pandas 开销。

    算法：Spearman 相关 = 对排名序列的 Pearson 相关。
    使用滑动窗口视图 + 向量化排名/相关，避免 pd.Series 创建和 pandas rolling。

    性能对比（250 个数据点，窗口=5）：
    - 旧版（pandas rolling.rank + rolling.corr）：~3ms/调用，嵌套时指数膨胀
    - 新版（纯 numpy 向量化）：~0.05ms/调用，加速 ~60x
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    n = int(n)
    min_periods = max(2, int(n * 0.8))
    length = len(a)
    if length < n:
        return np.full(length, np.nan)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    result = np.full(length, np.nan)

    # 滑动窗口视图：shape (n_windows, window_size)
    try:
        a_win = sliding_window_view(a, n)
        b_win = sliding_window_view(b, n)
    except AttributeError:
        # numpy < 1.20 回退
        return _ts_corr_np_fallback(a, b, n, min_periods)

    n_windows = a_win.shape[0]

    # ---- 向量化排名（每行独立） ----
    # 将 NaN 排到末尾后 argsort → 再 argsort 得到秩（向量化）
    a_safe = np.where(np.isnan(a_win), np.inf, a_win)
    b_safe = np.where(np.isnan(b_win), np.inf, b_win)
    a_order = np.argsort(a_safe, axis=1, kind='stable')
    b_order = np.argsort(b_safe, axis=1, kind='stable')
    row_indices = np.arange(n_windows)[:, None]
    ar = np.empty_like(a_order, dtype=np.float64)
    br = np.empty_like(b_order, dtype=np.float64)
    ar[row_indices, a_order] = np.arange(1, n + 1, dtype=np.float64)
    br[row_indices, b_order] = np.arange(1, n + 1, dtype=np.float64)
    # 还原 NaN 位置
    ar[np.isnan(a_win)] = np.nan
    br[np.isnan(b_win)] = np.nan

    # ---- 向量化 Pearson 相关（对排名序列） ----
    valid = (~np.isnan(ar)) & (~np.isnan(br))
    n_valid = valid.sum(axis=1)
    ok = n_valid >= min_periods

    # 只计算有效窗口的相关系数
    if not np.any(ok):
        return result

    ar_ok = ar[ok]
    br_ok = br[ok]

    # 去均值
    ar_mean = np.nansum(ar_ok, axis=1, keepdims=True) / n_valid[ok, None]
    br_mean = np.nansum(br_ok, axis=1, keepdims=True) / n_valid[ok, None]
    ca = ar_ok - ar_mean
    cb = br_ok - br_mean

    # 用 nansum 处理剩余 NaN
    cov = np.nansum(ca * cb, axis=1)
    var_a = np.nansum(ca * ca, axis=1)
    var_b = np.nansum(cb * cb, axis=1)
    denom = np.sqrt(var_a * var_b)

    corr = np.where(denom > 1e-10, cov / denom, np.nan)

    # 写回结果（滑动窗口的最后一个位置）
    result[n - 1 :][ok] = corr
    return result


def _ts_corr_np_fallback(a, b, n, min_periods):
    """numpy < 1.20 的回退实现（使用手动循环）。"""
    length = len(a)
    result = np.full(length, np.nan)
    for i in range(n - 1, length):
        wa = a[i - n + 1 : i + 1]
        wb = b[i - n + 1 : i + 1]
        mask = ~(np.isnan(wa) | np.isnan(wb))
        if mask.sum() < min_periods:
            continue
        sa = wa[mask]
        sb = wb[mask]
        ra = np.argsort(np.argsort(sa)) + 1
        rb = np.argsort(np.argsort(sb)) + 1
        ca = ra - ra.mean()
        cb = rb - rb.mean()
        den = np.sqrt((ca * ca).sum() * (cb * cb).sum())
        if den > 1e-10:
            result[i] = (ca * cb).sum() / den
    return result


def _pair_max_np(a, b):
    """Numpy版逐元素最大 — fmax 自动处理 NaN。"""
    return np.fmax(a, b)


def _pair_min_np(a, b):
    """Numpy版逐元素最小 — fmin 自动处理 NaN。"""
    return np.fmin(a, b)


def _sigmoid_np(a):
    """Numpy版 sigmoid — scipy.special.expit 已支持 ndarray。"""
    return expit(a)


def _tanh_np(a):
    """Numpy版 tanh — clip 防溢出。"""
    return np.tanh(np.clip(a, -500, 500))


def create_pset_numpy(n_factors: int, extended: bool = True) -> gp.PrimitiveSet:
    """Build a numpy-optimized PrimitiveSet for GP evaluation.

    与 create_pset 功能等价，但所有原语接受/返回 numpy ndarray，
    消除 pandas 索引对齐开销，加速约 3-5x。

    注意：numpy 版不支持 mask-first，因为 GP 评估路径中
    涨跌停过滤在 IC 计算阶段而非因子计算阶段处理。

    Args:
        n_factors: 基础因子数量
        extended: 是否包含扩展算子（时间序列窗口等）

    Terminals
    ---------
    ``factor_0`` … ``factor_{n_factors-1}`` – placeholders that receive
    pre-computed numpy ndarray of the corresponding base factor.

    Primitives (base set, 9)
    ------------------------
    =========  ========  ============================================
    Name       Arity     Description
    =========  ========  ============================================
    add        2         element-wise addition
    sub        2         element-wise subtraction
    mul        2         element-wise multiplication
    div        2         safe division (0 → NaN)
    neg        1         element-wise negation
    abs        1         absolute value
    log        1         safe natural log
    sqrt       1         safe square root
    rank       1         cross-sectional percentile rank
    =========  ========  ============================================

    Extended primitives (+16 when extended=True)
    --------------------------------------------
    =============  ========  =========================================
    Name           Arity     Description
    =============  ========  =========================================
    ts_mean_5      1         5-period rolling mean
    ts_mean_10     1         10-period rolling mean
    ts_mean_20     1         20-period rolling mean
    ts_std_5       1         5-period rolling std
    ts_std_10      1         10-period rolling std
    ts_std_20      1         20-period rolling std
    ts_delay_1     1         1-period lag
    ts_delay_5     1         5-period lag
    ts_delta_1     1         1-period difference
    ts_delta_5     1         5-period difference
    ts_corr_5      2         5-period rolling Spearman correlation
    ts_corr_10     2         10-period rolling Spearman correlation
    ts_corr_20     2         20-period rolling Spearman correlation
    max            2         element-wise max
    min            2         element-wise min
    sigmoid        1         sigmoid activation
    tanh           1         hyperbolic tangent activation
    =============  ========  =========================================
    """
    pset = gp.PrimitiveSet("MAIN", n_factors)

    # ---- Base primitives (9) — numpy 版 ----
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(safe_div_np, 2, name="div")
    pset.addPrimitive(np.negative, 1, name="neg")
    pset.addPrimitive(np.abs, 1, name="abs")
    pset.addPrimitive(safe_log_np, 1, name="log")
    pset.addPrimitive(safe_sqrt_np, 1, name="sqrt")
    pset.addPrimitive(pct_rank_np, 1, name="rank")

    # ---- Extended primitives (+16) ----
    if extended:
        pset.addPrimitive(partial(ts_mean_np, n=5), 1, name="ts_mean_5")
        pset.addPrimitive(partial(ts_mean_np, n=10), 1, name="ts_mean_10")
        pset.addPrimitive(partial(ts_mean_np, n=20), 1, name="ts_mean_20")
        pset.addPrimitive(partial(ts_std_np, n=5), 1, name="ts_std_5")
        pset.addPrimitive(partial(ts_std_np, n=10), 1, name="ts_std_10")
        pset.addPrimitive(partial(ts_std_np, n=20), 1, name="ts_std_20")
        pset.addPrimitive(partial(ts_delay_np, n=1), 1, name="ts_delay_1")
        pset.addPrimitive(partial(ts_delay_np, n=5), 1, name="ts_delay_5")
        pset.addPrimitive(partial(ts_delta_np, n=1), 1, name="ts_delta_1")
        pset.addPrimitive(partial(ts_delta_np, n=5), 1, name="ts_delta_5")
        pset.addPrimitive(partial(ts_corr_np, n=5), 2, name="ts_corr_5")
        pset.addPrimitive(partial(ts_corr_np, n=10), 2, name="ts_corr_10")
        pset.addPrimitive(partial(ts_corr_np, n=20), 2, name="ts_corr_20")
        pset.addPrimitive(_pair_max_np, 2, name="max")
        pset.addPrimitive(_pair_min_np, 2, name="min")
        pset.addPrimitive(_sigmoid_np, 1, name="sigmoid")
        pset.addPrimitive(_tanh_np, 1, name="tanh")

    # rename ARG0…ARG{N-1} → factor_0…factor_{N-1}
    renames = {f"ARG{i}": f"factor_{i}" for i in range(n_factors)}
    pset.renameArguments(**renames)

    return pset


# ---------------------------------------------------------------------------
# Expression similarity (structural) for diversity penalty
# ---------------------------------------------------------------------------


def expression_similarity(expr_a: str, expr_b: str) -> float:
    """Compute a simple structural similarity between two placeholder expressions.

    Uses token-overlap Jaccard similarity on the string representation.
    Returns a value in [0, 1] where 1 means identical.
    """
    if not expr_a or not expr_b:
        return 0.0
    tokens_a = set(
        expr_a.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
    )
    tokens_b = set(
        expr_b.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
    )
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Operator-weighted complexity (P1: replaces naive node count)
# Based on: Nomura et al. (2026) generalization bounds — structure-selection
# term should reflect operator computational cost, not just node count.
# ---------------------------------------------------------------------------

# Weight by computational cost category:
#   1.0 = basic arithmetic (O(n) elementwise)
#   2.0 = transcendental / sort-based (log, sqrt, rank)
#   3.0 = windowed aggregation (ts_mean, ts_std, sigmoid, tanh)
#   4.0 = pairwise windowed (ts_corr)
#   0.5 = terminal nodes (leaf, low cost)
_OPERATOR_WEIGHTS: dict = {
    # --- Basic arithmetic (weight 1.0) ---
    "add": 1.0, "sub": 1.0, "mul": 1.0, "div": 1.0,
    "neg": 1.0, "abs": 1.0,
    "max": 1.0, "min": 1.0,
    # --- Transcendental / sort-based (weight 2.0) ---
    "log": 2.0, "sqrt": 2.0, "rank": 2.0,
    # --- Windowed aggregation (weight 3.0) ---
    "ts_mean_5": 3.0, "ts_mean_10": 3.0, "ts_mean_20": 3.0,
    "ts_std_5": 3.0, "ts_std_10": 3.0, "ts_std_20": 3.0,
    "ts_delay_1": 3.0, "ts_delay_5": 3.0,
    "ts_delta_1": 3.0, "ts_delta_5": 3.0,
    "sigmoid": 3.0, "tanh": 3.0,
    # --- Pairwise windowed (weight 4.0) ---
    "ts_corr_5": 4.0, "ts_corr_10": 4.0, "ts_corr_20": 4.0,
}

# Default weight for unknown operators and terminals
_DEFAULT_OP_WEIGHT = 1.0
_TERMINAL_WEIGHT = 0.5


def compute_weighted_complexity(tree) -> float:
    """Compute operator-weighted complexity of a DEAP PrimitiveTree.

    Unlike naive ``len(tree)`` which counts every node equally, this
    assigns higher weights to expensive operators (``ts_corr``=4.0,
    ``log``/``sqrt``/``rank``=2.0, ``ts_mean``=3.0) and lower weights
    to terminals (0.5).

    This gives a more accurate proxy for the structure-selection term
    in the generalization bound (Nomura et al., 2026), improving
    parsimony pressure precision.

    Examples
    --------
    >>> # add(factor_0, factor_1) = 1.0 + 0.5 + 0.5 = 2.0
    >>> # ts_corr_5(factor_0, factor_1) = 4.0 + 0.5 + 0.5 = 5.0
    """
    total = 0.0
    for node in tree:
        name = getattr(node, "name", str(node))
        arity = getattr(node, "arity", 0)
        if arity == 0:
            # Terminal node (factor_N or constant)
            total += _TERMINAL_WEIGHT
        else:
            total += _OPERATOR_WEIGHTS.get(name, _DEFAULT_OP_WEIGHT)
    return total


def count_duplicate_subtrees(tree) -> dict:
    """检测表达式树中的重复子树

    遍历所有子树（包括内部节点和叶子），用 Zobrist 哈希检测重复。
    重复子树是 GP "表达式膨胀"（bloat）的典型特征：
    同一个子表达式在树中多次出现，增加计算成本但不提升表达能力。

    通过在适应度评估中对重复子树施加惩罚，GP 会自然倾向于
    产生无重复子表达式的紧凑表达式。

    Returns
    -------
    dict
        n_duplicates : int
            重复子树数量（不含第一次出现）
        duplicate_ratio : float
            重复子树占比 [0, 1]
        unique_subtrees : int
            唯一子树数量
        total_subtrees : int
            总子树数量
    """
    subtree_hashes: set = set()
    n_duplicates = 0
    total_subtrees = 0

    def _hash_subtree(start: int) -> tuple:
        nonlocal n_duplicates, total_subtrees
        node = tree[start]
        arity = getattr(node, "arity", 0)
        name = getattr(node, "name", str(node))
        node_val = _get_zobrist_value(name)

        if arity == 0:
            h = node_val
            next_idx = start + 1
        else:
            children = []
            idx = start + 1
            for _ in range(arity):
                child_hash, idx = _hash_subtree(idx)
                children.append(child_hash)

            if name in _COMMUTATIVE_OPS:
                children.sort()

            result = node_val
            for pos, ch in enumerate(children):
                seed = _POSITION_SEEDS[pos % len(_POSITION_SEEDS)]
                result ^= (ch * seed) & _MASK64

            h = result
            next_idx = idx

        total_subtrees += 1
        if h in subtree_hashes:
            n_duplicates += 1
        else:
            subtree_hashes.add(h)

        return h, next_idx

    _hash_subtree(0)

    unique_subtrees = len(subtree_hashes)
    duplicate_ratio = n_duplicates / total_subtrees if total_subtrees > 0 else 0.0

    return {
        "n_duplicates": n_duplicates,
        "duplicate_ratio": duplicate_ratio,
        "unique_subtrees": unique_subtrees,
        "total_subtrees": total_subtrees,
    }


def replace_duplicate_subtrees(
    tree, pset, max_subtree_depth: int = 1, max_replacements: int = 5
) -> tuple:
    """检测并替换个体内部的重复子树

    遍历所有子树，用 Zobrist 哈希检测重复。
    对于重复的子树（保留第一次出现），将后续出现替换为一个新的随机子树。

    设计决策：
    - **终端重复**：仅当可用终端 ≥ 3 时才替换。终端少时（如仅1个基础因子），
      factor_0 重复是不可避免的，强行替换只会用随机子树增加复杂度，
      导致表达式膨胀和评估变慢（10-20x）。
    - **算子子树重复**：始终替换，这些是真正的结构冗余。
    - **替换上限**：每个个体最多替换 max_replacements 个子树，防止复杂度爆炸。

    参考: Poli & McPhee (2008) "Bloat in Genetic Programming" —
    intron removal via subtree replacement.

    Parameters
    ----------
    tree : PrimitiveTree
        DEAP 表达式树
    pset : PrimitiveSet
        原语集合，用于生成随机子树
    max_subtree_depth : int
        替换子树的最大深度（默认 1，保持表达式简洁）
    max_replacements : int
        每个个体最多替换的子树数量（默认 5，防止复杂度爆炸）

    Returns
    -------
    tuple
        new_nodes : list or None
            替换后的节点列表（如果无重复返回 None）
        n_replaced : int
            替换的重复子树数量
    """
    seen_hashes: dict = {}
    duplicates = []  # [(start, end), ...]

    def _scan_subtree(start: int) -> tuple:
        node = tree[start]
        arity = getattr(node, "arity", 0)
        name = getattr(node, "name", str(node))
        node_val = _get_zobrist_value(name)

        if arity == 0:
            h = node_val
            end = start
        else:
            children = []
            idx = start + 1
            for _ in range(arity):
                child_hash, idx = _scan_subtree(idx)
                children.append(child_hash)

            if name in _COMMUTATIVE_OPS:
                children.sort()

            result = node_val
            for pos, ch in enumerate(children):
                seed = _POSITION_SEEDS[pos % len(_POSITION_SEEDS)]
                result ^= (ch * seed) & _MASK64

            h = result
            end = idx - 1

        # 检测所有子树重复（包括终端和算子子树）
        # 终端重复（如 factor_0 多次出现）在展开后会生成重复的复杂表达式，
        # 因为 base factor 本身可能是复杂表达式（如 Alpha101 因子）
        if h in seen_hashes:
            duplicates.append((start, end))
        else:
            seen_hashes[h] = start

        # 返回 end + 1 作为下一个子树的起始索引
        # （end 是当前子树最后一个节点的索引）
        return h, end + 1

    _scan_subtree(0)

    if not duplicates:
        return None, 0

    # 过滤嵌套的重复子树
    # 如果子树 A 包含子树 B，替换 A 后 B 的位置就失效了
    # 只保留不嵌套的（最外层的）重复子树
    sorted_dups = sorted(duplicates, key=lambda x: x[0])
    filtered = []
    for start, end in sorted_dups:
        nested = False
        for f_start, f_end in filtered:
            if f_start <= start and end <= f_end:
                nested = True
                break
        if not nested:
            filtered.append((start, end))

    nodes = list(tree)

    # Collect available terminals for replacing duplicate terminals.
    # Only replace terminal duplicates when there are enough alternatives (≥ 3).
    # With few terminals (e.g. 1 base factor), factor_0 repeats are unavoidable;
    # replacing them with random subtrees only inflates complexity (10-20x slower).
    available_terminals = []
    try:
        available_terminals = list(pset.terminals[pset.ret])
    except (KeyError, AttributeError):
        pass

    can_replace_terminals = len(available_terminals) >= 3

    # Filter: keep only operator-subtree duplicates, or terminal duplicates
    # when enough terminals are available
    replaceable = []
    for start, end in filtered:
        node = tree[start]
        arity = getattr(node, "arity", 0)
        if arity == 0 and not can_replace_terminals:
            continue  # Skip terminal duplicates when few terminals
        replaceable.append((start, end))

    if not replaceable:
        return None, 0

    # 从后往前替换（避免索引偏移），限制总替换数防止复杂度爆炸
    n_replaced = 0
    for start, end in sorted(replaceable, key=lambda x: x[0], reverse=True):
        if n_replaced >= max_replacements:
            break

        node = tree[start]
        arity = getattr(node, "arity", 0)

        if arity == 0:
            # 重复终端：用不同的终端替换
            name = getattr(node, "name", str(node))
            other_terminals = [
                t for t in available_terminals
                if getattr(t, "name", str(t)) != name
            ]
            if other_terminals:
                nodes[start:end + 1] = [random.choice(other_terminals)]
            else:
                new_subtree = gp.genHalfAndHalf(
                    pset, min_=0, max_=max_subtree_depth
                )
                nodes[start:end + 1] = new_subtree
        else:
            # 重复算子子树：用随机子树替换
            new_subtree = gp.genHalfAndHalf(
                pset, min_=0, max_=max_subtree_depth
            )
            nodes[start:end + 1] = new_subtree
        n_replaced += 1

    return nodes, n_replaced


# ---------------------------------------------------------------------------
# Zobrist hash for efficient duplicate detection (cache key)
# Based on: Burlacu (2025) "Zobrist Hash-based Duplicate Detection in SR"
# ---------------------------------------------------------------------------

_MASK64 = 0xFFFFFFFFFFFFFFFF
# Fixed-seed RNG for reproducible Zobrist table
_ZOBRIST_RNG = random.Random(0x5EED_1234)
_ZOBRIST_TABLE: dict = {}

# Commutative operations where argument order doesn't matter
_COMMUTATIVE_OPS = frozenset({"add", "mul", "max", "min"})

# Position-dependent seeds for non-commutative argument mixing
_POSITION_SEEDS = (
    0x9E3779B97F4A7C15,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x96C7D2C0B3F6A5D8,
)


def _get_zobrist_value(name: str) -> int:
    """Get or create a 64-bit Zobrist value for a node name."""
    val = _ZOBRIST_TABLE.get(name)
    if val is None:
        val = _ZOBRIST_RNG.getrandbits(64)
        _ZOBRIST_TABLE[name] = val
    return val


def zobrist_hash(tree) -> int:
    """Compute a Zobrist hash for a DEAP PrimitiveTree.

    Detects structurally identical trees, including isomorphic forms of
    commutative operations (``add``, ``mul``, ``max``, ``min``).

    For commutative ops, children hashes are sorted before mixing, so
    ``add(a, b)`` and ``add(b, a)`` produce the same hash.

    For non-commutative ops (``sub``, ``div``, …), position-dependent
    seeds ensure ``sub(a, b)`` ≠ ``sub(b, a)``.

    Collision rate is bounded by 2⁻⁶⁴, which is negligible.
    """
    def _hash_subtree(start: int) -> tuple:
        node = tree[start]
        arity = getattr(node, "arity", 0)
        name = getattr(node, "name", str(node))
        node_val = _get_zobrist_value(name)

        if arity == 0:
            return node_val, start + 1

        children = []
        idx = start + 1
        for _ in range(arity):
            child_hash, idx = _hash_subtree(idx)
            children.append(child_hash)

        # For commutative ops, sort to normalize argument order
        if name in _COMMUTATIVE_OPS:
            children.sort()

        # Mix children with position-dependent seeds
        result = node_val
        for pos, ch in enumerate(children):
            seed = _POSITION_SEEDS[pos % len(_POSITION_SEEDS)]
            result ^= (ch * seed) & _MASK64

        return result, idx

    h, _ = _hash_subtree(0)
    return h


# ---------------------------------------------------------------------------
# SymPy-based expression simplification (post-processing)
# Based on: SymPy simplify() + canonical form for deduplication
# ---------------------------------------------------------------------------

try:
    import sympy as sp

    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False
    logger.info("SymPy未安装，表达式简化功能将不可用")

# Mapping from GP primitive names to SymPy constructors
_GP_TO_SYMPY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "neg": lambda a: -a,
    "abs": lambda a: sp.Abs(a),
    "log": lambda a: sp.log(a),
    "sqrt": lambda a: sp.sqrt(a),
    "max": lambda a, b: sp.Max(a, b),
    "min": lambda a, b: sp.Min(a, b),
    "tanh": lambda a: sp.tanh(a),
    "sigmoid": lambda a: 1 / (1 + sp.exp(-a)),
}


def _tokenize_prefix(expr_str: str) -> list:
    """Tokenize a DEAP prefix expression like 'add(factor_0, sub(factor_1, factor_0))'."""
    tokens = []
    i = 0
    n = len(expr_str)
    while i < n:
        c = expr_str[i]
        if c.isspace():
            i += 1
        elif c in "(),":
            tokens.append(c)
            i += 1
        else:
            j = i
            while j < n and expr_str[j] not in "(), \t\n":
                j += 1
            tokens.append(expr_str[i:j])
            i = j
    return tokens


def _parse_prefix_tokens(tokens: list, idx: int, symbols: dict) -> tuple:
    """Parse tokenized prefix expression into a SymPy expression.

    Returns (sympy_expr, next_idx).
    """
    token = tokens[idx]

    # Check if this is a function call: name(args...)
    if idx + 1 < len(tokens) and tokens[idx + 1] == "(":
        func_name = token
        idx += 2  # skip name and '('
        args = []
        while tokens[idx] != ")":
            if tokens[idx] == ",":
                idx += 1
                continue
            arg, idx = _parse_prefix_tokens(tokens, idx, symbols)
            args.append(arg)
        idx += 1  # skip ')'

        # Apply known function or treat as symbolic
        if func_name in _GP_TO_SYMPY:
            return _GP_TO_SYMPY[func_name](*args), idx
        else:
            # Unknown function (ts_mean_5, rank, etc.) → symbolic Function
            func = sp.Function(func_name)
            return func(*args), idx

    # Terminal: symbol or number
    if token in symbols:
        return symbols[token], idx + 1

    # Try to parse as number (use Integer for whole numbers to enable simplification)
    try:
        val = float(token)
        if val == int(val) and abs(val) < 1e15:
            return sp.Integer(int(val)), idx + 1
        return sp.Float(val), idx + 1
    except ValueError:
        return sp.Symbol(token), idx + 1


def _sympy_to_prefix(expr) -> str:
    """Convert a SymPy expression back to GP prefix notation."""
    # Atoms
    if expr.is_Symbol:
        return str(expr)
    if expr.is_Number:
        return str(expr)

    # Addition → add/sub chain
    if expr.is_Add:
        args = list(expr.args)
        result = _sympy_to_prefix(args[0])
        for arg in args[1:]:
            # Check for negative coefficient → subtraction
            if arg.is_Mul and len(arg.args) == 2 and arg.args[0] == -1:
                result = f"sub({result}, {_sympy_to_prefix(arg.args[1])})"
            elif arg.is_Number and arg < 0:
                result = f"sub({result}, {_sympy_to_prefix(-arg)})"
            else:
                result = f"add({result}, {_sympy_to_prefix(arg)})"
        return result

    # Multiplication → mul/div chain
    if expr.is_Mul:
        args = list(expr.args)
        numerators = []
        denominators = []
        for arg in args:
            if arg.is_Pow and len(arg.args) == 2 and arg.args[1] == -1:
                denominators.append(arg.args[0])
            elif arg.is_Number and arg < 0:
                # Negative coefficient → neg
                numerators.append(sp.Mul(-arg, *args[args.index(arg) + 1:]))
                break
            else:
                numerators.append(arg)

        if not numerators:
            numerators = [sp.Integer(1)]

        # Build numerator
        num_str = _sympy_to_prefix(numerators[0])
        for n in numerators[1:]:
            num_str = f"mul({num_str}, {_sympy_to_prefix(n)})"

        if not denominators:
            return num_str

        # Build denominator
        den_str = _sympy_to_prefix(denominators[0])
        for d in denominators[1:]:
            den_str = f"mul({den_str}, {_sympy_to_prefix(d)})"

        return f"div({num_str}, {den_str})"

    # Power
    if expr.is_Pow:
        base, exp = expr.args
        if exp == sp.Rational(1, 2):
            return f"sqrt({_sympy_to_prefix(base)})"
        if exp == -1:
            return f"div(1, {_sympy_to_prefix(base)})"
        # General power not in GP primitive set
        return f"pow({_sympy_to_prefix(base)}, {_sympy_to_prefix(exp)})"

    # Abs
    if expr.func is sp.Abs:
        return f"abs({_sympy_to_prefix(expr.args[0])})"

    # log
    if expr.func is sp.log:
        return f"log({_sympy_to_prefix(expr.args[0])})"

    # Max / Min
    if expr.func is sp.Max:
        args = expr.args
        result = _sympy_to_prefix(args[0])
        for a in args[1:]:
            result = f"max({result}, {_sympy_to_prefix(a)})"
        return result

    if expr.func is sp.Min:
        args = expr.args
        result = _sympy_to_prefix(args[0])
        for a in args[1:]:
            result = f"min({result}, {_sympy_to_prefix(a)})"
        return result

    # tanh
    if expr.func is sp.tanh:
        return f"tanh({_sympy_to_prefix(expr.args[0])})"

    # exp (from sigmoid expansion)
    if expr.func is sp.exp:
        return f"exp({_sympy_to_prefix(expr.args[0])})"

    # Symbolic function (ts_mean_5, rank, etc.)
    if expr.is_Function:
        func_name = expr.func.__name__
        args_str = ", ".join(_sympy_to_prefix(a) for a in expr.args)
        return f"{func_name}({args_str})"

    # Fallback: string representation
    return str(expr)


def simplify_gp_expression(expr_str: str) -> str:
    """Simplify a GP prefix expression using SymPy.

    Parses the DEAP prefix notation (e.g. ``add(factor_0, sub(factor_1, factor_0))``)
    into a SymPy expression, applies :func:`sympy.simplify`, and converts the
    result back to prefix notation.

    SymPy auto-simplifies during construction (e.g. ``x + 0`` → ``x``),
    so we compare the original string against the converted result rather
    than comparing op counts.

    If SymPy is unavailable or simplification fails, the original expression
    is returned unchanged.

    Examples
    --------
    >>> simplify_gp_expression("add(factor_0, sub(factor_1, factor_0))")
    'factor_1'
    >>> simplify_gp_expression("mul(factor_0, div(factor_0, factor_0))")
    'factor_0'
    """
    if not _SYMPY_AVAILABLE:
        return expr_str

    try:
        # Collect all factor_N symbols from the expression
        tokens = _tokenize_prefix(expr_str)
        symbol_names = set()
        for t in tokens:
            if t.startswith("factor_"):
                symbol_names.add(t)

        symbols = {name: sp.Symbol(name) for name in symbol_names}

        # Parse prefix → SymPy (auto-simplifies during construction)
        sympy_expr, _ = _parse_prefix_tokens(tokens, 0, symbols)

        # Explicit simplify for additional reductions
        simplified = sp.simplify(sympy_expr)

        # Convert back to prefix notation
        result = _sympy_to_prefix(simplified)

        # Only return simplified result if it differs from the original
        # (i.e. actual simplification occurred)
        if result != expr_str:
            return result
        return expr_str
    except Exception as e:
        logger.warning(f"表达式简化失败 '{expr_str[:80]}': {e}")
        return expr_str


def sympy_canonical_key(tree) -> str:
    """Compute a SymPy canonical-form key for a DEAP PrimitiveTree.

    This is used for **algebraic-equivalence deduplication** during
    evaluation (P1): unlike :func:`zobrist_hash` which only detects
    structural isomorphism, this detects algebraic equivalence such as
    ``add(a, sub(b, a))`` ≡ ``b``.

    The canonical key is the SymPy ``srepr()`` of the simplified
    expression, which is deterministic and order-independent.

    If SymPy is unavailable or simplification fails, returns the
    plain prefix string (degraded dedup, still correct).

    Performance note
    ----------------
    ``sp.simplify`` is ~1-5ms for small expressions. Callers should
    only invoke this when the fast ``zobrist_hash`` lookup misses,
    so the overhead is amortized over cache hits.
    """
    expr_str = tree_to_placeholder_expr(tree)
    if not _SYMPY_AVAILABLE:
        return expr_str

    try:
        tokens = _tokenize_prefix(expr_str)
        symbol_names = {t for t in tokens if t.startswith("factor_")}
        symbols = {name: sp.Symbol(name) for name in symbol_names}

        sympy_expr, _ = _parse_prefix_tokens(tokens, 0, symbols)
        simplified = sp.simplify(sympy_expr)
        # srepr gives a deterministic, canonical string representation
        return sp.srepr(simplified)
    except Exception:
        # Fallback: use prefix string as key (still correct, just less dedup)
        return expr_str


def _parse_math_expr_to_sympy(expr_str: str):
    """Parse a standard math expression into a SymPy expression.

    Extracts method chains (e.g. ``close.rolling(20).mean()``) as atomic
    placeholders, distinguishes function names from variable names (to avoid
    ``open`` being parsed as a builtin), and returns the SymPy expression
    along with the placeholder→chain mapping.

    Returns
    -------
    tuple or None
        ``(sympy_expr, chains)`` on success, ``None`` on failure.
    """
    if not _SYMPY_AVAILABLE:
        return None

    import ast as _ast

    try:
        tree = _ast.parse(expr_str, mode="eval")
    except SyntaxError:
        return None

    chains = {}

    class _MethodChainExtractor(_ast.NodeTransformer):
        def visit_Call(self, node):
            # 方法调用 (func 是 Attribute，如 close.rolling(20).mean())
            if isinstance(node.func, _ast.Attribute):
                chain_str = _ast.unparse(node)
                existing = next(
                    (k for k, v in chains.items() if v == chain_str), None
                )
                if existing:
                    return _ast.Name(id=existing, ctx=_ast.Load())
                placeholder = f"_v{len(chains)}"
                chains[placeholder] = chain_str
                return _ast.Name(id=placeholder, ctx=_ast.Load())
            # 普通函数调用 (如 safe_divide(a, b))，递归处理参数
            self.generic_visit(node)
            return node

    extractor = _MethodChainExtractor()
    new_tree = extractor.visit(tree)
    math_expr = _ast.unparse(new_tree)

    try:
        # 区分函数名和变量名，避免 open 等内置名被误解析
        function_names = set()
        variable_names = set()
        for node in _ast.walk(new_tree):
            if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
                function_names.add(node.func.id)
            elif isinstance(node, _ast.Name):
                variable_names.add(node.id)

        local_dict = {}
        for name in function_names:
            local_dict[name] = sp.Function(name)
        for name in variable_names - function_names:
            local_dict[name] = sp.Symbol(name)

        sympy_expr = sp.parse_expr(
            math_expr, local_dict=local_dict, evaluate=True
        )
        return sympy_expr, chains
    except Exception as e:
        logger.warning(f"SymPy解析失败 '{math_expr}': {e}")
        return None


def simplify_math_expression(expr_str: str) -> tuple:
    """Simplify a standard math expression (with optional pandas method chains).

    Extracts method chains (e.g. ``close.rolling(20).mean()``) as atomic
    placeholders, simplifies the remaining pure-math part with SymPy, then
    substitutes the placeholders back.

    Unlike :func:`simplify_gp_expression` which handles DEAP prefix notation,
    this works on standard Python math syntax such as ``(close - open) / close``.

    Returns
    -------
    tuple
        ``(simplified_expr, changed)`` where *changed* is True when the
        simplified result differs from the input.
    """
    import re as _re

    parsed = _parse_math_expr_to_sympy(expr_str)
    if parsed is None:
        return expr_str, False

    sympy_expr, chains = parsed

    try:
        simplified = sp.simplify(sympy_expr)
        # 尝试 expand 获取更简形式（如 (close-open)/close → 1-open/close）
        expanded = sp.expand(simplified)
        if len(str(expanded)) <= len(str(simplified)):
            simplified = expanded
        result = str(simplified)
    except Exception as e:
        logger.warning(f"SymPy简化失败: {e}")
        return expr_str, False

    # 替换占位符回原方法链（用词边界精确匹配）
    for placeholder, chain in chains.items():
        result = _re.sub(
            r"\b" + _re.escape(placeholder) + r"\b",
            lambda m, c=chain: c,
            result,
        )

    # 规范化比较（去除空白差异）
    original_normalized = _re.sub(r"\s+", "", expr_str)
    result_normalized = _re.sub(r"\s+", "", result)
    if result_normalized == original_normalized:
        return expr_str, False
    return result, True


def math_expression_canonical_key(expr_str: str) -> Optional[str]:
    """Compute a canonical-form key for a standard math expression.

    Two expressions that are algebraically equivalent (e.g.
    ``(close - open) / close`` and ``1 - open/close``) will produce the
    same key, enabling duplicate detection across the factor library.

    Returns ``None`` if SymPy is unavailable or parsing fails.
    """
    parsed = _parse_math_expr_to_sympy(expr_str)
    if parsed is None:
        return None

    sympy_expr, _ = parsed
    try:
        simplified = sp.simplify(sympy_expr)
        return sp.srepr(simplified)
    except Exception:
        return None

