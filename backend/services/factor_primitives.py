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
from deap import gp
from typing import Optional
from functools import partial
from scipy.stats import spearmanr

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
    """Sigmoid activation (clamps output to [0, 1] range)."""
    x = np.clip(a, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


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
    for var_name in sorted(base_factor_codes, key=len, reverse=True):
        code = base_factor_codes[var_name]
        expr = expr.replace(var_name, f"({code})")
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
