"""
DEAP GP primitives for factor mining.

Defines protected operators and the PrimitiveSet used by the genetic
programming engine to build factor expressions with guaranteed syntactic
correctness.
"""

import numpy as np
import pandas as pd
from deap import gp


# ---------------------------------------------------------------------------
# Protected (safe) operators – avoid division-by-zero / log-of-negative
# ---------------------------------------------------------------------------

def safe_div(a, b):
    """x / y  with y=0 mapped to NaN."""
    return a / (b.replace(0, np.nan) + 1e-8)


def safe_log(a):
    """log(x)  with x <= 0 mapped to NaN."""
    return np.log(a.clip(lower=1e-10))


def safe_sqrt(a):
    """sqrt(x)  with x < 0 mapped to NaN."""
    return np.sqrt(a.clip(lower=0))


def pct_rank(a):
    """Cross-sectional percentile rank."""
    return a.rank(pct=True)


# ---------------------------------------------------------------------------
# PrimitiveSet factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Time-series window operators (Phase 7)
# ---------------------------------------------------------------------------

def ts_mean(a, n=5):
    """Rolling mean over *n* periods."""
    return a.rolling(window=int(n), min_periods=1).mean()


def ts_std(a, n=5):
    """Rolling std over *n* periods."""
    return a.rolling(window=int(n), min_periods=1).std()


def ts_delay(a, n=1):
    """Lag *a* by *n* periods (REF in 麦语言)."""
    return a.shift(int(n))


def ts_delta(a, n=1):
    """Difference: a - a.shift(n)."""
    return a - a.shift(int(n))


def ts_corr(a, b, n=5):
    """Rolling Pearson correlation between *a* and *b* over *n* periods."""
    return a.rolling(window=int(n), min_periods=2).corr(b)


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
    """Hyperbolic tangent activation (output in [-1, 1])."""
    return np.tanh(np.clip(a, -500, 500))


# ---------------------------------------------------------------------------
# PrimitiveSet factory
# ---------------------------------------------------------------------------

def create_pset(n_factors: int, extended: bool = True) -> gp.PrimitiveSet:
    """Build a DEAP ``PrimitiveSet`` for factor expressions.

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
    ts_corr_5      2         5-period rolling correlation
    ts_corr_10     2         10-period rolling correlation
    ts_corr_20     2         20-period rolling correlation
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
    pset.addPrimitive(np.add,       2, name="add")
    pset.addPrimitive(np.subtract,  2, name="sub")
    pset.addPrimitive(np.multiply,  2, name="mul")
    pset.addPrimitive(safe_div,     2, name="div")

    # unary
    pset.addPrimitive(np.negative,  1, name="neg")
    pset.addPrimitive(np.abs,       1, name="abs")
    pset.addPrimitive(safe_log,     1, name="log")
    pset.addPrimitive(safe_sqrt,    1, name="sqrt")
    pset.addPrimitive(pct_rank,     1, name="rank")

    # ---- Extended primitives (Phase 7, +16) ----
    if extended:
        # Time-series window operations (unary, fixed window)
        pset.addPrimitive(lambda a: ts_mean(a, 5),   1, name="ts_mean_5")
        pset.addPrimitive(lambda a: ts_mean(a, 10),  1, name="ts_mean_10")
        pset.addPrimitive(lambda a: ts_mean(a, 20),  1, name="ts_mean_20")
        pset.addPrimitive(lambda a: ts_std(a, 5),    1, name="ts_std_5")
        pset.addPrimitive(lambda a: ts_std(a, 10),   1, name="ts_std_10")
        pset.addPrimitive(lambda a: ts_std(a, 20),   1, name="ts_std_20")
        pset.addPrimitive(lambda a: ts_delay(a, 1),  1, name="ts_delay_1")
        pset.addPrimitive(lambda a: ts_delay(a, 5),  1, name="ts_delay_5")
        pset.addPrimitive(lambda a: ts_delta(a, 1),  1, name="ts_delta_1")
        pset.addPrimitive(lambda a: ts_delta(a, 5),  1, name="ts_delta_5")

        # Time-series correlation (binary, fixed window)
        pset.addPrimitive(lambda a, b: ts_corr(a, b, 5),   2, name="ts_corr_5")
        pset.addPrimitive(lambda a, b: ts_corr(a, b, 10),  2, name="ts_corr_10")
        pset.addPrimitive(lambda a, b: ts_corr(a, b, 20),  2, name="ts_corr_20")

        # Pairwise operations
        pset.addPrimitive(_pair_max,  2, name="max")
        pset.addPrimitive(_pair_min,  2, name="min")

        # Activation functions
        pset.addPrimitive(_sigmoid,   1, name="sigmoid")
        pset.addPrimitive(_tanh,      1, name="tanh")

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
    tokens_a = set(expr_a.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split())
    tokens_b = set(expr_b.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
