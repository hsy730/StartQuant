"""
IC（信息系数）计算公共工具 — 统一IC计算方式

项目规范5：IC计算在30+处各自实现，提取统一入口
TODO: 逐步迁移各服务中的IC计算到此处
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def calculate_ic(
    factor: pd.Series,
    returns: pd.Series,
    method: str = "pearson",
    min_samples: int = 10,
) -> Optional[float]:
    """
    计算单期IC（因子与收益率的相关系数）

    Args:
        factor: 因子值Series
        returns: 收益率Series
        method: "pearson" 或 "spearman"
        min_samples: 最小有效样本量

    Returns:
        IC值，样本不足时返回None
    """
    valid = factor.notna() & returns.notna()
    n = valid.sum()
    if n < min_samples:
        return None

    if method == "spearman":
        return float(factor[valid].rank().corr(returns[valid].rank()))
    else:
        return float(factor[valid].corr(returns[valid]))


def calculate_rank_ic(
    factor: pd.Series,
    returns: pd.Series,
    min_samples: int = 10,
) -> Optional[float]:
    """计算Rank IC（Spearman相关系数）"""
    return calculate_ic(factor, returns, method="spearman", min_samples=min_samples)


def calculate_rolling_ic(
    factor: pd.Series,
    returns: pd.Series,
    window: int = 20,
    method: str = "pearson",
) -> pd.Series:
    """
    计算滚动IC

    Args:
        factor: 因子值Series
        returns: 收益率Series
        window: 滚动窗口大小
        method: "pearson" 或 "spearman"

    Returns:
        滚动IC Series
    """
    aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    if method == "spearman":
        # Spearman相关 = 两个变量排名的Pearson相关
        # 必须对factor和returns都做排名，而非只对returns排名
        factor_rank = aligned["factor"].rank()
        returns_rank = aligned["returns"].rank()
        return factor_rank.rolling(window).corr(returns_rank)
    else:
        return aligned["factor"].rolling(window).corr(aligned["returns"])
