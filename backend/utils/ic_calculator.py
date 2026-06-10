"""
IC（信息系数）计算公共工具 — 统一IC计算方式

项目规范5：IC计算在30+处各自实现，提取统一入口
TODO: 逐步迁移各服务中的IC计算到此处
"""
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import spearmanr


def calculate_ic(
    factor: pd.Series,
    returns: pd.Series,
    method: str = "spearman",
    min_samples: int = 10,
) -> Optional[float]:
    """
    计算单期IC（因子与收益率的相关系数）

    Args:
        factor: 因子值Series
        returns: 收益率Series
        method: "pearson" 或 "spearman"（默认spearman，符合业界标准）
        min_samples: 最小有效样本量

    Returns:
        IC值，样本不足时返回None
    """
    valid = factor.notna() & returns.notna()
    n = valid.sum()
    if n < min_samples:
        return None

    if method == "spearman":
        ic = float(factor[valid].rank().corr(returns[valid].rank()))
        return ic if not np.isnan(ic) else None
    elif method == "pearson":
        ic = float(factor[valid].corr(returns[valid]))
        return ic if not np.isnan(ic) else None
    else:
        raise ValueError(f"Unsupported IC method: {method}. Use 'spearman' or 'pearson'.")


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
    method: str = "spearman",
) -> pd.Series:
    """
    计算滚动IC

    Args:
        factor: 因子值Series
        returns: 收益率Series
        window: 滚动窗口大小
        method: "pearson" 或 "spearman"（默认spearman，符合业界标准）

    Returns:
        滚动IC Series
    """
    aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    if method == "spearman":
        # 逐窗口计算Spearman相关（全局rank+rolling Pearson不等于per-window Spearman）
        min_periods = max(2, window // 2)

        def _rolling_spearman(x):
            valid_x = x.dropna()
            if len(valid_x) < min_periods:
                return np.nan
            y = aligned["returns"].loc[x.index]
            valid_y = y.reindex(valid_x.index).dropna()
            # 重新对齐：只保留两个序列都有效的位置
            common_idx = valid_x.index.intersection(valid_y.index)
            if len(common_idx) < min_periods:
                return np.nan
            vx = valid_x.loc[common_idx]
            vy = valid_y.loc[common_idx]
            return spearmanr(vx, vy)[0]

        return aligned["factor"].rolling(window, min_periods=min_periods).apply(
            _rolling_spearman, raw=False
        )
    else:
        min_periods = max(2, window // 2)
        return aligned["factor"].rolling(window, min_periods=min_periods).corr(aligned["returns"])
