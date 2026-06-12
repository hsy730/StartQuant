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
        ic_val, _ = spearmanr(factor[valid], returns[valid])
        return float(ic_val) if not np.isnan(ic_val) else None
    elif method == "pearson":
        ic = float(factor[valid].corr(returns[valid]))
        return ic if not np.isnan(ic) else None
    else:
        raise ValueError(
            f"Unsupported IC method: {method}. Use 'spearman' or 'pearson'."
        )


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

        return (
            aligned["factor"]
            .rolling(window, min_periods=min_periods)
            .apply(_rolling_spearman, raw=False)
        )
    else:
        min_periods = max(2, window // 2)
        return (
            aligned["factor"]
            .rolling(window, min_periods=min_periods)
            .corr(aligned["returns"])
        )


def calculate_cross_sectional_ic(
    factor_data: dict,
    factor_name: str,
    return_column: str = "future_return",
    min_stocks: int = 5,
    min_dates: int = 2,
    method: str = "spearman",
) -> Optional[dict]:
    """
    计算横截面IC — 每个时间截面上计算因子与收益率的相关系数

    项目规范5/7.1：横截面IC计算逻辑在7处重复实现，提取统一入口。
    所有需要横截面IC的服务应统一调用此函数。

    Args:
        factor_data: {stock_code: DataFrame} 格式的因子数据，
                     每个DataFrame应包含因子列和收益率列
        factor_name: 因子列名
        return_column: 收益率列名，默认 "future_return"
        min_stocks: 每个截面最少股票数
        min_dates: 最少有效截面数
        method: "spearman" 或 "pearson"（默认spearman，符合业界标准）

    Returns:
        {
            "ic_list": [float, ...],      # 每日IC列表
            "mean_ic": float,             # IC均值
            "ic_std": float,              # IC标准差
            "n_dates": int,               # 有效截面数
            "ic_positive_ratio": float,   # IC>0占比
        }
        如果有效截面不足，返回 None
    """
    ic_list = []
    all_dates = set()
    for code, df in factor_data.items():
        if factor_name in df.columns and return_column in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index.tolist())
            elif "date" in df.columns:
                all_dates.update(df["date"].tolist())

    for date in sorted(all_dates):
        factor_vals = []
        return_vals = []
        for code, df in factor_data.items():
            if factor_name not in df.columns or return_column not in df.columns:
                continue
            if isinstance(df.index, pd.DatetimeIndex):
                if date in df.index:
                    row = df.loc[date]
                    if isinstance(row, pd.DataFrame):
                        for _, r in row.iterrows():
                            f, ret = r.get(factor_name), r.get(return_column)
                            if pd.notna(f) and pd.notna(ret):
                                factor_vals.append(f)
                                return_vals.append(ret)
                    else:
                        f, ret = row.get(factor_name), row.get(return_column)
                        if pd.notna(f) and pd.notna(ret):
                            factor_vals.append(f)
                            return_vals.append(ret)
            elif "date" in df.columns:
                date_rows = df[df["date"] == date]
                for _, r in date_rows.iterrows():
                    f, ret = r.get(factor_name), r.get(return_column)
                    if pd.notna(f) and pd.notna(ret):
                        factor_vals.append(f)
                        return_vals.append(ret)

        if len(factor_vals) < min_stocks:
            continue

        fv = np.array(factor_vals)
        rv = np.array(return_vals)
        # 规则7.6/7.40：零标准差阈值统一使用 1e-10
        if np.std(fv) < 1e-10 or np.std(rv) < 1e-10:
            continue

        if method == "spearman":
            ic_val, _ = spearmanr(fv, rv)
        else:
            ic_val = np.corrcoef(fv, rv)[0, 1]

        if not np.isnan(ic_val):
            ic_list.append(float(ic_val))

    if len(ic_list) < min_dates:
        return None

    ic_series = pd.Series(ic_list)
    return {
        "ic_list": ic_list,
        "mean_ic": float(ic_series.mean()),
        "ic_std": float(ic_series.std(ddof=1)) if len(ic_list) > 1 else 0.0,
        "n_dates": len(ic_list),
        "ic_positive_ratio": float((ic_series > 0).mean()),
    }
