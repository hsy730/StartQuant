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
        # 纯 numpy 向量化滚动 Spearman 相关
        # 算法：Spearman = 对排名序列的 Pearson 相关
        # 使用 sliding_window_view + 向量化排名/相关，避免 .rolling().apply(spearmanr)
        # 性能：~222 个窗口从 ~5s 降至 ~50ms（加速 ~100x）
        from numpy.lib.stride_tricks import sliding_window_view

        min_periods = max(2, window // 2)
        f_arr = aligned["factor"].values.astype(np.float64)
        r_arr = aligned["returns"].values.astype(np.float64)
        length = len(f_arr)

        if length < window:
            return pd.Series(dtype=float)

        result = np.full(length, np.nan)

        try:
            f_win = sliding_window_view(f_arr, window)   # (n_windows, window)
            r_win = sliding_window_view(r_arr, window)
        except AttributeError:
            # numpy < 1.20 不支持 sliding_window_view，回退到旧实现
            def _rolling_spearman(x):
                valid_x = x.dropna()
                if len(valid_x) < min_periods:
                    return np.nan
                y = aligned["returns"].loc[x.index]
                valid_y = y.reindex(valid_x.index).dropna()
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

        n_windows = f_win.shape[0]

        # ---- 向量化排名（每行独立） ----
        # NaN → inf 排到末尾，argsort → 再 argsort 得到秩
        f_safe = np.where(np.isnan(f_win), np.inf, f_win)
        r_safe = np.where(np.isnan(r_win), np.inf, r_win)
        f_order = np.argsort(f_safe, axis=1, kind='stable')
        r_order = np.argsort(r_safe, axis=1, kind='stable')
        row_idx = np.arange(n_windows)[:, None]
        f_rank = np.empty_like(f_order, dtype=np.float64)
        r_rank = np.empty_like(r_order, dtype=np.float64)
        f_rank[row_idx, f_order] = np.arange(1, window + 1, dtype=np.float64)
        r_rank[row_idx, r_order] = np.arange(1, window + 1, dtype=np.float64)
        # 还原 NaN 位置
        f_rank[np.isnan(f_win)] = np.nan
        r_rank[np.isnan(r_win)] = np.nan

        # ---- 向量化 Pearson 相关（对排名序列） ----
        valid = (~np.isnan(f_rank)) & (~np.isnan(r_rank))
        n_valid = valid.sum(axis=1)
        ok = n_valid >= min_periods

        if not np.any(ok):
            return pd.Series(result, index=aligned.index)

        fr_ok = f_rank[ok]
        rr_ok = r_rank[ok]
        nv_ok = n_valid[ok]

        # 去均值
        fr_mean = np.nansum(fr_ok, axis=1, keepdims=True) / nv_ok[:, None]
        rr_mean = np.nansum(rr_ok, axis=1, keepdims=True) / nv_ok[:, None]
        fc = fr_ok - fr_mean
        rc = rr_ok - rr_mean

        # 协方差 / 标准差
        cov = np.nansum(fc * rc, axis=1)
        var_f = np.nansum(fc * fc, axis=1)
        var_r = np.nansum(rc * rc, axis=1)
        denom = np.sqrt(var_f * var_r)

        corr = np.where(denom > 1e-10, cov / denom, np.nan)

        # 写回结果（滑动窗口的最后一个位置对应原始索引）
        result[window - 1:][ok] = corr
        return pd.Series(result, index=aligned.index)
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
    # 向量化构建面板数据，替代三层嵌套循环
    all_rows = []
    for code, df in factor_data.items():
        if factor_name not in df.columns or return_column not in df.columns:
            continue
        stock_data = df[[factor_name, return_column]].copy()
        stock_data["_stock_code"] = code
        if isinstance(df.index, pd.DatetimeIndex):
            stock_data["_date"] = df.index
        elif "date" in df.columns:
            stock_data["_date"] = df["date"]
        else:
            stock_data["_date"] = df.index
        all_rows.append(stock_data)

    if not all_rows:
        return None

    panel = pd.concat(all_rows, ignore_index=False)
    panel["_date"] = pd.to_datetime(panel["_date"])

    # 清理无效值
    valid_mask = (
        panel[factor_name].notna()
        & panel[return_column].notna()
        & ~np.isinf(panel[factor_name])
        & ~np.isinf(panel[return_column])
    )
    panel = panel[valid_mask]

    return _calculate_cross_sectional_ic_from_panel(
        panel, factor_name, return_column, date_column="_date",
        min_stocks=min_stocks, min_dates=min_dates, method=method,
    )


def calculate_cross_sectional_ic_from_panel(
    panel_df: pd.DataFrame,
    factor_column: str,
    return_column: str,
    date_column: Optional[str] = None,
    min_stocks: int = 5,
    min_dates: int = 2,
    method: str = "spearman",
) -> Optional[dict]:
    """
    从面板DataFrame计算横截面IC（统一入口）

    当调用方已经构建好面板DataFrame时，直接使用此函数，
    避免再经过 dict → panel 的转换开销。

    Args:
        panel_df: 面板DataFrame，包含因子列、收益率列和日期信息
        factor_column: 因子列名
        return_column: 收益率列名
        date_column: 日期列名，None时使用DataFrame索引
        min_stocks: 每个截面最少股票数
        min_dates: 最少有效截面数
        method: "spearman" 或 "pearson"

    Returns:
        同 calculate_cross_sectional_ic
    """
    panel = panel_df.copy()
    if date_column is not None and date_column in panel.columns:
        panel["_date"] = pd.to_datetime(panel[date_column])
    else:
        panel["_date"] = pd.to_datetime(panel.index)

    return _calculate_cross_sectional_ic_from_panel(
        panel, factor_column, return_column, date_column="_date",
        min_stocks=min_stocks, min_dates=min_dates, method=method,
    )


def _calculate_cross_sectional_ic_from_panel(
    panel: pd.DataFrame,
    factor_column: str,
    return_column: str,
    date_column: str = "_date",
    min_stocks: int = 5,
    min_dates: int = 2,
    method: str = "spearman",
) -> Optional[dict]:
    """内部实现：从面板DataFrame向量化计算横截面IC"""
    # 按日期分组，向量化计算每日IC
    ic_list = []

    def _daily_ic(group):
        if len(group) < min_stocks:
            return np.nan
        fv = group[factor_column].values
        rv = group[return_column].values
        # 规则7.6/7.40：零标准差阈值统一使用 1e-10
        if np.std(fv) < 1e-10 or np.std(rv) < 1e-10:
            return np.nan
        if method == "spearman":
            ic_val, _ = spearmanr(fv, rv)
        else:
            ic_val = np.corrcoef(fv, rv)[0, 1]
        return ic_val if not np.isnan(ic_val) else np.nan

    daily_ic = panel.groupby(date_column).apply(_daily_ic)
    daily_ic = daily_ic.dropna()

    ic_list = daily_ic.tolist()

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
