"""
收益率计算工具

统一提供未来收益率、因子IC统计量等跨服务复用的计算方法。
"""

import numpy as np
import pandas as pd
from typing import Dict, Sequence, Tuple
from scipy import stats as scipy_stats

from backend.utils.safe_math import safe_divide, safe_ir


def calculate_future_returns(
    df: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10, 20),
    price_col: str = "close",
) -> pd.DataFrame:
    """
    计算多期未来收益率

    统一入口，消除各服务中重复的未来收益率计算代码。
    注意：shift(-N) 用于获取未来收益（结果变量），属于合法操作，不是前视偏差。

    Args:
        df: 包含价格数据的 DataFrame
        periods: 收益率计算周期列表
        price_col: 价格列名

    Returns:
        添加了 future_return_N 列的 DataFrame（副本）
    """
    result = df.copy()
    if isinstance(result.index, pd.MultiIndex):
        # MultiIndex 下按资产分组计算，避免跨资产比较价格
        # 必须先提取为纯 DatetimeIndex Series 再计算 pct_change/shift
        asset_level = 1
        for p in periods:
            col_name = f"future_return_{p}"
            new_col = pd.Series(np.nan, index=result.index, name=col_name)
            for asset_code in result.index.get_level_values(asset_level).unique():
                # 使用 xs 提取单资产数据（返回 DatetimeIndex DataFrame），再计算
                asset_df = result.xs(asset_code, level=asset_level)
                ret = asset_df[price_col].pct_change(p).shift(-p)
                # 将结果按日期索引对齐赋值
                for date in ret.index:
                    new_col.loc[(date, asset_code)] = ret.loc[date] if not pd.isna(ret.loc[date]) else np.nan
            result[col_name] = new_col
    else:
        for p in periods:
            result[f"future_return_{p}"] = result[price_col].pct_change(p).shift(-p)
    return result


def calculate_ic_stats(
    ic_series: pd.Series,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    计算IC序列的统计量（统一入口）

    消除 analysis_service、statistics_service、weighted_ic_service、
    alphalens_analysis_service 中重复的IC统计量计算代码。

    Args:
        ic_series: IC值序列
        confidence_level: 置信水平

    Returns:
        包含 IC 均值、标准差、IR、t统计量、p值、置信区间的字典
    """
    ic_clean = ic_series.dropna()
    n = len(ic_clean)

    if n < 2:
        return {
            "mean_ic": None,
            "std_ic": None,
            "ir": None,
            "t_statistic": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_samples": n,
            "positive_ratio": None,
        }

    mean_ic = float(ic_clean.mean())
    std_ic = float(ic_clean.std())

    # 常数IC序列：std接近0时（规则7.15）
    if std_ic < 1e-10:
        positive_ratio = float((ic_clean > 0).mean())
        if abs(mean_ic) > 1e-10:
            t_statistic = float("inf")
            p_value = 0.0
        else:
            t_statistic = 0.0
            p_value = 1.0
        return {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ir": None,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "ci_lower": None,
            "ci_upper": None,
            "n_samples": n,
            "positive_ratio": positive_ratio,
        }

    ir = safe_ir(float(mean_ic), float(std_ic), default=None)

    # t检验
    se = std_ic / np.sqrt(n)
    # se guaranteed positive: std_ic >= 1e-10 (after guard above), n >= 2
    t_statistic = float(mean_ic) / float(se)
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_statistic), df=n - 1))

    # 置信区间
    alpha = 1 - confidence_level
    t_critical = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean_ic - t_critical * se
    ci_upper = mean_ic + t_critical * se

    positive_ratio = float((ic_clean > 0).mean())

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": ir,
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_samples": n,
        "positive_ratio": positive_ratio,
    }


def calculate_rolling_ir(
    ic_series: pd.Series,
    window: int = 20,
    min_periods: int = 10,
) -> Tuple[float, float, float]:
    """
    向量化计算滚动IC均值、IC标准差和IR

    消除 factor_validation_service、factor_generator_service、
    factor_stability_service 中重复的滚动IR计算代码。

    Args:
        ic_series: IC值序列
        window: 滚动窗口大小
        min_periods: 最小观测数

    Returns:
        (ic_mean, ic_std, ir) 元组
    """
    ic_clean = ic_series.dropna()
    if len(ic_clean) < min_periods:
        return None, None, None

    # 逐窗口计算IR后取均值，而非 mean(rolling_mean)/mean(rolling_std)
    # 由Jensen不等式 E[X/Y] ≠ E[X]/E[Y]，后者会产生偏差
    rolling_mean = ic_clean.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = ic_clean.rolling(window=window, min_periods=min_periods).std()
    rolling_ir = safe_divide(rolling_mean, rolling_std, default=None)
    rolling_ir_clean = rolling_ir.dropna()

    if len(rolling_ir_clean) == 0:
        return None, None, None

    ic_mean = float(rolling_mean.dropna().mean())
    ic_std = float(rolling_std.dropna().mean())
    ir = float(rolling_ir_clean.mean())

    return ic_mean, ic_std, ir
