"""
收益率计算工具

统一提供未来收益率、因子IC统计量等跨服务复用的计算方法。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple
from scipy import stats as scipy_stats


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
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "ir": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_samples": n,
            "positive_ratio": 0.0,
        }

    mean_ic = float(ic_clean.mean())
    std_ic = float(ic_clean.std())
    ir = mean_ic / std_ic if std_ic > 0 else 0.0

    # t检验
    se = std_ic / np.sqrt(n)
    t_statistic = mean_ic / se if se > 0 else 0.0
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
        return 0.0, 0.0, 0.0

    rolling_ic = ic_clean.rolling(window=window, min_periods=min_periods).mean()
    ic_mean = float(rolling_ic.mean())

    rolling_std = ic_clean.rolling(window=window, min_periods=min_periods).std()
    ic_std = float(rolling_std.mean())

    ir = ic_mean / ic_std if ic_std > 0 else 0.0

    return ic_mean, ic_std, ir
