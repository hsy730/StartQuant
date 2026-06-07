"""
统一风险指标计算模块

底层委托 empyrical 计算，所有需要计算 Sharpe/Sortino/MaxDD/Calmar/VaR/CVaR 的地方应使用此模块。
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

import empyrical


def calculate_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    annual_trading_days: int = 252,
) -> Dict[str, float]:
    """
    计算标准风险指标（统一入口）

    Args:
        returns: 日收益率序列
        risk_free_rate: 年化无风险利率
        annual_trading_days: 年化交易日数

    Returns:
        包含所有风险指标的字典
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return _empty_metrics()

    returns_arr = returns_clean.values

    # 标准差为零时直接返回空指标，避免 empyrical 产生极大值
    if np.std(returns_arr) == 0:
        return _empty_metrics()

    result = {
        "total_return": float(empyrical.cum_returns_final(returns_arr)),
        "annual_return": float(empyrical.annual_return(
            returns_arr, period='daily', annualization=annual_trading_days
        )),
        "volatility": float(empyrical.annual_volatility(
            returns_arr, period='daily', annualization=annual_trading_days
        )),
        "sharpe_ratio": float(empyrical.sharpe_ratio(
            returns_arr, risk_free=risk_free_rate / annual_trading_days,
            period='daily', annualization=annual_trading_days
        )),
        "sortino_ratio": float(empyrical.sortino_ratio(
            returns_arr, required_return=risk_free_rate / annual_trading_days,
            period='daily', annualization=annual_trading_days
        )),
        "max_drawdown": float(empyrical.max_drawdown(returns_arr)),
        "calmar_ratio": float(empyrical.calmar_ratio(
            returns_arr, period='daily', annualization=annual_trading_days
        )),
        "win_rate": float((returns_arr > 0).mean()),
        "var_95": float(empyrical.value_at_risk(returns_arr, cutoff=0.05)),
        "cvar_95": float(empyrical.conditional_value_at_risk(returns_arr, cutoff=0.05)),
    }

    # empyrical 在某些边界条件下可能返回非有限值，统一转为 None（符合规则6）
    for key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
        if not np.isfinite(result[key]):
            result[key] = None

    return result


def _empty_metrics() -> Dict[str, Optional[float]]:
    """返回空的风险指标（不可计算的指标返回None，符合规则6）"""
    return {
        "total_return": None,
        "annual_return": None,
        "volatility": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "max_drawdown": None,
        "calmar_ratio": None,
        "win_rate": None,
        "var_95": None,
        "cvar_95": None,
    }


def calculate_relative_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.03,
    annual_trading_days: int = 252,
) -> Dict[str, Optional[float]]:
    """
    计算相对风险指标（需要基准收益率）

    统一入口 — 所有需要计算 alpha/beta/information_ratio/tracking_error 的地方应使用此函数。

    Args:
        strategy_returns: 策略日收益率序列
        benchmark_returns: 基准日收益率序列
        risk_free_rate: 年化无风险利率
        annual_trading_days: 年化交易日数

    Returns:
        包含相对风险指标的字典
    """
    aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    if len(aligned) < 2:
        return _empty_relative_metrics()

    strategy_arr = aligned["strategy"].values
    benchmark_arr = aligned["benchmark"].values

    result = {}

    # Tracking Error
    excess_returns = aligned["strategy"] - aligned["benchmark"]
    tracking_error = float(excess_returns.std() * np.sqrt(annual_trading_days))
    result["tracking_error"] = tracking_error if np.isfinite(tracking_error) else None

    # Excess Return
    excess_return = float(excess_returns.mean() * annual_trading_days)
    result["excess_return"] = excess_return if np.isfinite(excess_return) else None

    # Information Ratio
    try:
        ir = float(empyrical.information_ratio(strategy_arr, benchmark_arr))
        result["information_ratio"] = ir if np.isfinite(ir) else None
    except Exception as e:
        logger.debug(f"信息比率计算失败: {e}")
        # Fallback: excess_return / tracking_error
        if tracking_error and tracking_error > 0 and result["excess_return"] is not None:
            result["information_ratio"] = result["excess_return"] / tracking_error
        else:
            result["information_ratio"] = None

    # Alpha & Beta
    try:
        alpha, beta = empyrical.alpha_beta_aligned(
            strategy_arr, benchmark_arr,
            risk_free=risk_free_rate / annual_trading_days,
            period='daily', annualization=annual_trading_days
        )
        result["alpha"] = float(alpha) if np.isfinite(alpha) else None
        result["beta"] = float(beta) if np.isfinite(beta) else None
    except Exception as e:
        logger.debug(f"alpha/beta计算失败: {e}")
        # Fallback: covariance / variance
        cov = float(aligned["strategy"].cov(aligned["benchmark"]))
        var = float(aligned["benchmark"].var())
        result["beta"] = cov / var if var > 0 else None
        result["alpha"] = None

    # Correlation
    correlation = float(aligned["strategy"].corr(aligned["benchmark"]))
    result["correlation"] = correlation if np.isfinite(correlation) else None

    return result


def _empty_relative_metrics() -> Dict[str, Optional[float]]:
    """返回空的相对风险指标"""
    return {
        "tracking_error": None,
        "excess_return": None,
        "information_ratio": None,
        "alpha": None,
        "beta": None,
        "correlation": None,
    }


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.03, annual_trading_days: int = 252) -> Optional[float]:
    """
    单独计算Sharpe比率（轻量接口，用于只需Sharpe的场景）

    Args:
        returns: 日收益率序列
        risk_free_rate: 年化无风险利率
        annual_trading_days: 年化交易日数

    Returns:
        Sharpe比率，不可计算时返回None
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return None
    returns_arr = returns_clean.values
    if np.std(returns_arr) == 0:
        return None
    result = float(empyrical.sharpe_ratio(
        returns_arr, risk_free=risk_free_rate / annual_trading_days,
        period='daily', annualization=annual_trading_days
    ))
    return result if np.isfinite(result) else None


def calculate_volatility(returns: pd.Series, annual_trading_days: int = 252) -> Optional[float]:
    """
    单独计算年化波动率（轻量接口）

    Args:
        returns: 日收益率序列
        annual_trading_days: 年化交易日数

    Returns:
        年化波动率，不可计算时返回None
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return None
    returns_arr = returns_clean.values
    result = float(empyrical.annual_volatility(
        returns_arr, period='daily', annualization=annual_trading_days
    ))
    return result if np.isfinite(result) else None
