"""
统一风险指标计算模块

底层委托 empyrical 计算，所有需要计算 Sharpe/Sortino/MaxDD/Calmar/VaR/CVaR 的地方应使用此模块。
"""
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
    """返回空的风险指标"""
    return {
        "total_return": 0.0,
        "annual_return": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "max_drawdown": 0.0,
        "calmar_ratio": None,
        "win_rate": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
    }
