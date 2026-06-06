"""
统一风险指标计算模块

整合 empyrical（优先）和手动计算（回退），避免跨文件重复实现。
所有需要计算 Sharpe/Sortino/MaxDD/Calmar/VaR/CVaR 的地方应使用此模块。
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional

try:
    import empyrical
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False


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

    if EMPYRICAL_AVAILABLE:
        return _calculate_with_empyrical(returns_arr, risk_free_rate, annual_trading_days)
    else:
        return _calculate_manual(returns_clean, risk_free_rate, annual_trading_days)


def _calculate_with_empyrical(
    returns_arr: np.ndarray,
    risk_free_rate: float,
    annual_trading_days: int,
) -> Dict[str, float]:
    """使用 empyrical 计算风险指标"""
    return {
        "total_return": float(empyrical.cum_returns_final(returns_arr)),
        "annual_return": float(empyrical.annual_return(
            returns_arr, period='daily', annualization=annual_trading_days
        )),
        "volatility": float(empyrical.annual_volatility(
            returns_arr, period='daily', annualization=annual_trading_days
        )),
        "sharpe_ratio": float(empyrical.sharpe_ratio(
            returns_arr, risk_free=risk_free_rate,
            period='daily', annualization=annual_trading_days
        )),
        "sortino_ratio": float(empyrical.sortino_ratio(
            returns_arr, required_return=risk_free_rate,
            period='daily', annualization=annual_trading_days
        )),
        "max_drawdown": float(empyrical.max_drawdown(returns_arr)),
        "calmar_ratio": float(empyrical.calmar_ratio(
            returns_arr, period='daily', annualization=annual_trading_days
        )),
        "win_rate": float((returns_arr > 0).mean()),
        "var_95": float(np.percentile(returns_arr, 5)),
        "cvar_95": float(returns_arr[returns_arr <= np.percentile(returns_arr, 5)].mean())
            if len(returns_arr) > 0 else 0.0,
    }


def _calculate_manual(
    returns_clean: pd.Series,
    risk_free_rate: float,
    annual_trading_days: int,
) -> Dict[str, float]:
    """手动计算风险指标（empyrical 不可用时的回退方案）"""
    n = len(returns_clean)

    # 总收益率
    total_return = float((1 + returns_clean).prod() - 1)

    # 年化收益率（处理本金亏光的情况）
    if total_return <= -1.0:
        annual_return = -1.0
    else:
        annual_return = float((1 + total_return) ** (annual_trading_days / n) - 1)

    # 波动率
    volatility = float(returns_clean.std() * np.sqrt(annual_trading_days))

    # 夏普比率
    daily_rf = risk_free_rate / annual_trading_days
    excess_returns = returns_clean - daily_rf
    sharpe_ratio = (
        float(excess_returns.mean() / excess_returns.std() * np.sqrt(annual_trading_days))
        if excess_returns.std() > 0 else 0.0
    )

    # 最大回撤
    equity = (1 + returns_clean).cumprod()
    peak = equity.cummax()
    drawdown = (peak - equity) / peak
    max_drawdown = float(drawdown.max())

    # 卡玛比率
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0.0001 else 0.0

    # Sortino（标准下行偏差公式，扣除无风险利率）
    downside_diff = (returns_clean - daily_rf).clip(upper=0)
    downside_std = float(np.sqrt((downside_diff ** 2).mean()) * np.sqrt(annual_trading_days))
    sortino_ratio = (
        float(excess_returns.mean() * annual_trading_days / downside_std)
        if downside_std > 0 else 0.0
    )

    # VaR / CVaR
    var_95 = float(returns_clean.quantile(0.05)) if n > 0 else 0.0
    cvar_95 = float(returns_clean[returns_clean <= var_95].mean()) if n > 0 else 0.0

    # 胜率
    win_rate = float((returns_clean > 0).mean())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar_ratio),
        "win_rate": win_rate,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def _empty_metrics() -> Dict[str, float]:
    """返回空的风险指标"""
    return {
        "total_return": 0.0,
        "annual_return": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
    }
