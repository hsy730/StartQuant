"""
回测服务核心引擎（薄编排层 - 委托VectorBT金标准）

集成了新的策略系统，支持预置策略和策略对比。
所有核心回测方法委托给VectorBTBacktestService；
VectorBT不可用时抛出明确错误，不再使用有Bug的自建fallback。
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from backend.strategies.base_strategy import BaseStrategy
from backend.services.strategy_registry import strategy_registry
from backend.services.strategy_comparison_service import strategy_comparison_service
from backend.services.position_analysis_service import position_analysis_service
from backend.services.export_service import export_service
from backend.services.factor_preprocessing_pipeline import (
    FactorPreprocessingPipeline,
    PreprocessingConfig,
)

try:
    from backend.services.vectorbt_backtest_service import (
        VectorBTBacktestService,
        check_vectorbt_available,
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

logger = logging.getLogger(__name__)


class BacktestService:
    """回测服务 — 薄编排层（委托VectorBT金标准）"""

    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.0003):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self._vbt = None

    def _get_vbt(self) -> VectorBTBacktestService:
        """获取VectorBT实例，不可用时抛出明确错误"""
        if self._vbt is None and VECTORBT_AVAILABLE:
            self._vbt = VectorBTBacktestService(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
            )
        if self._vbt is None:
            raise ImportError(
                "VectorBT不可用，请安装: pip install vectorbt。"
                "自建回测引擎已移除（存在已知Bug），VectorBT是唯一支持的回测引擎。"
            )
        return self._vbt

    # ==================== 单因子回测 (委托VectorBT) ====================

    def single_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        percentile: int = 50,
        direction: str = "long",
        n_quantiles: int = 5,
        use_tradable_mask: bool = True,
    ) -> Dict:
        result = self._get_vbt().single_factor_backtest(
            df=df, factor_name=factor_name,
            percentile=percentile, direction=direction,
            n_quantiles=n_quantiles,
            use_tradable_mask=use_tradable_mask,
        )
        result["engine"] = "vectorbt"
        return result

    # ==================== 横截面回测 (委托VectorBT) ====================

    def cross_sectional_backtest(self, df: pd.DataFrame, factor_name: str, top_percentile: float = 0.2, direction: str = "long") -> Dict:
        result = self._get_vbt().cross_sectional_backtest(df=df, factor_name=factor_name, top_percentile=top_percentile, direction=direction)
        result["engine"] = "vectorbt"
        return result

    # ==================== 多因子回测 (委托VectorBT) ====================

    def multi_factor_backtest(self, df: pd.DataFrame, factor_names: List[str], weights: Optional[List[float]] = None, method: str = "equal_weight", percentile: int = 50, direction: str = "long") -> Dict:
        result = self._get_vbt().multi_factor_backtest(df=df, factor_names=factor_names, weights=weights, method=method, percentile=percentile, direction=direction)
        result["engine"] = "vectorbt"
        return result

    # ==================== 性能指标计算 (纯计算，保持自建) ====================

    def calculate_metrics(self, returns: pd.Series, annual_trading_days: int = 252, risk_free_rate: float = 0.03) -> Dict:
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return self._empty_metrics()
        total_return = (1 + returns_clean).prod() - 1
        n_days = len(returns_clean)
        annual_return = (1 + total_return) ** (annual_trading_days / n_days) - 1 if n_days > 0 else 0.0
        volatility = returns_clean.std() * np.sqrt(annual_trading_days)
        daily_rf = risk_free_rate / annual_trading_days
        excess_returns = returns_clean - daily_rf
        sharpe_ratio = excess_returns.mean() * annual_trading_days / volatility if volatility > 0 else 0.0
        equity = (1 + returns_clean).cumprod()
        peak = equity.cummax()
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        win_rate = (returns_clean > 0).mean()
        downside_returns = returns_clean[returns_clean < 0]
        sortino_ratio = (returns_clean.mean() * annual_trading_days - risk_free_rate) / (downside_returns.std() * np.sqrt(annual_trading_days)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0.0
        var_95 = returns_clean.quantile(0.05)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()
        return {
            "total_return": total_return, "annual_return": annual_return, "volatility": volatility,
            "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown, "calmar_ratio": calmar_ratio,
            "win_rate": win_rate, "sortino_ratio": sortino_ratio, "var_95": var_95, "cvar_95": cvar_95,
        }

    def _empty_metrics(self) -> Dict:
        return {"total_return": 0.0, "annual_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "calmar_ratio": 0.0, "win_rate": 0.0, "sortino_ratio": 0.0, "var_95": 0.0, "cvar_95": 0.0}

    # ==================== 回撤计算 ====================

    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        peak = equity_curve.cummax()
        return (peak - equity_curve) / peak

    # ==================== 信号生成 ====================

    def generate_signals(self, df: pd.DataFrame, factor_name: str, method: str = "percentile", threshold: float = 0.5, direction: str = "long") -> pd.Series:
        if method == "percentile":
            rank = df[factor_name].rolling(252, min_periods=1).rank(pct=True)
            signals = (rank >= threshold).astype(int) if direction == "long" else (rank <= threshold).astype(int)
        else:
            signals = (df[factor_name] >= threshold).astype(int) if direction == "long" else (df[factor_name] <= threshold).astype(int)
        return signals

    # ==================== 基准对比 ====================

    def calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series, annual_trading_days: int = 252) -> Dict:
        aligned_data = pd.DataFrame({"strategy": returns, "benchmark": benchmark_returns}).dropna()
        if len(aligned_data) == 0:
            return {"excess_return": 0.0, "tracking_error": 0.0, "information_ratio": 0.0}
        excess_returns = aligned_data["strategy"] - aligned_data["benchmark"]
        excess_return = excess_returns.mean() * annual_trading_days
        tracking_error = excess_returns.std() * np.sqrt(annual_trading_days)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0
        correlation = aligned_data["strategy"].corr(aligned_data["benchmark"])
        covariance = aligned_data["strategy"].cov(aligned_data["benchmark"])
        benchmark_variance = aligned_data["benchmark"].var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        return {"excess_return": excess_return, "tracking_error": tracking_error, "information_ratio": information_ratio, "correlation": correlation, "beta": beta}

    # ==================== 月度收益计算 ====================

    def calculate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        if len(returns) == 0:
            return pd.DataFrame()
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        monthly_returns = (1 + returns).resample("M").prod() - 1
        monthly_df = monthly_returns.to_frame(name="return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        return monthly_df.pivot(index="year", columns="month", values="return")

    # ==================== 策略系统支持 ====================

    def run_strategy(self, df: pd.DataFrame, strategy_name: str, strategy_params: Optional[Dict] = None) -> Dict:
        if strategy_params is None:
            strategy_params = {}
        strategy = strategy_registry.get_strategy(strategy_name, **strategy_params)
        backtest_result = strategy.backtest(df)
        metrics = strategy.calculate_metrics(backtest_result["portfolio_returns"])
        return {"strategy_name": strategy_name, "backtest": backtest_result, "metrics": metrics}

    def run_strategy_comparison(self, df: pd.DataFrame, strategy_names: List[str], strategy_params: Optional[Dict[str, Dict]] = None) -> Dict:
        return strategy_comparison_service.compare_strategies(df=df, strategy_names=strategy_names, strategy_params=strategy_params)

    def analyze_positions(self, positions: pd.Series, initial_capital: float = 1000000) -> Dict:
        return position_analysis_service.analyze_positions(positions=positions, initial_capital=initial_capital)

    def export_to_excel(self, backtest_result: Dict, output_path: str, strategy_name: str = "策略"):
        metrics = backtest_result.get("metrics")
        export_service.export_backtest_to_excel(backtest_result=backtest_result, output_path=output_path, metrics=metrics, strategy_name=strategy_name)

    def export_comparison_to_excel(self, comparison_result: Dict, output_path: str):
        export_service.export_comparison_to_excel(comparison_result=comparison_result, output_path=output_path)


def check_backtest_engine() -> str:
    """检查当前使用的回测引擎"""
    if VECTORBT_AVAILABLE:
        return "vectorbt"
    return "fallback"
