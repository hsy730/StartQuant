"""
回测服务核心引擎（薄编排层 - 优先委托VectorBT金标准）

集成了新的策略系统，支持预置策略和策略对比。
当vectorbt可用时，核心回测方法委托给VectorBTBacktestService；
不可用时回退到自建逻辑。
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
    """回测服务 — 薄编排层（P2改造: 委托VectorBT金标准）"""

    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.0003):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self._vbt = None

    def _get_vbt(self):
        if self._vbt is None and VECTORBT_AVAILABLE:
            self._vbt = VectorBTBacktestService(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
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
        vbt = self._get_vbt()
        if vbt is not None:
            result = vbt.single_factor_backtest(
                df=df, factor_name=factor_name,
                percentile=percentile, direction=direction,
                n_quantiles=n_quantiles,
                use_tradable_mask=use_tradable_mask,
            )
            result["engine"] = "vectorbt"
            return result
        return self._single_factor_fallback(df, factor_name, percentile, direction, n_quantiles, use_tradable_mask)

    def _single_factor_fallback(self, df, factor_name, percentile, direction, n_quantiles, use_tradable_mask):
        logger.info("Backtest using fallback engine (vectorbt unavailable)")
        df = df.copy()
        if "date" in df.columns:
            df = df.sort_values("date")
        elif df.index.name == "date":
            df = df.sort_index()
        if use_tradable_mask and "tradable_mask" in df.columns:
            tradable_mask = df["tradable_mask"]
            if tradable_mask.sum() == 0:
                raise ValueError("tradable_mask全为False！所有日期都不可交易")
        else:
            tradable_mask = pd.Series(True, index=df.index)
        if use_tradable_mask and "tradable_mask" in df.columns:
            factor_clean = df[factor_name].where(tradable_mask)
            df["factor_rank"] = factor_clean.rolling(window=252, min_periods=150).rank(pct=True)
        else:
            df["factor_rank"] = df[factor_name].rolling(window=252, min_periods=1).rank(pct=True)
        df["quantile"] = pd.qcut(df["factor_rank"], q=n_quantiles, labels=False, duplicates="drop")
        df["next_return_raw"] = df["close"].pct_change(1).shift(-1)
        if use_tradable_mask and "tradable_mask" in df.columns:
            mask_today = tradable_mask
            mask_tomorrow = tradable_mask.shift(-1)
            valid_return_mask = mask_today & mask_tomorrow.fillna(False)
            df["next_return"] = df["next_return_raw"].where(valid_return_mask)
        else:
            df["next_return"] = df["next_return_raw"]
        quantile_returns = {}
        for q in range(n_quantiles):
            layer_mask = (df["quantile"] == q) & df["next_return"].notna()
            quantile_returns[f"Q{q + 1}"] = df.loc[layer_mask, "next_return"]
        percentile_threshold = percentile / 100.0
        signal_mask = df["factor_rank"] >= percentile_threshold if direction == "long" else df["factor_rank"] <= percentile_threshold
        if use_tradable_mask and "tradable_mask" in df.columns:
            signal_mask = signal_mask & tradable_mask
        portfolio_returns = df["next_return"].copy()
        portfolio_returns[~signal_mask] = 0
        portfolio_returns = portfolio_returns.clip(lower=-0.5, upper=0.5)
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital
        trades_count = int(signal_mask.astype(int).diff().abs().sum())
        return {
            "quantile_returns": quantile_returns,
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity,
            "trades_count": trades_count,
            "signal_mask": signal_mask,
            "factor_rank": df["factor_rank"],
            "mask_statistics": {
                "total_days": len(df),
                "tradable_days": int(tradable_mask.sum()) if "tradable_mask" in df.columns else len(df),
                "tradable_ratio": float(tradable_mask.mean()) if "tradable_mask" in df.columns else 1.0,
                "limit_up_days": int(df["is_limit_up"].sum()) if "is_limit_up" in df.columns else 0,
                "limit_down_days": int(df["is_limit_down"].sum()) if "is_limit_down" in df.columns else 0,
                "suspended_days": int(df["is_suspended"].sum()) if "is_suspended" in df.columns else 0,
            },
            "engine": "fallback",
        }

    # ==================== 横截面回测 (委托VectorBT) ====================

    def cross_sectional_backtest(self, df: pd.DataFrame, factor_name: str, top_percentile: float = 0.2, direction: str = "long") -> Dict:
        vbt = self._get_vbt()
        if vbt is not None:
            result = vbt.cross_sectional_backtest(df=df, factor_name=factor_name, top_percentile=top_percentile, direction=direction)
            result["engine"] = "vectorbt"
            return result
        return self._cross_sectional_fallback(df, factor_name, top_percentile, direction)

    def _cross_sectional_fallback(self, df, factor_name, top_percentile, direction):
        logger.info("Cross-sectional backtest using fallback engine")
        if "date" not in df.columns:
            df = df.reset_index()
        df["next_return"] = df.groupby("stock_code")["close"].pct_change(1).shift(-1)
        daily_returns = []
        for date, group in df.groupby("date"):
            factor_values = group[factor_name].dropna()
            if len(factor_values) == 0:
                continue
            ranks = factor_values.rank(pct=True)
            selected_stocks = ranks[ranks >= (1 - top_percentile)].index if direction == "long" else ranks[ranks <= top_percentile].index
            selected_returns = group.loc[selected_stocks, "next_return"]
            daily_returns.append({"date": date, "return": selected_returns.mean() if len(selected_returns) > 0 else 0.0})
        returns_df = pd.DataFrame(daily_returns).set_index("date").sort_index()
        portfolio_returns = returns_df["return"]
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital
        return {
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity,
            "trades_count": len(daily_returns),
            "daily_selected_count": len(daily_returns),
            "engine": "fallback",
        }

    # ==================== 多因子回测 (委托VectorBT) ====================

    def multi_factor_backtest(self, df: pd.DataFrame, factor_names: List[str], weights: Optional[List[float]] = None, method: str = "equal_weight", percentile: int = 50, direction: str = "long") -> Dict:
        vbt = self._get_vbt()
        if vbt is not None:
            result = vbt.multi_factor_backtest(df=df, factor_names=factor_names, weights=weights, method=method, percentile=percentile, direction=direction)
            result["engine"] = "vectorbt"
            return result
        return self._multi_factor_fallback(df, factor_names, weights, method, percentile, direction)

    def _multi_factor_fallback(self, df, factor_names, weights, method, percentile, direction):
        logger.info("Multi-factor backtest using fallback engine")
        df = df.copy()
        if "date" in df.columns:
            df = df.sort_values("date")
        preprocessing_pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
            winsorize_method="mad", enable_market_cap_neutralization=("market_cap" in df.columns),
            enable_industry_neutralization=("industry" in df.columns), standardize_method="zscore", cross_sectional=True,
        ))
        factor_columns_to_process = [fn for fn in factor_names if fn in df.columns]
        if factor_columns_to_process:
            df, _ = preprocessing_pipeline.process_factor_dataframe(
                df=df, factor_columns=factor_columns_to_process,
                date_column="date" if "date" in df.columns else None, parallel=False,
            )
            for factor_name in factor_columns_to_process:
                if f"{factor_name}_std" not in df.columns:
                    df[f"{factor_name}_std"] = df[factor_name]
        std_factor_names = [f"{fn}_std" for fn in factor_names]
        if weights is None:
            if method == "equal_weight":
                weights = [1.0 / len(factor_names)] * len(factor_names)
            elif method == "risk_parity":
                inv_vol = []
                for fn in std_factor_names:
                    vol = df[fn].std()
                    inv_vol.append(1.0 / (vol + 1e-8))
                total = sum(inv_vol)
                weights = [v / total for v in inv_vol]
            else:
                weights = [1.0 / len(factor_names)] * len(factor_names)
        df["composite_score"] = sum(df[fn] * w for fn, w in zip(std_factor_names, weights))
        df["score_rank"] = df["composite_score"].rolling(252, min_periods=1).rank(pct=True)
        percentile_threshold = percentile / 100.0
        signal_mask = df["score_rank"] >= percentile_threshold if direction == "long" else df["score_rank"] <= percentile_threshold
        df["next_return"] = df["close"].pct_change(1).shift(-1)
        portfolio_returns = df["next_return"].copy()
        portfolio_returns[~signal_mask] = 0
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital
        trades_count = int(signal_mask.astype(int).diff().abs().sum())
        return {
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity,
            "trades_count": trades_count,
            "composite_score": df["composite_score"],
            "signal_mask": signal_mask,
            "factor_weights": dict(zip(factor_names, weights)),
            "engine": "fallback",
        }

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
