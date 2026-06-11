"""
策略模块测试 - 覆盖均值回归策略和动量策略
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

from backend.strategies.mean_reversion_strategy import MeanReversionStrategy  # noqa: E402
from backend.strategies.momentum_strategy import MomentumStrategy  # noqa: E402


def make_price_df(n=200, seed=42):
    """生成模拟价格数据"""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "close": close,
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.3),
            "low": close - abs(np.random.randn(n) * 0.3),
            "volume": np.random.randint(100000, 1000000, n),
        },
        index=dates,
    )


class TestMeanReversionStrategy:
    """均值回归策略测试"""

    def test_generate_signals_should_return_series(self):
        """信号生成应返回Series"""
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0)
        df = make_price_df()
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)

    def test_calculate_weights_should_return_series(self):
        """权重计算应返回Series"""
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0)
        df = make_price_df()
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(df)

    def test_constant_price_should_not_crash(self):
        """恒定价格（std=0）不应崩溃"""
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0)
        df = pd.DataFrame(
            {
                "close": [100.0] * 200,
                "open": [100.0] * 200,
                "high": [100.0] * 200,
                "low": [100.0] * 200,
                "volume": [1000000] * 200,
            },
            index=pd.date_range(start="2023-01-01", periods=200, freq="B"),
        )
        signals = strategy.generate_signals(df)
        # 恒定价格时zscore应为NaN，信号应为0
        assert not signals.isna().any(), "信号不应包含NaN"

    def test_extreme_price_drop_should_generate_buy_signal(self):
        """极端下跌应产生买入信号"""
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=1.5)
        dates = pd.date_range(start="2023-01-01", periods=200, freq="B")
        # 先稳定再突然下跌，下跌后价格继续波动（确保rolling_std非零）
        close = np.concatenate(
            [
                np.ones(100) * 100,  # 稳定期
                np.ones(5) * 80,  # 突然下跌
                80 + np.random.randn(95) * 2,  # 下跌后继续波动
            ]
        )
        df = pd.DataFrame(
            {
                "close": close,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "volume": 1000000,
            },
            index=dates,
        )
        signals = strategy.generate_signals(df)
        # 下跌后应有买入信号（zscore < -entry_threshold）
        assert (signals == 1).any()

    def test_short_data_should_not_crash(self):
        """短数据不应崩溃"""
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0)
        df = make_price_df(n=10)
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.Series)

    def test_backtest_should_return_result_dict(self):
        """回测应返回包含关键字段的结果字典"""
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0)
        df = make_price_df()
        result = strategy.backtest(df)
        assert isinstance(result, dict)
        assert "portfolio_returns" in result
        assert "equity_curve" in result
        assert "weights" in result
        assert "signals" in result

    def test_calculate_metrics_should_return_metrics(self):
        """指标计算应返回性能指标"""
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0)
        df = make_price_df()
        result = strategy.backtest(df)
        metrics = strategy.calculate_metrics(result["portfolio_returns"])
        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "total_return" in metrics


class TestMomentumStrategy:
    """动量策略测试"""

    def test_generate_signals_should_return_series(self):
        """信号生成应返回Series"""
        strategy = MomentumStrategy(momentum_window=20)
        df = make_price_df()
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)

    def test_calculate_weights_should_return_series(self):
        """权重计算应返回Series"""
        strategy = MomentumStrategy(momentum_window=20)
        df = make_price_df()
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)
        assert isinstance(weights, pd.Series)

    def test_uptrend_should_generate_buy_signal(self):
        """上涨趋势应产生买入信号"""
        strategy = MomentumStrategy(momentum_window=20)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        close = 100 + np.arange(100) * 0.5  # 持续上涨
        df = pd.DataFrame(
            {
                "close": close,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "volume": 1000000,
            },
            index=dates,
        )
        signals = strategy.generate_signals(df)
        # 上涨趋势中应有买入信号
        assert (signals.iloc[-30:] == 1).any()

    def test_downtrend_should_generate_sell_signal(self):
        """下跌趋势应产生卖出信号"""
        strategy = MomentumStrategy(momentum_window=20)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
        close = 200 - np.arange(100) * 0.5  # 持续下跌
        df = pd.DataFrame(
            {
                "close": close,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "volume": 1000000,
            },
            index=dates,
        )
        signals = strategy.generate_signals(df)
        # 下跌趋势中应有卖出信号
        assert (signals.iloc[-30:] == -1).any()

    def test_short_data_should_not_crash(self):
        """短数据不应崩溃"""
        strategy = MomentumStrategy(momentum_window=20)
        df = make_price_df(n=10)
        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.Series)

    def test_backtest_should_return_result_dict(self):
        """回测应返回包含关键字段的结果字典"""
        strategy = MomentumStrategy(momentum_window=20)
        df = make_price_df()
        result = strategy.backtest(df)
        assert isinstance(result, dict)
        assert "portfolio_returns" in result
        assert "equity_curve" in result

    def test_calculate_metrics_should_return_metrics(self):
        """指标计算应返回性能指标"""
        strategy = MomentumStrategy(momentum_window=20)
        df = make_price_df()
        result = strategy.backtest(df)
        metrics = strategy.calculate_metrics(result["portfolio_returns"])
        assert isinstance(metrics, dict)
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "total_return" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
