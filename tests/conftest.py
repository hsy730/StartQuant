"""
测试共享fixture

提供通用测试数据生成函数，消除各测试文件中重复的数据生成代码。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_seed():
    """固定随机种子"""
    np.random.seed(42)
    return 42


@pytest.fixture
def make_factor_series():
    """生成模拟因子值序列的工厂函数"""

    def _make(n=200, mean=0.0, std=0.1, seed=42):
        np.random.seed(seed)
        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        return pd.Series(np.random.randn(n) * std + mean, index=dates)

    return _make


@pytest.fixture
def make_ohlcv_df():
    """生成模拟OHLCV数据的工厂函数"""

    def _make(n=200, seed=42):
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

    return _make


@pytest.fixture
def make_cross_sectional_df():
    """生成模拟横截面数据的工厂函数"""

    def _make(n_dates=100, n_stocks=50, seed=42):
        np.random.seed(seed)
        dates = pd.date_range(start="2023-01-01", periods=n_dates, freq="B")
        stock_codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
        data = []
        for date in dates:
            for stock in stock_codes:
                data.append(
                    {
                        "date": date,
                        "stock_code": stock,
                        "factor_1": np.random.randn() * 10 + 5,
                        "factor_2": np.random.randn() * 20 - 3,
                        "market_cap": np.random.lognormal(mean=10, sigma=1),
                        "industry": np.random.choice(["Tech", "Finance", "Health", "Energy"]),
                    }
                )
        return pd.DataFrame(data)

    return _make


@pytest.fixture
def make_returns_series():
    """生成模拟收益率序列的工厂函数"""

    def _make(n=252, mean=0.001, std=0.01, seed=42):
        np.random.seed(seed)
        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        return pd.Series(np.random.randn(n) * std + mean, index=dates)

    return _make
