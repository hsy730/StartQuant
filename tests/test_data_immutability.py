"""
数据不可变性单元测试

覆盖场景：
- 服务层方法不应就地修改传入的DataFrame/dict
- data_preprocessing_service.incremental_update 不修改传入的DataFrame
- factor_data 传入服务后原始数据不被污染
- backtest API 中 stock_data 不被因子列污染

项目规则5：输入数据不可变 — 禁止就地修改传入的DataFrame
"""
import pytest
import numpy as np
import pandas as pd

from backend.services.data_preprocessing_service import DataPreprocessingService
from backend.strategies.base_strategy import BaseStrategy


class TestDataPreprocessingImmutability:
    """DataPreprocessingService 不应修改传入的 DataFrame"""

    def test_incremental_update_not_mutate_existing_df(self):
        """incremental_update 不应修改传入的 existing_df"""
        service = DataPreprocessingService()

        dates1 = pd.date_range("2023-01-01", periods=10, freq="B")
        dates2 = pd.date_range("2023-01-15", periods=5, freq="B")

        existing_df = pd.DataFrame({"date": dates1, "value": range(10)})
        new_df = pd.DataFrame({"date": dates2, "value": range(10, 15)})

        original_existing = existing_df.copy()
        original_new = new_df.copy()

        try:
            service.incremental_update(existing_df, new_df, date_column="date")
        except Exception:
            pass  # 忽略执行错误，只关注数据是否被修改

        assert existing_df.equals(original_existing), \
            "incremental_update 不应修改传入的 existing_df"
        assert new_df.equals(original_new), \
            "incremental_update 不应修改传入的 new_df"

    def test_detect_outliers_not_mutate_input(self):
        """detect_outliers 不应修改传入的 DataFrame"""
        service = DataPreprocessingService()

        data = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 100, 200],
        })
        original = data.copy()

        try:
            service.detect_outliers(data, "value", n_sigma=2.0, method="std")
        except Exception:
            pass

        assert data.equals(original), "detect_outliers 不应修改传入的 DataFrame"


class TestBaseStrategyImmutability:
    """BaseStrategy.backtest 不应修改传入的 DataFrame"""

    def test_backtest_not_mutate_input_df(self):
        """backtest 方法不应修改传入的 df"""

        class BuyHold(BaseStrategy):
            def generate_signals(self, df):
                return pd.Series(1, index=df.index)
            def calculate_weights(self, df, signals):
                return pd.Series(1.0, index=df.index)

        strategy = BuyHold(commission_rate=0.0003)

        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df = pd.DataFrame({
            "open": close + np.random.randn(50) * 0.1,
            "high": close + abs(np.random.randn(50) * 0.3),
            "low": close - abs(np.random.randn(50) * 0.3),
            "close": close,
            "volume": np.random.randint(100000, 1000000, 50).astype(float),
        }, index=dates)

        original = df.copy()
        strategy.backtest(df)

        assert df.equals(original), "backtest 不应修改传入的 DataFrame"


class TestFactorDataImmutability:
    """服务层方法不应修改传入的 factor_data 字典"""

    def test_factor_data_dict_not_mutated(self):
        """传入服务层的 factor_data 字典不应被修改"""
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        np.random.seed(42)

        factor_data = {}
        for i in range(3):
            code = f"{600000 + i:06d}"
            df = pd.DataFrame({
                "close": 100 + np.cumsum(np.random.randn(50) * 0.5),
                "factor_1": np.random.randn(50) * 10 + 5,
            }, index=dates)
            factor_data[code] = df

        # 深拷贝原始数据用于后续比较
        original_data = {code: df.copy() for code, df in factor_data.items()}

        # 传入各种服务方法（可能内部修改数据）
        from backend.services.factor_correlation_service import FactorCorrelationService
        service = FactorCorrelationService()
        try:
            service.analyze_correlation(factor_data, ["factor_1"])
        except Exception:
            pass

        # 验证原始数据未被修改
        for code in factor_data:
            assert factor_data[code].equals(original_data[code]), \
                f"factor_data['{code}'] 不应被服务方法修改"
