"""
Alpha101 因子模块单元测试

验证 Alpha101 因子定义和计算的正确性
"""

import pytest
import numpy as np
import pandas as pd

from backend.services.alpha101_factors import get_alpha101_factors
from backend.services.factor_service import FactorCalculator


class TestAlpha101FactorDefinitions:
    """Alpha101 因子定义测试"""

    def test_get_alpha101_factors_returns_dict(self):
        result = get_alpha101_factors()
        assert isinstance(result, dict)

    def test_alpha101_has_expected_categories(self):
        result = get_alpha101_factors()
        expected_categories = [
            "Alpha101-动量反转",
            "Alpha101-量价关系",
            "Alpha101-波动率",
            "Alpha101-趋势强度",
            "Alpha101-综合信号",
        ]
        for cat in expected_categories:
            assert cat in result, f"Missing category: {cat}"

    def test_alpha101_factor_count(self):
        result = get_alpha101_factors()
        total = sum(len(v) for v in result.values())
        assert total >= 50, f"Expected at least 50 Alpha101 factors, got {total}"

    def test_each_factor_has_required_fields(self):
        result = get_alpha101_factors()
        for cat, factors in result.items():
            for f in factors:
                assert "name" in f, f"Missing 'name' in {cat}"
                assert "code" in f, f"Missing 'code' in {cat}"
                assert "description" in f, f"Missing 'description' in {cat}"
                assert f["name"].startswith("alpha"), f"Factor name should start with 'alpha': {f['name']}"

    def test_no_duplicate_factor_names(self):
        result = get_alpha101_factors()
        names = []
        for cat, factors in result.items():
            for f in factors:
                names.append(f["name"])
        assert len(names) == len(set(names)), "Duplicate factor names found"


class TestAlpha101HelperFunctions:
    """Alpha101 辅助函数测试"""

    def setup_method(self):
        self.calculator = FactorCalculator()
        np.random.seed(42)
        n = 200
        self.test_df = pd.DataFrame(
            {
                "open": np.cumsum(np.random.randn(n)) + 100,
                "high": np.cumsum(np.random.randn(n)) + 101,
                "low": np.cumsum(np.random.randn(n)) + 99,
                "close": np.cumsum(np.random.randn(n)) + 100,
                "volume": np.abs(np.random.randn(n)) * 1000000 + 500000,
            }
        )

    def test_tsrank_returns_percentile(self):
        result = self.calculator.calculate(self.test_df, "TSRANK(close, 10)")
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_corr_returns_bounded(self):
        result = self.calculator.calculate(self.test_df, "CORR(close, volume, 10)")
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= -1.01).all() and (valid <= 1.01).all()

    def test_delta_returns_difference(self):
        result = self.calculator.calculate(self.test_df, "DELTA(close, 5)")
        assert isinstance(result, pd.Series)
        close = self.test_df["close"]
        expected = close - close.shift(5)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_sign_returns_sign(self):
        result = self.calculator.calculate(self.test_df, "SIGN(DELTA(close, 1))")
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_signedpower_preserves_sign(self):
        result = self.calculator.calculate(self.test_df, "SIGNEDPOWER(DELTA(close, 1), 2)")
        assert isinstance(result, pd.Series)
        delta = self.test_df["close"] - self.test_df["close"].shift(1)
        expected = np.sign(delta) * np.power(np.abs(delta), 2)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_returns_pct_change(self):
        result = self.calculator.calculate(self.test_df, "RETURNS(close)")
        assert isinstance(result, pd.Series)
        expected = self.test_df["close"].pct_change()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_scale_normalizes(self):
        result = self.calculator.calculate(self.test_df, "SCALE(close)")
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert np.abs(valid).sum() == pytest.approx(1.0, abs=1e-6)

    def test_decay_linear_weights(self):
        result = self.calculator.calculate(self.test_df, "DECAY_LINEAR(close, 5)")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() > 0

    def test_vwap_approximation(self):
        result = self.calculator.calculate(self.test_df, "VWAP")
        assert isinstance(result, pd.Series)
        expected = (self.test_df["high"] + self.test_df["low"] + self.test_df["close"]) / 3
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestAlpha101FactorCalculation:
    """Alpha101 因子计算集成测试"""

    def setup_method(self):
        self.calculator = FactorCalculator()
        np.random.seed(42)
        n = 200
        self.test_df = pd.DataFrame(
            {
                "open": np.cumsum(np.random.randn(n)) + 100,
                "high": np.cumsum(np.random.randn(n)) + 101,
                "low": np.cumsum(np.random.randn(n)) + 99,
                "close": np.cumsum(np.random.randn(n)) + 100,
                "volume": np.abs(np.random.randn(n)) * 1000000 + 500000,
            }
        )

    def test_all_alpha101_factors_compute(self):
        alpha101 = get_alpha101_factors()
        for cat, factors in alpha101.items():
            for f in factors:
                result = self.calculator.calculate(self.test_df, f["code"])
                assert isinstance(result, pd.Series), f"{f['name']} did not return Series"
                assert result.notna().sum() > 0, f"{f['name']} returned all NaN"

    def test_alpha001_computation(self):
        result = self.calculator.calculate(
            self.test_df, "TSRANK(SIGNEDPOWER(IF(RETURNS(close) < 0, STD(RETURNS(close), 20), close), 2), 5) - 0.5"
        )
        assert isinstance(result, pd.Series)
        assert result.notna().sum() > 0

    def test_alpha101_computation(self):
        result = self.calculator.calculate(self.test_df, "(close - open) / ((high - low) + 1e-10) * volume")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() > 0

    def test_alpha012_sign_volume_price(self):
        result = self.calculator.calculate(self.test_df, "SIGN(DELTA(volume, 1)) * (-1 * DELTA(close, 1))")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() > 0

    def test_no_inf_in_results(self):
        alpha101 = get_alpha101_factors()
        for cat, factors in alpha101.items():
            for f in factors:
                result = self.calculator.calculate(self.test_df, f["code"])
                valid = result.dropna()
                if len(valid) > 0:
                    assert not np.isinf(valid).any(), f"{f['name']} contains inf values"


class TestAlpha101EdgeCases:
    """Alpha101 边界情况测试"""

    def setup_method(self):
        self.calculator = FactorCalculator()

    def test_constant_price_series(self):
        df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [100.0] * 50,
                "low": [100.0] * 50,
                "close": [100.0] * 50,
                "volume": [1000000] * 50,
            }
        )
        result = self.calculator.calculate(df, "-1 * CORR(open, volume, 10)")
        assert isinstance(result, pd.Series)

    def test_short_data_series(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000000, 1100000, 1200000],
            }
        )
        result = self.calculator.calculate(df, "DELTA(close, 1)")
        assert isinstance(result, pd.Series)
        assert len(result) == 3
