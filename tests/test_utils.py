"""
工具模块测试 - 覆盖序列化、板块识别、收益率计算工具
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.utils.serialization import safe_numeric_value, sanitize_dict  # noqa: E402
from backend.core.market_board import (  # noqa: E402
    MarketBoard,
    detect_market_board,
    get_board_n_sigma,
    get_board_slippage_multiplier,
)
from backend.utils.returns import (  # noqa: E402
    calculate_future_returns,
    calculate_ic_stats,
    calculate_rolling_ir,
)


class TestSafeNumericValue:
    """safe_numeric_value 测试"""

    def test_normal_float(self):
        assert safe_numeric_value(3.14) == 3.14

    def test_int(self):
        assert safe_numeric_value(42) == 42.0

    def test_numpy_float(self):
        assert safe_numeric_value(np.float64(1.5)) == 1.5

    def test_nan_returns_none(self):
        assert safe_numeric_value(float("nan")) is None

    def test_inf_returns_none(self):
        assert safe_numeric_value(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert safe_numeric_value(float("-inf")) is None

    def test_numpy_nan_returns_none(self):
        assert safe_numeric_value(np.nan) is None

    def test_none_returns_none(self):
        assert safe_numeric_value(None) is None

    def test_string_returns_default(self):
        assert safe_numeric_value("abc") is None

    def test_string_with_custom_default(self):
        assert safe_numeric_value("abc", default=0.0) == 0.0

    def test_zero_is_valid(self):
        assert safe_numeric_value(0.0) == 0.0

    def test_negative_is_valid(self):
        assert safe_numeric_value(-1.5) == -1.5


class TestSanitizeDict:
    """sanitize_dict 测试"""

    def test_simple_dict(self):
        result = sanitize_dict({"a": 1, "b": 2.5})
        assert result == {"a": 1, "b": 2.5}

    def test_nan_in_dict(self):
        result = sanitize_dict({"a": float("nan")})
        assert result["a"] is None

    def test_inf_in_dict(self):
        result = sanitize_dict({"a": float("inf")})
        assert result["a"] is None

    def test_numpy_int(self):
        result = sanitize_dict({"a": np.int64(42)})
        assert result["a"] == 42
        assert isinstance(result["a"], int)

    def test_numpy_array(self):
        result = sanitize_dict({"a": np.array([1, 2, 3])})
        assert result["a"] == [1.0, 2.0, 3.0]

    def test_nested_dict(self):
        result = sanitize_dict({"outer": {"inner": np.float64(1.5)}})
        assert result == {"outer": {"inner": 1.5}}

    def test_list_in_dict(self):
        result = sanitize_dict({"a": [1, np.float64(2), float("nan")]})
        assert result["a"] == [1, 2.0, None]

    def test_none_value(self):
        result = sanitize_dict({"a": None})
        assert result["a"] is None

    def test_string_value(self):
        result = sanitize_dict({"a": "hello"})
        assert result["a"] == "hello"

    def test_bool_value(self):
        result = sanitize_dict({"a": True})
        assert result["a"] is True

    def test_timestamp(self):
        ts = pd.Timestamp("2023-01-01")
        result = sanitize_dict({"a": ts})
        assert isinstance(result["a"], str)


class TestMarketBoard:
    """板块识别测试"""

    def test_shanghai_main(self):
        assert detect_market_board("600519") == MarketBoard.MAIN

    def test_shenzhen_main(self):
        assert detect_market_board("000001") == MarketBoard.MAIN

    def test_chinext(self):
        assert detect_market_board("300750") == MarketBoard.CHINEXT

    def test_star(self):
        assert detect_market_board("688981") == MarketBoard.STAR

    def test_bse_8(self):
        assert detect_market_board("830799") == MarketBoard.BSE

    def test_bse_4(self):
        assert detect_market_board("430047") == MarketBoard.BSE

    def test_unknown(self):
        assert detect_market_board("999999") == MarketBoard.UNKNOWN

    def test_empty_string(self):
        assert detect_market_board("") == MarketBoard.UNKNOWN

    def test_none(self):
        assert detect_market_board(None) == MarketBoard.UNKNOWN


class TestGetBoardNSigma:
    """板块自适应n_sigma参数测试"""

    def test_main_board(self):
        assert get_board_n_sigma(MarketBoard.MAIN) == 3.0

    def test_chinext(self):
        assert get_board_n_sigma(MarketBoard.CHINEXT) == 2.8

    def test_star(self):
        assert get_board_n_sigma(MarketBoard.STAR) == 2.7

    def test_bse(self):
        assert get_board_n_sigma(MarketBoard.BSE) == 2.5

    def test_unknown_uses_main_default(self):
        assert get_board_n_sigma(MarketBoard.UNKNOWN) == 3.0

    def test_sigma_ordering(self):
        """高波动板块n_sigma应更小"""
        assert get_board_n_sigma(MarketBoard.MAIN) > get_board_n_sigma(MarketBoard.CHINEXT)
        assert get_board_n_sigma(MarketBoard.CHINEXT) > get_board_n_sigma(MarketBoard.STAR)
        assert get_board_n_sigma(MarketBoard.STAR) > get_board_n_sigma(MarketBoard.BSE)


class TestGetBoardSlippageMultiplier:
    """板块滑点乘数测试"""

    def test_main_board_baseline(self):
        assert get_board_slippage_multiplier(MarketBoard.MAIN) == 1.0

    def test_higher_volatility_higher_slippage(self):
        """高波动板块滑点应更大"""
        assert get_board_slippage_multiplier(MarketBoard.CHINEXT) > 1.0
        assert get_board_slippage_multiplier(MarketBoard.STAR) > get_board_slippage_multiplier(MarketBoard.CHINEXT)
        assert get_board_slippage_multiplier(MarketBoard.BSE) > get_board_slippage_multiplier(MarketBoard.STAR)


class TestCalculateFutureReturns:
    """未来收益率计算测试"""

    def test_single_period(self):
        df = pd.DataFrame({"close": [100, 102, 101, 103, 105]})
        result = calculate_future_returns(df, periods=[1])
        assert "future_return_1" in result.columns
        # 第0行的1期未来收益 = 102/100 - 1 = 0.02
        assert abs(result["future_return_1"].iloc[0] - 0.02) < 1e-10

    def test_multiple_periods(self):
        df = pd.DataFrame({"close": [100, 102, 101, 103, 105]})
        result = calculate_future_returns(df, periods=[1, 5])
        assert "future_return_1" in result.columns
        assert "future_return_5" in result.columns

    def test_does_not_modify_input(self):
        """不应修改输入DataFrame"""
        df = pd.DataFrame({"close": [100, 102, 101, 103, 105]})
        original_cols = df.columns.tolist()
        _ = calculate_future_returns(df, periods=[1])
        assert df.columns.tolist() == original_cols

    def test_last_row_should_be_nan(self):
        """最后一行的未来收益应为NaN"""
        df = pd.DataFrame({"close": [100, 102, 101, 103, 105]})
        result = calculate_future_returns(df, periods=[1])
        assert pd.isna(result["future_return_1"].iloc[-1])


class TestCalculateICStats:
    """IC统计量计算测试"""

    def test_normal_ic_series(self):
        ic = pd.Series(np.random.randn(100) * 0.05 + 0.03)
        result = calculate_ic_stats(ic)
        assert "mean_ic" in result
        assert "std_ic" in result
        assert "ir" in result
        assert "t_statistic" in result
        assert "p_value" in result
        assert "n_samples" in result

    def test_empty_series(self):
        ic = pd.Series([], dtype=float)
        result = calculate_ic_stats(ic)
        assert result["mean_ic"] is None
        assert result["n_samples"] == 0

    def test_single_value(self):
        ic = pd.Series([0.05])
        result = calculate_ic_stats(ic)
        assert result["n_samples"] == 1
        assert result["mean_ic"] is None

    def test_all_nan(self):
        ic = pd.Series([np.nan] * 50)
        result = calculate_ic_stats(ic)
        assert result["mean_ic"] is None

    def test_constant_ic_should_have_near_zero_std(self):
        """恒定IC的std应接近0（浮点精度限制）"""
        ic = pd.Series([0.05] * 50)
        result = calculate_ic_stats(ic)
        assert abs(result["std_ic"]) < 1e-10

    def test_positive_ic_should_have_positive_mean(self):
        """正IC序列的均值应为正"""
        ic = pd.Series(np.abs(np.random.randn(100)) * 0.05)
        result = calculate_ic_stats(ic)
        assert result["mean_ic"] > 0


class TestCalculateRollingIR:
    """滚动IR计算测试"""

    def test_normal_ic_series(self):
        ic = pd.Series(np.random.randn(100) * 0.05 + 0.03)
        ic_mean, ic_std, ir = calculate_rolling_ir(ic, window=20)
        assert isinstance(ic_mean, float)
        assert isinstance(ic_std, float)
        assert isinstance(ir, float)

    def test_short_series(self):
        """短序列应返回None（不可计算）"""
        ic = pd.Series([0.05, 0.03])
        ic_mean, ic_std, ir = calculate_rolling_ir(ic, window=20, min_periods=10)
        assert ic_mean is None
        assert ic_std is None
        assert ir is None

    def test_constant_ic(self):
        """恒定IC的IR不可计算（std=0），应返回None"""
        ic = pd.Series([0.05] * 100)
        ic_mean, ic_std, ir = calculate_rolling_ir(ic, window=20)
        assert ir is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
