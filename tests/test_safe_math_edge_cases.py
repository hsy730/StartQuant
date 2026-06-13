"""
safe_math 模块边界条件单元测试

覆盖 safe_float, safe_divide, safe_ir, safe_series_divide 的边界情况。
"""
import numpy as np
import pandas as pd
import pytest
from backend.utils.safe_math import safe_float, safe_divide, safe_ir, safe_series_divide


class TestSafeFloat:
    """safe_float 边界条件测试"""

    def test_none_returns_default(self):
        assert safe_float(None, default=0.0) == 0.0
        assert safe_float(None, default=None) is None

    def test_nan_returns_default(self):
        assert safe_float(float("nan"), default=0.0) == 0.0
        assert safe_float(np.nan, default=None) is None

    def test_inf_returns_default(self):
        assert safe_float(float("inf"), default=0.0) == 0.0
        assert safe_float(float("-inf"), default=0.0) == 0.0
        assert safe_float(np.inf, default=None) is None

    def test_valid_float_passes_through(self):
        assert safe_float(3.14) == 3.14
        assert safe_float(-1.0) == -1.0
        assert safe_float(0.0) == 0.0

    def test_numpy_scalar(self):
        assert safe_float(np.float64(3.14)) == 3.14
        assert safe_float(np.float64(0.0)) == 0.0

    def test_numpy_nan_scalar(self):
        """numpy>=2.0 中 np.float64(nan) 不是 float 子类"""
        result = safe_float(np.float64(np.nan), default=None)
        assert result is None

    def test_non_numeric_returns_default(self):
        assert safe_float("abc", default=0.0) == 0.0
        assert safe_float([1, 2, 3], default=None) is None


class TestSafeDivide:
    """safe_divide 边界条件测试"""

    def test_scalar_division_zero_denominator(self):
        assert safe_divide(0.05, 0.0) is None
        assert safe_divide(0.05, 0.0, default=0.0) == 0.0

    def test_scalar_division_near_zero(self):
        """浮点噪声：7e-18 ≈ 0 但 > 1e-10"""
        result = safe_divide(0.05, 7e-18)
        assert result is None

    def test_scalar_division_nan_denominator(self):
        assert safe_divide(0.05, np.nan) is None
        assert safe_divide(0.05, np.nan, default=0.0) == 0.0

    def test_scalar_nan_numerator(self):
        """NaN 分子：NaN/x = NaN，不是"不可计算"（语义不同）"""
        result = safe_divide(np.nan, 1.0)
        assert np.isnan(result)

    def test_scalar_none_numerator(self):
        """None 分子：语义为"缺失"，返回 default"""
        assert safe_divide(None, 1.0) is None
        assert safe_divide(None, 1.0, default=0.0) == 0.0

    def test_scalar_none_denominator(self):
        assert safe_divide(1.0, None) is None

    def test_scalar_inf(self):
        """分母 inf 不应触发 safe_divide 保护（|inf| > 1e-10 = True）"""
        result = safe_divide(1.0, float("inf"))
        assert result == 0.0  # 1/∞ = 0 在浮点中

    def test_scalar_just_above_threshold(self):
        """分母刚好超过阈值：合法除法"""
        result = safe_divide(0.05, 1e-9, default=None)
        assert result == 0.05 / 1e-9

    def test_series_division(self):
        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([2, 0, 5])
        result = safe_divide(s1, s2, default=None)
        assert result.iloc[0] == 5.0
        # safe_divide 在 Series 中使用 .mask(invalid, default)，
        # 当 default=None 时，pandas 将 None 存储为 NaN
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 6.0

    def test_series_nan_division(self):
        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([2, np.nan, 5])
        result = safe_divide(s1, s2, default=0.0)
        assert result.iloc[0] == 5.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 6.0

    def test_array_division(self):
        a1 = np.array([10.0, 20.0, 30.0])
        a2 = np.array([2.0, 0.0, 5.0])
        result = safe_divide(a1, a2, default=0.0)
        assert result[0] == 5.0
        assert result[1] == 0.0
        assert result[2] == 6.0

    def test_none_numerator_with_series_denominator(self):
        s = pd.Series([0.0, 1.0, np.nan])
        result = safe_divide(None, s, default=0.0)
        assert all(result == 0.0)


class TestSafeIR:
    """safe_ir 边界条件测试"""

    def test_normal_ir(self):
        assert safe_ir(0.05, 0.02) == 2.5

    def test_zero_std_returns_none(self):
        """IC标准差为0：IR不可计算"""
        assert safe_ir(0.05, 0.0) is None
        assert safe_ir(0.05, 0.0, default=None) is None

    def test_near_zero_std(self):
        """接近0的标准差"""
        assert safe_ir(0.05, 7e-18) is None

    def test_nan_std(self):
        assert safe_ir(0.05, np.nan) is None

    def test_nan_mean(self):
        assert safe_ir(np.nan, 0.02) is None

    def test_none_inputs(self):
        assert safe_ir(None, 0.02) is None
        assert safe_ir(0.05, None) is None

    def test_stable_constant_factor(self):
        """IC=0.05 恒定：IC_std=0，IR不可计算（规则7.10）"""
        result = safe_ir(0.05, 0.001)
        assert result is not None  # 0.001 > 1e-10
        assert result == 50.0

    def test_numpy_nan_input(self):
        """numpy>=2.0 兼容性"""
        result = safe_ir(np.float64(np.nan), 0.02, default=None)
        assert result is None


class TestSafeSeriesDivide:
    """safe_series_divide 边界条件测试"""

    def test_basic_division(self):
        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([2, 0, 5])
        result = safe_series_divide(s1, s2)
        assert result.iloc[0] == 5.0
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 6.0

    def test_fill_value_zero(self):
        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([2, 0, 5])
        result = safe_series_divide(s1, s2, fill_value=0.0)
        assert result.iloc[1] == 0.0

    def test_nan_denominator(self):
        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([2, np.nan, 5])
        result = safe_series_divide(s1, s2)
        assert np.isnan(result.iloc[1])

    def test_array_input(self):
        a1 = np.array([10.0, 20.0, 30.0])
        a2 = np.array([2.0, 0.0, 5.0])
        result = safe_series_divide(a1, a2)
        assert result[0] == 5.0
        assert np.isnan(result[1])
        assert result[2] == 6.0

    def test_scalar_denominator_zero(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = safe_series_divide(s, 0.0)
        assert all(np.isnan(result))

    def test_scalar_denominator_none(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = safe_series_divide(s, None)
        assert all(np.isnan(result))

    def test_scalar_numerator_with_zero_denominator(self):
        result = safe_series_divide(1.0, 0.0)
        assert np.isnan(result)