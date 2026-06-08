"""
safe_math.py 安全除法工具测试

项目规则要求所有除法使用 safe_divide，所有 IR 计算使用 safe_ir，
禁止裸除法和 +1e-10 hack。
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf

from backend.utils.safe_math import safe_divide, safe_ir


class TestSafeDivideScalar:
    """safe_divide 标量除法测试"""

    def test_safe_divide_normal_division_should_return_correct_result(self):
        """正常除法应返回正确结果"""
        result = safe_divide(10.0, 2.0)
        assert result == pytest.approx(5.0)

    def test_safe_divide_divide_by_zero_should_return_default_none(self):
        """除以零应返回默认值 None"""
        result = safe_divide(10.0, 0.0)
        assert result is None

    def test_safe_divide_divide_by_zero_custom_default_should_return_custom(self):
        """除以零且自定义默认值应返回自定义值"""
        result = safe_divide(10.0, 0.0, default=0.0)
        assert result == 0.0

    def test_safe_divide_divide_by_nan_should_return_default(self):
        """除以 NaN 应返回默认值"""
        result = safe_divide(10.0, np.nan)
        assert result is None

    def test_safe_divide_divide_by_none_should_return_default(self):
        """除以 None 应返回默认值"""
        result = safe_divide(10.0, None)
        assert result is None

    def test_safe_divide_near_zero_denominator_should_return_default(self):
        """近零分母（1e-18）应返回默认值"""
        result = safe_divide(10.0, 1e-18)
        assert result is None

    def test_safe_divide_custom_min_threshold_should_respect_threshold(self):
        """自定义 min_threshold 应按阈值判断"""
        # 分母 1e-8 大于默认阈值 1e-10 → 正常计算
        result_default = safe_divide(10.0, 1e-8)
        assert result_default == pytest.approx(10.0 / 1e-8)

        # 分母 1e-8 小于自定义阈值 1e-6 → 返回 default
        result_custom = safe_divide(10.0, 1e-8, min_threshold=1e-6)
        assert result_custom is None

    def test_safe_divide_negative_denominator_should_return_correct_result(self):
        """负分母应正常计算"""
        result = safe_divide(10.0, -2.0)
        assert result == pytest.approx(-5.0)

    def test_safe_divide_both_zero_should_return_default(self):
        """0/0 应返回默认值"""
        result = safe_divide(0.0, 0.0)
        assert result is None

    def test_safe_divide_very_large_numerator_should_return_correct_result(self):
        """极大分子应正常计算"""
        result = safe_divide(1e300, 1e100)
        assert result == pytest.approx(1e200)

    def test_safe_divide_very_large_denominator_should_return_correct_result(self):
        """极大分母应正常计算（结果趋近于0）"""
        result = safe_divide(1.0, 1e300)
        assert result == pytest.approx(0.0, abs=1e-200)

    def test_safe_divide_negative_near_zero_should_return_default(self):
        """负近零分母应返回默认值"""
        result = safe_divide(10.0, -1e-18)
        assert result is None

    def test_safe_divide_integer_inputs_should_work(self):
        """整数输入应正常工作"""
        result = safe_divide(10, 3)
        assert result == pytest.approx(10 / 3)

    def test_safe_divide_nan_numerator_should_produce_nan(self):
        """NaN 分子除以有效分母应产生 NaN（不是 default）"""
        result = safe_divide(np.nan, 2.0)
        assert np.isnan(result)


class TestSafeDivideSeries:
    """safe_divide pd.Series 除法测试"""

    def test_safe_divide_series_normal_division_should_return_correct_result(self):
        """正常 Series 除法应返回正确结果"""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, 5.0, 10.0])
        result = safe_divide(numerator, denominator)
        expected = pd.Series([5.0, 4.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_divide_series_zero_in_denominator_should_return_default(self):
        """Series 分母含零元素应返回默认值"""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, 0.0, 10.0])
        result = safe_divide(numerator, denominator, default=0.0)
        assert result.iloc[0] == pytest.approx(5.0)
        assert result.iloc[1] == 0.0  # 零分母位置
        assert result.iloc[2] == pytest.approx(3.0)

    def test_safe_divide_series_nan_in_denominator_should_return_default(self):
        """Series 分母含 NaN 应返回默认值"""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, np.nan, 10.0])
        result = safe_divide(numerator, denominator, default=None)
        assert result.iloc[0] == pytest.approx(5.0)
        assert pd.isna(result.iloc[1])  # None 在 Series 中变成 NaN
        assert result.iloc[2] == pytest.approx(3.0)

    def test_safe_divide_series_mixed_valid_invalid_should_handle_correctly(self):
        """Series 混合有效/无效分母应正确处理"""
        numerator = pd.Series([10.0, 20.0, 30.0, 40.0])
        denominator = pd.Series([2.0, 0.0, np.nan, 1e-18])
        result = safe_divide(numerator, denominator, default=0.0)
        assert result.iloc[0] == pytest.approx(5.0)  # 正常
        assert result.iloc[1] == 0.0  # 零分母
        assert result.iloc[2] == 0.0  # NaN 分母
        assert result.iloc[3] == 0.0  # 近零分母

    def test_safe_divide_series_default_none_should_produce_nan(self):
        """Series 中默认值为 None 时，无效位置应为 NaN"""
        numerator = pd.Series([10.0, 20.0])
        denominator = pd.Series([2.0, 0.0])
        result = safe_divide(numerator, denominator, default=None)
        assert result.iloc[0] == pytest.approx(5.0)
        assert pd.isna(result.iloc[1])

    def test_safe_divide_series_all_valid_should_return_all_results(self):
        """Series 全部有效分母应全部正常计算"""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, 4.0, 6.0])
        result = safe_divide(numerator, denominator)
        expected = pd.Series([5.0, 5.0, 5.0])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_divide_series_all_zero_denominator_should_return_all_defaults(self):
        """Series 全零分母应全部返回默认值"""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([0.0, 0.0, 0.0])
        result = safe_divide(numerator, denominator, default=0.0)
        assert (result == 0.0).all()


class TestSafeDivideArray:
    """safe_divide np.ndarray 除法测试"""

    def test_safe_divide_array_normal_division_should_return_correct_result(self):
        """正常 ndarray 除法应返回正确结果"""
        numerator = np.array([10.0, 20.0, 30.0])
        denominator = np.array([2.0, 5.0, 10.0])
        result = safe_divide(numerator, denominator)
        expected = np.array([5.0, 4.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_divide_array_zero_elements_should_return_default(self):
        """ndarray 分母含零元素应返回默认值"""
        numerator = np.array([10.0, 20.0, 30.0])
        denominator = np.array([2.0, 0.0, 10.0])
        result = safe_divide(numerator, denominator, default=0.0)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == 0.0  # 零分母位置
        assert result[2] == pytest.approx(3.0)

    def test_safe_divide_array_nan_elements_should_return_default(self):
        """ndarray 分母含 NaN 应返回默认值"""
        numerator = np.array([10.0, 20.0, 30.0])
        denominator = np.array([2.0, np.nan, 10.0])
        result = safe_divide(numerator, denominator, default=0.0)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == 0.0  # NaN 分母位置
        assert result[2] == pytest.approx(3.0)

    def test_safe_divide_array_mixed_valid_invalid_should_handle_correctly(self):
        """ndarray 混合有效/无效分母应正确处理"""
        numerator = np.array([10.0, 20.0, 30.0, 40.0])
        denominator = np.array([2.0, 0.0, np.nan, 1e-18])
        result = safe_divide(numerator, denominator, default=-1.0)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == -1.0  # 零分母
        assert result[2] == -1.0  # NaN 分母
        assert result[3] == -1.0  # 近零分母

    def test_safe_divide_array_default_none_should_produce_nan(self):
        """ndarray 中默认值为 None 时，无效位置应为 NaN"""
        numerator = np.array([10.0, 20.0])
        denominator = np.array([2.0, 0.0])
        result = safe_divide(numerator, denominator, default=None)
        assert result[0] == pytest.approx(5.0)
        assert np.isnan(result[1])

    def test_safe_divide_array_all_zero_denominator_should_return_all_defaults(self):
        """ndarray 全零分母应全部返回默认值"""
        numerator = np.array([10.0, 20.0, 30.0])
        denominator = np.array([0.0, 0.0, 0.0])
        result = safe_divide(numerator, denominator, default=0.0)
        assert (result == 0.0).all()


class TestSafeIR:
    """safe_ir 信息比率测试"""

    def test_safe_ir_normal_should_return_correct_ir(self):
        """正常 IC 均值和标准差应返回正确 IR"""
        result = safe_ir(0.05, 0.1)
        assert result == pytest.approx(0.5)

    def test_safe_ir_zero_std_should_return_default_none(self):
        """零标准差应返回默认值 None"""
        result = safe_ir(0.05, 0.0)
        assert result is None

    def test_safe_ir_nan_std_should_return_default(self):
        """NaN 标准差应返回默认值"""
        result = safe_ir(0.05, np.nan)
        assert result is None

    def test_safe_ir_negative_ir_should_return_negative(self):
        """负 IC 均值应返回负 IR"""
        result = safe_ir(-0.05, 0.1)
        assert result == pytest.approx(-0.5)

    def test_safe_ir_custom_default_should_return_custom(self):
        """自定义默认值应在不可计算时返回"""
        result = safe_ir(0.05, 0.0, default=0.0)
        assert result == 0.0

    def test_safe_ir_zero_ic_mean_should_return_zero(self):
        """零 IC 均值应返回零 IR"""
        result = safe_ir(0.0, 0.1)
        assert result == pytest.approx(0.0)

    def test_safe_ir_both_zero_should_return_default(self):
        """IC 均值和标准差都为零应返回默认值"""
        result = safe_ir(0.0, 0.0)
        assert result is None

    def test_safe_ir_near_zero_std_should_return_default(self):
        """近零标准差应返回默认值"""
        result = safe_ir(0.05, 1e-18)
        assert result is None

    def test_safe_ir_large_values_should_calculate_correctly(self):
        """大数值 IR 应正确计算"""
        result = safe_ir(50.0, 100.0)
        assert result == pytest.approx(0.5)

    def test_safe_ir_none_std_should_return_default(self):
        """None 标准差应返回默认值"""
        result = safe_ir(0.05, None)
        assert result is None


class TestSafeDivideEdgeCases:
    """safe_divide 边界情况测试"""

    def test_safe_divide_negative_numerator_negative_denominator_should_return_positive(self):
        """负分子负分母应返回正值"""
        result = safe_divide(-10.0, -2.0)
        assert result == pytest.approx(5.0)

    def test_safe_divide_inf_denominator_should_return_near_zero(self):
        """无穷大分母应正常计算（结果趋近于0）"""
        result = safe_divide(10.0, np.inf)
        assert result == pytest.approx(0.0)

    def test_safe_divide_inf_numerator_should_return_inf(self):
        """无穷大分子除以有限分母应返回无穷大"""
        result = safe_divide(np.inf, 2.0)
        assert np.isinf(result) and result > 0

    def test_safe_divide_negative_inf_denominator_should_return_near_zero(self):
        """负无穷大分母应正常计算"""
        result = safe_divide(10.0, np.NINF)
        assert result == pytest.approx(0.0)

    def test_safe_divide_scalar_default_zero_with_zero_denominator(self):
        """默认值为 0.0 时，零分母应返回 0.0"""
        result = safe_divide(5.0, 0.0, default=0.0)
        assert result == 0.0

    def test_safe_divide_scalar_default_negative_one(self):
        """默认值为 -1.0 时，零分母应返回 -1.0"""
        result = safe_divide(5.0, 0.0, default=-1.0)
        assert result == -1.0

    def test_safe_divide_very_small_but_above_threshold_should_calculate(self):
        """刚好大于阈值的分母应正常计算"""
        # 默认 min_threshold=1e-10，分母 2e-10 > 1e-10
        result = safe_divide(10.0, 2e-10)
        assert result == pytest.approx(10.0 / 2e-10)

    def test_safe_divide_exactly_at_threshold_should_return_default(self):
        """恰好等于阈值的分母应返回默认值（abs < threshold 不满足，但 1e-10 < 1e-10 为 False）"""
        # abs(1e-10) < 1e-10 → False，所以应正常计算
        result = safe_divide(10.0, 1e-10)
        assert result == pytest.approx(10.0 / 1e-10)

    def test_safe_divide_just_below_threshold_should_return_default(self):
        """略小于阈值的分母应返回默认值"""
        result = safe_divide(10.0, 0.9e-10)
        assert result is None

    def test_safe_divide_series_preserves_index(self):
        """Series 除法应保留原始索引"""
        numerator = pd.Series([10.0, 20.0], index=["a", "b"])
        denominator = pd.Series([2.0, 5.0], index=["a", "b"])
        result = safe_divide(numerator, denominator)
        assert list(result.index) == ["a", "b"]

    def test_safe_divide_float_nan_denominator_scalar(self):
        """Python float('nan') 作为分母应返回默认值"""
        result = safe_divide(10.0, float('nan'))
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
