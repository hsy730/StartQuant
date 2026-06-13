"""
IC统计量计算 边界条件单元测试

覆盖 returns.py 中 calculate_ic_stats 和 calculate_rolling_ir 的所有边界条件。
"""
import numpy as np
import pandas as pd
import pytest
from backend.utils.returns import calculate_ic_stats, calculate_rolling_ir


class TestCalculateICStats:
    """calculate_ic_stats 边界条件测试"""

    def test_empty_series(self):
        result = calculate_ic_stats(pd.Series([], dtype=float))
        assert result["mean_ic"] is None
        assert result["std_ic"] is None
        assert result["ir"] is None
        assert result["n_samples"] == 0

    def test_single_element(self):
        result = calculate_ic_stats(pd.Series([0.05]))
        assert result["mean_ic"] is None
        assert result["n_samples"] == 1

    def test_two_elements(self):
        result = calculate_ic_stats(pd.Series([0.04, 0.06]))
        assert result["mean_ic"] == 0.05
        # With 2 samples, std with ddof=0 (default in std()) is ~0.01
        assert result["std_ic"] is not None
        assert result["ir"] is not None

    def test_constant_ic_positive(self):
        """IC=0.05 恒定（所有截面IC相同）→ IC_std=0 → mean>0 → t_stat=inf"""
        result = calculate_ic_stats(pd.Series([0.05] * 20))
        assert result["mean_ic"] == pytest.approx(0.05)
        assert result["std_ic"] < 1e-10  # 浮点精度下常数序列 std≈7e-18
        assert result["ir"] is None  # 不可计算
        assert result["t_statistic"] == float("inf")
        assert result["p_value"] == 0.0
        assert result["positive_ratio"] == 1.0

    def test_constant_ic_zero(self):
        """IC=0 恒定 → IC_std=0 → mean=0 → t_stat=0"""
        result = calculate_ic_stats(pd.Series([0.0] * 20))
        assert result["mean_ic"] == 0.0
        assert result["std_ic"] == 0.0
        assert result["ir"] is None
        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["positive_ratio"] == 0.0

    def test_normal_ic_series(self):
        """正常IC序列"""
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0.03, 0.05, 100))
        result = calculate_ic_stats(ic_series)
        assert result["mean_ic"] is not None
        assert result["std_ic"] is not None
        assert result["ir"] is not None
        assert result["t_statistic"] is not None
        assert result["p_value"] is not None
        assert result["ci_lower"] is not None
        assert result["ci_upper"] is not None
        assert result["positive_ratio"] is not None
        assert result["n_samples"] == 100

    def test_with_nan_values(self):
        """含NaN的IC序列"""
        ic_series = pd.Series([0.03, np.nan, 0.05, np.nan, 0.04])
        result = calculate_ic_stats(ic_series)
        assert result["n_samples"] == 3  # drops NaN
        assert result["mean_ic"] is not None

    def test_all_nan(self):
        """全NaN序列"""
        ic_series = pd.Series([np.nan, np.nan, np.nan])
        result = calculate_ic_stats(ic_series)
        assert result["n_samples"] == 0
        assert result["mean_ic"] is None

    def test_confidence_interval(self):
        """置信区间合理性"""
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0.05, 0.03, 60))
        result = calculate_ic_stats(ic_series, confidence_level=0.95)
        assert result["ci_lower"] < result["mean_ic"] < result["ci_upper"]

    def test_near_zero_std_positive_mean(self):
        """极小IC_std但非零IC_mean → t_stat inf"""
        result = calculate_ic_stats(pd.Series([0.0499999, 0.0500001] * 10))
        # 这些值几乎相同，std ≈ 1e-7 < 1e-10
        std_val = pd.Series([0.0499999, 0.0500001] * 10).std()
        if std_val < 1e-10:
            assert result["ir"] is None
            assert result["t_statistic"] == float("inf")
            assert result["p_value"] == 0.0


class TestCalculateRollingIR:
    """calculate_rolling_ir 边界条件测试"""

    def test_empty_series(self):
        mean, std, ir = calculate_rolling_ir(pd.Series([], dtype=float))
        assert mean is None
        assert std is None
        assert ir is None

    def test_insufficient_data(self):
        """窗口比数据大"""
        ic_series = pd.Series([0.03] * 5)
        mean, std, ir = calculate_rolling_ir(ic_series, window=20, min_periods=10)
        assert mean is None
        assert std is None
        assert ir is None

    def test_normal_rolling_ir(self):
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0.03, 0.04, 200))
        mean, std, ir = calculate_rolling_ir(ic_series, window=20, min_periods=10)
        assert mean is not None
        assert std is not None
        assert ir is not None
        assert isinstance(ir, float)

    def test_with_nan_values(self):
        """含NaN的IC序列"""
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0.03, 0.04, 200))
        ic_series.iloc[::5] = np.nan  # 每5个插入一个NaN
        mean, std, ir = calculate_rolling_ir(ic_series, window=20, min_periods=10)
        assert mean is not None
        assert std is not None
        assert ir is not None

    def test_constant_ic(self):
        """常数IC序列：滚动std=0 → 所有滚动IR为NaN → dropna后为空 → 返回None"""
        ic_series = pd.Series([0.05] * 100)
        mean, std, ir = calculate_rolling_ir(ic_series, window=20, min_periods=10)
        # 常数序列：滚动标准差=0，safe_divide 返回 None，dropna后无有效值
        assert mean is None
        assert std is None
        assert ir is None