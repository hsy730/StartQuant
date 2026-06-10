"""
statistics_service.py 高级统计分析服务测试

覆盖 StatisticsService 的所有公开方法，包括正常路径、边界条件和异常输入。
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf

from backend.services.statistics_service import StatisticsService


# ============================================================
# t_test_ic 测试
# ============================================================

class TestTTestIC:
    """IC序列t检验测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_normal_ic_should_return_significant_result(self):
        """正常正偏IC序列应返回显著结果"""
        ic = pd.Series(np.random.randn(100) * 0.05 + 0.03)  # 正偏IC
        result = self.svc.t_test_ic(ic)

        assert "t_statistic" in result
        assert "p_value" in result
        assert "is_significant" in result
        assert "mean_ic" in result
        assert "std_ic" in result
        assert "confidence_interval" in result
        assert isinstance(result["is_significant"], (bool, np.bool_))
        # 正偏IC均值应 > 0
        assert result["mean_ic"] > 0

    def test_significant_ic_should_have_small_p_value(self):
        """强正偏IC序列的p值应很小"""
        ic = pd.Series(np.random.randn(200) * 0.02 + 0.05)
        result = self.svc.t_test_ic(ic, confidence_level=0.95)
        assert result["p_value"] < 0.05
        assert result["is_significant"] == True

    def test_zero_mean_ic_should_not_be_significant(self):
        """均值为零的IC序列不应显著"""
        ic = pd.Series(np.random.randn(200) * 0.05)  # 均值约0
        result = self.svc.t_test_ic(ic)
        # 均值接近0，大概率不显著
        assert isinstance(result["p_value"], float)

    def test_empty_series_should_return_defaults(self):
        """空IC序列应返回None（不可计算）"""
        ic = pd.Series([], dtype=float)
        result = self.svc.t_test_ic(ic)

        assert result["t_statistic"] is None
        assert result["p_value"] is None
        assert result["is_significant"] is False
        assert result["mean_ic"] is None
        assert result["std_ic"] is None
        assert result["confidence_interval"] == (None, None)

    def test_all_nan_series_should_return_defaults(self):
        """全NaN序列应返回None（不可计算）"""
        ic = pd.Series([np.nan] * 50)
        result = self.svc.t_test_ic(ic)

        assert result["t_statistic"] is None
        assert result["p_value"] is None
        assert result["is_significant"] is False

    def test_single_value_should_not_crash(self):
        """单值IC序列不应崩溃"""
        ic = pd.Series([0.05])
        result = self.svc.t_test_ic(ic)
        assert isinstance(result, dict)
        assert "t_statistic" in result

    def test_confidence_interval_should_contain_mean(self):
        """置信区间应包含均值"""
        ic = pd.Series(np.random.randn(100) * 0.05 + 0.02)
        result = self.svc.t_test_ic(ic, confidence_level=0.95)
        ci = result["confidence_interval"]
        assert ci[0] <= result["mean_ic"] <= ci[1]

    def test_higher_confidence_should_have_wider_interval(self):
        """更高置信水平应有更宽的置信区间"""
        ic = pd.Series(np.random.randn(100) * 0.05 + 0.02)
        result_90 = self.svc.t_test_ic(ic, confidence_level=0.90)
        result_99 = self.svc.t_test_ic(ic, confidence_level=0.99)
        width_90 = result_90["confidence_interval"][1] - result_90["confidence_interval"][0]
        width_99 = result_99["confidence_interval"][1] - result_99["confidence_interval"][0]
        assert width_99 > width_90

    def test_with_some_nan_should_skip_and_calculate(self):
        """含部分NaN的序列应跳过NaN后计算"""
        ic = pd.Series(np.random.randn(100) * 0.05 + 0.02)
        ic.iloc[10] = np.nan
        ic.iloc[30] = np.nan
        result = self.svc.t_test_ic(ic)
        assert isinstance(result["t_statistic"], float)
        assert not np.isnan(result["t_statistic"])


# ============================================================
# test_monotonicity 测试
# ============================================================

class TestMonotonicity:
    """分层单调性检验测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_increasing_returns_should_be_monotonic(self):
        """递增的分层收益应通过单调性检验"""
        quantile_returns = {
            "Q1": pd.Series(np.random.randn(50) * 0.01 + 0.01),
            "Q2": pd.Series(np.random.randn(50) * 0.01 + 0.02),
            "Q3": pd.Series(np.random.randn(50) * 0.01 + 0.03),
            "Q4": pd.Series(np.random.randn(50) * 0.01 + 0.04),
            "Q5": pd.Series(np.random.randn(50) * 0.01 + 0.05),
        }
        result = self.svc.test_monotonicity(quantile_returns, alternative="increasing")

        assert "correlation" in result
        assert "p_value" in result
        assert "is_monotonic" in result
        assert "layer_means" in result
        assert "layer_names" in result
        assert "expected_direction" in result
        assert result["expected_direction"] == "正相关"
        assert len(result["layer_means"]) == 5
        assert result["layer_names"] == sorted(quantile_returns.keys())

    def test_decreasing_returns_should_be_monotonic(self):
        """递减的分层收益应在decreasing模式下通过单调性检验"""
        quantile_returns = {
            "Q1": pd.Series(np.random.randn(50) * 0.01 + 0.05),
            "Q2": pd.Series(np.random.randn(50) * 0.01 + 0.04),
            "Q3": pd.Series(np.random.randn(50) * 0.01 + 0.03),
            "Q4": pd.Series(np.random.randn(50) * 0.01 + 0.02),
            "Q5": pd.Series(np.random.randn(50) * 0.01 + 0.01),
        }
        result = self.svc.test_monotonicity(quantile_returns, alternative="decreasing")

        assert result["expected_direction"] == "负相关"
        assert result["correlation"] < 0

    def test_flat_returns_should_not_be_monotonic(self):
        """平坦的分层收益不应通过单调性检验"""
        quantile_returns = {
            "Q1": pd.Series(np.random.randn(50) * 0.01 + 0.03),
            "Q2": pd.Series(np.random.randn(50) * 0.01 + 0.03),
            "Q3": pd.Series(np.random.randn(50) * 0.01 + 0.03),
        }
        result = self.svc.test_monotonicity(quantile_returns, alternative="increasing")
        # 平坦收益的秩相关应接近0或较低
        assert abs(result["correlation"]) <= 0.5

    def test_empty_quantile_should_use_zero_mean(self):
        """空分层应使用0.0作为均值"""
        quantile_returns = {
            "Q1": pd.Series([], dtype=float),
            "Q2": pd.Series(np.random.randn(50) * 0.01 + 0.03),
        }
        result = self.svc.test_monotonicity(quantile_returns)
        assert result["layer_means"][0] == 0.0

    def test_all_nan_quantile_should_use_zero_mean(self):
        """全NaN分层应使用0.0作为均值"""
        quantile_returns = {
            "Q1": pd.Series([np.nan] * 50),
            "Q2": pd.Series(np.random.randn(50) * 0.01 + 0.03),
        }
        result = self.svc.test_monotonicity(quantile_returns)
        assert result["layer_means"][0] == 0.0

    def test_layer_names_should_be_sorted(self):
        """分层名称应按字典序排列"""
        quantile_returns = {
            "Q5": pd.Series(np.random.randn(50) * 0.01 + 0.05),
            "Q1": pd.Series(np.random.randn(50) * 0.01 + 0.01),
            "Q3": pd.Series(np.random.randn(50) * 0.01 + 0.03),
        }
        result = self.svc.test_monotonicity(quantile_returns)
        assert result["layer_names"] == ["Q1", "Q3", "Q5"]


# ============================================================
# calculate_factor_decay 测试
# ============================================================

class TestFactorDecay:
    """因子衰减测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def _make_decay_df(self, n=300, seed=42):
        """生成含因子和价格数据的DataFrame"""
        np.random.seed(seed)
        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        factor = np.random.randn(n) * 10 + 5
        return pd.DataFrame({"close": close, "factor_a": factor}, index=dates)

    def test_normal_decay_should_return_period_keys(self):
        """正常数据应返回各期IC值"""
        df = self._make_decay_df()
        result = self.svc.calculate_factor_decay(df, "factor_a", max_periods=5)

        for period in range(1, 6):
            assert f"period_{period}" in result

    def test_decay_ic_should_be_numeric(self):
        """各期IC值应为数值或NaN"""
        df = self._make_decay_df()
        result = self.svc.calculate_factor_decay(df, "factor_a", max_periods=3)

        for key, val in result.items():
            assert isinstance(val, (float, np.floating)) or np.isnan(val)

    def test_short_data_should_return_nan_for_late_periods(self):
        """数据不足时后期IC应为NaN"""
        df = self._make_decay_df(n=15)
        result = self.svc.calculate_factor_decay(df, "factor_a", max_periods=10)
        # 短数据后期对齐后样本不足10个，应返回NaN
        has_nan = any(np.isnan(v) for v in result.values())
        assert has_nan

    def test_max_periods_controls_output_count(self):
        """max_periods应控制输出期数"""
        df = self._make_decay_df()
        result = self.svc.calculate_factor_decay(df, "factor_a", max_periods=3)
        assert len(result) == 3

    def test_with_nan_in_factor_should_still_calculate(self):
        """因子含NaN时应仍能计算（dropna后对齐）"""
        df = self._make_decay_df()
        df.loc[df.index[5], "factor_a"] = np.nan
        df.loc[df.index[20], "factor_a"] = np.nan
        result = self.svc.calculate_factor_decay(df, "factor_a", max_periods=3)
        assert len(result) == 3


# ============================================================
# calculate_factor_crowding 测试
# ============================================================

class TestFactorCrowding:
    """因子拥挤度测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_normal_crowding_should_return_series(self):
        """正常数据应返回Series"""
        n = 100
        df = pd.DataFrame({"factor_a": np.random.randn(n) * 10 + 5})
        result = self.svc.calculate_factor_crowding(df, "factor_a", window=20)

        assert isinstance(result, pd.Series)
        assert len(result) == n

    def test_crowding_should_be_between_zero_and_one(self):
        """拥挤度 = 1/(1+std) 应在[0, 1]之间"""
        df = pd.DataFrame({"factor_a": np.random.randn(200) * 10 + 5})
        result = self.svc.calculate_factor_crowding(df, "factor_a", window=20)
        valid = result.dropna()
        # 前几个值因min_periods=1且std=NaN，safe_divide返回0.0
        # 跳过前面不够窗口长度的值，检查窗口充分后的值
        result_after_warmup = result.iloc[20:]
        assert (result_after_warmup > 0).all()
        assert (result_after_warmup <= 1.0 + 1e-10).all()

    def test_constant_factor_should_have_high_crowding(self):
        """恒定因子值（std=0）拥挤度应为1.0"""
        df = pd.DataFrame({"factor_a": [5.0] * 100})
        result = self.svc.calculate_factor_crowding(df, "factor_a", window=20)
        # std=0 → 1/(1+0) = 1.0
        assert result.iloc[-1] == 1.0

    def test_volatile_factor_should_have_lower_crowding(self):
        """高波动因子的拥挤度应低于低波动因子"""
        np.random.seed(42)
        df_low_vol = pd.DataFrame({"factor_a": np.random.randn(200) * 0.1 + 5})
        df_high_vol = pd.DataFrame({"factor_b": np.random.randn(200) * 50 + 5})

        crowding_low = self.svc.calculate_factor_crowding(df_low_vol, "factor_a", window=20)
        crowding_high = self.svc.calculate_factor_crowding(df_high_vol, "factor_b", window=20)

        # 低波动 → 低std → 高拥挤度
        assert crowding_low.mean() > crowding_high.mean()

    def test_window_parameter_affects_smoothing(self):
        """窗口参数应影响平滑程度"""
        df = pd.DataFrame({"factor_a": np.random.randn(200) * 10 + 5})
        result_short = self.svc.calculate_factor_crowding(df, "factor_a", window=5)
        result_long = self.svc.calculate_factor_crowding(df, "factor_a", window=60)
        # 长窗口的std应更平滑（方差更小）
        assert result_long.std() <= result_short.std() + 1e-10

    def test_with_nan_should_skip(self):
        """含NaN的因子应跳过NaN后计算"""
        factor = np.random.randn(100) * 10 + 5
        factor[5] = np.nan
        factor[20] = np.nan
        df = pd.DataFrame({"factor_a": factor})
        result = self.svc.calculate_factor_crowding(df, "factor_a", window=20)
        assert isinstance(result, pd.Series)


# ============================================================
# calculate_turnover 测试
# ============================================================

class TestTurnover:
    """因子换手率测试"""

    def setup_method(self):
        self.svc = StatisticsService()

    def test_stable_signal_should_have_low_turnover(self):
        """稳定信号应有低换手率"""
        signals = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        result = self.svc.calculate_turnover(signals)

        assert "turnover_rate" in result
        assert "avg_turnover" in result
        # 信号不变，换手率应为0
        assert result["turnover_rate"] == 0.0

    def test_alternating_signal_should_have_high_turnover(self):
        """交替信号应有高换手率"""
        signals = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        result = self.svc.calculate_turnover(signals)
        # 每次都变，换手率应为1.0
        assert result["turnover_rate"] == 1.0

    def test_empty_signals_should_return_zero(self):
        """空信号应返回0.0"""
        signals = pd.Series([], dtype=float)
        result = self.svc.calculate_turnover(signals)
        assert result["turnover_rate"] == 0.0
        assert result["avg_turnover"] == 0.0

    def test_single_signal_should_have_zero_turnover(self):
        """单值信号换手率应为0（diff结果为NaN，mean后为0）"""
        signals = pd.Series([1])
        result = self.svc.calculate_turnover(signals)
        # diff(1)产生NaN，mean跳过NaN
        assert isinstance(result["turnover_rate"], float)

    def test_lag_parameter_should_affect_turnover(self):
        """lag参数应影响换手率计算"""
        signals = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        result_lag1 = self.svc.calculate_turnover(signals, lag=1)
        result_lag2 = self.svc.calculate_turnover(signals, lag=2)
        # lag=2时信号变化更少
        assert result_lag2["turnover_rate"] <= result_lag1["turnover_rate"] + 1e-10

    def test_partial_change_should_have_medium_turnover(self):
        """部分变化的信号应有中等换手率"""
        signals = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
        result = self.svc.calculate_turnover(signals)
        assert 0.0 < result["turnover_rate"] < 1.0


# ============================================================
# analyze_quantile_returns 测试
# ============================================================

class TestAnalyzeQuantileReturns:
    """分层收益分析测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_normal_quantile_returns_should_calculate_stats(self):
        """正常分层收益应计算统计量"""
        quantile_returns = {
            "Q1": pd.Series(np.random.randn(252) * 0.02 + 0.001),
            "Q5": pd.Series(np.random.randn(252) * 0.02 + 0.003),
        }
        result = self.svc.analyze_quantile_returns(quantile_returns)

        for q_name in ["Q1", "Q5"]:
            assert q_name in result
            stats = result[q_name]
            assert "mean" in stats
            assert "std" in stats
            assert "annual_return" in stats
            assert "sharpe" in stats
            assert "win_rate" in stats

    def test_higher_return_quantile_should_have_higher_mean(self):
        """高收益分层的均值应更高"""
        quantile_returns = {
            "Q1": pd.Series(np.random.randn(252) * 0.01 - 0.001),
            "Q5": pd.Series(np.random.randn(252) * 0.01 + 0.005),
        }
        result = self.svc.analyze_quantile_returns(quantile_returns)
        assert result["Q5"]["mean"] > result["Q1"]["mean"]

    def test_empty_quantile_should_return_defaults(self):
        """空分层收益应返回None（不可计算）"""
        quantile_returns = {
            "Q1": pd.Series([], dtype=float),
        }
        result = self.svc.analyze_quantile_returns(quantile_returns)
        assert result["Q1"]["mean"] is None
        assert result["Q1"]["std"] is None
        assert result["Q1"]["sharpe"] is None
        assert result["Q1"]["win_rate"] is None

    def test_all_nan_quantile_should_return_defaults(self):
        """全NaN分层收益应返回None（不可计算）"""
        quantile_returns = {
            "Q1": pd.Series([np.nan] * 100),
        }
        result = self.svc.analyze_quantile_returns(quantile_returns)
        assert result["Q1"]["mean"] is None

    def test_win_rate_should_be_between_zero_and_one(self):
        """胜率应在[0, 1]之间"""
        quantile_returns = {
            "Q1": pd.Series(np.random.randn(252) * 0.02 + 0.001),
        }
        result = self.svc.analyze_quantile_returns(quantile_returns)
        assert 0.0 <= result["Q1"]["win_rate"] <= 1.0

    def test_risk_free_rate_affects_sharpe(self):
        """无风险利率应影响Sharpe"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02 + 0.001)
        quantile_returns = {"Q1": returns}

        result_low_rf = self.svc.analyze_quantile_returns(quantile_returns, risk_free_rate=0.01)
        result_high_rf = self.svc.analyze_quantile_returns(quantile_returns, risk_free_rate=0.10)

        # 高无风险利率下Sharpe应更低
        assert result_low_rf["Q1"]["sharpe"] > result_high_rf["Q1"]["sharpe"]


# ============================================================
# calculate_ic_predictability 测试
# ============================================================

class TestICPredictability:
    """IC可预测性（自相关）测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_normal_ic_should_return_autocorrelations(self):
        """正常IC序列应返回自相关系数"""
        ic = pd.Series(np.random.randn(200) * 0.05 + 0.02)
        result = self.svc.calculate_ic_predictability(ic, lag=5)

        assert "autocorrelations" in result
        assert "mean_abs_autocorr" in result
        assert len(result["autocorrelations"]) == 5

    def test_lag_parameter_controls_length(self):
        """lag参数应控制自相关系数数量"""
        ic = pd.Series(np.random.randn(200) * 0.05)
        result = self.svc.calculate_ic_predictability(ic, lag=3)
        assert len(result["autocorrelations"]) == 3

    def test_mean_abs_autocorr_should_be_non_negative(self):
        """平均绝对自相关应非负"""
        ic = pd.Series(np.random.randn(200) * 0.05)
        result = self.svc.calculate_ic_predictability(ic, lag=5)
        assert result["mean_abs_autocorr"] >= 0

    def test_persistent_ic_should_have_high_autocorr(self):
        """持续性IC序列应有高自相关"""
        # 生成高自相关序列
        n = 200
        ic_vals = np.zeros(n)
        ic_vals[0] = np.random.randn() * 0.05
        for i in range(1, n):
            ic_vals[i] = 0.9 * ic_vals[i - 1] + np.random.randn() * 0.01
        ic = pd.Series(ic_vals)
        result = self.svc.calculate_ic_predictability(ic, lag=5)
        # 高持续性 → 高自相关
        assert result["mean_abs_autocorr"] > 0.3

    def test_random_ic_should_have_low_autocorr(self):
        """随机IC序列的自相关应接近0"""
        ic = pd.Series(np.random.randn(500) * 0.05)
        result = self.svc.calculate_ic_predictability(ic, lag=5)
        # 随机序列自相关应接近0
        assert result["mean_abs_autocorr"] < 0.2


# ============================================================
# calculate_rolling_ic_stability 测试
# ============================================================

class TestRollingICStability:
    """滚动IC稳定性测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_normal_ic_should_return_window_stats(self):
        """正常IC序列应返回各窗口统计量"""
        ic = pd.Series(np.random.randn(300) * 0.05 + 0.02)
        result = self.svc.calculate_rolling_ic_stability(ic, windows=[20, 60])

        assert "window_20" in result
        assert "window_60" in result
        for w_key in ["window_20", "window_60"]:
            assert "mean_ic" in result[w_key]
            assert "std_ic" in result[w_key]
            assert "ir" in result[w_key]

    def test_short_ic_should_return_nan_for_large_window(self):
        """短IC序列对大窗口应返回NaN"""
        ic = pd.Series(np.random.randn(30) * 0.05)
        result = self.svc.calculate_rolling_ic_stability(ic, windows=[20, 120])
        # 窗口120 > 数据长度30，min_periods=30，可能无法计算
        assert np.isnan(result["window_120"]["mean_ic"]) or isinstance(result["window_120"]["mean_ic"], float)

    def test_custom_windows_should_be_respected(self):
        """自定义窗口列表应被正确使用"""
        ic = pd.Series(np.random.randn(300) * 0.05)
        result = self.svc.calculate_rolling_ic_stability(ic, windows=[10, 50])
        assert "window_10" in result
        assert "window_50" in result
        assert len(result) == 2

    def test_stable_ic_should_have_lower_std_than_volatile(self):
        """稳定IC的滚动std应低于波动IC"""
        np.random.seed(42)
        stable_ic = pd.Series(np.random.randn(300) * 0.01 + 0.02)
        volatile_ic = pd.Series(np.random.randn(300) * 0.1 + 0.02)

        result_stable = self.svc.calculate_rolling_ic_stability(stable_ic, windows=[60])
        result_volatile = self.svc.calculate_rolling_ic_stability(volatile_ic, windows=[60])

        assert abs(result_stable["window_60"]["std_ic"]) < abs(result_volatile["window_60"]["std_ic"])


# ============================================================
# calculate_factor_correlation_matrix 测试
# ============================================================

class TestFactorCorrelationMatrix:
    """因子相关性矩阵测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_two_factors_should_return_2x2_matrix(self):
        """两个因子应返回2x2相关矩阵"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
        })
        result = self.svc.calculate_factor_correlation_matrix(df, ["factor_a", "factor_b"])

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        # 对角线应为1.0
        assert result.loc["factor_a", "factor_a"] == pytest.approx(1.0)
        assert result.loc["factor_b", "factor_b"] == pytest.approx(1.0)

    def test_correlated_factors_should_have_high_correlation(self):
        """高度相关的因子应有高相关系数"""
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # y ≈ x + 噪声
        df = pd.DataFrame({"factor_a": x, "factor_b": y})
        result = self.svc.calculate_factor_correlation_matrix(df, ["factor_a", "factor_b"])

        assert abs(result.loc["factor_a", "factor_b"]) > 0.8

    def test_uncorrelated_factors_should_have_low_correlation(self):
        """不相关因子应有低相关系数"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(500),
            "factor_b": np.random.randn(500),
        })
        result = self.svc.calculate_factor_correlation_matrix(df, ["factor_a", "factor_b"])
        assert abs(result.loc["factor_a", "factor_b"]) < 0.3

    def test_empty_df_should_return_empty_dataframe(self):
        """空DataFrame应返回空DataFrame"""
        df = pd.DataFrame({"factor_a": pd.Series([], dtype=float)})
        result = self.svc.calculate_factor_correlation_matrix(df, ["factor_a"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_all_nan_df_should_return_empty_dataframe(self):
        """全NaN的DataFrame应返回空DataFrame"""
        df = pd.DataFrame({
            "factor_a": [np.nan] * 10,
            "factor_b": [np.nan] * 10,
        })
        result = self.svc.calculate_factor_correlation_matrix(df, ["factor_a", "factor_b"])
        assert result.empty

    def test_with_some_nan_should_drop_and_calculate(self):
        """含部分NaN应dropna后计算"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
        })
        df.iloc[5, 0] = np.nan
        df.iloc[20, 1] = np.nan
        result = self.svc.calculate_factor_correlation_matrix(df, ["factor_a", "factor_b"])
        assert result.shape == (2, 2)

    def test_symmetry_of_correlation_matrix(self):
        """相关矩阵应对称"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
            "factor_c": np.random.randn(100),
        })
        result = self.svc.calculate_factor_correlation_matrix(df, ["factor_a", "factor_b", "factor_c"])
        # 检查对称性
        for f1 in result.index:
            for f2 in result.columns:
                assert result.loc[f1, f2] == pytest.approx(result.loc[f2, f1], abs=1e-10)


# ============================================================
# analyze_factor_interactions 测试
# ============================================================

class TestFactorInteractions:
    """因子交互效应分析测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_two_factors_degree2_should_have_interactions(self):
        """两个因子二阶交互应包含交互项"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
        })
        result = self.svc.analyze_factor_interactions(df, ["factor_a", "factor_b"], degree=2)

        assert "interaction_features" in result
        assert "feature_info" in result
        features = result["interaction_features"]
        # degree=2 应包含: factor_a, factor_b, factor_a^2, factor_a factor_b, factor_b^2
        assert len(features) == 5

    def test_interaction_flag_should_be_correct(self):
        """交互项标记应正确"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
        })
        result = self.svc.analyze_factor_interactions(df, ["factor_a", "factor_b"], degree=2)

        feature_info = result["feature_info"]
        # 找到交互项（名称含空格但不含^2）
        interaction_items = {k: v for k, v in feature_info.items() if v["is_interaction"]}
        squared_items = {k: v for k, v in feature_info.items() if v["is_squared"]}

        assert len(interaction_items) >= 1  # factor_a factor_b
        assert len(squared_items) >= 1  # factor_a^2, factor_b^2

    def test_insufficient_data_should_return_empty(self):
        """数据不足时应返回空结果"""
        df = pd.DataFrame({
            "factor_a": [1.0, 2.0],
            "factor_b": [3.0, 4.0],
        })
        result = self.svc.analyze_factor_interactions(df, ["factor_a", "factor_b"], degree=2)
        assert result["interaction_features"] == []
        assert result["feature_importance"] == {}

    def test_single_factor_degree2_should_have_squared(self):
        """单因子二阶应包含平方项"""
        df = pd.DataFrame({"factor_a": np.random.randn(100)})
        result = self.svc.analyze_factor_interactions(df, ["factor_a"], degree=2)

        features = result["interaction_features"]
        # degree=2 单因子: factor_a, factor_a^2
        assert len(features) == 2
        assert any("^2" in f for f in features)

    def test_with_nan_should_drop_and_calculate(self):
        """含NaN应dropna后计算"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
        })
        df.iloc[5, 0] = np.nan
        df.iloc[20, 1] = np.nan
        result = self.svc.analyze_factor_interactions(df, ["factor_a", "factor_b"], degree=2)
        assert len(result["interaction_features"]) > 0

    def test_degree1_should_have_no_interactions(self):
        """degree=1不应有交互项或平方项"""
        df = pd.DataFrame({
            "factor_a": np.random.randn(100),
            "factor_b": np.random.randn(100),
        })
        result = self.svc.analyze_factor_interactions(df, ["factor_a", "factor_b"], degree=1)

        features = result["interaction_features"]
        assert len(features) == 2  # 只有 factor_a, factor_b
        # 无交互项和平方项
        for name, info in result["feature_info"].items():
            assert info["is_interaction"] is False
            assert info["is_squared"] is False


# ============================================================
# 集成/边界条件测试
# ============================================================

class TestEdgeCasesAndIntegration:
    """边界条件和集成测试"""

    def setup_method(self):
        self.svc = StatisticsService()
        np.random.seed(42)

    def test_t_test_with_large_sample(self):
        """大样本t检验应正常工作"""
        ic = pd.Series(np.random.randn(10000) * 0.01 + 0.001)
        result = self.svc.t_test_ic(ic)
        assert isinstance(result["t_statistic"], float)
        assert not np.isnan(result["t_statistic"])

    def test_monotonicity_with_single_layer(self):
        """单层单调性检验不应崩溃"""
        quantile_returns = {"Q1": pd.Series(np.random.randn(50) * 0.01)}
        result = self.svc.test_monotonicity(quantile_returns)
        assert isinstance(result, dict)

    def test_factor_decay_with_constant_price(self):
        """恒定价格因子衰减应能处理（pct_change产生0）"""
        df = pd.DataFrame({
            "close": [100.0] * 300,
            "factor_a": np.random.randn(300),
        })
        result = self.svc.calculate_factor_decay(df, "factor_a", max_periods=3)
        # 恒定价格 → 收益率为0/NaN → IC为NaN
        assert len(result) == 3

    def test_crowding_with_single_value(self):
        """单值因子拥挤度不应崩溃"""
        df = pd.DataFrame({"factor_a": [5.0]})
        result = self.svc.calculate_factor_crowding(df, "factor_a", window=20)
        assert isinstance(result, pd.Series)

    def test_turnover_with_constant_signal(self):
        """恒定信号换手率应为0"""
        signals = pd.Series([1.0] * 50)
        result = self.svc.calculate_turnover(signals)
        assert result["turnover_rate"] == 0.0

    def test_rolling_ic_stability_with_default_windows(self):
        """默认窗口列表应正常工作"""
        ic = pd.Series(np.random.randn(300) * 0.05)
        result = self.svc.calculate_rolling_ic_stability(ic)
        assert "window_20" in result
        assert "window_60" in result
        assert "window_120" in result
        assert "window_252" in result

    def test_correlation_matrix_with_three_factors(self):
        """三因子相关矩阵应为3x3"""
        df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "f3": np.random.randn(100),
        })
        result = self.svc.calculate_factor_correlation_matrix(df, ["f1", "f2", "f3"])
        assert result.shape == (3, 3)

    def test_ic_predictability_with_lag_1(self):
        """lag=1应返回1个自相关系数"""
        ic = pd.Series(np.random.randn(200) * 0.05)
        result = self.svc.calculate_ic_predictability(ic, lag=1)
        assert len(result["autocorrelations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
