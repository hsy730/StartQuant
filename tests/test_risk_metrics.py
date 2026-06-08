"""
risk_metrics.py 统一风险指标入口测试

项目规则要求所有风险指标（Sharpe/Sortino/MaxDD/Calmar/VaR/CVaR）
通过 risk_metrics.py 统一入口，底层委托 empyrical。
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

from backend.services.risk_metrics import calculate_risk_metrics, _empty_metrics


class TestCalculateRiskMetrics:
    """calculate_risk_metrics 统一入口测试"""

    def setup_method(self):
        np.random.seed(42)

    def test_normal_returns_should_calculate_all_metrics(self):
        """正常收益率序列应计算所有指标"""
        returns = pd.Series(np.random.randn(252) * 0.01)  # 日收益率约1%波动
        result = calculate_risk_metrics(returns)

        expected_keys = [
            "total_return", "annual_return", "volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate",
            "var_95", "cvar_95"
        ]
        for key in expected_keys:
            assert key in result, f"缺少指标: {key}"
            assert isinstance(result[key], float), f"指标 {key} 应为 float"

    def test_positive_returns_should_have_higher_sharpe_than_negative(self):
        """正收益序列的Sharpe应高于负收益序列"""
        np.random.seed(42)
        returns_pos = pd.Series(np.random.randn(252) * 0.01 + 0.002)  # 正偏收益
        returns_neg = pd.Series(np.random.randn(252) * 0.01 - 0.005)  # 负偏收益
        result_pos = calculate_risk_metrics(returns_pos, risk_free_rate=0.01)
        result_neg = calculate_risk_metrics(returns_neg, risk_free_rate=0.01)
        assert result_pos["sharpe_ratio"] > result_neg["sharpe_ratio"]

    def test_negative_returns_should_have_negative_sharpe(self):
        """负收益序列的Sharpe应为负"""
        returns = pd.Series(np.random.randn(100) * 0.005 - 0.002)  # 负偏收益
        result = calculate_risk_metrics(returns)
        assert result["sharpe_ratio"] < 0

    def test_max_drawdown_should_be_non_positive(self):
        """最大回撤应 <= 0"""
        returns = pd.Series(np.random.randn(252) * 0.02)
        result = calculate_risk_metrics(returns)
        assert result["max_drawdown"] <= 0

    def test_var_95_should_be_negative_for_normal_returns(self):
        """正常收益序列的VaR(95%)应为负"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        result = calculate_risk_metrics(returns)
        assert result["var_95"] < 0

    def test_cvar_95_should_be_more_negative_than_var(self):
        """CVaR应比VaR更负（尾部均值 < 尾部分位数）"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        result = calculate_risk_metrics(returns)
        assert result["cvar_95"] <= result["var_95"]

    def test_empty_series_should_return_empty_metrics(self):
        """空序列应返回空指标"""
        returns = pd.Series([], dtype=float)
        result = calculate_risk_metrics(returns)
        expected = _empty_metrics()
        for key in expected:
            assert result[key] == expected[key]

    def test_all_nan_series_should_return_empty_metrics(self):
        """全NaN序列应返回空指标"""
        returns = pd.Series([np.nan] * 100)
        result = calculate_risk_metrics(returns)
        expected = _empty_metrics()
        for key in expected:
            assert result[key] == expected[key]

    def test_single_value_should_not_crash(self):
        """单值序列不应崩溃"""
        returns = pd.Series([0.01])
        result = calculate_risk_metrics(returns)
        assert isinstance(result, dict)

    def test_constant_returns_should_handle_zero_std(self):
        """恒定收益率（std=0）不应崩溃"""
        returns = pd.Series([0.01] * 50)
        result = calculate_risk_metrics(returns)
        assert isinstance(result, dict)
        # std=0 时 empyrical 可能返回 NaN，比率指标应转为 None（规则6）
        for key, val in result.items():
            assert val is None or isinstance(val, float), f"{key} 应为 float 或 None，实际为 {type(val)}"

    def test_extreme_returns_should_not_produce_inf(self):
        """极端收益率不应产生inf"""
        returns = pd.Series([0.5, -0.3, 0.1, -0.2, 0.05, -0.15, 0.08, -0.1, 0.02, -0.05])
        result = calculate_risk_metrics(returns)
        for key, val in result.items():
            assert not np.isinf(val), f"{key} 不应为 inf"

    def test_risk_free_rate_affects_sharpe(self):
        """无风险利率应影响Sharpe"""
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        result_low_rf = calculate_risk_metrics(returns, risk_free_rate=0.01)
        result_high_rf = calculate_risk_metrics(returns, risk_free_rate=0.10)
        # 高无风险利率下Sharpe应更低
        assert result_low_rf["sharpe_ratio"] > result_high_rf["sharpe_ratio"]

    def test_with_nan_values_should_skip_and_calculate(self):
        """含NaN的序列应跳过NaN后计算"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        returns.iloc[10] = np.nan
        returns.iloc[50] = np.nan
        result = calculate_risk_metrics(returns)
        assert isinstance(result, dict)
        assert not np.isnan(result["sharpe_ratio"])


class TestEmptyMetrics:
    """_empty_metrics 测试"""

    def test_empty_metrics_all_zero_or_none(self):
        """空指标中所有值应为None（符合规范6：不可计算→None）"""
        result = _empty_metrics()
        for key, val in result.items():
            assert val is None, f"{key} 应为 None（不可计算）"

    def test_empty_metrics_has_all_keys(self):
        """空指标应包含所有键"""
        result = _empty_metrics()
        expected_keys = [
            "total_return", "annual_return", "volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate",
            "var_95", "cvar_95"
        ]
        for key in expected_keys:
            assert key in result

    def test_empty_metrics_values_are_valid(self):
        """空指标值应为float或None"""
        result = _empty_metrics()
        for key, val in result.items():
            assert isinstance(val, float) or val is None, f"{key} 应为 float 或 None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
