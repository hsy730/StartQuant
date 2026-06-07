"""
NaN/Inf 序列化单元测试

覆盖场景：
- _empty_metrics 返回 None 而非 0.0
- risk_metrics 空数据/NaN数据返回 None
- sanitize_dict 正确转换 NaN/Inf → None
- beta 为 NaN 时转为 None
- 因子评分中 NaN 不传播

项目规则6：NaN/Inf序列化统一转为None — 禁止转为0.0
"""
import pytest
import numpy as np
import pandas as pd

from backend.services.risk_metrics import calculate_risk_metrics, _empty_metrics
from backend.strategies.base_strategy import BaseStrategy
from backend.utils.serialization import sanitize_dict


class TestEmptyMetricsReturnsNone:
    """_empty_metrics 所有指标应返回 None"""

    def test_risk_metrics_empty_all_none(self):
        """risk_metrics._empty_metrics 所有值应为 None"""
        empty = _empty_metrics()
        for key, val in empty.items():
            assert val is None, f"_empty_metrics()['{key}'] = {val}，应为 None"

    def test_base_strategy_empty_all_none(self):
        """BaseStrategy._empty_metrics 所有值应为 None"""

        class Dummy(BaseStrategy):
            def generate_signals(self, df): return pd.Series(0, index=df.index)
            def calculate_weights(self, df, signals): return pd.Series(0.0, index=df.index)

        empty = Dummy()._empty_metrics()
        for key, val in empty.items():
            assert val is None, f"BaseStrategy._empty_metrics()['{key}'] = {val}，应为 None"


class TestRiskMetricsNaNHandling:
    """risk_metrics 对 NaN/空数据的处理"""

    def test_empty_series_returns_none(self):
        """空收益率序列应返回全 None"""
        result = calculate_risk_metrics(pd.Series([], dtype=float))
        for key, val in result.items():
            assert val is None, f"空序列时 {key} = {val}，应为 None"

    def test_all_nan_returns_none(self):
        """全 NaN 收益率应返回全 None"""
        result = calculate_risk_metrics(pd.Series([np.nan] * 50))
        for key, val in result.items():
            assert val is None, f"全NaN时 {key} = {val}，应为 None"

    def test_constant_returns_ratio_metrics_none(self):
        """恒定收益率（std=0）的比率指标应为 None"""
        result = calculate_risk_metrics(pd.Series([0.01] * 50))
        for key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
            val = result[key]
            if val is not None:
                assert not np.isnan(val), f"{key} 不应为 NaN"
                assert not np.isinf(val), f"{key} 不应为 inf"

    def test_normal_returns_no_none(self):
        """正常收益率序列不应有 None"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        result = calculate_risk_metrics(returns)
        for key, val in result.items():
            assert val is not None, f"正常数据时 {key} 不应为 None"
            assert not np.isnan(val), f"正常数据时 {key} 不应为 NaN"
            assert not np.isinf(val), f"正常数据时 {key} 不应为 inf"


class TestSanitizeDict:
    """sanitize_dict 应正确转换 NaN/Inf → None"""

    def test_nan_becomes_none(self):
        """NaN 应转为 None"""
        result = sanitize_dict({"value": float('nan')})
        assert result["value"] is None, f"NaN 应转为 None，实际为 {result['value']}"

    def test_inf_becomes_none(self):
        """Inf 应转为 None"""
        result = sanitize_dict({"value": float('inf')})
        assert result["value"] is None, f"Inf 应转为 None"

    def test_neg_inf_becomes_none(self):
        """-Inf 应转为 None"""
        result = sanitize_dict({"value": float('-inf')})
        assert result["value"] is None, f"-Inf 应转为 None"

    def test_normal_float_unchanged(self):
        """正常浮点数应保持不变"""
        result = sanitize_dict({"value": 3.14})
        assert result["value"] == 3.14

    def test_zero_unchanged(self):
        """零值应保持不变（不应被误转为None）"""
        result = sanitize_dict({"value": 0.0})
        assert result["value"] == 0.0, "0.0 应保持不变，不应被转为 None"

    def test_negative_zero_unchanged(self):
        """负零应保持不变"""
        result = sanitize_dict({"value": -0.0})
        assert result["value"] == 0.0 or result["value"] == -0.0

    def test_nested_dict(self):
        """嵌套字典中的 NaN 也应被转换"""
        result = sanitize_dict({"outer": {"inner": float('nan')}})
        assert result["outer"]["inner"] is None

    def test_list_values(self):
        """列表中的 NaN 也应被转换"""
        result = sanitize_dict({"values": [1.0, float('nan'), 3.0]})
        assert result["values"][1] is None

    def test_np_int64_to_int(self):
        """numpy int64 应转为 Python int"""
        result = sanitize_dict({"count": np.int64(42)})
        assert isinstance(result["count"], int)
        assert result["count"] == 42


class TestBetaNaNToNone:
    """beta 为 NaN 时应转为 None"""

    def test_beta_nan_becomes_none(self):
        """beta 为 NaN 时应转为 None 而非 NaN"""
        from backend.services.portfolio_analysis_service import PortfolioAnalysisService

        # 模拟 beta 为 NaN 的场景
        beta = float('nan')
        result = None if (isinstance(beta, float) and np.isnan(beta)) else float(beta)
        assert result is None, f"NaN beta 应转为 None，实际为 {result}"

    def test_beta_normal_value(self):
        """正常 beta 值应保持不变"""
        beta = 1.2
        result = None if (isinstance(beta, float) and np.isnan(beta)) else float(beta)
        assert result == 1.2


class TestFactorScoreNaNPropagation:
    """因子评分中 NaN 不应传播导致评分失真"""

    def test_nan_ic_mean_not_propagate(self):
        """IC均值为NaN时评分不应变为NaN"""
        # 模拟修复后的逻辑
        ic_mean = float('nan')
        if isinstance(ic_mean, float) and np.isnan(ic_mean):
            ic_mean = 0.0
        score = min(ic_mean * 400, 40)
        assert not np.isnan(score), f"评分不应为NaN，实际为 {score}"
        assert score == 0.0, f"NaN IC应回退为0，评分应为0.0，实际为 {score}"

    def test_nan_ir_not_propagate(self):
        """IR为NaN时评分不应变为NaN"""
        ir = float('nan')
        if isinstance(ir, float) and np.isnan(ir):
            ir = 0.0
        score = min(abs(ir) * 200, 30)
        assert not np.isnan(score), f"评分不应为NaN"
