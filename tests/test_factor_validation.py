"""
factor_validation_service.py 因子验证服务测试

覆盖IC验证、Rank IC验证、IR验证、稳定性验证等核心功能。
使用公共API validate_factor() 进行测试。
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

from backend.services.factor_validation_service import FactorValidationService  # noqa: E402


def make_factor_return_data(n=200, seed=42):
    """生成模拟因子和收益率数据"""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    factor_values = pd.Series(np.random.randn(n) * 0.1, index=dates)
    # 收益率与因子有一定相关性
    return_values = factor_values * 0.05 + np.random.randn(n) * 0.02
    return_values = pd.Series(return_values, index=dates)
    return factor_values, return_values


class TestFactorValidationService:
    """因子验证服务测试"""

    def setup_method(self):
        self.service = FactorValidationService()

    def test_validate_factor_with_normal_data(self):
        """正常数据的完整因子验证"""
        factor_values, return_values = make_factor_return_data()
        result = self.service.validate_factor(factor_values, return_values)
        assert isinstance(result, dict)
        assert "ic_validation" in result
        assert "rank_ic_validation" in result
        assert "ir_validation" in result
        assert "stability_validation" in result
        assert "turnover_validation" in result
        assert "correlation_validation" in result
        assert "lookahead_bias" in result
        assert "overall_passed" in result
        assert "score" in result

    def test_ic_validation_structure(self):
        """IC验证结果应包含必要字段"""
        factor_values, return_values = make_factor_return_data()
        result = self.service.validate_factor(factor_values, return_values)
        ic_result = result["ic_validation"]
        assert "passed" in ic_result
        assert "ic" in ic_result
        assert "p_value" in ic_result
        assert "is_significant" in ic_result

    def test_rank_ic_validation_structure(self):
        """Rank IC验证结果应包含必要字段"""
        factor_values, return_values = make_factor_return_data()
        result = self.service.validate_factor(factor_values, return_values)
        rank_ic_result = result["rank_ic_validation"]
        assert "passed" in rank_ic_result
        assert "rank_ic" in rank_ic_result
        assert "p_value" in rank_ic_result

    def test_ir_validation_structure(self):
        """IR验证结果应包含必要字段"""
        factor_values, return_values = make_factor_return_data()
        result = self.service.validate_factor(factor_values, return_values)
        ir_result = result["ir_validation"]
        assert "passed" in ir_result
        assert "ir" in ir_result

    def test_stability_validation_structure(self):
        """稳定性验证结果应包含必要字段"""
        factor_values, return_values = make_factor_return_data()
        result = self.service.validate_factor(factor_values, return_values)
        stab_result = result["stability_validation"]
        assert "passed" in stab_result
        assert "stability_score" in stab_result

    def test_validate_with_short_data(self):
        """短数据应返回有效结果（可能标记为不通过）"""
        factor_values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        return_values = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03])
        result = self.service.validate_factor(factor_values, return_values)
        assert isinstance(result, dict)
        assert "ic_validation" in result

    def test_validate_with_nan_data(self):
        """含NaN数据应跳过NaN后计算"""
        factor_values, return_values = make_factor_return_data()
        factor_values.iloc[10:20] = np.nan
        result = self.service.validate_factor(factor_values, return_values)
        assert isinstance(result, dict)
        ic = result["ic_validation"]["ic"]
        assert not np.isnan(ic)

    def test_validate_with_constant_factor(self):
        """恒定因子值（std=0）应被安全处理"""
        factor_values = pd.Series([0.5] * 100)
        return_values = pd.Series(np.random.randn(100) * 0.02)
        result = self.service.validate_factor(factor_values, return_values)
        assert isinstance(result, dict)
        # 恒定因子的IC应为None（不可计算）或0或NaN
        ic = result["ic_validation"].get("ic", 0)
        assert ic is None or ic == 0 or (isinstance(ic, float) and np.isnan(ic)) or abs(ic) < 1e-10

    def test_validate_with_existing_factors(self):
        """传入已有因子时应进行相关性验证"""
        factor_values, return_values = make_factor_return_data()
        existing = {"existing_factor": factor_values * 0.9 + np.random.randn(len(factor_values)) * 0.01}
        result = self.service.validate_factor(factor_values, return_values, existing_factors=existing)
        assert isinstance(result, dict)
        assert "correlation_validation" in result
        assert result["correlation_validation"]["max_correlation"] > 0

    def test_batch_validate(self):
        """批量验证应返回每个因子的结果"""
        factor_values, return_values = make_factor_return_data()
        factors = {
            "factor_a": factor_values,
            "factor_b": factor_values * 0.5 + np.random.randn(len(factor_values)) * 0.05,
        }
        results = self.service.batch_validate(factors, return_values)
        assert isinstance(results, dict)
        assert "factor_a" in results
        assert "factor_b" in results

    def test_score_is_non_negative(self):
        """综合得分应为非负数"""
        factor_values, return_values = make_factor_return_data()
        result = self.service.validate_factor(factor_values, return_values)
        assert result["score"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
