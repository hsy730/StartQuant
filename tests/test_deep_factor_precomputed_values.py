"""
Deep Factor预计算因子值回归测试

防护Bug: deep_factor_mining_service返回的expression是不可计算的模型描述字符串，
导致FactorAnalyzer重新计算时IC/IR全为0。

修复: 在返回结果中添加 _precomputed_factor_values 字段。
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

# 标记torch是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from backend.services.deep_factor_mining_service import DeepFactorMiningService  # noqa: E402


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch未安装")
class TestDeepFactorPrecomputedValues:
    """深度隐式因子预计算值测试"""

    def test_validate_factors_returns_precomputed_values(self):
        """_validate_factors返回的结果应包含_precomputed_factor_values"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        data = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
                "return": np.random.randn(n) * 0.01,
            },
            index=dates,
        )

        service = DeepFactorMiningService(
            base_factors=["close"],
            data=data,
            return_column="return",
            n_epochs=2,
            batch_size=8,
            seq_length=5,
            d_model=16,
            n_heads=2,
            n_layers=1,
            n_latent_factors=2,
        )

        # 模拟训练后的模型和因子值
        mock_fv = pd.Series(np.random.randn(n), index=dates)
        factor_info = {
            "expression": "Transformer_test_factor_1",
            "fitness": 0.3,
            "validation": {},
            "source": "deep_implicit",
            "factor_type": "implicit",
            "_precomputed_factor_values": mock_fv,
        }

        # 验证返回结果包含 _precomputed_factor_values
        assert "_precomputed_factor_values" in factor_info
        assert isinstance(factor_info["_precomputed_factor_values"], pd.Series)
        assert len(factor_info["_precomputed_factor_values"]) == n

    def test_precomputed_values_used_by_factor_analyzer(self):
        """FactorAnalyzer应能使用precomputed_values计算IC"""
        from backend.services.factor_analyzer import FactorAnalyzer

        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
        factor = pd.Series(np.random.randn(n), index=dates)
        returns = factor * 0.05 + np.random.randn(n) * 0.01
        data = pd.DataFrame({"factor": factor, "return": returns}, index=dates)

        class DummyCalc:
            def calculate(self, data, expr):
                raise RuntimeError("不应调用calculate，应使用precomputed_values")

        analyzer = FactorAnalyzer(DummyCalc(), data, returns)

        # 使用precomputed_values时应直接计算，不调用factor_calculator
        fv = factor
        result = analyzer._unified_validate("Transformer_test", precomputed_values=fv)

        # 应有有效的IC和IR
        assert result["ic"] is not None
        assert result["ic"] >= 0
        if result["ir"] is not None:
            assert result["ir"] >= 0

    def test_mining_result_dict_conversion_includes_precomputed(self):
        """MiningResult.from_legacy_dict应能识别_precomputed_factor_values"""
        from backend.services.mining_models import MiningResult

        mock_fv = pd.Series([1.0, 2.0, 3.0])
        legacy_dict = {
            "success": True,
            "best_factors": [
                {
                    "expression": "test",
                    "fitness": 0.5,
                    "_precomputed_factor_values": mock_fv,
                }
            ],
        }

        result = MiningResult.from_legacy_dict(legacy_dict)
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.precomputed_values is not None
        assert candidate.precomputed_values.equals(mock_fv)
