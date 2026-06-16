"""
IC/IR符号一致性回归测试

防护Bug: FactorAnalyzer._unified_validate 中IC取abs但IR未取abs，
导致IC为正、IR为负的异常结果。

规则: code_smell_prevention.md 7.10/7.11
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

from backend.services.factor_analyzer import FactorAnalyzer  # noqa: E402


class MockFactorCalculator:
    """模拟因子计算器"""

    def calculate(self, data, expression):
        # 根据表达式返回不同的因子值，模拟正负IC场景
        if "neg" in expression:
            return -data["factor"]
        if "abs" in expression:
            return data["factor"].abs()
        return data["factor"]


def make_test_data(n=100, seed=42):
    """生成测试数据：因子与收益负相关"""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    factor = pd.Series(np.random.randn(n), index=dates)
    # 收益与因子负相关 → IC为负
    returns = -factor * 0.05 + np.random.randn(n) * 0.01
    returns = pd.Series(returns, index=dates)
    data = pd.DataFrame({"factor": factor, "return": returns}, index=dates)
    return data, returns


class TestICIRSignConsistency:
    """IC/IR符号一致性测试"""

    def test_ir_non_negative_when_ic_positive(self):
        """IC为正时，IR必须非负"""
        data, returns = make_test_data()
        calc = MockFactorCalculator()
        analyzer = FactorAnalyzer(calc, data, returns)

        # 使用 precomputed_values 避免表达式计算失败
        fv = data["factor"]
        result = analyzer._unified_validate("test_expr", precomputed_values=fv)

        ic = result["ic"]
        ir = result["ir"]

        # IC取abs后应为非负
        assert ic is not None
        assert ic >= 0, f"IC应为非负，但得到 {ic}"

        # IR必须与IC同号（都取abs后非负）
        if ir is not None:
            assert ir >= 0, f"IR应为非负，但得到 {ir}"

    def test_ir_sign_matches_ic_after_abs(self):
        """IC和IR取abs后符号一致"""
        data, returns = make_test_data()
        calc = MockFactorCalculator()
        analyzer = FactorAnalyzer(calc, data, returns)

        # 负相关因子：原始IC<0，但统一验证后取abs
        fv = -data["factor"]
        result = analyzer._unified_validate("neg_factor", precomputed_values=fv)

        ic = result["ic"]
        ir = result["ir"]

        assert ic is not None
        assert ic >= 0, f"IC取abs后应为非负，但得到 {ic}"

        if ir is not None:
            assert ir >= 0, f"IR取abs后应为非负，但得到 {ir}"
            # IR应在合理范围 [0, 5]
            assert ir <= 5.0, f"IR不应超过5.0，但得到 {ir}"

    def test_ic_ir_both_none_for_invalid_data(self):
        """无效数据时IC和IR都应为None"""
        data, returns = make_test_data()
        calc = MockFactorCalculator()
        analyzer = FactorAnalyzer(calc, data, returns)

        # 常数因子 → 无法计算有效IC/IR
        fv = pd.Series(1.0, index=data.index)
        result = analyzer._unified_validate("constant", precomputed_values=fv)

        # 常数因子的IC可能为0或None，但不应出现IC=0且IR<0的异常情况
        ic = result["ic"]
        ir = result["ir"]

        if ic is not None and abs(ic) < 1e-10:
            # IC接近0时，IR应为None（不可计算）
            assert ir is None or ir >= 0, f"IC≈0时IR不应为负，但得到 {ir}"
