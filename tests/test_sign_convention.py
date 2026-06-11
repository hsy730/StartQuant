"""
回撤/风险指标符号约定单元测试

覆盖场景：
- 回撤应为负值（与empyrical约定一致）
- risk_metrics.max_drawdown 应为负值
- BacktestService.calculate_drawdown 应为负值
- VaR/CVaR 使用 empyrical 计算（规则0）
- 波动率使用 empyrical 计算（规则0）

项目规则0：开源库优先 — VaR/CVaR/波动率应委托empyrical
"""

import numpy as np
import pandas as pd

from backend.services.risk_metrics import calculate_risk_metrics
from backend.services.backtest_service import BacktestService


class TestDrawdownSignConvention:
    """回撤符号约定：应为负值，与empyrical一致"""

    def test_risk_metrics_max_drawdown_negative(self):
        """calculate_risk_metrics 的 max_drawdown 应为负值"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02 - 0.001)  # 略负偏
        result = calculate_risk_metrics(returns)
        assert result["max_drawdown"] <= 0, f"max_drawdown 应为负值，实际为 {result['max_drawdown']}"

    def test_backtest_service_drawdown_negative(self):
        """BacktestService.calculate_drawdown 应返回负值"""
        service = BacktestService()
        equity = pd.Series([100, 110, 105, 95, 90, 100], dtype=float)
        drawdown = service.calculate_drawdown(equity)

        valid = drawdown.dropna()
        assert (valid <= 0).all(), f"回撤应为负值，发现正值: {valid[valid > 0].tolist()}"

    def test_drawdown_zero_at_peak(self):
        """在净值创新高时回撤应为0"""
        service = BacktestService()
        equity = pd.Series([100, 110, 120, 130], dtype=float)
        drawdown = service.calculate_drawdown(equity)

        # 持续创新高，回撤应全为0
        valid = drawdown.dropna()
        assert (valid == 0).all(), f"持续创新高时回撤应为0，实际为 {valid.tolist()}"

    def test_drawdown_after_decline(self):
        """净值下跌后回撤应为负"""
        service = BacktestService()
        equity = pd.Series([100, 110, 100], dtype=float)
        drawdown = service.calculate_drawdown(equity)

        # 第3期：净值100，peak=110，回撤=(100-110)/110 ≈ -0.0909
        assert drawdown.iloc[2] < 0, "下跌后回撤应为负"
        expected = (100 - 110) / 110
        assert abs(drawdown.iloc[2] - expected) < 1e-6, f"回撤值应为 {expected:.4f}，实际为 {drawdown.iloc[2]:.4f}"

    def test_drawdown_no_inf_when_zero_equity(self):
        """净值含0时回撤不应产生inf"""
        service = BacktestService()
        equity = pd.Series([100, 50, 0, 50, 100], dtype=float)
        drawdown = service.calculate_drawdown(equity)

        assert not np.isinf(drawdown.dropna()).any(), "回撤不应包含inf"


class TestRiskMetricsUseEmpyrical:
    """风险指标应使用 empyrical 计算（规则0）"""

    def test_var_95_reasonable(self):
        """VaR(95%) 应为合理负值"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)
        result = calculate_risk_metrics(returns)
        assert result["var_95"] < 0, "VaR(95%) 应为负值"
        assert result["var_95"] > -0.1, "VaR(95%) 不应极端（正常波动约1%）"

    def test_cvar_more_negative_than_var(self):
        """CVaR 应比 VaR 更负（尾部均值 < 尾部分位数）"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)
        result = calculate_risk_metrics(returns)
        assert result["cvar_95"] <= result["var_95"], f"CVaR({result['cvar_95']}) 应 <= VaR({result['var_95']})"

    def test_volatility_positive(self):
        """波动率应为正值"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)
        result = calculate_risk_metrics(returns)
        assert result["volatility"] > 0, "波动率应为正值"

    def test_sharpe_uses_risk_free_rate(self):
        """Sharpe 应扣除无风险利率"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)

        result_low_rf = calculate_risk_metrics(returns, risk_free_rate=0.01)
        result_high_rf = calculate_risk_metrics(returns, risk_free_rate=0.05)

        # 高无风险利率 → 低Sharpe
        assert result_low_rf["sharpe_ratio"] > result_high_rf["sharpe_ratio"], "高无风险利率应导致更低的Sharpe比率"

    def test_consistency_with_empyrical_max_drawdown(self):
        """calculate_risk_metrics 的 max_drawdown 应与 empyrical 一致"""
        import empyrical

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02 - 0.001)

        result = calculate_risk_metrics(returns)
        empyrical_dd = float(empyrical.max_drawdown(returns))

        # 应一致（都为负值）
        assert (
            abs(result["max_drawdown"] - empyrical_dd) < 0.01
        ), f"max_drawdown 应与 empyrical 一致: 计算={result['max_drawdown']}, empyrical={empyrical_dd}"
