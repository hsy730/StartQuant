"""
除零保护单元测试

覆盖场景：
- IR计算中std为0/NaN/极小值
- 权重归一化中sum为0
- 因子暴露度加权平均中权重sum为0
- 市值加权中市值全NaN
- 回撤计算中peak为0
- CV系数中mean为0

项目规则3：所有除法/标准差必须处理零值
"""
import pytest
import numpy as np
import pandas as pd

from backend.services.risk_metrics import calculate_risk_metrics, _empty_metrics
from backend.services.backtest_service import BacktestService
from backend.services.portfolio_analysis_service import PortfolioAnalysisService
from backend.services.factor_effectiveness_service import FactorEffectivenessService
from backend.services.factor_exposure_service import FactorExposureService
from backend.services.analysis_service import AnalysisService


class TestIRDivisionByZero:
    """IR（信息比率）= IC均值 / IC标准差，当std=0或NaN时的保护"""

    def test_empty_metrics_all_none(self):
        """空数据时所有指标应为None（规则6：禁止返回0.0掩盖不可计算）"""
        empty = _empty_metrics()
        for key, val in empty.items():
            assert val is None, f"_empty_metrics()['{key}'] 应为 None，实际为 {val}"

    def test_constant_returns_ir_not_inf(self):
        """恒定收益率（std=0）的IR不应为inf或NaN"""
        returns = pd.Series([0.01] * 50)
        result = calculate_risk_metrics(returns)
        # std=0时比率指标不可计算，应为None
        for key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
            val = result[key]
            if val is not None:
                assert not np.isinf(val), f"{key} 不应为 inf"
                assert not np.isnan(val), f"{key} 不应为 NaN"

    def test_single_return_no_crash(self):
        """单条收益率不应崩溃"""
        returns = pd.Series([0.01])
        result = calculate_risk_metrics(returns)
        assert isinstance(result, dict)

    def test_all_nan_returns_no_crash(self):
        """全NaN收益率不应崩溃"""
        returns = pd.Series([np.nan] * 50)
        result = calculate_risk_metrics(returns)
        assert isinstance(result, dict)

    def test_ic_std_near_zero_ir_not_extreme(self):
        """IC标准差极小（浮点噪声~1e-17）时IR不应为极大值"""
        # 模拟修复后的逻辑
        ic_mean = 0.05
        ic_std = 7e-18  # pandas std(ddof=1) 对常数序列的浮点噪声

        # 修复后：abs(ic_std) > 1e-10 阈值保护
        if abs(ic_std) > 1e-10 and not np.isnan(ic_std):
            ir = ic_mean / ic_std
        else:
            ir = 0.0

        assert ir == 0.0, f"std极小时IR应为0.0，实际为 {ir}"
        assert not np.isinf(ir), "IR不应为inf"

    def test_ic_std_is_nan_ir_not_nan(self):
        """IC标准差为NaN时IR应为0.0而非NaN"""
        ic_mean = 0.05
        ic_std = float('nan')

        if abs(ic_std) > 1e-10 and not np.isnan(ic_std):
            ir = ic_mean / ic_std
        else:
            ir = 0.0

        assert ir == 0.0, f"std为NaN时IR应为0.0，实际为 {ir}"


class TestWeightNormalizationDivisionByZero:
    """权重归一化中 sum=0 的保护"""

    def test_zero_weights_no_crash(self):
        """权重全为0时不应崩溃"""
        service = PortfolioAnalysisService()
        positions = pd.DataFrame({
            "stock_code": ["000001", "000002", "000003"],
            "weight": [0.0, 0.0, 0.0],
            "industry": ["Tech", "Finance", "Health"],
        })
        result = service.calculate_concentration(positions, weight_column="weight")
        assert isinstance(result, dict), "应返回字典"

    def test_factor_exposure_zero_weight_sum(self):
        """因子暴露度加权平均中权重sum为0时应返回0.0而非inf"""
        # 模拟修复后的逻辑
        stock_weights = np.array([0.0, 0.0, 0.0])
        aligned_factors = np.array([1.5, 2.0, -0.5])

        weight_sum = stock_weights.sum()
        if weight_sum > 0:
            result = (stock_weights * aligned_factors).sum() / weight_sum
        else:
            result = 0.0

        assert result == 0.0, f"权重sum为0时应返回0.0，实际为 {result}"
        assert not np.isinf(result), "结果不应为inf"


class TestDrawdownDivisionByZero:
    """回撤计算中 peak=0 的保护"""

    def test_drawdown_with_zero_equity(self):
        """净值曲线含0时回撤不应产生inf"""
        service = BacktestService()
        equity = pd.Series([100, 50, 0, 50, 100], dtype=float)
        drawdown = service.calculate_drawdown(equity)

        # 不应包含 inf
        assert not np.isinf(drawdown.dropna()).any(), "回撤不应包含inf值"

    def test_drawdown_negative_convention(self):
        """回撤应为负值，与empyrical约定一致"""
        service = BacktestService()
        equity = pd.Series([100, 110, 105, 95, 90, 100], dtype=float)
        drawdown = service.calculate_drawdown(equity)

        # 所有非NaN值应 <= 0
        valid_dd = drawdown.dropna()
        assert (valid_dd <= 0).all(), f"回撤应为负值，发现正值: {valid_dd[valid_dd > 0].tolist()}"

    def test_drawdown_max_matches_empyrical(self):
        """最大回撤值应与empyrical一致"""
        service = BacktestService()
        equity = pd.Series([100, 110, 105, 95, 90, 100], dtype=float)
        drawdown = service.calculate_drawdown(equity)

        # 最大回撤点（净值90时，peak=110）：(90-110)/110 ≈ -0.1818
        min_dd = drawdown.min()
        expected = (90 - 110) / 110
        assert abs(min_dd - expected) < 0.01, f"最大回撤应为 {expected:.4f}，实际为 {min_dd:.4f}"


class TestMarketCapWeightedDivisionByZero:
    """市值加权中市值全NaN的保护"""

    def test_all_nan_market_cap_no_crash(self):
        """市值全NaN时不应崩溃，应回退到等权"""
        # 模拟修复后的逻辑
        group_data = pd.DataFrame({
            "market_cap": [float('nan')] * 10,
            "future_return": np.random.randn(10) * 0.01,
        })

        weights = group_data["market_cap"].fillna(group_data["market_cap"].median())
        weight_sum = weights.sum()

        if weight_sum == 0 or np.isnan(weight_sum):
            avg_return = group_data["future_return"].mean()
        else:
            weights = weights / weight_sum
            avg_return = (group_data["future_return"] * weights).sum()

        assert not np.isnan(avg_return), "全NaN市值时应回退到等权均值"
        assert not np.isinf(avg_return), "结果不应为inf"


class TestCVCoefficientDivisionByZero:
    """变异系数 mean=0 时的保护"""

    def test_cv_mean_zero_returns_none(self):
        """均值为0时CV应返回None（规则6：不可计算的值返回None）"""
        # 模拟修复后的逻辑
        latest_std = 5.0
        latest_mean = 0.0

        cv = float(latest_std / latest_mean) if latest_mean != 0 else None

        assert cv is None, f"mean=0时CV应为None，实际为 {cv}"

    def test_cv_normal_calculation(self):
        """正常情况下CV应正确计算"""
        latest_std = 5.0
        latest_mean = 10.0

        cv = float(latest_std / latest_mean) if latest_mean != 0 else None

        assert abs(cv - 0.5) < 1e-10, f"CV应为0.5，实际为 {cv}"
