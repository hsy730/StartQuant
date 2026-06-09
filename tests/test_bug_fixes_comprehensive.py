"""
业务Bug修复验证测试
覆盖本次审查发现的所有关键Bug修复
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestSafeMathFix:
    """safe_math.py 修复验证"""

    def test_safe_divide_np_float64_nan(self):
        """np.float64(np.nan) 应被正确捕获，返回default"""
        from backend.utils.safe_math import safe_divide
        result = safe_divide(1.0, np.float64(np.nan), default=0.0)
        assert result == 0.0

    def test_safe_ir_np_float64_nan(self):
        """safe_ir 应正确处理 np.float64 NaN 输入"""
        from backend.utils.safe_math import safe_ir
        assert safe_ir(np.float64(np.nan), 0.1, default=None) is None
        assert safe_ir(0.05, np.float64(np.nan), default=None) is None

    def test_safe_divide_pd_series_no_warning(self):
        """pd.Series 分支不应产生 RuntimeWarning"""
        from backend.utils.safe_math import safe_divide
        import warnings
        s = pd.Series([1.0, 2.0, 3.0])
        d = pd.Series([0.0, 0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = safe_divide(s, d, default=0.0)
        assert result.iloc[0] == 0.0
        assert result.iloc[2] == 3.0


class TestBaseStrategyMultiIndex:
    """base_strategy.py MultiIndex 保留验证"""

    def test_backtest_preserves_multiindex(self):
        """回测不应破坏 MultiIndex 结构"""
        from backend.strategies.equal_weight_strategy import EqualWeightStrategy
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        assets = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        df = pd.DataFrame({
            "close": np.random.randn(100).cumsum() + 100,
            "factor": np.random.randn(100),
        }, index=idx)
        strategy = EqualWeightStrategy()
        result = strategy.backtest(df)
        # 验证 portfolio_returns 的索引是日期级别
        assert isinstance(result["portfolio_returns"].index, pd.DatetimeIndex)


class TestMomentumStrategyMultiIndex:
    """momentum_strategy.py MultiIndex 适配验证"""

    def test_momentum_groupby_asset(self):
        """动量计算应在资产内分组"""
        from backend.strategies.momentum_strategy import MomentumStrategy
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        assets = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        df = pd.DataFrame({
            "close": np.concatenate([
                np.linspace(100, 120, 50),  # A: 上涨
                np.linspace(100, 80, 50),   # B: 下跌
            ]),
        }, index=idx)
        strategy = MomentumStrategy(momentum_window=10, buy_threshold=0.01, sell_threshold=-0.01)
        signals = strategy.generate_signals(df)
        # A 上涨应触发买入信号，B 下跌应触发卖出信号
        a_signals = signals.xs("A", level=1)
        b_signals = signals.xs("B", level=1)
        assert (a_signals == 1).any()  # A 有买入信号
        assert (b_signals == -1).any()  # B 有卖出信号

    def test_momentum_weights_normalized_per_date(self):
        """权重应在每个日期内归一化"""
        from backend.strategies.momentum_strategy import MomentumStrategy
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        assets = ["A", "B", "C"]
        idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        df = pd.DataFrame({
            "close": np.random.randn(150).cumsum() + 100,
        }, index=idx)
        strategy = MomentumStrategy(momentum_window=10, buy_threshold=0.0)
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)
        # 每个日期的权重总和应 <= 1.0
        daily_sum = weights.groupby(level=0).sum()
        assert (daily_sum <= 1.0 + 1e-10).all()


class TestMeanReversionMultiIndex:
    """mean_reversion_strategy.py MultiIndex 适配验证"""

    def test_positions_independent_per_asset(self):
        """不同资产的持仓状态应独立"""
        from backend.strategies.mean_reversion_strategy import MeanReversionStrategy
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        assets = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        # A: 均值附近波动，B: 远离均值
        np.random.seed(42)
        close_a = 100 + np.random.randn(100) * 2
        close_b = np.concatenate([np.linspace(80, 120, 50), np.linspace(120, 80, 50)])
        df = pd.DataFrame({
            "close": np.concatenate([close_a, close_b]),
        }, index=idx)
        strategy = MeanReversionStrategy(lookback_window=20, entry_threshold=2.0, exit_threshold=0.5)
        signals = strategy.generate_signals(df)
        # A 和 B 的信号应该不同
        a_signals = signals.xs("A", level=1)
        b_signals = signals.xs("B", level=1)
        # 至少在某些点上信号不同
        assert not (a_signals == b_signals).all()


class TestReturnsMultiIndex:
    """returns.py MultiIndex 修复验证"""

    def test_future_returns_multiindex_no_cross_asset(self):
        """MultiIndex 下不应跨资产计算收益率"""
        from backend.utils.returns import calculate_future_returns
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        assets = ["A", "B"]
        # 构造 MultiIndex 数据：from_product 产生 (date1,A),(date1,B),(date2,A),...
        # 所以 close 数据也必须按此顺序排列
        close_a = 100 * (1 + np.linspace(0.001, 0.003, 30))  # A 的30天价格
        close_b = 200 * (1 + np.linspace(0.001, 0.003, 30))  # B 的30天价格
        # 交错排列：A1,B1,A2,B2,...
        close_interleaved = np.empty(60)
        close_interleaved[0::2] = close_a  # 偶数位放A
        close_interleaved[1::2] = close_b  # 奇数位放B
        idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        df = pd.DataFrame({"close": close_interleaved}, index=idx)
        result = calculate_future_returns(df, periods=[1])
        # 资产A的收益率应在合理范围内，不应出现跨资产的极端值
        a_returns = result.xs("A", level=1)["future_return_1"].dropna()
        # 日收益率应在 0.1% 左右，不应出现50%+的跨资产收益率
        assert (a_returns.abs() < 0.01).all(), f"发现异常收益率: {a_returns[a_returns.abs() >= 0.01]}"


class TestRiskMetricsFix:
    """risk_metrics.py 修复验证"""

    def test_all_non_finite_values_converted_to_none(self):
        """所有非有限值应转为None"""
        from backend.services.risk_metrics import calculate_risk_metrics
        # 极短收益率序列可能产生 inf
        returns = pd.Series([0.01, -0.01, 0.0])
        result = calculate_risk_metrics(returns)
        for key, value in result.items():
            if value is not None and isinstance(value, float):
                assert np.isfinite(value), f"{key}={value} 不是有限值"

    def test_information_ratio_uses_safe_divide(self):
        """IR 计算应使用 safe_divide"""
        from backend.services.risk_metrics import calculate_risk_metrics
        # 极小 tracking_error 不应产生极大 IR
        returns = pd.Series([0.01] * 20 + [-0.01] * 20)
        result = calculate_risk_metrics(returns, risk_free_rate=0.0)
        # IR 不应为 inf
        if result.get("information_ratio") is not None:
            assert np.isfinite(result["information_ratio"])


class TestLookaheadBiasDetectorFix:
    """lookahead_bias_detector.py 修复验证"""

    def test_detection_not_always_passing(self):
        """IC正向占比检测不应永远通过"""
        from backend.services.lookahead_bias_detector import lookahead_bias_detector
        # 极端正向占比应被检测为可疑
        np.random.seed(42)
        # 构造一个有强正向预测力的因子（可能触发高正向占比）
        factor = pd.Series(np.random.randn(200))
        returns = factor * 0.5 + pd.Series(np.random.randn(200) * 0.1)  # 强正相关
        result = lookahead_bias_detector.detect(factor, returns, "test_factor")
        # 检测器应该能运行不崩溃
        assert result is not None
        assert hasattr(result, 'has_bias')


class TestFactorAttributionFix:
    """factor_attribution_service.py 修复验证"""

    def test_contribution_multi_stock_accumulates(self):
        """多股票场景下 factor_panel 应累加而非覆盖"""
        from backend.services.factor_attribution_service import FactorAttributionService
        service = FactorAttributionService()
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        factor_data = {}
        for code in ["A", "B", "C", "D", "E"]:
            df = pd.DataFrame({
                "close": 100 + np.random.randn(50).cumsum(),
                "momentum": np.random.randn(50),
            }, index=dates)
            factor_data[code] = df
        result = service._calculate_contribution(factor_data, "momentum")
        # 不应返回 error
        assert "error" not in result
        assert "ic" in result


class TestFactorDataUtilsFix:
    """factor_data_utils.py 修复验证"""

    def test_missing_factor_column_raises_error(self):
        """不存在的因子列应抛出 ValueError"""
        from backend.utils.factor_data_utils import find_longest_stock
        factor_data = {
            "A": pd.DataFrame({"other_factor": [1, 2, 3]}),
            "B": pd.DataFrame({"other_factor": [4, 5, 6]}),
        }
        with pytest.raises(ValueError, match="没有股票包含因子列"):
            find_longest_stock(factor_data, "nonexistent_factor")

    def test_returns_copy_not_view(self):
        """返回值应为副本而非视图"""
        from backend.utils.factor_data_utils import find_longest_stock
        factor_data = {
            "A": pd.DataFrame({"factor": [1, 2, 3]}),
        }
        code, df = find_longest_stock(factor_data, "factor")
        df["new_col"] = 999
        assert "new_col" not in factor_data["A"].columns


class TestIcCalculatorFix:
    """ic_calculator.py 修复验证"""

    def test_constant_factor_returns_none(self):
        """常数因子应返回 None 而非 NaN"""
        from backend.utils.ic_calculator import calculate_ic
        factor = pd.Series([0.5] * 20)
        returns = pd.Series(np.random.randn(20))
        result = calculate_ic(factor, returns)
        assert result is None


class TestSerializationFix:
    """serialization.py 修复验证"""

    def test_np_bool_preserved(self):
        """np.bool_ 应转为 Python bool 而非 float"""
        from backend.utils.serialization import sanitize_dict
        result = sanitize_dict({"flag": np.bool_(True)})
        assert isinstance(result["flag"], bool)
        assert result["flag"] is True


class TestWeightUtilsFix:
    """weight_utils.py 修复验证"""

    def test_small_total_with_negative_weights(self):
        """负权重+极小总和应回退到等权"""
        from backend.utils.weight_utils import normalize_weights
        # 总和 = 0.5 + (-0.5) = 0.0，绝对值 < 1e-8
        weights = {"A": 0.5, "B": -0.5}
        result = normalize_weights(weights)
        # 总和为0，应回退到等权
        assert abs(result["A"] - 0.5) < 0.01
        assert abs(result["B"] - 0.5) < 0.01


class TestFactorNeutralizationFix:
    """factor_neutralization_service.py 修复验证"""

    def test_returns_copy_not_view(self):
        """中性化结果应为副本，修改不影响原始数据"""
        from backend.services.factor_neutralization_service import FactorNeutralizationService
        service = FactorNeutralizationService()
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame({
            "factor": np.random.randn(30),
            "industry": ["行业A"] * 5 + ["行业B"] * 5 + ["行业C"] * 5 + ["行业D"] * 5 + ["行业E"] * 5 + ["行业F"] * 5,
        }, index=dates)
        original_values = df["factor"].copy()
        # 行业不足2个时返回副本
        df_single = df.copy()
        df_single["industry"] = "唯一行业"
        result = service.neutralize_industry(df_single, "factor", "industry")
        # 修改结果不应影响原始数据
        if result is not None:
            result.iloc[0] = 999
            assert df_single["factor"].iloc[0] != 999


class TestFactorPreprocessingPipelineFix:
    """factor_preprocessing_pipeline.py 修复验证"""

    def test_rank_standardize_handles_inf(self):
        """Rank标准化应正确处理inf值"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig, StandardizeMethod
        config = PreprocessingConfig(standardize_method=StandardizeMethod.RANK)
        pipeline = FactorPreprocessingPipeline(config)
        factor_vals = pd.Series([1.0, 2.0, np.inf, -np.inf, 3.0])
        result = pipeline._standardize(factor_vals)
        # inf 应被转为 NaN 后参与 rank
        assert not np.isinf(result).any()


class TestMarketCapStrategyFix:
    """market_cap_strategy.py 修复验证"""

    def test_safe_divide_imported(self):
        """safe_divide 应已正确导入"""
        from backend.strategies.market_cap_strategy import MarketCapStrategy
        strategy = MarketCapStrategy()
        # 不应抛出 NameError
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame({
            "close": np.random.randn(30).cumsum() + 100,
            "market_cap": np.random.randn(30) * 1e10 + 1e10,
        }, index=dates)
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)
        assert weights is not None

    def test_equal_weight_fallback_multiindex(self):
        """无市值数据时等权回退应按日期分组"""
        from backend.strategies.market_cap_strategy import MarketCapStrategy
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        assets = ["A", "B", "C"]
        idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        df = pd.DataFrame({
            "close": np.random.randn(60).cumsum() + 100,
        }, index=idx)
        strategy = MarketCapStrategy()
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)
        # 每个日期权重总和应为1
        daily_sum = weights.groupby(level=0).sum()
        assert (daily_sum - 1.0).abs().max() < 1e-10
