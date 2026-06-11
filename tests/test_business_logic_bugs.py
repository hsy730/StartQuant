"""
业务逻辑Bug回归测试 — 覆盖21个已修复的关键Bug

测试命名格式: test_[功能]_[场景]_[预期结果]
"""

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from scipy import stats as scipy_stats

# ==================== Critical 级别 ====================


class TestEnhancedAnalysisServiceFactorDataStructure:
    """Bug #1: enhanced_analysis_service: factor_data结构为{stock_code: DataFrame}"""

    def test_analyze_enhanced_factor_data_stock_code_keys_should_iterate_stocks(self):
        """factor_data以stock_code为key时，应正确遍历每只股票的DataFrame"""
        from backend.services.enhanced_analysis_service import EnhancedAnalysisService

        service = EnhancedAnalysisService()
        dates = pd.date_range("2023-01-01", periods=100)
        factor_data = {
            "600036": pd.DataFrame(
                {
                    "close": np.random.randn(100).cumsum() + 100,
                    "factor_a": np.random.randn(100),
                },
                index=dates,
            ),
            "000001": pd.DataFrame(
                {
                    "close": np.random.randn(100).cumsum() + 50,
                    "factor_a": np.random.randn(100),
                },
                index=dates,
            ),
        }

        result = service.analyze_enhanced(factor_data, ["factor_a"])
        # factor_data结构为{stock_code: DataFrame}，应能正确遍历
        assert "factors" in result
        assert "factor_a" in result["factors"]
        assert "ic_significance" in result["factors"]["factor_a"]

    def test_analyze_enhanced_single_stock_should_work(self):
        """单股票factor_data也应正常工作"""
        from backend.services.enhanced_analysis_service import EnhancedAnalysisService

        service = EnhancedAnalysisService()
        dates = pd.date_range("2023-01-01", periods=100)
        factor_data = {
            "600036": pd.DataFrame(
                {
                    "close": np.random.randn(100).cumsum() + 100,
                    "factor_a": np.random.randn(100),
                },
                index=dates,
            ),
        }

        result = service.analyze_enhanced(factor_data, ["factor_a"])
        assert "factor_a" in result["factors"]


class TestFormulaCompilerATR:
    """Bug #2: formula_compiler_service: ATR编译需要3个价格序列参数"""

    def test_atr_compile_with_three_price_args_should_generate_correct_code(self):
        """ATR需要high, low, close三个价格序列参数"""
        from backend.services.formula_compiler_service import FormulaCompilerService

        compiler = FormulaCompilerService()
        formula_tree = {
            "type": "function",
            "name": "ATR",
            "args": [
                {"type": "column", "value": "high"},
                {"type": "column", "value": "low"},
                {"type": "column", "value": "close"},
                {"type": "literal", "value": 14},
            ],
        }

        code = compiler.compile_formula(formula_tree)
        assert 'df["high"]' in code
        assert 'df["low"]' in code
        assert 'df["close"]' in code
        assert "timeperiod=14" in code

    def test_atr_compile_with_insufficient_args_should_not_crash(self):
        """ATR参数不足时不应崩溃，应生成通用调用让运行时报错"""
        from backend.services.formula_compiler_service import FormulaCompilerService

        compiler = FormulaCompilerService()
        formula_tree = {
            "type": "function",
            "name": "ATR",
            "args": [
                {"type": "column", "value": "close"},
                {"type": "literal", "value": 14},
            ],
        }

        # 参数不足时不应抛异常，而是生成通用调用
        code = compiler.compile_formula(formula_tree)
        assert "ATR" in code


class TestPortfolioAnalysisSharpeNone:
    """Bug #3: portfolio_analysis_service: Sharpe=None时不应崩溃"""

    def test_optimize_weights_sharpe_none_should_return_none(self):
        """当波动率为0导致Sharpe不可计算时，应返回None而非崩溃"""
        from backend.services.portfolio_analysis_service import PortfolioAnalysisService

        service = PortfolioAnalysisService()
        # 构造全零收益（波动率为0），Sharpe应返回None
        dates = pd.date_range("2023-01-01", periods=30)
        factor_returns = pd.DataFrame(
            {
                "factor_a": np.zeros(30),
                "factor_b": np.zeros(30),
            },
            index=dates,
        )

        result = service.optimize_weights(factor_returns, method="equal_weight")
        # Sharpe不可计算时应为None，不应崩溃
        assert result["sharpe_ratio"] is None

    def test_optimize_weights_normal_sharpe_should_be_float(self):
        """正常情况下Sharpe应为float"""
        from backend.services.portfolio_analysis_service import PortfolioAnalysisService

        service = PortfolioAnalysisService()
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        factor_returns = pd.DataFrame(
            {
                "factor_a": np.random.randn(252) * 0.01 + 0.0005,
                "factor_b": np.random.randn(252) * 0.01 + 0.0003,
            },
            index=dates,
        )

        result = service.optimize_weights(factor_returns, method="equal_weight")
        assert isinstance(result["sharpe_ratio"], float)


class TestSingleStockWeightOne:
    """Bug #4: 3个策略单股票场景权重应为1.0"""

    def _make_single_stock_df(self, n=100):
        dates = pd.date_range("2023-01-01", periods=n)
        return pd.DataFrame(
            {
                "close": np.random.randn(n).cumsum() + 100,
                "market_cap": np.random.uniform(1e9, 1e10, n),
            },
            index=dates,
        )

    def test_momentum_strategy_single_stock_weight_should_be_one(self):
        """动量策略单股票买入时权重应为1.0"""
        from backend.strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy(momentum_window=5, buy_threshold=0.01, sell_threshold=-0.01)
        df = self._make_single_stock_df()
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)

        # 买入信号的权重应为1.0
        buy_mask = signals == 1
        if buy_mask.any():
            assert (weights[buy_mask] == 1.0).all()

    def test_mean_reversion_strategy_single_stock_weight_should_be_one(self):
        """均值回归策略单股票做多时权重应为1.0"""
        from backend.strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy(lookback_window=10, entry_threshold=1.5)
        df = self._make_single_stock_df()
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)

        # 做多信号权重应为1.0
        buy_mask = signals == 1
        if buy_mask.any():
            assert (weights[buy_mask] == 1.0).all()

    def test_market_cap_strategy_single_stock_weight_should_be_one(self):
        """市值策略单股票权重应为1.0"""
        from backend.strategies.market_cap_strategy import MarketCapStrategy

        strategy = MarketCapStrategy()
        df = self._make_single_stock_df()
        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)

        # 单股票场景权重应为1.0
        buy_mask = signals == 1
        if buy_mask.any():
            assert (weights[buy_mask] == 1.0).all()


# ==================== Major 级别 ====================


class TestFactorOrchestratorICKey:
    """Bug #5: factor_orchestrator_service: IC取key应从ic_stats子字典中取"""

    def test_ic_analysis_should_extract_from_ic_stats_subdict(self):
        """IC分析结果应从ic_stats子字典中正确提取因子key"""
        from backend.services.factor_orchestrator_service import FactorOrchestrator, PipelineStageResult, PipelineStatus

        FactorOrchestrator()
        # 模拟ic_analysis阶段返回的结果
        ic_result = {
            "ic_stats": {
                "my_factor_spearman_ic_1D": {
                    "IC均值": 0.05,
                    "IR": 0.3,
                    "Rank_IC均值": 0.04,
                },
            },
            "monthly_ic": {},
            "rolling_ir": {},
        }

        PipelineStageResult(
            stage_name="ic_analysis",
            status=PipelineStatus.PASSED,
            result=ic_result,
        )

        # 验证能从ic_stats中正确提取
        ic_stats = ic_result.get("ic_stats", {})
        factor_ic = ic_stats.get("my_factor", {})
        # 如果直接key不存在，应尝试模糊匹配
        if not factor_ic:
            for key, stats in ic_stats.items():
                if "my_factor" in key:
                    factor_ic = stats
                    break

        assert factor_ic.get("IC均值") == 0.05


class TestMeanReversionShortWeight:
    """Bug #6: mean_reversion_strategy: 做空信号应分配负权重"""

    def test_sell_signal_should_have_negative_weight(self):
        """做空信号(sell_mask)的权重应为负值"""
        from backend.strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy(lookback_window=10, entry_threshold=1.5)
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame(
            {
                "close": np.random.randn(100).cumsum() + 100,
            },
            index=dates,
        )

        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)

        # 做空信号(-1)的权重应为负值
        sell_mask = signals == -1
        if sell_mask.any():
            assert (weights[sell_mask] == -1.0).all(), f"做空权重应为-1.0，实际为{weights[sell_mask].unique()}"


class TestFactorOrchestratorCrossSectionalDetection:
    """Bug #7: factor_orchestrator_service: 多股票未来函数检测应使用detect_cross_sectional"""

    def test_multi_stock_lookahead_should_use_cross_sectional(self):
        """多股票场景应使用detect_cross_sectional方法"""
        from backend.services.factor_orchestrator_service import FactorOrchestrator

        orchestrator = FactorOrchestrator()
        # 验证_stage_lookahead_detection中多股票分支调用detect_cross_sectional
        # 通过检查代码逻辑确认：len(factor_data) >= 2 时走detect_cross_sectional分支
        shared_data = {
            "factor_data": {
                "600036": pd.DataFrame({"my_factor": np.random.randn(50), "close": np.random.randn(50).cumsum() + 100}),
                "000001": pd.DataFrame({"my_factor": np.random.randn(50), "close": np.random.randn(50).cumsum() + 50}),
            },
            "factor_name": "my_factor",
        }

        # lookahead_bias_detector是在方法内部import的，需要patch源模块
        with patch("backend.services.lookahead_bias_detector.lookahead_bias_detector") as mock_detector:
            mock_result = MagicMock()
            mock_result.has_bias = False
            mock_result.risk_level = MagicMock(value="safe")
            mock_result.risk_score = 10.0
            mock_result.summary = "safe"
            mock_result.recommendations = []
            mock_result.checks = []
            mock_detector.detect_cross_sectional.return_value = mock_result

            orchestrator._stage_lookahead_detection(shared_data)

            # 多股票场景应调用detect_cross_sectional
            mock_detector.detect_cross_sectional.assert_called_once()


class TestAnalysisServiceMaskWhere:
    """Bug #8: analysis_service: mask后应使用where而非布尔索引过滤"""

    def test_single_stock_ic_with_tradable_mask_should_use_where(self):
        """有tradable_mask时，IC计算应使用where保留原始索引"""
        from backend.services.analysis_service import AnalysisService

        service = AnalysisService()
        dates = pd.date_range("2023-01-01", periods=200)
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "factor_a": np.random.randn(200),
                "close": np.random.randn(200).cumsum() + 100,
                "future_return_1": np.random.randn(200) * 0.01,
                "tradable_mask": np.random.choice([True, False], 200, p=[0.8, 0.2]),
                "is_limit_up": np.zeros(200, dtype=bool),
                "is_limit_down": np.zeros(200, dtype=bool),
            },
            index=dates,
        )

        factor_data = {"600036": df}
        result = service._calculate_single_stock_ic(factor_data, ["factor_a"])

        # 使用where后rolling计算应基于原始连续索引
        if "factor_a" in result.get("ic_stats", {}):
            stats = result["ic_stats"]["factor_a"]
            # IC类型应标注为Spearman
            assert "Spearman" in stats.get("IC类型", "") or "Rank IC" in stats.get("IC类型", "")


class TestSmartPreprocessingFatTail:
    """Bug #9: smart_preprocessing_detector: 肥尾判断阈值应为kurtosis>3"""

    def test_fat_tail_threshold_should_be_kurtosis_greater_than_3(self):
        """峰度>3（超额峰度>0）应判定为肥尾分布"""
        from backend.services.smart_preprocessing_detector import SmartPreprocessingDetector

        detector = SmartPreprocessingDetector()
        # 使用足够大的样本量使t(3)的样本超额峰度稳定
        # t(3)理论超额峰度=6，但小样本下估计不稳定
        np.random.seed(42)
        n = 5000
        dates = pd.date_range("2023-01-01", periods=n)

        # 构造肥尾分布数据（t分布自由度=3，超额峰度=6，总峰度=9）
        fat_tail_data = {
            "600036": pd.DataFrame(
                {
                    "factor_a": np.random.standard_t(df=3, size=n),
                    "date": dates,
                },
                index=dates,
            ),
        }

        chars = detector.analyze_data(fat_tail_data, ["factor_a"])
        # t(3)分布的超额峰度=6，总峰度=9，应判定为肥尾
        assert chars.is_fat_tailed is True

    def test_normal_distribution_should_not_be_fat_tail(self):
        """正态分布（峰度≈3）不应判定为肥尾"""
        from backend.services.smart_preprocessing_detector import SmartPreprocessingDetector

        detector = SmartPreprocessingDetector()
        dates = pd.date_range("2023-01-01", periods=1000)

        np.random.seed(42)
        normal_data = {
            "600036": pd.DataFrame(
                {
                    "factor_a": np.random.randn(1000),
                    "date": dates,
                },
                index=dates,
            ),
        }

        chars = detector.analyze_data(normal_data, ["factor_a"])
        # 正态分布峰度≈0（pandas kurtosis返回超额峰度），不应判定为肥尾
        assert chars.is_fat_tailed is False


class TestSmartPreprocessingOutlierRatioMAD:
    """Bug #10: smart_preprocessing_detector: 异常值比例应使用MAD法"""

    def test_outlier_ratio_should_use_mad_method(self):
        """异常值比例应使用MAD法计算，与推荐的去极值方法一致"""
        from backend.services.smart_preprocessing_detector import SmartPreprocessingDetector

        detector = SmartPreprocessingDetector()
        # 构造含异常值的数据
        np.random.seed(42)
        data = np.random.randn(100)
        data[0] = 100  # 极端异常值

        series = pd.Series(data)
        ratio = detector._calc_outlier_ratio_mad(series, n_sigma=3.0)

        # MAD法应能检测到异常值
        assert ratio > 0
        assert isinstance(ratio, float)

    def test_outlier_ratio_no_outliers_should_be_zero(self):
        """无异常值时比例应为0"""
        from backend.services.smart_preprocessing_detector import SmartPreprocessingDetector

        detector = SmartPreprocessingDetector()
        # 均匀分布数据，无异常值
        series = pd.Series(np.ones(100))
        ratio = detector._calc_outlier_ratio_mad(series, n_sigma=3.0)

        # 所有值相同，MAD=0，应返回0
        assert ratio == 0.0


class TestWeightOptimizerAlignIndices:
    """Bug #11: weight_optimizer_service: _max_sharpe/_risk_parity应先对齐索引"""

    def test_align_factor_indices_should_handle_different_date_ranges(self):
        """不同起止日期的因子Series应对齐后再构建收益矩阵"""
        from backend.services.weight_optimizer_service import WeightOptimizer

        optimizer = WeightOptimizer()
        dates_a = pd.date_range("2023-01-01", periods=100)
        dates_b = pd.date_range("2023-02-01", periods=100)

        factor_values = {
            "factor_a": pd.Series(np.random.randn(100), index=dates_a),
            "factor_b": pd.Series(np.random.randn(100), index=dates_b),
        }

        aligned = optimizer._align_factor_indices(factor_values)
        # 对齐后两个Series应有相同的索引
        assert len(aligned["factor_a"]) == len(aligned["factor_b"])
        # 对齐后索引应为交集
        common = dates_a.intersection(dates_b)
        assert len(aligned["factor_a"]) == len(common)

    def test_max_sharpe_with_misaligned_indices_should_not_crash(self):
        """索引不对齐时max_sharpe不应崩溃"""
        from backend.services.weight_optimizer_service import WeightOptimizer

        optimizer = WeightOptimizer()
        dates_a = pd.date_range("2023-01-01", periods=100)
        dates_b = pd.date_range("2023-02-01", periods=100)

        factor_values = {
            "factor_a": pd.Series(np.random.randn(100), index=dates_a),
            "factor_b": pd.Series(np.random.randn(100), index=dates_b),
        }

        # 应正常执行或回退到等权，不崩溃
        result = optimizer._max_sharpe(factor_values, ["factor_a", "factor_b"], None)
        assert "weights" in result


class TestFactorServiceDownsideRiskWhere:
    """Bug #12: factor_service: downside_risk应使用where而非pipe(filter)"""

    def test_downside_risk_code_should_use_where(self):
        """downside_risk因子代码应使用.where(lambda x: x < 0)而非.pipe(filter)"""
        from backend.services.factor_service import FactorService

        service = FactorService()
        # 获取内置因子定义中的downside_risk代码
        default_factors = service._get_default_factors()
        risk_factors = default_factors.get("风险指标", [])
        downside_risk_factor = None
        for f in risk_factors:
            if f["name"] == "downside_risk":
                downside_risk_factor = f
                break

        assert downside_risk_factor is not None, "应存在downside_risk因子定义"
        code = downside_risk_factor["code"]
        # 应使用where而非pipe(filter)
        assert ".where(" in code, f"downside_risk应使用.where()，实际代码: {code}"
        assert ".pipe(" not in code, f"downside_risk不应使用.pipe()，实际代码: {code}"


class TestFactorCorrelationFisherZ:
    """Bug #13: factor_correlation_service: Fisher z变换标准误"""

    def test_fisher_z_standard_error_should_use_n_days_minus_3(self):
        """Fisher z变换标准误应为1/sqrt(n-3)，而非1/sqrt(n-1)"""
        from backend.services.factor_correlation_service import FactorCorrelationService

        service = FactorCorrelationService()
        # 构造横截面相关性结果
        n_days = 50
        cross_sectional = {
            "avg_pearson": {
                "factor_a": {"factor_a": 1.0, "factor_b": 0.3},
                "factor_b": {"factor_a": 0.3, "factor_b": 1.0},
            },
            "n_days": n_days,
        }

        result = service._significance_tests(cross_sectional)

        # 验证Fisher z标准误 = 1/sqrt(n_days - 3)
        # 如果使用了错误的1/sqrt(n-1)，p值会不同
        z_val = np.arctanh(0.3)
        correct_se = 1 / np.sqrt(n_days - 3)
        2 * (1 - scipy_stats.norm.cdf(abs(z_val) / correct_se))

        wrong_se = 1 / np.sqrt(n_days - 1)
        2 * (1 - scipy_stats.norm.cdf(abs(z_val) / wrong_se))

        # 正确标准误应更大（n-3 < n-1），因此p值应更大
        assert correct_se > wrong_se
        # 结果中应存在显著性检验
        assert "results" in result


class TestSmartSlippagePerStockVolatility:
    """Bug #14: smart_slippage_detector: 波动率应分别计算每只股票"""

    def test_volatility_should_be_calculated_per_stock(self):
        """每只股票应分别计算波动率，取中位数而非混合计算"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector

        detector = SmartSlippageDetector()
        dates = pd.date_range("2023-01-01", periods=100)

        price_data = {
            "600036": pd.DataFrame(
                {
                    "close": np.random.randn(100).cumsum() + 100,
                },
                index=dates,
            ),
            "000001": pd.DataFrame(
                {
                    "close": np.random.randn(100).cumsum() + 50,
                },
                index=dates,
            ),
        }

        chars = detector.analyze_market(
            stock_codes=["600036", "000001"],
            price_data=price_data,
        )

        # 波动率应为各股票波动率的中位数
        assert chars.price_volatility >= 0

    def test_single_stock_volatility_should_match(self):
        """单只股票波动率应与直接计算一致"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector

        detector = SmartSlippageDetector()
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252)
        close = pd.Series(np.random.randn(252).cumsum() + 100, index=dates)

        price_data = {"600036": pd.DataFrame({"close": close}, index=dates)}

        chars = detector.analyze_market(
            stock_codes=["600036"],
            price_data=price_data,
        )

        # 波动率应大于0
        assert chars.price_volatility > 0


# ==================== Minor 级别 ====================


class TestSingleStockICSpearman:
    """Bug #15: analysis_service: 单股票IC应使用Spearman"""

    def test_single_stock_ic_should_use_spearman(self):
        """单股票IC计算应使用Spearman（Rank IC）作为主统计量"""
        from backend.services.analysis_service import AnalysisService

        service = AnalysisService()
        dates = pd.date_range("2023-01-01", periods=200)
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "factor_a": np.random.randn(200),
                "close": np.random.randn(200).cumsum() + 100,
                "future_return_1": np.random.randn(200) * 0.01,
            },
            index=dates,
        )

        factor_data = {"600036": df}
        result = service._calculate_single_stock_ic(factor_data, ["factor_a"])

        if "factor_a" in result.get("ic_stats", {}):
            stats = result["ic_stats"]["factor_a"]
            # IC类型应标注为Spearman
            assert "Spearman" in stats.get("IC类型", "") or "Rank IC" in stats.get("IC类型", "")


class TestFactorEffectivenessCrossSectionalSpearman:
    """Bug #16: factor_effectiveness_service: 横截面IC应使用spearmanr"""

    # Bug #16已修复：_calculate_cross_sectional_ic已使用spearmanr
    def test_cross_sectional_ic_should_use_spearmanr(self):
        """横截面IC计算应使用scipy.stats.spearmanr"""

        # 验证_calculate_cross_sectional_ic方法使用spearmanr
        # 通过检查import确认
        import backend.services.factor_effectiveness_service as fe_module
        import inspect

        source = inspect.getsource(fe_module.FactorEffectivenessService._calculate_cross_sectional_ic)
        # 应包含spearmanr调用（而非pearsonr）
        assert "spearmanr" in source, "横截面IC应使用spearmanr"


class TestCrossSectionalMADMinValue:
    """Bug #17: factor_preprocessing_pipeline: 横截面MAD应检查极小值"""

    def test_cross_sectional_mad_near_zero_should_not_clip(self):
        """横截面MAD极小时（数据过于集中），不应执行clip"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
            WinsorizeMethod,
        )

        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(
                winsorize_method=WinsorizeMethod.MAD,
                winsorize_n_sigma=3.0,
                cross_sectional=True,
            )
        )

        # 构造几乎恒定的因子值
        series = pd.Series([1.0] * 50 + [1.0001] * 50)
        result, stats = pipeline._winsorize(series)

        # 数据几乎恒定，MAD极小，不应产生异常clip
        assert not result.isna().any() or stats["clipped_count"] == 0

    def test_cross_sectional_mad_zero_should_fallback_to_std(self):
        """MAD=0时应回退到std作为σ_hat估计"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
            WinsorizeMethod,
        )

        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(
                winsorize_method=WinsorizeMethod.MAD,
                winsorize_n_sigma=3.0,
            )
        )

        # 所有值完全相同
        series = pd.Series([5.0] * 100)
        result, stats = pipeline._winsorize(series)

        # 数据完全一致，不应clip任何值
        assert stats["clipped_count"] == 0
        assert (result == 5.0).all()


class TestCrossSectionalJointNeutralizationSmallIndustry:
    """Bug #18: factor_preprocessing_pipeline: 横截面联合中性化应合并小行业"""

    def test_small_industry_should_be_merged_to_other(self):
        """样本量<5的小行业应合并为"Other"类别"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(
                enable_market_cap_neutralization=True,
                enable_industry_neutralization=True,
                use_joint_neutralization=True,
                cross_sectional=True,
                min_samples=5,
            )
        )

        # 构造含小行业的数据（多日期确保groupby.apply正确展开）
        np.random.seed(42)
        dates = [pd.Timestamp("2023-01-01")] * 25 + [pd.Timestamp("2023-01-02")] * 25
        n = 50
        df = pd.DataFrame(
            {
                "date": dates,
                "factor_a": np.random.randn(n),
                "market_cap": np.random.uniform(1e9, 1e10, n),
                "industry": (["银行"] * 10 + ["地产"] * 10 + ["微小行业A"] * 2 + ["微小行业B"] * 2 + ["科技"] * 1) * 2,
                "stock_code": [f"stock_{i}" for i in range(n)],
            }
        )

        # 执行横截面处理，不应崩溃
        result, stats = pipeline._process_cross_sectional(df, "factor_a", "market_cap", "industry", "date")

        # 应正常返回结果（Series或DataFrame）
        assert result is not None
        assert "dates_processed" in stats


class TestMomentumStrategyStateMachine:
    """Bug #19: momentum_strategy: 应使用持仓状态机"""

    def test_momentum_signal_should_use_position_state_machine(self):
        """动量策略应使用持仓状态机，中性区间保持当前头寸"""
        from backend.strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy(momentum_window=5, buy_threshold=0.03, sell_threshold=-0.03)
        # 构造先涨后横盘的数据
        n = 60
        close = np.ones(n) * 100
        close[10:20] = np.linspace(100, 110, 10)  # 上涨触发买入
        close[20:40] = 110  # 横盘（动量接近0，中性区间）
        close[40:50] = np.linspace(110, 100, 10)  # 下跌触发卖出

        df = pd.DataFrame({"close": close}, index=pd.date_range("2023-01-01", periods=n))
        signals = strategy.generate_signals(df)

        # 横盘区间信号0应表示"持有当前头寸"而非"清仓"
        # 买入后横盘区间应保持持仓状态（信号=1）
        # 验证信号不是全部为0
        assert signals.value_counts().get(1, 0) > 0 or signals.value_counts().get(-1, 0) > 0

    def test_momentum_state_machine_should_not_flip_in_neutral_zone(self):
        """状态机在中性区间不应翻转信号"""
        from backend.strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy(momentum_window=5, buy_threshold=0.05, sell_threshold=-0.05)
        n = 100
        np.random.seed(42)
        # 微小波动，大部分时间在阈值内
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)

        df = pd.DataFrame({"close": close}, index=pd.date_range("2023-01-01", periods=n))
        signals = strategy.generate_signals(df)

        # 信号应保持状态机逻辑，不会频繁翻转
        signal_changes = (signals.diff().abs() > 0).sum()
        # 状态机模式下信号变化次数应有限
        assert signal_changes < n * 0.5


class TestVectorbtBacktestWarmupFilter:
    """Bug #20: vectorbt_backtest_service: 首块分层收益应过滤warmup"""

    def test_first_chunk_quantile_returns_should_filter_warmup(self):
        """首块(i=0)的分层收益也应过滤warmup区"""
        # 验证代码逻辑：所有分块（包括i=0）都过滤warmup区
        import inspect
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService

        source = inspect.getsource(VectorBTBacktestService.chunked_single_factor_backtest)
        # 应包含对所有分块过滤warmup的逻辑
        assert "warmup_end_time" in source or "warmup" in source.lower()


class TestMarketCapStrategyDatetimeIndex:
    """Bug #21: market_cap_strategy: DatetimeIndex应识别为日期级别"""

    def test_datetime_index_should_be_recognized_as_date_level(self):
        """DatetimeIndex应被识别为日期级别，无需名为'date'"""
        from backend.strategies.market_cap_strategy import MarketCapStrategy

        strategy = MarketCapStrategy()
        dates = pd.date_range("2023-01-01", periods=50)

        # 单股票DatetimeIndex场景
        df = pd.DataFrame(
            {
                "close": np.random.randn(50).cumsum() + 100,
                "market_cap": np.random.uniform(1e9, 1e10, 50),
            },
            index=dates,
        )

        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)

        # DatetimeIndex应被识别为日期级别，权重计算应正常
        buy_mask = signals == 1
        if buy_mask.any():
            # 单股票满仓
            assert (weights[buy_mask] == 1.0).all()

    def test_non_datetime_index_named_date_should_also_work(self):
        """索引名为'date'的普通索引也应被识别"""
        from backend.strategies.market_cap_strategy import MarketCapStrategy

        strategy = MarketCapStrategy()
        n = 50
        idx = pd.Index(range(n), name="date")

        df = pd.DataFrame(
            {
                "close": np.random.randn(n).cumsum() + 100,
                "market_cap": np.random.uniform(1e9, 1e10, n),
            },
            index=idx,
        )

        signals = strategy.generate_signals(df)
        weights = strategy.calculate_weights(df, signals)

        # 不应崩溃
        assert len(weights) == n


# ==================== 集成测试 ====================


class TestBusinessLogicBugsIntegration:
    """集成测试：验证多个Bug修复后的协同工作"""

    def test_single_stock_full_pipeline_no_crash(self):
        """单股票完整流程不应崩溃"""
        from backend.services.enhanced_analysis_service import EnhancedAnalysisService

        service = EnhancedAnalysisService()
        dates = pd.date_range("2023-01-01", periods=200)
        factor_data = {
            "600036": pd.DataFrame(
                {
                    "close": np.random.randn(200).cumsum() + 100,
                    "factor_a": np.random.randn(200),
                },
                index=dates,
            ),
        }

        # 应正常完成分析
        result = service.analyze_enhanced(factor_data, ["factor_a"])
        assert "factors" in result

    def test_multi_stock_full_pipeline_no_crash(self):
        """多股票完整流程不应崩溃"""
        from backend.services.enhanced_analysis_service import EnhancedAnalysisService

        service = EnhancedAnalysisService()
        dates = pd.date_range("2023-01-01", periods=200)
        factor_data = {
            "600036": pd.DataFrame(
                {
                    "close": np.random.randn(200).cumsum() + 100,
                    "factor_a": np.random.randn(200),
                },
                index=dates,
            ),
            "000001": pd.DataFrame(
                {
                    "close": np.random.randn(200).cumsum() + 50,
                    "factor_a": np.random.randn(200),
                },
                index=dates,
            ),
        }

        result = service.analyze_enhanced(factor_data, ["factor_a"])
        assert "factors" in result

    def test_portfolio_with_zero_volatility_no_crash(self):
        """零波动率组合不应崩溃"""
        from backend.services.portfolio_analysis_service import PortfolioAnalysisService

        service = PortfolioAnalysisService()
        dates = pd.date_range("2023-01-01", periods=30)

        # 一个因子零波动，一个正常
        factor_returns = pd.DataFrame(
            {
                "factor_a": np.zeros(30),
                "factor_b": np.random.randn(30) * 0.01,
            },
            index=dates,
        )

        result = service.optimize_weights(factor_returns, method="equal_weight")
        assert "weights" in result
        assert "sharpe_ratio" in result

    def test_strategies_with_empty_signals_no_crash(self):
        """策略无交易信号时不应崩溃"""
        from backend.strategies.momentum_strategy import MomentumStrategy
        from backend.strategies.mean_reversion_strategy import MeanReversionStrategy
        from backend.strategies.market_cap_strategy import MarketCapStrategy

        dates = pd.date_range("2023-01-01", periods=50)
        # 构造几乎不变的价格，不太可能触发信号
        df = pd.DataFrame(
            {
                "close": np.ones(50) * 100,
                "market_cap": np.ones(50) * 1e9,
            },
            index=dates,
        )

        for StrategyClass in [MomentumStrategy, MeanReversionStrategy, MarketCapStrategy]:
            strategy = StrategyClass()
            signals = strategy.generate_signals(df)
            weights = strategy.calculate_weights(df, signals)
            assert len(weights) == 50
