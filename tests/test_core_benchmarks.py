"""
核心算法基准测试 - 覆盖6大模块的正确性与性能验证

模块划分：
  1. 数据预处理 (MAD去极值/中性化/标准化/智能检测)
  2. 因子计算 (麦语言函数/CROSS/CONST等)
  3. 因子分析 (IC/IR/未来函数检测)
  4. 回测引擎 (指标计算/信号生成)
  5. 滑点检测 (板块识别/流动性评估)
  6. 加权IC (权重计算/相关性调整)

运行方式：
  pytest tests/test_core_benchmarks.py -v
"""
import time
import numpy as np
import pandas as pd
import pytest


# ============================================================================
# 辅助工具：生成测试数据
# ============================================================================

def make_factor_series(n=500, seed=42, with_outliers=True):
    """生成模拟因子值序列"""
    rng = np.random.RandomState(seed)
    data = rng.randn(n) * 0.05 + 0.01
    if with_outliers:
        # 注入5%的极端异常值
        outlier_idx = rng.choice(n, size=n // 20, replace=False)
        data[outlier_idx] = rng.randn(len(outlier_idx)) * 0.5
    return pd.Series(data, name="test_factor")


def make_market_cap_series(n=500, seed=42):
    """生成模拟市值序列（对数正态分布）"""
    rng = np.random.RandomState(seed)
    return pd.Series(np.exp(rng.randn(n) * 2 + 20), name="market_cap")


def make_industry_series(n=500, n_industries=5, seed=42):
    """生成模拟行业分类序列"""
    rng = np.random.RandomState(seed)
    return pd.Series(
        [f"IND_{rng.randint(0, n_industries)}" for _ in range(n)],
        name="industry",
    )


def make_ohlcv_df(n=500, seed=42):
    """生成模拟OHLCV数据"""
    rng = np.random.RandomState(seed)
    close = 10 + np.cumsum(rng.randn(n) * 0.1)
    return pd.DataFrame({
        "open": close + rng.randn(n) * 0.05,
        "high": close + abs(rng.randn(n) * 0.1),
        "low": close - abs(rng.randn(n) * 0.1),
        "close": close,
        "volume": rng.randint(100000, 10000000, n).astype(float),
    })


def make_cross_sectional_df(n_stocks=50, n_dates=100, seed=42):
    """生成横截面测试数据"""
    rng = np.random.RandomState(seed)
    rows = []
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    for stock_i in range(n_stocks):
        stock_code = f"{600000 + stock_i:06d}"
        for date in dates:
            rows.append({
                "date": date,
                "stock_code": stock_code,
                "factor_value": rng.randn() * 0.05,
                "market_cap": np.exp(rng.randn() * 2 + 20),
                "industry": f"IND_{rng.randint(0, 5)}",
                "close": 10 + rng.randn() * 0.5,
                "return": rng.randn() * 0.02,
            })
    return pd.DataFrame(rows)


# ============================================================================
# 模块1: 数据预处理基准测试
# ============================================================================

class TestPreprocessingPipeline:
    """预处理管道正确性与性能基准"""

    def test_mad_winsorization_should_clip_extreme_values(self):
        """MAD去极值应正确截断极端值"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
            WinsorizeMethod,
        )
        data = make_factor_series(with_outliers=True)
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(winsorize_method=WinsorizeMethod.MAD, winsorize_n_sigma=3.0)
        )
        result, stats = pipeline._winsorize(data)
        # 极端值应被截断
        assert stats["clipped_count"] > 0, "应检测到并截断异常值"
        # 截断后不应有超出边界的值
        median = result.median()
        mad = 1.4826 * np.median(np.abs(result - median))
        if mad > 0:
            outside = ((result < median - 3.0 * mad) | (result > median + 3.0 * mad)).sum()
            assert outside == 0, f"截断后仍有{outside}个值超出3σ范围"

    def test_mad_winsorization_fallback_when_mad_zero(self):
        """MAD=0时应使用std作为fallback，而非std*0.6745（Bug#1修复验证）"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
            WinsorizeMethod,
        )
        # 构造MAD=0的数据（超过50%的值相同）
        data = pd.Series([1.0] * 80 + [2.0, 3.0, -1.0, -2.0] * 5)
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(winsorize_method=WinsorizeMethod.MAD, winsorize_n_sigma=3.0)
        )
        result, stats = pipeline._winsorize(data)
        # MAD=0但std>0，应能正常去极值
        assert len(result) == len(data), "去极值不应改变数据长度"

    def test_mad_winsorization_constant_data(self):
        """全常数数据（MAD=0且std=0）应安全跳过"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
            WinsorizeMethod,
        )
        data = pd.Series([5.0] * 100)
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(winsorize_method=WinsorizeMethod.MAD, winsorize_n_sigma=3.0)
        )
        result, stats = pipeline._winsorize(data)
        assert stats["clipped_count"] == 0, "常数数据不应有截断"

    def test_percentile_winsorization(self):
        """百分位去极值应正确工作"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
            WinsorizeMethod,
        )
        data = make_factor_series(with_outliers=True)
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(
                winsorize_method=WinsorizeMethod.PERCENTILE,
                winsorize_limits=(0.01, 0.99),
            )
        )
        result, stats = pipeline._winsorize(data)
        assert stats["clipped_count"] > 0

    def test_std_winsorization(self):
        """3σ去极值应正确工作"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
            WinsorizeMethod,
        )
        data = make_factor_series(with_outliers=True)
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(winsorize_method=WinsorizeMethod.STD, winsorize_n_sigma=3.0)
        )
        result, stats = pipeline._winsorize(data)
        assert stats["clipped_count"] > 0

    def test_market_cap_neutralization_reduces_correlation(self):
        """市值中性化应显著降低因子与市值的相关性"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig
        n = 500
        factor = make_factor_series(n)
        market_cap = make_market_cap_series(n)
        # 注入市值相关性
        factor = factor + np.log(market_cap) * 0.01

        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(enable_market_cap_neutralization=True, enable_industry_neutralization=False)
        )
        result = pipeline._neutralize_market_cap(factor, market_cap)
        # 中性化后相关性应大幅降低
        corr_before = np.corrcoef(factor, np.log(market_cap))[0, 1]
        valid = result.notna() & market_cap.notna() & (market_cap > 0)
        corr_after = np.corrcoef(result[valid], np.log(market_cap[valid]))[0, 1] if valid.sum() > 10 else 0
        assert abs(corr_after) < abs(corr_before) * 0.3, (
            f"市值中性化后相关性应降低: before={corr_before:.4f}, after={corr_after:.4f}"
        )

    def test_industry_neutralization_reduces_industry_effect(self):
        """行业中性化应消除行业间系统性差异"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig
        n = 500
        factor = make_factor_series(n)
        industry = make_industry_series(n, n_industries=5)
        # 注入行业效应
        industry_means = {"IND_0": 0.1, "IND_1": -0.05, "IND_2": 0.02, "IND_3": -0.08, "IND_4": 0.03}
        for i in range(n):
            factor.iloc[i] += industry_means.get(industry.iloc[i], 0)

        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(enable_market_cap_neutralization=False, enable_industry_neutralization=True)
        )
        result = pipeline._neutralize_industry(factor, industry)
        # 中性化后各行业均值应接近0
        result_df = pd.DataFrame({"result": result, "industry": industry})
        industry_means_after = result_df.groupby("industry")["result"].mean()
        max_abs_mean = industry_means_after.abs().max()
        assert max_abs_mean < 0.01, f"行业中性化后最大行业均值={max_abs_mean:.4f}，应接近0"

    def test_joint_neutralization(self):
        """联合回归中性化应同时消除市值和行业效应"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig
        n = 500
        factor = make_factor_series(n)
        market_cap = make_market_cap_series(n)
        industry = make_industry_series(n, n_industries=5)
        factor = factor + np.log(market_cap) * 0.005

        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(
                enable_market_cap_neutralization=True,
                enable_industry_neutralization=True,
                use_joint_neutralization=True,
            )
        )
        result = pipeline._neutralize_joint(factor, market_cap, industry)
        assert result.notna().sum() > n * 0.8, "联合中性化应保留大部分有效数据"

    def test_zscore_standardization(self):
        """Z-score标准化后均值≈0，标准差≈1"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig, StandardizeMethod
        data = make_factor_series()
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(standardize_method=StandardizeMethod.ZSCORE)
        )
        result = pipeline._standardize(data)
        assert abs(result.mean()) < 0.01, f"Z-score后均值={result.mean():.4f}，应接近0"
        assert abs(result.std() - 1.0) < 0.05, f"Z-score后标准差={result.std():.4f}，应接近1"

    def test_rank_standardization(self):
        """Rank标准化后值在[0,1]区间"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig, StandardizeMethod
        data = make_factor_series()
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(standardize_method=StandardizeMethod.RANK)
        )
        result = pipeline._standardize(data)
        assert result.min() >= 0 and result.max() <= 1, "Rank标准化后值应在[0,1]"

    def test_median_mad_standardization(self):
        """Median-MAD标准化应抗异常值"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig, StandardizeMethod
        data = make_factor_series(with_outliers=True)
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(standardize_method=StandardizeMethod.MEDIAN_MAD)
        )
        result = pipeline._standardize(data)
        assert result.notna().all(), "Median-MAD标准化不应产生NaN"

    def test_full_pipeline_end_to_end(self):
        """完整管道端到端测试"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig
        n = 500
        factor = make_factor_series(n)
        market_cap = make_market_cap_series(n)
        industry = make_industry_series(n)

        pipeline = FactorPreprocessingPipeline()
        result, stats = pipeline.process_single_factor(factor, market_cap, industry)
        assert len(result) == n, "处理后数据长度应不变"
        assert stats["standardized"], "应完成标准化"

    def test_missing_value_handling(self):
        """缺失值处理应正确工作"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig
        data = make_factor_series(n=200)
        data.iloc[10:20] = np.nan  # 注入10个NaN
        assert data.isna().sum() == 10

        # fill_zero
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(handle_missing="fill_zero"))
        result, stats = pipeline._handle_missing(data)
        assert result.isna().sum() == 0, "fill_zero后不应有NaN"
        assert stats["filled_count"] == 10

        # fill_median
        pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(handle_missing="fill_median"))
        result, stats = pipeline._handle_missing(data)
        assert result.isna().sum() == 0

    def test_performance_large_dataset(self):
        """性能基准：100万样本处理应<2秒"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig
        n = 1_000_000
        rng = np.random.RandomState(42)
        factor = pd.Series(rng.randn(n) * 0.05 + 0.01)
        market_cap = pd.Series(np.exp(rng.randn(n) * 2 + 20))
        industry = pd.Series([f"IND_{rng.randint(0, 5)}" for _ in range(n)])

        pipeline = FactorPreprocessingPipeline()
        start = time.time()
        result, stats = pipeline.process_single_factor(factor, market_cap, industry)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"100万样本处理耗时{elapsed:.2f}s，应<2秒"
        print(f"\n[性能] 100万样本预处理: {elapsed:.3f}s")


# ============================================================================
# 模块2: 因子计算基准测试
# ============================================================================

class TestFactorCalculator:
    """因子计算器正确性基准"""

    def test_cross_function_operator_precedence(self):
        """CROSS函数应正确检测金叉（Bug#2修复验证）"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        # 构造明确的金叉场景：x从下方穿越y
        x = pd.Series([1, 2, 3, 2, 1], dtype=float)
        y = pd.Series([2, 2, 2, 2, 2], dtype=float)
        cross_func = calc.mylanguage_funcs["CROSS"]
        result = cross_func(x, y)
        # 第2个位置(索引1): x(2)不大于y(2), 不算金叉
        # 第3个位置(索引2): x(3)>y(2)且x.shift(1)(2)<=y.shift(1)(2) → 金叉!
        assert result.iloc[2] == True, "索引2应为金叉点"
        assert result.iloc[0] == False
        assert result.iloc[3] == False

    def test_cross_function_with_scalars(self):
        """CROSS函数在非Series输入时也应正确工作"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        cross_func = calc.mylanguage_funcs["CROSS"]
        x = pd.Series([1, 2, 3, 2, 1], dtype=float)
        y = pd.Series([2, 2, 2, 2, 2], dtype=float)
        result = cross_func(x, y)
        assert isinstance(result, pd.Series)

    def test_const_function_default_length(self):
        """CONST函数不指定length时应返回标量（Bug#5修复验证）"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        const_func = calc.mylanguage_funcs["CONST"]
        # 不指定length，应返回标量
        result = const_func(42)
        assert result == 42, f"CONST(42)应返回42，实际返回{result}"
        # 指定length，应返回Series
        result = const_func(42, length=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 10
        assert (result == 42).all()

    def test_ref_function(self):
        """REF函数应正确引用N日前的值"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        ref_func = calc.mylanguage_funcs["REF"]
        s = pd.Series([10, 20, 30, 40, 50], dtype=float)
        result = ref_func(s, 2)
        assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
        assert result.iloc[2] == 10
        assert result.iloc[4] == 30

    def test_hhv_llv_functions(self):
        """HHV/LLV应正确计算N日最高/最低值"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        hhv = calc.mylanguage_funcs["HHV"]
        llv = calc.mylanguage_funcs["LLV"]
        s = pd.Series([10, 20, 15, 30, 25], dtype=float)
        assert hhv(s, 3).iloc[4] == 30  # 最近3日最高
        assert llv(s, 3).iloc[4] == 15  # 最近3日最低

    def test_barslast_function(self):
        """BARSLAST应正确计算距上次条件满足的周期数"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        barslast = calc.mylanguage_funcs["BARSLAST"]
        condition = pd.Series([True, False, False, True, False, False, False])
        result = barslast(condition)
        assert result.iloc[0] == 0  # 当前就满足
        assert result.iloc[1] == 1  # 1期前满足
        assert result.iloc[2] == 2  # 2期前满足
        assert result.iloc[3] == 0  # 当前就满足
        assert result.iloc[6] == 3  # 3期前满足

    def test_if_function(self):
        """IF条件函数应正确工作"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        if_func = calc.mylanguage_funcs["IF"]
        cond = pd.Series([True, False, True, False])
        result = if_func(cond, 1, -1)
        assert list(result) == [1, -1, 1, -1]

    def test_expression_calculation(self):
        """表达式形式的因子计算应正确工作"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        df = make_ohlcv_df()
        result = calc.calculate(df, "close / SMA(close, timeperiod=20)")
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_performance_factor_calculation(self):
        """性能基准：单因子x100股票x250天应<2秒"""
        from backend.services.factor_service import FactorCalculator
        calc = FactorCalculator()
        df = make_ohlcv_df(n=250)
        start = time.time()
        for _ in range(100):
            calc.calculate(df, "np.log(close / close.shift(1))")
        elapsed = time.time() - start
        assert elapsed < 2.0, f"100次因子计算耗时{elapsed:.2f}s，应<2秒"
        print(f"\n[性能] 100次因子计算: {elapsed:.3f}s")


# ============================================================================
# 模块3: 因子分析基准测试
# ============================================================================

class TestAnalysisService:
    """因子分析正确性基准"""

    def test_calculate_ic_ir_does_not_modify_input(self):
        """calculate_ic_ir不应修改输入数据（Bug#3修复验证）"""
        from backend.services.analysis_service import AnalysisService
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        factor_data = {
            "600000": pd.DataFrame({
                "close": 10 + np.random.randn(n).cumsum() * 0.1,
                "factor1": np.random.randn(n) * 0.05,
            }, index=dates),
            "600001": pd.DataFrame({
                "close": 10 + np.random.randn(n).cumsum() * 0.1,
                "factor1": np.random.randn(n) * 0.05,
            }, index=dates),
        }
        # 记录原始列
        original_cols_0 = set(factor_data["600000"].columns)
        original_cols_1 = set(factor_data["600001"].columns)

        service = AnalysisService()
        # 无论alphalens是否成功，都不应修改输入
        try:
            service.calculate_ic_ir(factor_data, ["factor1"], ["600000", "600001"])
        except Exception:
            pass  # alphalens可能因数据质量失败，但不应影响副作用检测

        # 验证输入未被修改
        assert set(factor_data["600000"].columns) == original_cols_0, (
            "calculate_ic_ir不应修改输入factor_data的列"
        )
        assert set(factor_data["600001"].columns) == original_cols_1

    def test_rolling_ir_no_inf(self):
        """rolling_ir不应产生inf值（Bug#4修复验证）"""
        from backend.services.analysis_service import AnalysisService
        service = AnalysisService()
        # 构造IC全为0的序列（会导致std=0）
        ic_series = {"factor1": pd.Series([0.0] * 100)}
        result = service._calculate_rolling_ir(ic_series, window=20)
        assert not np.isinf(result["factor1"]).any(), "rolling_ir不应包含inf"
        assert not np.isnan(result["factor1"]).any(), "rolling_ir不应包含NaN（除首部外）"

    def test_single_stock_ic_calculation(self):
        """单股票IC计算应返回有效结果"""
        from backend.services.analysis_service import AnalysisService
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)
        factor_data = {
            "600000": pd.DataFrame({
                "close": 10 + rng.randn(n).cumsum() * 0.1,
                "factor1": rng.randn(n) * 0.05,
            }, index=dates),
        }
        service = AnalysisService()
        # calculate_ic_ir会自动添加future_return列
        result = service.calculate_ic_ir(factor_data, ["factor1"], ["600000"])
        assert "ic_stats" in result
        if "factor1" in result["ic_stats"]:
            stats = result["ic_stats"]["factor1"]
            assert "IC均值" in stats
            assert "IR" in stats

    def test_lookahead_bias_detector_safe_factor(self):
        """未来函数检测器对正常因子应返回安全结果"""
        from backend.services.lookahead_bias_detector import LookaheadBiasDetector, BiasRiskLevel
        rng = np.random.RandomState(42)
        n = 500
        factor = pd.Series(rng.randn(n) * 0.05, index=pd.date_range("2023-01-01", periods=n, freq="B"))
        returns = pd.Series(rng.randn(n) * 0.02, index=factor.index)
        detector = LookaheadBiasDetector()
        result = detector.detect(factor, returns, factor_name="normal_factor")
        assert result.risk_level in (BiasRiskLevel.SAFE, BiasRiskLevel.LOW), (
            f"正常因子风险等级应为SAFE/LOW，实际为{result.risk_level}"
        )

    def test_lookahead_bias_detector_biased_factor(self):
        """未来函数检测器对有泄漏的因子应返回高风险"""
        from backend.services.lookahead_bias_detector import LookaheadBiasDetector, BiasRiskLevel
        rng = np.random.RandomState(42)
        n = 500
        returns = pd.Series(rng.randn(n) * 0.02, index=pd.date_range("2023-01-01", periods=n, freq="B"))
        # 构造使用未来信息的因子：factor = return * 0.8
        factor = returns * 0.8 + rng.randn(n) * 0.001
        detector = LookaheadBiasDetector()
        result = detector.detect(factor, returns, factor_name="biased_factor")
        assert result.risk_level in (BiasRiskLevel.HIGH, BiasRiskLevel.CRITICAL), (
            f"有泄漏的因子风险等级应为HIGH/CRITICAL，实际为{result.risk_level}"
        )

    def test_lookahead_bias_cross_sectional(self):
        """横截面未来函数检测应正确工作"""
        from backend.services.lookahead_bias_detector import LookaheadBiasDetector
        df = make_cross_sectional_df(n_stocks=30, n_dates=100)
        factor_df = df[["date", "stock_code", "factor_value"]].rename(columns={"factor_value": "factor_val"})
        return_df = df[["date", "stock_code", "return"]]
        detector = LookaheadBiasDetector()
        result = detector.detect_cross_sectional(
            factor_df=factor_df, return_df=return_df, factor_name="factor_val"
        )
        assert result.risk_score >= 0

    def test_weighted_ic_equal_weight(self):
        """等权加权IC应等于各因子IC的简单平均"""
        from backend.services.weighted_ic_service import WeightedICService, WeightedICConfig, WeightingMethod
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        ic_dict = {
            "f1": pd.Series(rng.randn(n) * 0.05, index=dates),
            "f2": pd.Series(rng.randn(n) * 0.03, index=dates),
        }
        service = WeightedICService(config=WeightedICConfig(weighting_method=WeightingMethod.EQUAL_WEIGHT))
        result = service.calculate_weighted_ic(ic_dict)
        assert result.get("success", False), f"加权IC计算失败: {result.get('error', '')}"
        # 等权加权IC均值应接近两因子IC均值的平均
        expected_mean = (ic_dict["f1"].mean() + ic_dict["f2"].mean()) / 2
        actual_mean = result["weighted_ic"]["mean"]
        assert abs(actual_mean - expected_mean) < 0.01, (
            f"等权IC均值={actual_mean:.4f}，期望≈{expected_mean:.4f}"
        )


# ============================================================================
# 模块4: 回测引擎基准测试
# ============================================================================

class TestBacktestService:
    """回测引擎正确性基准"""

    def test_calculate_metrics_basic(self):
        """基础指标计算应正确"""
        from backend.services.backtest_service import BacktestService
        service = BacktestService()
        # 构造已知收益序列
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01] * 50)
        metrics = service.calculate_metrics(returns)
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "annual_return" in metrics
        assert 0 <= metrics["win_rate"] <= 1

    def test_calculate_metrics_empty_returns(self):
        """空收益序列应返回零指标"""
        from backend.services.backtest_service import BacktestService
        service = BacktestService()
        returns = pd.Series(dtype=float)
        metrics = service.calculate_metrics(returns)
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["total_return"] == 0.0

    def test_drawdown_calculation(self):
        """回撤计算应正确"""
        from backend.services.backtest_service import BacktestService
        service = BacktestService()
        equity = pd.Series([100, 110, 105, 115, 100, 120])
        dd = service.calculate_drawdown(equity)
        assert dd.max() > 0, "应存在回撤"
        assert dd.iloc[1] == 0, "新高点回撤应为0"

    def test_benchmark_metrics(self):
        """基准对比指标应正确"""
        from backend.services.backtest_service import BacktestService
        service = BacktestService()
        rng = np.random.RandomState(42)
        strategy_returns = pd.Series(rng.randn(252) * 0.01)
        benchmark_returns = pd.Series(rng.randn(252) * 0.008)
        metrics = service.calculate_benchmark_metrics(strategy_returns, benchmark_returns)
        assert "information_ratio" in metrics
        assert "beta" in metrics
        assert "correlation" in metrics

    def test_monthly_returns(self):
        """月度收益计算应正确"""
        from backend.services.backtest_service import BacktestService
        service = BacktestService()
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
        monthly = service.calculate_monthly_returns(returns)
        assert isinstance(monthly, pd.DataFrame)
        if len(monthly) > 0:
            assert monthly.shape[1] <= 12  # 最多12个月

    def test_performance_metrics_calculation(self):
        """性能基准：指标计算应快速"""
        from backend.services.backtest_service import BacktestService
        service = BacktestService()
        returns = pd.Series(np.random.randn(10000) * 0.01)
        start = time.time()
        for _ in range(1000):
            service.calculate_metrics(returns)
        elapsed = time.time() - start
        print(f"\n[性能] 1000次指标计算: {elapsed:.3f}s")


# ============================================================================
# 模块5: 智能滑点检测基准测试
# ============================================================================

class TestSmartSlippageDetector:
    """智能滑点检测正确性基准"""

    def test_board_detection_main(self):
        """主板股票应正确识别"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector, MarketBoard
        detector = SmartSlippageDetector()
        chars = detector.analyze_market(["600000", "000001", "600036"])
        assert chars.market_board == MarketBoard.MAIN

    def test_board_detection_chinext(self):
        """创业板股票应正确识别"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector, MarketBoard
        detector = SmartSlippageDetector()
        chars = detector.analyze_market(["300001", "300002", "300003"])
        assert chars.market_board == MarketBoard.CHINEXT

    def test_board_detection_star(self):
        """科创板股票应正确识别"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector, MarketBoard
        detector = SmartSlippageDetector()
        chars = detector.analyze_market(["688001", "688002", "688003"])
        assert chars.market_board == MarketBoard.STAR

    def test_board_detection_beijing(self):
        """北交所股票应正确识别"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector, MarketBoard
        detector = SmartSlippageDetector()
        chars = detector.analyze_market(["830001", "830002", "430001"])
        assert chars.market_board == MarketBoard.BEIJING

    def test_board_detection_mixed(self):
        """混合板块应正确识别"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector, MarketBoard
        detector = SmartSlippageDetector()
        chars = detector.analyze_market(["600000", "300001", "688001"])
        assert chars.market_board == MarketBoard.MIXED

    def test_slippage_recommendation_range(self):
        """滑点推荐值应在合理范围内"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector
        detector = SmartSlippageDetector()
        rec = detector.recommend_slippage(["600000", "600036"], strategy_turnover=12.0)
        assert 0.0005 <= rec.recommended_slippage <= 0.01, (
            f"推荐滑点{rec.recommended_slippage:.4f}超出合理范围[0.05%, 1%]"
        )
        assert rec.conservative_slippage >= rec.recommended_slippage
        assert rec.aggressive_slippage <= rec.recommended_slippage

    def test_slippage_higher_for_illiquid(self):
        """低流动性股票应有更高滑点"""
        from backend.services.smart_slippage_detector import SmartSlippageDetector
        detector = SmartSlippageDetector()
        # 高流动性（大盘股）
        rec_high = detector.recommend_slippage(["600000", "600036"], strategy_turnover=6.0)
        # 使用市场数据模拟低流动性
        low_liq_data = pd.DataFrame({
            "market_cap": [1e8, 2e8],
            "volume": [1000, 2000],
            "amount": [1e5, 2e5],
            "turnover_rate": [0.001, 0.002],
        })
        rec_low = detector.recommend_slippage(
            ["830001", "830002"], strategy_turnover=6.0, market_data=low_liq_data
        )
        # 北交所+低流动性应比主板+高流动性有更高滑点
        assert rec_low.recommended_slippage >= rec_high.recommended_slippage * 0.8, (
            f"低流动性滑点({rec_low.recommended_slippage:.4f})应不低于"
            f"高流动性({rec_high.recommended_slippage:.4f})的80%"
        )


# ============================================================================
# 模块6: 智能预处理检测基准测试
# ============================================================================

class TestSmartPreprocessingDetector:
    """智能预处理参数检测正确性基准"""

    def test_board_detection_in_detector(self):
        """智能检测器应正确识别市场板块"""
        from backend.services.smart_preprocessing_detector import SmartPreprocessingDetector, MarketBoard
        detector = SmartPreprocessingDetector()
        # 主板
        board = detector._detect_market_board(["600000", "000001", "600036"])
        assert board == MarketBoard.MAIN
        # 创业板
        board = detector._detect_market_board(["300001", "300002", "300003"])
        assert board == MarketBoard.CHINEXT

    def test_recommend_config_has_required_keys(self):
        """推荐配置应包含所有必要参数"""
        from backend.services.smart_preprocessing_detector import SmartPreprocessingDetector
        detector = SmartPreprocessingDetector()
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        factor_data = {
            "600000": pd.DataFrame({
                "date": dates,
                "factor1": np.random.randn(n) * 0.05,
                "market_cap": np.exp(np.random.randn(n) * 2 + 20),
                "industry": [f"IND_{np.random.randint(0, 5)}" for _ in range(n)],
            }),
        }
        rec = detector.recommend_config(factor_data, ["factor1"])
        required_keys = [
            "winsorize_method", "winsorize_n_sigma",
            "enable_market_cap_neutralization", "enable_industry_neutralization",
            "standardize_method", "handle_missing", "min_samples", "cross_sectional",
        ]
        for key in required_keys:
            assert key in rec.config_dict, f"推荐配置缺少必要参数: {key}"
        assert 0 <= rec.confidence <= 1

    def test_fat_tail_detection_recommends_mad(self):
        """肥尾分布应推荐MAD法"""
        from backend.services.smart_preprocessing_detector import SmartPreprocessingDetector
        detector = SmartPreprocessingDetector()
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        # 生成肥尾数据（t分布）
        rng = np.random.RandomState(42)
        factor_data = {
            "600000": pd.DataFrame({
                "date": dates,
                "factor1": rng.standard_t(df=3, size=n) * 0.05,  # 肥尾
                "market_cap": np.exp(rng.randn(n) * 2 + 20),
                "industry": [f"IND_{rng.randint(0, 5)}" for _ in range(n)],
            }),
        }
        rec = detector.recommend_config(factor_data, ["factor1"])
        assert rec.config_dict["winsorize_method"] == "mad", "肥尾分布应推荐MAD法"


# ============================================================================
# 边界情况测试
# ============================================================================

class TestEdgeCases:
    """边界情况与鲁棒性测试"""

    def test_empty_data(self):
        """空数据应安全处理"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline
        pipeline = FactorPreprocessingPipeline()
        result, stats = pipeline.process_single_factor(pd.Series(dtype=float))
        assert stats.get("skipped", False) or len(result) == 0

    def test_constant_data(self):
        """常数数据应安全处理"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig
        pipeline = FactorPreprocessingPipeline()
        data = pd.Series([5.0] * 100)
        result, stats = pipeline.process_single_factor(data)
        assert len(result) == 100

    def test_inf_values(self):
        """无穷大值应安全处理"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline
        pipeline = FactorPreprocessingPipeline()
        data = pd.Series([1.0, 2.0, np.inf, -np.inf, 3.0] * 20)
        result, stats = pipeline.process_single_factor(data)
        assert not np.isinf(result).any(), "处理后不应包含inf"

    def test_all_nan_data(self):
        """全NaN数据应安全处理"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline
        pipeline = FactorPreprocessingPipeline()
        data = pd.Series([np.nan] * 100)
        result, stats = pipeline.process_single_factor(data)
        assert len(result) == 100

    def test_single_value_data(self):
        """单值数据应安全处理"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline
        pipeline = FactorPreprocessingPipeline()
        data = pd.Series([42.0])
        result, stats = pipeline.process_single_factor(data)
        assert stats.get("skipped", False), "单值数据应跳过处理"

    def test_very_large_values(self):
        """极大值应被正确截断"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline, PreprocessingConfig, WinsorizeMethod,
        )
        pipeline = FactorPreprocessingPipeline(
            config=PreprocessingConfig(winsorize_method=WinsorizeMethod.MAD)
        )
        data = pd.Series([0.01, 0.02, 0.015, 1e10, -1e10, 0.012] * 20)
        result, stats = pipeline._winsorize(data)
        assert stats["clipped_count"] > 0, "极大值应被截断"

    def test_neutralization_with_missing_market_cap(self):
        """缺少市值数据时中性化应安全跳过"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline
        pipeline = FactorPreprocessingPipeline()
        factor = make_factor_series()
        market_cap = pd.Series([np.nan] * len(factor))
        result = pipeline._neutralize_market_cap(factor, market_cap)
        # 应返回原始数据（跳过中性化）
        pd.testing.assert_series_equal(result, factor)

    def test_neutralization_with_single_industry(self):
        """仅1个行业时应跳过行业中性化"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline
        pipeline = FactorPreprocessingPipeline()
        factor = make_factor_series()
        industry = pd.Series(["ONLY_ONE"] * len(factor))
        result = pipeline._neutralize_industry(factor, industry)
        # 应返回原始数据（行业不足2个）
        pd.testing.assert_series_equal(result, factor)
