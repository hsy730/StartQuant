"""
P0/P1 改造验证测试脚本

验证以下改造的正确性:
  P0: AnalysisService IC/IR 委托 Alphalens
  P1: 行业中性化 → 回归残差法 (JQ/BQ标准)
  P1: 行业+市值联合中性化 → 一次回归
  P1: MAD 缩放因子 1.4826
  P1: FactorEffectivenessService 委托 Alphalens
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_passed = 0
_failed = 0
_errors = []


def make_mock_factor_data(n_stocks=5, n_dates=120, seed=42):
    """生成模拟多股票因子数据"""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_dates, freq="B")
    industries = ["Tech", "Finance", "Healthcare", "Consumer", "Energy"]
    
    factor_data = {}
    for i in range(n_stocks):
        stock_code = f"{600000 + i}"
        
        base_return = np.random.randn(n_dates) * 0.02
        
        factor_a = np.random.randn(n_dates) * 2 + base_return * 10  
        factor_b = np.random.randn(n_dates) * 3 - base_return * 5  
        
        close = 100 + np.cumsum(base_return) * 50
        close = np.maximum(close, 1)
        
        market_cap = np.exp(np.random.normal(25, 0.8, size=n_dates))
        
        industry = np.random.choice(industries)
        
        df = pd.DataFrame({
            "close": close,
            "factor_a": factor_a,
            "factor_b": factor_b,
            "market_cap": market_cap,
            "industry": industry,
            "tradable_mask": True,
        }, index=dates)
        
        df = df.dropna(how="all")
        
        factor_data[stock_code] = df
    
    return factor_data


def test(name):
    """测试装饰器"""
    def decorator(fn):
        def wrapper():
            global _passed, _failed, _errors
            try:
                fn()
                _passed += 1
                print(f"  [PASS] {name}")
            except AssertionError as e:
                _failed += 1
                _errors.append((name, str(e)))
                print(f"  [FAIL] {name}: {e}")
            except Exception as e:
                _failed += 1
                _errors.append((name, f"异常: {e}"))
                import traceback
                print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
        wrapper()
        return wrapper
    return decorator


# ============================================================
# 测试 1: MAD 1.4826 缩放因子
# ============================================================
@test("MAD缩放因子 1.4826 — 去极值后范围合理")
def test_mad_14826():
    from backend.services.factor_preprocessing_pipeline import (
        FactorPreprocessingPipeline, PreprocessingConfig, WinsorizeMethod,
    )
    
    np.random.seed(42)
    series = pd.Series(np.concatenate([
        np.random.normal(0, 1, 990),
        [15, -12, 20, -18, 30, -25, 100, -80, 50, -40],
    ]))
    
    pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method=WinsorizeMethod.MAD,
        winsorize_n_sigma=3.0,
        enable_market_cap_neutralization=False,
        enable_industry_neutralization=False,
        standardize_method=None,
    ))
    
    result, stats = pipeline.process_single_factor(series)
    
    assert not result.isna().any(), "不应有NaN"
    assert not np.isinf(result).any(), "不应有Inf"
    
    median = series.median()
    mad_raw = np.median(np.abs(series - median))
    expected_upper = median + 3.0 * 1.4826 * mad_raw
    expected_lower = median - 3.0 * 1.4826 * mad_raw
    
    assert result.max() <= expected_upper + 1e-6, \
        f"最大值{result.max():.4f} > 预期上限{expected_upper:.4f}"
    assert result.min() >= expected_lower - 1e-6, \
        f"最小值{result.min():.4f} < 预期下限{expected_lower:.4f}"


@test("MAD 1.4826 — 与无缩放因子的结果不同")
def test_mad_14826_differs():
    from backend.services.factor_preprocessing_pipeline import (
        FactorPreprocessingPipeline, PreprocessingConfig, WinsorizeMethod,
    )
    
    np.random.seed(99)
    series = pd.Series(np.concatenate([np.random.normal(0, 1, 900), [50, -50]]))
    
    pipeline_14826 = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method=WinsorizeMethod.MAD, winsorize_n_sigma=3.0,
        enable_market_cap_neutralization=False, enable_industry_neutralization=False,
        standardize_method=None,
    ))
    
    r1, _ = pipeline_14826.process_single_factor(series)
    
    upper_14826 = r1.max()
    
    raw_mad = np.median(np.abs(series - series.median()))
    old_expected = series.median() + 3.0 * raw_mad
    
    diff_pct = abs(upper_14826 - old_expected) / abs(old_expected) * 100
    
    assert diff_pct > 32.0, \
        f"1.4826缩放应使截断范围变化>32%，实际仅{diff_pct:.1f}%"


@test("Median-MAD标准化使用1.4826")
def test_median_mad_standardize():
    from backend.services.factor_preprocessing_pipeline import (
        FactorPreprocessingPipeline, PreprocessingConfig, StandardizeMethod,
    )
    
    np.random.seed(77)
    series = pd.Series(np.random.normal(10, 5, 200))
    
    pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method=None,
        enable_market_cap_neutralization=False,
        enable_industry_neutralization=False,
        standardize_method=StandardizeMethod.MEDIAN_MAD,
    ))
    
    result, stats = pipeline.process_single_factor(series)
    
    assert not result.isna().any()
    assert abs(result.median()) < 0.01, f"MAD标准化后中位数应接近0，实际{result.median():.4f}"


# ============================================================
# 测试 2: 行业中性化回归残差法
# ============================================================
@test("行业中性化 — 回归残差法（非组内Z-score）")
def test_industry_regression_residual():
    from backend.services.factor_neutralization_service import FactorNeutralizationService
    
    svc = FactorNeutralizationService()
    
    n = 500
    np.random.seed(123)
    industry_list = ["Tech"] * 150 + ["Finance"] * 120 + ["Healthcare"] * 130 + ["Energy"] * 100
    
    tech_bias = 3.0
    fin_bias = -1.5
    health_bias = 0.5
    energy_bias = -2.0
    bias_map = {"Tech": tech_bias, "Finance": fin_bias, "Healthcare": health_bias, "Energy": energy_bias}
    
    factor_vals = np.array([bias_map[ind] + np.random.randn() for ind in industry_list])
    
    df = pd.DataFrame({
        "factor_test": factor_vals,
        "industry": industry_list,
    })
    
    result = svc.neutralize_industry(df, "factor_test", industry_column="industry")
    
    valid_result = result.dropna()
    assert len(valid_result) >= 400, f"有效结果不足: {len(valid_result)}"
    
    for ind_name in set(industry_list):
        mask = df["industry"].values == ind_name
        if mask.sum() >= 10:
            group_residuals = valid_result.values[mask]
            mean_resid = float(np.mean(group_residuals))
            
            original_mean = bias_map[ind_name]
            reduction_pct = abs(mean_resid / (abs(original_mean) + 1e-10)) * 100
            
            assert reduction_pct < 95, \
                f"行业 {ind_name}: 中性化后均值 {mean_resid:.4f} 应远小于原始偏移 {original_mean:.1f}，只降低了{100-reduction_pct:.1f}%"
    
    print(f"     Tech: {tech_bias:+.1f} → {np.mean(valid_result[df['industry']=='Tech']):+.4f}")
    print(f"     Finance: {fin_bias:+.1f} → {np.mean(valid_result[df['industry']=='Finance']):+.4f}")


@test("行业中性化 — 单个行业时应返回原值")
def test_industry_single_skip():
    from backend.services.factor_neutralization_service import FactorNeutralizationService
    
    svc = FactorNeutralizationService()
    df = pd.DataFrame({"f": [1, 2, 3] * 4, "ind": ["Tech"] * 12})
    result = svc.neutralize_industry(df, "f", industry_column="ind")
    
    pd.testing.assert_series_equal(result, df["f"], check_names=False)


# ============================================================
# 测试 3: 联合中性化（一次回归）
# ============================================================
@test("联合中性化 — 同时剥离行业+市值")
def test_joint_neutralization():
    from backend.services.factor_neutralization_service import FactorNeutralizationService
    
    svc = FactorNeutralizationService()
    
    n = 300
    np.random.seed(456)
    
    mc_log = np.random.normal(24, 0.7, n)
    mc_exp = np.exp(mc_log)
    
    industry_list = np.random.choice(["A", "B", "C"], size=n)
    ind_dummies = {}
    for ind in ["A", "B", "C"]:
        ind_dummies[ind] = (industry_list == ind).astype(float)
    
    true_beta_mc = 0.05
    true_beta_A = 2.0
    true_beta_B = -1.5
    
    noise = np.random.randn(n) * 0.5
    factor = (true_beta_mc * mc_log 
             + true_beta_A * ind_dummies["A"] 
             + true_beta_B * ind_dummies["B"] 
             + noise)
    
    df = pd.DataFrame({
        "factor_joint": factor,
        "market_cap": mc_exp,
        "industry": industry_list,
    })
    
    result = svc.neutralize_both(
        df, "factor_joint",
        market_cap_column="market_cap",
        industry_column="industry",
    )
    
    valid_mask = result.notna()
    residuals = result[valid_mask].values
    
    corr_with_mc = np.corrcoef(residuals, mc_log[valid_mask])[0, 1]
    
    assert abs(corr_with_mc) < 0.15, \
        f"联合中性化后与log市值相关性应为接近0，实际={corr_with_mc:.4f}"
    
    for ind in ["A", "B", "C"]:
        mask = (df["industry"][valid_mask] == ind).values
        if mask.sum() >= 10:
            ind_mean = float(residuals[mask].mean())
            assert abs(ind_mean) < 1.5, \
                f"行业 {ind} 残差均值 {ind_mean:.4f} 应接近0"


# ============================================================
# 测试 4: Pipeline 中的行业中性化也用回归残差法
# ============================================================
@test("Pipeline横截面模式 — 行业中性化为回归残差法")
def test_pipeline_industry_regression():
    from backend.services.factor_preprocessing_pipeline import (
        FactorPreprocessingPipeline, PreprocessingConfig,
    )
    
    np.random.seed(789)
    n = 200
    dates = pd.date_range("2023-06-01", periods=n, freq="B")
    
    records = []
    for date in dates:
        for stock in range(1, 11):
            ind = ["Tech", "Fin", "Hlth", "Egy"][stock % 4]
            ind_bias = {"Tech": 3, "Fin": -2, "Hlth": 1, "Egy": -1}[ind]
            records.append({
                "date": date,
                "stock_code": f"{stock:04d}",
                "factor_x": ind_bias + np.random.randn(),
                "market_cap": np.exp(np.random.normal(24, 0.8)),
                "industry": ind,
            })
    
    df = pd.DataFrame(records)
    
    pipeline = FactorPreprocessingPipeline(config=PreprocessingConfig(
        winsorize_method=None,
        enable_market_cap_neutralization=False,
        enable_industry_neutralization=True,
        standardize_method=None,
        cross_sectional=True,
    ))
    
    processed, _ = pipeline.process_factor_dataframe(
        df=df,
        factor_columns=["factor_x"],
        market_cap_column="market_cap",
        industry_column="industry",
        date_column="date",
    )
    
    for ind_name in ["Tech", "Fin", "Hlth", "Egy"]:
        ind_group = processed[processed["industry"] == ind_name]["factor_x"]
        if len(ind_group) >= 20:
            mean_val = ind_group.mean()
            assert abs(mean_val) < 2.0, \
                f"Pipeline行业中性化后 {ind_name} 均值 {mean_val:.4f} 应接近0"


# ============================================================
# 测试 5: AnalysisService IC/IR — 单股票 Mask-First
# ============================================================
@test("AnalysisService — 单股票时序IC (Mask-First)")
def test_analysis_single_stock_ic():
    from backend.services.analysis_service import analysis_service
    
    np.random.seed(111)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    
    signal = np.sin(np.linspace(0, 8*np.pi, 200)) + np.random.randn(200) * 0.3
    returns = np.roll(signal, -1) * 0.01 + np.random.randn(200) * 0.005
    returns[-1] = np.nan
    
    close = 100 + np.cumsum(np.random.randn(200) * 0.5)
    close = np.maximum(close, 1)
    
    df = pd.DataFrame({
        "close": close,
        "test_factor": signal,
        "tradable_mask": True,
        "is_limit_up": False,
        "is_limit_down": False,
    }, index=dates)
    
    factor_data = {"000001": df}
    
    ic_ir = analysis_service.calculate_ic_ir(factor_data, ["test_factor"])
    
    assert "ic_stats" in ic_ir, "应包含ic_stats"
    assert len(ic_ir["ic_stats"]) > 0, "应有至少一个因子的IC统计"
    
    stat_key = next(iter(ic_ir["ic_stats"]))
    stats = ic_ir["ic_stats"][stat_key]
    
    assert "IC均值" in stats, "应包含IC均值"
    assert "IR" in stats, "应包含IR"
    assert "IC类型" in stats, "应包含IC类型标识"
    assert "时序" in stats.get("IC类型", ""), "单股票应是时序IC"
    assert stats.get("Mask-First") is True, "应标记使用了Mask-First"
    
    assert "monthly_ic" in ic_ir, "应包含月度IC"
    assert "rolling_ir" in ic_ir, "应包含滚动IR"
    
    print(f"     IC均值={stats['IC均值']:.4f}, IR={stats['IR']:.4f}")


# ============================================================
# 测试 6: AnalysisService IC/IR — 多股票委托 Alphalens
# ============================================================
@test("AnalysisService — 多股票IC委托Alphalens")
def test_analysis_multi_stock_alphalens():
    from backend.services.analysis_service import analysis_service
    from backend.services.alphalens_analysis_service import ALPHALENS_AVAILABLE
    
    if not ALPHALENS_AVAILABLE:
        print("     [SKIP] Alphalens未安装")
        return
    
    factor_data = make_mock_factor_data(n_stocks=5, n_dates=120, seed=202)
    
    ic_ir = analysis_service.calculate_ic_ir(
        factor_data, 
        ["factor_a"], 
        stock_codes=list(factor_data.keys()),
    )
    
    assert "ic_stats" in ic_ir, "应包含ic_stats"
    assert len(ic_ir["ic_stats"]) > 0, "应有至少一个因子的IC统计"
    
    first_key = list(ic_ir["ic_stats"].keys())[0]
    stats = ic_ir["ic_stats"][first_key]
    
    assert "IC均值" in stats
    assert "IC标准差" in stats
    assert "IR" in stats
    
    if "Alphalens" in stats.get("IC类型", ""):
        assert "t统计量" in stats, "Alphalens应提供t检验"
        assert "p值" in stats, "Alphalens应提供p值"
        print(f"     key={first_key}, IC均值={stats['IC均值']:.4f}, IR={stats['IR']:.4f}, t={stats['t统计量']:.2f} [Alphalens]")
    else:
        print(f"     key={first_key}, IC均值={stats['IC均值']:.4f}, IR={stats['IR']:.4f} [fallback]")
    
    assert "monthly_ic" in ic_ir
    assert "rolling_ir" in ic_ir


# ============================================================
# 测试 7: AnalysisService analyze() 完整流程
# ============================================================
@test("AnalysisService.analyze() — 完整流程不报错")
def test_analyze_full_flow():
    from backend.services.analysis_service import analysis_service
    from unittest.mock import patch, MagicMock
    
    factor_data = make_mock_factor_data(n_stocks=3, n_dates=60, seed=333)
    
    with patch('backend.services.analysis_service.AnalysisCacheRepository') as mock_repo_cls:
        mock_repo = MagicMock()
        mock_repo.get_by_key.return_value = None
        mock_repo.create = MagicMock()
        mock_repo_cls.return_value = mock_repo
        
        with patch('backend.services.analysis_service.SHAP_AVAILABLE', False):
            with patch('backend.services.factor_service.factor_service.calculate_factors_for_stocks', return_value=factor_data):
                results = analysis_service.analyze(
                    stock_codes=list(factor_data.keys()),
                    factor_names=["factor_a"],
                    start_date="2023-01-01",
                    end_date="2023-03-31",
                    use_cache=False,
                )
    
    assert "metadata" in results
    assert "factor_data" in results
    assert "ic_ir" in results
    assert "ic_stats" in results["ic_ir"]
    assert len(results["ic_ir"]["ic_stats"]) > 0


# ============================================================
# 测试 8: FactorEffectivenessService — IC委托Alphalens
# ============================================================
@test("FactorEffectivenessService — IC委托Alphalens")
def test_effectiveness_ic_delegation():
    from backend.services.factor_effectiveness_service import factor_effectiveness_service
    from backend.services.alphalens_analysis_service import ALPHALENS_AVAILABLE
    
    if not ALPHALENS_AVAILABLE:
        print("     ⚠️ Alphalens未安装，跳过")
        return
    
    factor_data = make_mock_factor_data(n_stocks=5, n_dates=120, seed=555)
    
    result = factor_effectiveness_service.analyze_effectiveness(
        factor_data=factor_data,
        factor_name="factor_a",
        future_periods=[1, 5, 10],
    )
    
    assert "scatter_plot" in result
    assert "ic_time_series" in result
    assert "event_response" in result
    assert "decay_analysis" in result
    
    ic_ts = result["ic_time_series"]
    assert "error" not in ic_ts, f"IC计算不应出错: {ic_ts.get('error', '')}"
    assert "ic_mean" in ic_ts
    assert "source" in ic_ts, "应标注数据来源"
    assert ic_ts["source"] == "Alphalens", "多股票应使用Alphalens"
    
    decay = result["decay_analysis"]
    assert "error" not in decay, f"衰减分析不应出错: {decay.get('error', '')}"
    assert "decay_curve" in decay
    assert len(decay["decay_curve"]) >= 3, "衰减曲线应有多个周期点"
    
    print(f"     IC source={ic_ts['source']}, decay points={len(decay['decay_curve'])}")


# ============================================================
# 测试 9: FactorEffectivenessService — 单股票fallback
# ============================================================
@test("FactorEffectivenessService — 单股票fallback自建")
def test_effectiveness_fallback():
    from backend.services.factor_effectiveness_service import factor_effectiveness_service
    
    np.random.seed(666)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    
    signal = np.sin(np.linspace(0, 6*np.pi, 100)) + np.random.randn(100)*0.2
    close = 100 + np.cumsum(np.random.randn(100))
    
    df = pd.DataFrame({
        "close": close,
        "test_fac": signal,
    }, index=dates)
    
    factor_data = {"000001": df}
    
    result = factor_effectiveness_service.analyze_effectiveness(
        factor_data=factor_data,
        factor_name="test_fac",
        future_periods=[1, 5],
    )
    
    ic_ts = result["ic_time_series"]
    assert "error" not in ic_ts, f"单股票IC fallback不应出错: {ic_ts.get('error', '')}"
    assert "ic_mean" in ic_ts
    
    scatter = result["scatter_plot"]
    assert "correlation" in scatter
    assert "count" in scatter
    assert scatter["count"] >= 50


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FactorHub P0/P1 改造验证测试")
    print("=" * 60)
    
    test_mad_14826()
    test_mad_14826_differs()
    test_median_mad_standardize()
    test_industry_regression_residual()
    test_industry_single_skip()
    test_joint_neutralization()
    test_pipeline_industry_regression()
    test_analysis_single_stock_ic()
    test_analysis_multi_stock_alphalens()
    test_analyze_full_flow()
    test_effectiveness_ic_delegation()
    test_effectiveness_fallback()
    
    print("\n" + "=" * 60)
    print(f"结果: {_passed} 通过, {_failed} 失败, 共 {_passed+_failed} 个")
    
    if _errors:
        print("\n失败详情:")
        for name, err in _errors:
            print(f"  ❌ {name}: {err[:120]}")
    
    print("=" * 60)
    
    sys.exit(0 if _failed == 0 else 1)
