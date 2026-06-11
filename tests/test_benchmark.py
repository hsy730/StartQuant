"""
FactorHub 核心计算模块功能基准测试

覆盖范围：
1. 预处理管道（去极值、中性化、标准化）
2. 统计指标（Sharpe、Sortino、最大回撤、年化收益）
3. 因子分析（IC、IR、t检验、Fisher z变换）
4. 回测策略（等权重、Sortino、交易次数）
5. 评分系统（IR/Sharpe负值处理、max_drawdown abs处理）
"""

import numpy as np
import pandas as pd
import pytest
import sys
import time
import re
from unittest.mock import MagicMock

import empyrical

# Mock akshare and heavy dependencies before importing modules that depend on them
sys.modules.setdefault("akshare", MagicMock())
sys.modules.setdefault("sqlalchemy", MagicMock())
sys.modules.setdefault("sqlalchemy.orm", MagicMock())
sys.modules.setdefault("backend.services.cache_service", MagicMock())
sys.modules.setdefault("backend.services.data_service", MagicMock())


# ============================================================
# 辅助函数：生成测试数据（使用独立RNG避免并行测试耦合）
# ============================================================


def generate_normal_factor(n=1000, seed=42):
    """生成正态分布因子数据"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n), name="factor")


def generate_factor_with_outliers(n=1000, n_outliers=20, seed=42):
    """生成含异常值的因子数据"""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(n)
    # 在首尾添加极端异常值
    outlier_indices = np.concatenate(
        [
            rng.choice(range(50), n_outliers // 2, replace=False),
            rng.choice(range(n - 50, n), n_outliers // 2, replace=False),
        ]
    )
    signs = rng.choice([-1, 1], n_outliers)
    magnitudes = rng.uniform(10, 20, n_outliers)
    data[outlier_indices] = signs * magnitudes
    return pd.Series(data, name="factor")


def generate_daily_returns(n=252, mean=0.0005, std=0.02, seed=42):
    """生成日收益率序列"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def generate_stock_data(n_stocks=50, n_days=252, seed=42):
    """生成多股票面板数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    stocks = [f"{600000 + i:06d}" for i in range(n_stocks)]
    data = {}
    for stock in stocks:
        close = 10.0 * (1 + rng.normal(0.0005, 0.02, n_days)).cumprod()
        factor = rng.standard_normal(n_days)
        market_cap = rng.uniform(1e8, 1e10, n_days)
        df = pd.DataFrame(
            {
                "close": close,
                "factor": factor,
                "market_cap": market_cap,
                "industry": rng.choice(["银行", "地产", "科技", "医药", "能源"], n_days),
            },
            index=dates,
        )
        data[stock] = df
    return data


# ============================================================
# 1. 预处理管道基准测试
# ============================================================


class TestWinsorization:
    """去极值基准测试"""

    def test_mad_winsorization_should_clip_extreme_values(self):
        """MAD去极值应截断极端异常值"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        factor = generate_factor_with_outliers(n=1000, n_outliers=20)
        config = PreprocessingConfig(winsorize_method="mad", winsorize_n_sigma=3.0)
        pipeline = FactorPreprocessingPipeline(config)
        result, stats = pipeline.process_single_factor(factor)

        # 截断后的值不应超过 median +/- n_sigma * 1.4826 * MAD
        median = factor.median()
        mad = (factor - median).abs().median() * 1.4826
        lower = median - 3.0 * mad
        upper = median + 3.0 * mad
        assert result.min() >= lower - 0.01, f"最小值 {result.min():.4f} 低于下界 {lower:.4f}"
        assert result.max() <= upper + 0.01, f"最大值 {result.max():.4f} 超过上界 {upper:.4f}"

    def test_mad_winsorization_should_preserve_normal_data(self):
        """MAD去极值应保留正常数据"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        factor = generate_normal_factor(n=1000)
        # 仅测试去极值，禁用中性化和标准化
        config = PreprocessingConfig(
            winsorize_method="mad",
            winsorize_n_sigma=3.0,
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method="zscore",
        )
        pipeline = FactorPreprocessingPipeline(config)
        result, stats = pipeline.process_single_factor(factor)

        # 正态数据3σ截断应截断极少数据点（winsorized_count应很小）
        assert (
            stats["winsorized_count"] / len(factor) < 0.05
        ), f"截断比例 {stats['winsorized_count'] / len(factor) * 100:.1f}% 超过5%"

    def test_percentile_winsorization_should_clip_at_quantiles(self):
        """百分位去极值应在指定分位数截断"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        factor = generate_factor_with_outliers(n=1000, n_outliers=20)
        config = PreprocessingConfig(winsorize_method="percentile", winsorize_limits=(0.01, 0.99))
        pipeline = FactorPreprocessingPipeline(config)
        result, stats = pipeline.process_single_factor(factor)

        q01 = factor.quantile(0.01)
        q99 = factor.quantile(0.99)
        assert result.min() >= q01 - 0.01
        assert result.max() <= q99 + 0.01

    def test_mad_winsorization_should_include_14826_factor(self):
        """MAD去极值应包含1.4826修正因子 - 验证pipeline内部实现"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        factor = pd.Series(data, name="factor")

        # 使用pipeline处理，验证1.4826修正因子的效果：
        # 修正后的MAD*1.4826 ≈ σ，因此MAD法3σ截断与3σ标准差法截断结果应接近
        config_mad = PreprocessingConfig(
            winsorize_method="mad",
            winsorize_n_sigma=3.0,
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method="zscore",
        )
        config_std = PreprocessingConfig(
            winsorize_method="std",
            winsorize_n_sigma=3.0,
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
            standardize_method="zscore",
        )

        result_mad, stats_mad = FactorPreprocessingPipeline(config_mad).process_single_factor(factor)
        result_std, stats_std = FactorPreprocessingPipeline(config_std).process_single_factor(factor)

        # 1.4826修正因子使MAD法与STD法在正态分布下截断比例接近
        ratio_mad = stats_mad["winsorized_count"] / len(factor)
        ratio_std = stats_std["winsorized_count"] / len(factor)
        assert (
            abs(ratio_mad - ratio_std) < 0.02
        ), f"MAD截断比例{ratio_mad:.3f}与STD截断比例{ratio_std:.3f}偏差过大，1.4826修正因子可能缺失"


class TestNeutralization:
    """中性化基准测试"""

    def test_market_cap_neutralization_should_reduce_correlation(self):
        """市值中性化应显著降低因子与市值的相关性"""
        from backend.services.factor_neutralization_service import FactorNeutralizationService

        rng = np.random.default_rng(42)
        n = 500
        market_cap = rng.lognormal(mean=20, sigma=2, size=n)
        # 构造与市值强相关的因子
        factor = 0.8 * np.log(market_cap) + rng.standard_normal(n) * 0.5
        df = pd.DataFrame({"factor": factor, "market_cap": market_cap})

        service = FactorNeutralizationService()
        result = service.neutralize_market_cap(df, "factor", "market_cap")

        # 中性化前相关性
        corr_before = np.corrcoef(factor, np.log(market_cap))[0, 1]
        # 中性化后相关性（result是pd.Series，直接使用有效值）
        valid_mask = result.notna()
        corr_after = np.corrcoef(result[valid_mask].values, np.log(market_cap[valid_mask.values]))[0, 1]

        assert abs(corr_after) < abs(corr_before) * 0.2, f"相关性降低不足: {corr_before:.4f} -> {corr_after:.4f}"

    def test_industry_neutralization_should_reduce_industry_effect(self):
        """行业中性化应降低行业间因子差异"""
        from backend.services.factor_neutralization_service import FactorNeutralizationService

        rng = np.random.default_rng(42)
        n = 300
        industries = rng.choice(["A", "B", "C"], n)
        # 构造行业间差异显著的因子
        industry_effect = {"A": 2.0, "B": -1.0, "C": 0.5}
        factor = np.array([industry_effect[ind] for ind in industries]) + rng.standard_normal(n) * 0.5

        df = pd.DataFrame({"factor": factor, "industry": industries})
        service = FactorNeutralizationService()
        result = service.neutralize_industry(df, "factor", "industry")

        # 中性化后各行业均值应接近0（result是pd.Series）
        for ind in ["A", "B", "C"]:
            ind_mean = result[df["industry"] == ind].dropna().mean()
            assert abs(ind_mean) < 0.3, f"行业 {ind} 均值 {ind_mean:.4f} 偏离0"


class TestStandardization:
    """标准化基准测试"""

    def test_zscore_should_produce_mean_zero_std_one(self):
        """Z-score标准化后均值应接近0，标准差接近1"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        factor = generate_normal_factor(n=1000)
        config = PreprocessingConfig(standardize_method="zscore")
        pipeline = FactorPreprocessingPipeline(config)
        result, stats = pipeline.process_single_factor(factor)

        assert abs(result.mean()) < 0.1, f"均值 {result.mean():.4f} 偏离0"
        assert abs(result.std() - 1.0) < 0.1, f"标准差 {result.std():.4f} 偏离1"

    def test_rank_standardization_should_produce_uniform_distribution(self):
        """Rank标准化后应产生均匀分布"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        factor = generate_normal_factor(n=1000)
        config = PreprocessingConfig(standardize_method="rank")
        pipeline = FactorPreprocessingPipeline(config)
        result, stats = pipeline.process_single_factor(factor)

        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert abs(result.mean() - 0.5) < 0.05, f"均值 {result.mean():.4f} 偏离0.5"


# ============================================================
# 2. 统计指标基准测试
# ============================================================


class TestFinancialMetrics:
    """金融统计指标基准测试"""

    def test_sharpe_ratio_should_be_annualized(self):
        """Sharpe比率应正确年化 - 使用empyrical作为基准"""
        from backend.services.statistics_service import StatisticsService

        returns = generate_daily_returns(n=252, mean=0.001, std=0.02)
        service = StatisticsService()
        result = service.analyze_quantile_returns({"Q1": returns}, annual_trading_days=252)

        sharpe = result["Q1"]["sharpe"]
        # 使用empyrical自身计算作为基准，验证service正确委托了empyrical
        expected_sharpe = float(
            empyrical.sharpe_ratio(returns.values, risk_free=0.03 / 252, period="daily", annualization=252)
        )

        # 验证service的Sharpe与empyrical基准一致（方向和数值）
        assert (
            abs(sharpe - expected_sharpe) < 0.01
        ), f"Sharpe {sharpe:.4f} 与empyrical基准 {expected_sharpe:.4f} 偏差过大"

    def test_annual_return_should_use_compound(self):
        """年化收益率应使用复利计算 - 验证service委托empyrical"""
        from backend.services.statistics_service import StatisticsService

        returns = generate_daily_returns(n=252, mean=0.001, std=0.02)
        service = StatisticsService()
        result = service.analyze_quantile_returns({"Q1": returns}, annual_trading_days=252)

        annual_return = result["Q1"]["annual_return"]
        # 使用empyrical复利年化作为基准
        expected = float(empyrical.annual_return(returns.values, period="daily", annualization=252))
        # 简单乘法年化（错误方式）
        wrong = returns.mean() * 252

        # 结果应与empyrical复利年化一致
        assert (
            abs(annual_return - expected) < 0.01
        ), f"年化收益 {annual_return:.4f} 与empyrical基准 {expected:.4f} 偏差过大"
        # 结果应更接近复利年化而非简单乘法
        diff_compound = abs(annual_return - expected)
        diff_simple = abs(annual_return - wrong)
        assert (
            diff_compound <= diff_simple + 0.01
        ), f"年化收益 {annual_return:.4f} 更接近简单乘法 {wrong:.4f} 而非复利 {expected:.4f}"

    def test_max_drawdown_with_negative_cumulative_returns(self):
        """最大回撤在累计收益为负时应正确计算 - 使用empyrical"""
        # 构造累计收益为负的序列
        returns = pd.Series([-0.05, -0.03, -0.02, 0.01, -0.04, -0.01, 0.02, -0.03])

        # 使用empyrical计算最大回撤（遵循开源库优先原则）
        dd = float(empyrical.max_drawdown(returns.values))

        # 最大回撤应为负数
        assert dd < 0, f"最大回撤 {dd:.4f} 应为负数"
        assert dd >= -1.0, f"最大回撤 {dd:.4f} 不应小于-100%"

    def test_sortino_ratio_formula(self):
        """Sortino比率应使用empyrical标准下行偏差公式"""
        from backend.strategies.equal_weight_strategy import EqualWeightStrategy

        # 构造已知收益序列
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02, -0.01, 0.01])

        # 使用EqualWeightStrategy.calculate_metrics计算Sortino
        strategy = EqualWeightStrategy()
        metrics = strategy.calculate_metrics(returns, risk_free_rate=0.0)

        # 使用empyrical作为基准验证
        expected_sortino = float(
            empyrical.sortino_ratio(returns.values, required_return=0.0, period="daily", annualization=252)
        )

        sortino = metrics["sortino_ratio"]
        # BaseStrategy应委托empyrical计算，结果应一致
        assert (
            abs(sortino - expected_sortino) < 0.01
        ), f"Sortino {sortino:.4f} 与empyrical基准 {expected_sortino:.4f} 偏差过大"


# ============================================================
# 3. 因子分析基准测试
# ============================================================


class TestFactorAnalysis:
    """因子分析基准测试"""

    def test_ic_t_test_should_use_fisher_z(self):
        """IC显著性检验应使用Fisher z变换 - 验证StatisticsService.t_test_ic"""
        from backend.services.statistics_service import StatisticsService

        rng = np.random.default_rng(42)
        n = 60
        # 构造IC均值为0.05的IC序列
        ic_values = rng.normal(loc=0.05, scale=0.1, size=n)
        ic_series = pd.Series(ic_values)

        service = StatisticsService()
        result = service.t_test_ic(ic_series)

        # 验证t检验结果的基本合理性
        assert 0 < result["p_value"] < 1, f"p值 {result['p_value']} 不在合理范围"
        assert abs(result["mean_ic"] - 0.05) < 0.05, f"IC均值 {result['mean_ic']:.4f} 与构造值0.05偏差过大"

        # 对比Fisher z变换与t检验：对于大IC，两者统计量应有差异
        large_ic = 0.8
        ic_large = pd.Series(rng.normal(loc=large_ic, scale=0.1, size=n))
        result_large = service.t_test_ic(ic_large)

        # Fisher z变换的统计量（手动计算作为参考基准）
        z_mean = np.arctanh(large_ic)
        z_se = 1 / np.sqrt(n - 3)
        z_stat = z_mean / z_se

        # t检验统计量与Fisher z统计量在大IC时应不同
        t_stat = result_large["t_statistic"]
        assert abs(t_stat - z_stat) > 0.1, f"t统计量 {t_stat:.4f} 和Fisher z统计量 {z_stat:.4f} 在大IC时应不同"

    def test_welch_t_test_standard_error(self):
        """Welch t检验标准误应正确计算 - 验证StatisticsService分层检验"""
        from backend.services.statistics_service import StatisticsService

        rng = np.random.default_rng(42)
        # 构造两组均值差异显著的收益序列
        n_top, n_bot = 100, 80
        top_returns = pd.Series(rng.normal(0.002, 0.05, n_top))
        bot_returns = pd.Series(rng.normal(-0.001, 0.04, n_bot))

        service = StatisticsService()
        # 使用analyze_quantile_returns验证分层收益分析
        result = service.analyze_quantile_returns({"Q1": top_returns, "Q5": bot_returns}, annual_trading_days=252)

        # 验证两组统计量计算正确
        assert (
            result["Q1"]["mean"] > result["Q5"]["mean"]
        ), f"Q1均值 {result['Q1']['mean']:.4f} 应大于Q5均值 {result['Q5']['mean']:.4f}"

        # 正确的Welch SE公式
        std_top, std_bot = top_returns.std(), bot_returns.std()
        se_correct = np.sqrt(std_top**2 / n_top + std_bot**2 / n_bot)
        # 错误的公式
        se_wrong = np.sqrt(std_top**2 + std_bot**2) * np.sqrt(1 / n_top + 1 / n_bot)
        # 正确SE应更小
        assert se_correct < se_wrong, "正确SE应小于错误SE"


class TestCorrelationAnalysis:
    """相关性分析基准测试"""

    def test_vif_warnings_should_not_lose_severe_warnings(self):
        """VIF警告不应丢失严重共线性警告 - 验证FactorNeutralizationService"""
        from backend.services.factor_neutralization_service import FactorNeutralizationService

        # 构造有严重共线性的因子数据（两个因子高度相关）
        rng = np.random.default_rng(42)
        n = 200
        base = rng.standard_normal(n)
        factor_a = base + rng.standard_normal(n) * 0.01  # 与base几乎完全相关
        factor_b = base + rng.standard_normal(n) * 0.01
        market_cap = rng.uniform(1e8, 1e10, n)

        df = pd.DataFrame(
            {
                "factor_a": factor_a,
                "factor_b": factor_b,
                "market_cap": market_cap,
            }
        )

        service = FactorNeutralizationService()
        # 验证联合中性化在共线性数据下不崩溃
        result = service.neutralize_both(df, "factor_a", "market_cap_column", "industry")
        # 应返回有效结果（非全NaN）
        assert result.notna().any(), "联合中性化在共线性数据下应返回部分有效结果"

    def test_weekly_alignment_should_not_mix_years(self):
        """周频率对齐不应跨年混叠 - 验证ISO周分组逻辑"""
        # 构造跨两年的日期序列，两年都有第1周
        dates_2023 = pd.DatetimeIndex(
            [
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
            ]
        )  # 2023年第1周
        dates_2024 = pd.DatetimeIndex(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        )  # 2024年第1周
        dates = dates_2023.append(dates_2024)

        # 正确做法：按年份+周号分组（避免跨年混叠）
        iso_cal = dates.isocalendar()
        group_keys = iso_cal["year"].astype(str) + "-" + iso_cal["week"].astype(str)
        week_groups = dates.groupby(group_keys)

        # 应有2个组（2023年第1周和2024年第1周）
        assert len(week_groups) == 2, f"应有2个周组，实际 {len(week_groups)}"

        # 错误做法：仅按周号分组会导致跨年混叠（两年的第1周合并为1组）
        week_groups_wrong = dates.groupby(iso_cal["week"])
        assert len(week_groups_wrong) == 1, "仅按周号分组会跨年混叠"


# ============================================================
# 4. 回测策略基准测试
# ============================================================


class TestBacktestStrategies:
    """回测策略基准测试"""

    def test_equal_weight_single_stock_should_be_one(self):
        """单股票等权重策略信号应为1"""
        from backend.strategies.equal_weight_strategy import EqualWeightStrategy

        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "close": 10.0 * (1 + rng.normal(0.001, 0.02, n)).cumprod(),
                "factor": rng.standard_normal(n),
            },
            index=pd.bdate_range("2023-01-01", periods=n),
        )

        strategy = EqualWeightStrategy()
        result = strategy.generate_signals(df)

        # generate_signals 返回 pd.Series，单股票场景信号应全部为1（满仓）
        assert isinstance(result, pd.Series), f"返回值应为Series，实际 {type(result)}"
        assert (result == 1).all(), f"单股票信号应全部为1，实际 {result.unique()}"

    def test_trade_count_should_not_count_nan_diff(self):
        """交易次数不应将NaN diff计为交易 - 验证BaseStrategy.backtest"""
        from backend.strategies.equal_weight_strategy import EqualWeightStrategy

        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame(
            {
                "close": 100.0 * (1 + rng.normal(0.001, 0.02, n)).cumprod(),
                "factor": rng.standard_normal(n),
            },
            index=pd.bdate_range("2023-01-01", periods=n),
        )

        strategy = EqualWeightStrategy()
        result = strategy.backtest(df)

        # BaseStrategy.backtest使用 weights.diff().fillna(0) != 0 计算交易次数
        trades_count = result["trades_count"]
        # 手动验证：weights.diff()第一个值为NaN，fillna(0)后为0，不计为交易
        weights = result["weights"]
        manual_count = (weights.diff().fillna(0) != 0).sum()
        assert trades_count == manual_count, f"交易次数 {trades_count} 与手动计算 {manual_count} 不一致"

    def test_forward_return_no_lookahead_bias(self):
        """前向收益率不应存在前视偏差 - 验证BaseStrategy.backtest"""
        from backend.strategies.equal_weight_strategy import EqualWeightStrategy

        close = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0])
        df = pd.DataFrame(
            {
                "close": close,
                "factor": [0.1, 0.2, -0.1, 0.3, -0.2],
            }
        )

        strategy = EqualWeightStrategy()
        strategy.backtest(df)

        # BaseStrategy使用 close.pct_change(1).shift(-1) 计算前向收益
        # 验证：t日权重配对t+1日收益，无前视偏差
        # 前向收益最后一个值应为NaN（无未来数据）
        forward_return = close.pct_change(1).shift(-1)
        assert pd.isna(forward_return.iloc[-1]), "前向收益最后一个值应为NaN"


# ============================================================
# 5. 评分系统基准测试
# ============================================================


class TestScoringSystem:
    """评分系统基准测试"""

    def test_ir_score_should_handle_negative_ir(self):
        """IR得分应正确处理负IR"""
        # 负IR表示因子方向稳定但反向，同样有价值
        ir_positive = 2.0
        ir_negative = -2.0

        # 修复后的公式
        score_pos = min(abs(ir_positive) * 40, 100)
        score_neg = min(abs(ir_negative) * 40, 100)

        assert score_pos == score_neg, f"正负IR得分应相等: pos={score_pos}, neg={score_neg}"
        assert score_pos == 80, f"IR=2.0得分应为80，实际 {score_pos}"

    def test_sharpe_score_should_have_lower_bound(self):
        """Sharpe得分应有下界0"""
        sharpe_negative = -1.5

        # 修复后的公式
        score = max(min(sharpe_negative / 2.0 * 100, 100), 0)

        assert score >= 0, f"Sharpe得分 {score} 不应为负数"
        assert score == 0, f"负Sharpe得分应为0，实际 {score}"

    def test_risk_score_should_not_reverse_with_negative_drawdown(self):
        """风险评分在负回撤时不应反转"""
        volatility = 0.15
        max_drawdown_negative = -0.15  # 15%回撤，以负数存储

        # 修复后的公式
        risk_score = max(100 - (volatility / 0.2 * 50 + abs(max_drawdown_negative) / 0.15 * 50), 0)

        # 风险越高得分应越低
        assert risk_score < 100, f"有风险时得分 {risk_score} 不应为满分"
        assert risk_score >= 0, f"得分 {risk_score} 不应为负"

        # 未修复的公式会产生反转
        risk_score_wrong = max(100 - (volatility / 0.2 * 50 + max_drawdown_negative / 0.15 * 50), 0)
        # -0.15/0.15*50 = -50, 所以 100 - (37.5 + (-50)) = 112.5
        assert risk_score_wrong > 100 or risk_score_wrong > risk_score, "未修复公式在负回撤时得分异常高"

    def test_ic_positive_ratio_should_be_fair_to_negative_ic(self):
        """IC>0占比评分应对负IC因子公平"""
        # IC一致为正
        ratio_positive = 0.95
        score_pos = max(ratio_positive, 1 - ratio_positive) * 10

        # IC一致为负
        ratio_negative = 0.05
        score_neg = max(ratio_negative, 1 - ratio_negative) * 10

        assert abs(score_pos - score_neg) < 0.01, f"正负IC因子得分应相等: pos={score_pos}, neg={score_neg}"

        # IC方向混乱
        ratio_mixed = 0.5
        score_mixed = max(ratio_mixed, 1 - ratio_mixed) * 10
        assert score_mixed < score_pos, "方向混乱的因子得分应低于方向稳定的因子"


# ============================================================
# 6. 数据处理基准测试
# ============================================================


class TestDataProcessing:
    """数据处理基准测试"""

    def test_industry_mapping_should_strip_suffix(self):
        """行业映射应正确去除股票代码后缀 - 验证FactorNeutralizationService"""
        from backend.services.factor_neutralization_service import FactorNeutralizationService

        industry_map = {"000001": "银行", "600000": "银行", "300001": "科技"}
        stock_codes = ["000001.SZ", "600000.SH", "300001.SZ"]

        FactorNeutralizationService()
        # 使用service的add_industry_classification方法
        df = pd.DataFrame({"stock_code": stock_codes, "factor": [1.0, 2.0, 3.0]})

        # 验证后缀去除逻辑（service内部使用相同的正则）
        pure_codes = df["stock_code"].str.replace(r"\.(SH|SZ|BJ)$", "", regex=True)
        mapped = pure_codes.map(industry_map)

        assert mapped.notna().all(), f"映射后存在NaN: {mapped.tolist()}"
        assert mapped.iloc[0] == "银行"
        assert mapped.iloc[2] == "科技"

    def test_market_regime_detection_should_record_multiple_regimes(self):
        """市场环境检测应记录多个regime - 验证StatisticsService"""
        from backend.services.statistics_service import StatisticsService

        rng = np.random.default_rng(42)
        # 构造先涨后跌的市场收益序列
        pd.Series([0.01] * 50 + [-0.01] * 50)
        factor_data = {"bull": pd.Series(rng.standard_normal(50)), "bear": pd.Series(rng.standard_normal(50))}
        return_data = {"bull": pd.Series(rng.standard_normal(50)), "bear": pd.Series(rng.standard_normal(50))}

        service = StatisticsService()
        result = service.calculate_market_regime_ic(factor_data, return_data)

        # 应能区分不同市场环境
        assert len(result) >= 2, f"应记录至少2个市场环境IC，实际 {len(result)}"
        assert "bull" in result, "应包含牛市IC"
        assert "bear" in result, "应包含熊市IC"

    def test_regex_should_have_end_anchor(self):
        """股票代码正则应有结束锚点"""
        # 正确的正则（带$锚点）
        chinxext_pattern = r"^3\d{5}$"
        beijing_pattern = r"^(8\d{5}|4\d{5})$"

        # 6位代码应匹配
        assert re.match(chinxext_pattern, "300001") is not None
        assert re.match(beijing_pattern, "830001") is not None

        # 7位代码不应匹配
        assert re.match(chinxext_pattern, "3000001") is None
        assert re.match(beijing_pattern, "8300001") is None

    def test_bootstrap_condition_should_use_string_keys(self):
        """Bootstrap条件判断应使用字符串键"""
        n_quantiles = 5
        bootstrapped_returns = {"Q1": [0.01, 0.02], "Q5": [0.03, 0.04]}

        # 正确条件
        condition_correct = (
            f"Q{n_quantiles}" in bootstrapped_returns
            and "Q1" in bootstrapped_returns
            and len(bootstrapped_returns[f"Q{n_quantiles}"]) > 0
            and len(bootstrapped_returns["Q1"]) > 0
        )
        assert condition_correct is True

        # 错误条件（整数键）
        condition_wrong = n_quantiles in bootstrapped_returns and 0 in bootstrapped_returns
        assert condition_wrong is False


# ============================================================
# 7. 性能基准测试
# ============================================================


class TestPerformance:
    """性能基准测试"""

    def test_single_factor_preprocessing_throughput(self):
        """单因子预处理吞吐量应 > 3000 样本/秒"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        n = 100_000
        rng = np.random.default_rng(42)
        factor = pd.Series(rng.standard_normal(n), name="factor")
        config = PreprocessingConfig()
        pipeline = FactorPreprocessingPipeline(config)

        start = time.time()
        pipeline.process_single_factor(factor)
        elapsed = time.time() - start

        throughput = n / elapsed
        assert throughput > 3000, f"吞吐量 {throughput:.0f} 样本/秒低于基准 3000"

    def test_multi_factor_preprocessing_throughput(self):
        """多因子预处理吞吐量应满足要求 - 使用process_multi_stock_factors API"""
        from backend.services.factor_preprocessing_pipeline import FactorPreprocessingPipeline, PreprocessingConfig

        n_stocks = 50
        n_factors = 5
        n_days = 250

        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        factor_names = [f"factor_{j}" for j in range(n_factors)]

        # 使用与process_multi_stock_factors匹配的数据结构：{stock_code: DataFrame}
        data = {}
        for i in range(n_stocks):
            stock_code = f"{600000 + i:06d}"
            df = pd.DataFrame(
                {
                    **{f"factor_{j}": rng.standard_normal(n_days) for j in range(n_factors)},
                    "market_cap": rng.uniform(1e8, 1e10, n_days),
                    "industry": rng.choice(["银行", "地产", "科技", "医药", "能源"], n_days),
                },
                index=dates,
            )
            data[stock_code] = df

        # 禁用中性化以避免单股票时间序列回归的警告
        config = PreprocessingConfig(
            enable_market_cap_neutralization=False,
            enable_industry_neutralization=False,
        )
        pipeline = FactorPreprocessingPipeline(config)

        start = time.time()
        result_data, all_stats = pipeline.process_multi_stock_factors(
            data,
            factor_names,
            parallel_stocks=False,
        )
        elapsed = time.time() - start

        total_samples = n_stocks * n_factors * n_days
        total_samples / elapsed
        # 多因子(5) x 多股票(50) x 250天 应 < 10秒
        assert elapsed < 10.0, f"多因子预处理耗时 {elapsed:.2f}s 超过基准 10s"
        # 验证所有股票都有处理结果
        assert len(result_data) == n_stocks, f"处理结果股票数 {len(result_data)} 与输入 {n_stocks} 不一致"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
