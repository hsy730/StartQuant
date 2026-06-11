"""
portfolio_analysis_service.py 组合分析服务测试

覆盖所有公开方法及关键私有方法：
- calculate_industry_exposure: 行业暴露度
- calculate_factor_exposure: 因子暴露度
- calculate_concentration: 集中度指标
- _calculate_gini: 基尼系数
- calculate_risk_metrics: 风险指标（委托 risk_metrics 统一入口）
- optimize_weights: 权重优化（5种方法）
- calculate_combined_factor_score: 综合因子得分
- compare_weight_methods: 多方法比较
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

from backend.services.portfolio_analysis_service import PortfolioAnalysisService  # noqa: E402

# ============================================================
# 测试辅助：构造真实数据
# ============================================================


def _make_positions(n=20, with_industry=True, with_stock_code=True):
    """构造持仓 DataFrame"""
    data = {}
    if with_stock_code:
        data["stock_code"] = [f"600{i:03d}" for i in range(n)]
    if with_industry:
        industries = ["银行", "地产", "科技", "消费", "医药"]
        data["industry"] = [industries[i % len(industries)] for i in range(n)]
    # 权重：递减分布，模拟真实持仓
    raw_weights = np.array([1.0 / (i + 1) for i in range(n)])
    data["weight"] = raw_weights / raw_weights.sum()
    return pd.DataFrame(data)


def _make_factor_returns(n_factors=3, n_periods=120):
    """构造因子收益率 DataFrame"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="B")
    factor_names = [f"factor_{i}" for i in range(n_factors)]
    data = np.random.randn(n_periods, n_factors) * 0.01
    return pd.DataFrame(data, index=dates, columns=factor_names)


def _make_returns(n=252):
    """构造日收益率序列"""
    np.random.seed(42)
    return pd.Series(np.random.randn(n) * 0.01 + 0.0005)


# ============================================================
# calculate_industry_exposure 测试
# ============================================================


class TestCalculateIndustryExposure:
    """行业暴露度计算测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()

    def test_normal_positions_should_return_industry_exposure(self):
        """正常持仓应返回行业暴露度字典"""
        positions = _make_positions(n=20, with_industry=True)
        result = self.service.calculate_industry_exposure(positions)

        assert "industry_exposure" in result
        assert "max_exposure" in result
        assert "min_exposure" in result
        assert "concentration" in result
        assert "top3_concentration" in result

    def test_industry_exposure_should_sum_to_one(self):
        """行业暴露度之和应约等于1"""
        positions = _make_positions(n=20, with_industry=True)
        result = self.service.calculate_industry_exposure(positions)
        total = sum(result["industry_exposure"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_max_exposure_should_be_largest_industry(self):
        """max_exposure 应等于最大行业暴露度"""
        positions = _make_positions(n=20, with_industry=True)
        result = self.service.calculate_industry_exposure(positions)
        max_val = max(result["industry_exposure"].values())
        assert result["max_exposure"] == pytest.approx(max_val)

    def test_min_exposure_should_be_smallest_industry(self):
        """min_exposure 应等于最小行业暴露度"""
        positions = _make_positions(n=20, with_industry=True)
        result = self.service.calculate_industry_exposure(positions)
        min_val = min(result["industry_exposure"].values())
        assert result["min_exposure"] == pytest.approx(min_val)

    def test_top3_concentration_should_be_sum_of_top3(self):
        """top3_concentration 应等于前3大行业暴露度之和"""
        positions = _make_positions(n=20, with_industry=True)
        result = self.service.calculate_industry_exposure(positions)
        top3 = sorted(result["industry_exposure"].values(), reverse=True)[:3]
        assert result["top3_concentration"] == pytest.approx(sum(top3))

    def test_missing_industry_column_should_return_error(self):
        """缺少行业列应返回错误字典"""
        positions = _make_positions(n=10, with_industry=False)
        result = self.service.calculate_industry_exposure(positions)
        assert "error" in result
        assert "industry" in result["error"]

    def test_missing_weight_column_should_return_error(self):
        """缺少权重列应返回错误字典"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000", "600001"],
                "industry": ["银行", "科技"],
            }
        )
        result = self.service.calculate_industry_exposure(positions, weight_column="missing_weight")
        assert "error" in result
        assert "missing_weight" in result["error"]

    def test_single_industry_should_have_exposure_one(self):
        """单一行业应暴露度为1"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000", "600001", "600002"],
                "industry": ["银行", "银行", "银行"],
                "weight": [0.4, 0.35, 0.25],
            }
        )
        result = self.service.calculate_industry_exposure(positions)
        assert result["industry_exposure"]["银行"] == pytest.approx(1.0)
        assert result["max_exposure"] == pytest.approx(1.0)
        assert result["min_exposure"] == pytest.approx(1.0)

    def test_zero_weights_should_not_crash(self):
        """全零权重不应崩溃"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000", "600001"],
                "industry": ["银行", "科技"],
                "weight": [0.0, 0.0],
            }
        )
        result = self.service.calculate_industry_exposure(positions)
        # 全零权重时 total_weight=0，走 else 分支
        assert isinstance(result, dict)

    def test_custom_column_names_should_work(self):
        """自定义列名应正常工作"""
        positions = pd.DataFrame(
            {
                "code": ["600000", "600001"],
                "sector": ["银行", "科技"],
                "pct": [0.6, 0.4],
            }
        )
        result = self.service.calculate_industry_exposure(positions, industry_column="sector", weight_column="pct")
        assert "industry_exposure" in result
        assert result["industry_exposure"]["银行"] == pytest.approx(0.6)

    def test_same_industry_multiple_stocks_should_aggregate(self):
        """同一行业多只股票应汇总权重"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000", "600001", "600002"],
                "industry": ["银行", "银行", "科技"],
                "weight": [0.3, 0.2, 0.5],
            }
        )
        result = self.service.calculate_industry_exposure(positions)
        assert result["industry_exposure"]["银行"] == pytest.approx(0.5)
        assert result["industry_exposure"]["科技"] == pytest.approx(0.5)


# ============================================================
# calculate_factor_exposure 测试
# ============================================================


class TestCalculateFactorExposure:
    """因子暴露度计算测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()

    def test_normal_factor_data_should_return_exposures(self):
        """正常因子数据应返回因子暴露度"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000", "600001", "600002"],
                "weight": [0.4, 0.35, 0.25],
            }
        )
        factor_data = {
            "momentum": pd.Series([0.5, 0.3, 0.2], index=["600000", "600001", "600002"]),
            "value": pd.Series([0.1, 0.4, 0.3], index=["600000", "600001", "600002"]),
        }
        result = self.service.calculate_factor_exposure(positions, factor_data)

        assert "factor_exposures" in result
        assert "max_exposure" in result
        assert "momentum" in result["factor_exposures"]
        assert "value" in result["factor_exposures"]

    def test_weighted_average_should_be_correct(self):
        """加权平均因子暴露度应正确计算"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [0.5, 0.5],
            }
        )
        factor_data = {
            "factor_1": pd.Series([1.0, 3.0], index=["A", "B"]),
        }
        result = self.service.calculate_factor_exposure(positions, factor_data)
        # 加权平均 = 0.5*1.0 + 0.5*3.0 = 2.0
        assert result["factor_exposures"]["factor_1"] == pytest.approx(2.0)

    def test_missing_weight_column_should_return_error(self):
        """缺少权重列应返回错误"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000", "600001"],
            }
        )
        factor_data = {"factor_1": pd.Series([0.5, 0.3], index=["600000", "600001"])}
        result = self.service.calculate_factor_exposure(positions, factor_data)
        assert "error" in result

    def test_partial_stock_overlap_should_use_valid_only(self):
        """部分股票不匹配时，应仅使用有效数据"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B", "C"],
                "weight": [0.4, 0.3, 0.3],
            }
        )
        # 因子数据只有 A 和 B，缺少 C
        factor_data = {
            "factor_1": pd.Series([1.0, 2.0], index=["A", "B"]),
        }
        result = self.service.calculate_factor_exposure(positions, factor_data)
        # C 的因子值为 NaN，被过滤；有效权重 A=0.4, B=0.3, weight_sum=0.7
        # 加权平均 = (0.4*1.0 + 0.3*2.0) / 0.7 = 1.0/0.7 ≈ 1.4286
        assert result["factor_exposures"]["factor_1"] == pytest.approx(1.0 / 0.7, abs=1e-4)

    def test_no_valid_data_should_return_zero(self):
        """无有效数据时因子暴露度应为0"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [0.5, 0.5],
            }
        )
        # 因子数据索引完全不匹配
        factor_data = {
            "factor_1": pd.Series([1.0, 2.0], index=["X", "Y"]),
        }
        result = self.service.calculate_factor_exposure(positions, factor_data)
        assert result["factor_exposures"]["factor_1"] == 0.0

    def test_scalar_factor_value_should_return_scalar(self):
        """标量因子值应直接返回标量"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [0.5, 0.5],
            }
        )
        factor_data = {
            "factor_1": 0.75,  # 标量
        }
        result = self.service.calculate_factor_exposure(positions, factor_data)
        assert result["factor_exposures"]["factor_1"] == pytest.approx(0.75)

    def test_max_exposure_should_be_max_absolute_value(self):
        """max_exposure 应为最大绝对因子暴露度"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [0.5, 0.5],
            }
        )
        factor_data = {
            "f1": pd.Series([1.0, 1.0], index=["A", "B"]),
            "f2": pd.Series([-3.0, -3.0], index=["A", "B"]),
        }
        result = self.service.calculate_factor_exposure(positions, factor_data)
        assert result["max_exposure"] == pytest.approx(3.0)

    def test_empty_factor_data_should_return_zero_max(self):
        """空因子数据应返回 max_exposure=0"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [0.5, 0.5],
            }
        )
        result = self.service.calculate_factor_exposure(positions, {})
        assert result["max_exposure"] == 0.0


# ============================================================
# calculate_concentration 测试
# ============================================================


class TestCalculateConcentration:
    """组合集中度计算测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()

    def test_normal_positions_should_return_all_metrics(self):
        """正常持仓应返回所有集中度指标"""
        positions = _make_positions(n=20)
        result = self.service.calculate_concentration(positions)

        assert "top10_concentration" in result
        assert "herfindahl_index" in result
        assert "gini_coefficient" in result

    def test_equal_weights_should_have_low_concentration(self):
        """等权持仓应有较低的集中度"""
        n = 20
        positions = pd.DataFrame(
            {
                "stock_code": [f"S{i}" for i in range(n)],
                "weight": [1.0 / n] * n,
            }
        )
        result = self.service.calculate_concentration(positions)

        # 等权时 HHI = n * (1/n)^2 = 1/n
        assert result["herfindahl_index"] == pytest.approx(1.0 / n, abs=1e-6)
        # 等权时基尼系数应为0
        assert result["gini_coefficient"] == pytest.approx(0.0, abs=1e-6)

    def test_single_stock_should_have_max_concentration(self):
        """单只股票持仓应有最大集中度"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A"],
                "weight": [1.0],
            }
        )
        result = self.service.calculate_concentration(positions)

        assert result["top10_concentration"] == pytest.approx(1.0)
        assert result["herfindahl_index"] == pytest.approx(1.0)
        assert result["gini_coefficient"] == pytest.approx(0.0)  # 单元素基尼为0

    def test_missing_weight_column_should_return_error(self):
        """缺少权重列应返回错误"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
            }
        )
        result = self.service.calculate_concentration(positions)
        assert "error" in result

    def test_empty_weights_should_return_zeros(self):
        """空权重应返回全零"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [np.nan, np.nan],
            }
        )
        result = self.service.calculate_concentration(positions)
        assert result["top10_concentration"] == 0.0
        assert result["herfindahl_index"] == 0.0
        assert result["gini_coefficient"] == 0.0

    def test_zero_weights_should_return_zeros(self):
        """全零权重应返回全零"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [0.0, 0.0],
            }
        )
        result = self.service.calculate_concentration(positions)
        assert result["top10_concentration"] == 0.0
        assert result["herfindahl_index"] == 0.0
        assert result["gini_coefficient"] == 0.0

    def test_top10_concentration_should_be_correct(self):
        """前十大持仓占比应正确计算"""
        n = 15
        weights = np.array([1.0 / (i + 1) for i in range(n)])
        weights = weights / weights.sum()
        positions = pd.DataFrame(
            {
                "stock_code": [f"S{i}" for i in range(n)],
                "weight": weights,
            }
        )
        result = self.service.calculate_concentration(positions)

        # 手动计算 top10
        sorted_weights = np.sort(np.abs(weights))[::-1]
        expected_top10 = sorted_weights[:10].sum() / np.abs(weights).sum()
        assert result["top10_concentration"] == pytest.approx(expected_top10, abs=1e-6)

    def test_herfindahl_index_range(self):
        """HHI 应在 [1/n, 1] 范围内"""
        n = 10
        positions = pd.DataFrame(
            {
                "stock_code": [f"S{i}" for i in range(n)],
                "weight": [1.0 / n] * n,
            }
        )
        result = self.service.calculate_concentration(positions)
        assert result["herfindahl_index"] >= 1.0 / n - 1e-6
        assert result["herfindahl_index"] <= 1.0 + 1e-6

    def test_concentrated_portfolio_should_have_higher_hhi(self):
        """集中持仓应有更高的HHI"""
        n = 10
        # 集中持仓
        concentrated = pd.DataFrame(
            {
                "stock_code": [f"S{i}" for i in range(n)],
                "weight": [0.5] + [0.5 / (n - 1)] * (n - 1),
            }
        )
        # 分散持仓
        dispersed = pd.DataFrame(
            {
                "stock_code": [f"S{i}" for i in range(n)],
                "weight": [1.0 / n] * n,
            }
        )
        result_c = self.service.calculate_concentration(concentrated)
        result_d = self.service.calculate_concentration(dispersed)
        assert result_c["herfindahl_index"] > result_d["herfindahl_index"]

    def test_negative_weights_should_use_absolute_values(self):
        """负权重应取绝对值计算"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B"],
                "weight": [-0.6, -0.4],
            }
        )
        result = self.service.calculate_concentration(positions)
        assert result["top10_concentration"] == pytest.approx(1.0)


# ============================================================
# _calculate_gini 测试
# ============================================================


class TestCalculateGini:
    """基尼系数计算测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()

    def test_equal_values_should_have_zero_gini(self):
        """等值数组基尼系数应为0"""
        values = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = self.service._calculate_gini(values)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_single_value_should_have_zero_gini(self):
        """单值数组基尼系数应为0"""
        values = np.array([1.0])
        result = self.service._calculate_gini(values)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_all_zero_should_return_zero(self):
        """全零数组应返回0"""
        values = np.array([0.0, 0.0, 0.0])
        result = self.service._calculate_gini(values)
        assert result == 0.0

    def test_extreme_inequality_should_have_high_gini(self):
        """极端不平等应有高基尼系数"""
        # 一个值占绝大部分
        values = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        result = self.service._calculate_gini(values)
        assert result > 0.5

    def test_gini_should_be_between_zero_and_one(self):
        """基尼系数应在 [0, 1] 范围内"""
        np.random.seed(42)
        for _ in range(10):
            values = np.random.rand(20)
            values = values / values.sum()
            result = self.service._calculate_gini(values)
            assert 0.0 <= result <= 1.0

    def test_two_values_unequal(self):
        """两个不等值的基尼系数应正确"""
        values = np.array([0.25, 0.75])
        result = self.service._calculate_gini(values)
        # 公式: (n+1 - 2*sum(cumsum)/cumsum[-1]) / n
        # sorted: [0.25, 0.75], cumsum: [0.25, 1.0]
        # (2+1 - 2*(0.25+1.0)/1.0) / 2 = (3 - 2*1.25) / 2 = (3-2.5)/2 = 0.25
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_gini_formula_consistency(self):
        """基尼系数公式一致性验证"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.service._calculate_gini(values)
        # sorted: [1,2,3,4,5], cumsum: [1,3,6,10,15]
        # n=5, (5+1 - 2*(1+3+6+10+15)/15) / 5 = (6 - 2*35/15) / 5
        # = (6 - 70/15) / 5 = (90/15 - 70/15) / 5 = (20/15) / 5 = 4/15 ≈ 0.2667
        assert result == pytest.approx(4.0 / 15.0, abs=1e-4)


# ============================================================
# calculate_risk_metrics 测试
# ============================================================


class TestCalculateRiskMetrics:
    """风险指标计算测试（委托 risk_metrics 统一入口）"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()

    def test_normal_returns_should_return_metrics(self):
        """正常收益率应返回风险指标"""
        returns = _make_returns(252)
        result = self.service.calculate_risk_metrics(returns)

        expected_keys = [
            "total_return",
            "annual_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "var_95",
            "cvar_95",
        ]
        for key in expected_keys:
            assert key in result

    def test_empty_returns_should_return_empty_metrics(self):
        """空收益率应返回空指标"""
        returns = pd.Series([], dtype=float)
        result = self.service.calculate_risk_metrics(returns)
        # _empty_metrics 所有值为 None
        for key, val in result.items():
            assert val is None, f"{key} 应为 None"

    def test_all_nan_returns_should_return_empty_metrics(self):
        """全NaN收益率应返回空指标"""
        returns = pd.Series([np.nan] * 50)
        result = self.service.calculate_risk_metrics(returns)
        for key, val in result.items():
            assert val is None

    def test_with_benchmark_should_include_relative_metrics(self):
        """有基准收益率时应包含相对指标"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
        benchmark = pd.Series(np.random.randn(252) * 0.008 + 0.0003)
        result = self.service.calculate_risk_metrics(returns, benchmark_returns=benchmark)

        assert "tracking_error" in result
        assert "beta" in result

    def test_without_benchmark_should_not_include_relative_metrics(self):
        """无基准收益率时不应包含相对指标"""
        returns = _make_returns(252)
        result = self.service.calculate_risk_metrics(returns)
        assert "tracking_error" not in result
        assert "beta" not in result

    def test_max_drawdown_should_be_non_positive(self):
        """最大回撤应 <= 0"""
        returns = _make_returns(252)
        result = self.service.calculate_risk_metrics(returns)
        assert result["max_drawdown"] <= 0


# ============================================================
# optimize_weights 测试
# ============================================================


class TestOptimizeWeights:
    """权重优化测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()
        self.factor_returns = _make_factor_returns(n_factors=3, n_periods=120)

    def test_equal_weight_should_return_uniform_weights(self):
        """等权重方法应返回均匀权重"""
        result = self.service.optimize_weights(self.factor_returns, method="equal_weight")

        assert "weights" in result
        assert "method" in result
        assert result["method"] == "equal_weight"
        n = len(self.factor_returns.columns)
        for w in result["weights"].values():
            assert w == pytest.approx(1.0 / n, abs=1e-6)

    def test_equal_weight_should_have_expected_metrics(self):
        """等权重应返回期望收益、波动率和夏普"""
        result = self.service.optimize_weights(self.factor_returns, method="equal_weight")

        assert "expected_return" in result
        assert "expected_volatility" in result
        assert "sharpe_ratio" in result
        assert isinstance(result["expected_return"], float)
        assert isinstance(result["expected_volatility"], float)
        assert isinstance(result["sharpe_ratio"], float)

    def test_ic_weight_should_assign_higher_weight_to_better_factors(self):
        """IC加权应给表现更好的因子更高权重"""
        # 构造因子值和收益率数据，使good_factor与收益有更高相关性
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        # good_factor: 与收益率高度相关
        returns_data = np.random.randn(120) * 0.01
        good_factor_vals = returns_data * 5 + np.random.randn(120) * 0.01  # 高相关
        bad_factor_vals = np.random.randn(120) * 0.01  # 无相关

        factor_returns = pd.DataFrame(
            {
                "good_factor": returns_data,
                "bad_factor": returns_data,
            },
            index=dates,
        )
        factor_values = pd.DataFrame(
            {
                "good_factor": good_factor_vals,
                "bad_factor": bad_factor_vals,
            },
            index=dates,
        )

        result = self.service.optimize_weights(
            factor_returns,
            method="ic_weight",
            factor_values=factor_values,
            stock_returns=pd.Series(returns_data, index=dates),
        )
        weights = result["weights"]
        # good_factor 的 IC 应更高，权重应更大
        assert weights["good_factor"] > weights["bad_factor"]

    def test_ic_weight_all_negative_ir_should_fallback_to_equal(self):
        """IC加权所有IR为负时应回退到等权重"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        factor_returns = pd.DataFrame(
            {
                "f1": np.random.randn(120) * 0.01 - 0.01,
                "f2": np.random.randn(120) * 0.01 - 0.02,
            },
            index=dates,
        )

        result = self.service.optimize_weights(factor_returns, method="ic_weight")
        n = len(factor_returns.columns)
        for w in result["weights"].values():
            assert w == pytest.approx(1.0 / n, abs=1e-6)

    def test_risk_parity_should_assign_inverse_vol_weights(self):
        """风险平价应给低波动因子更高权重"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        factor_returns = pd.DataFrame(
            {
                "low_vol": np.random.randn(120) * 0.005,  # 低波动
                "high_vol": np.random.randn(120) * 0.03,  # 高波动
            },
            index=dates,
        )

        result = self.service.optimize_weights(factor_returns, method="risk_parity")
        weights = result["weights"]
        # 低波动因子应获得更高权重
        assert weights["low_vol"] > weights["high_vol"]

    def test_risk_parity_zero_vol_should_fallback_to_equal(self):
        """风险平价零波动率应回退到等权重"""
        # 构造 std 精确为 0 的数据（单行数据，std=0）
        factor_returns = pd.DataFrame(
            {
                "f1": [0.01],
                "f2": [0.02],
            }
        )

        result = self.service.optimize_weights(factor_returns, method="risk_parity")
        n = len(factor_returns.columns)
        # 单行数据 std=0，replace(0, np.nan) 后 isna().all()=True，回退到等权
        for w in result["weights"].values():
            assert w == pytest.approx(1.0 / n, abs=1e-6)

    def test_max_sharpe_should_return_valid_weights(self):
        """最大夏普应返回有效权重"""
        result = self.service.optimize_weights(self.factor_returns, method="max_sharpe")

        assert "weights" in result
        weights = result["weights"]
        # 权重和应约等于1
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.05)
        # 所有权重非负
        for w in weights.values():
            assert w >= -0.01  # 允许微小数值误差

    def test_min_variance_should_return_valid_weights(self):
        """最小方差应返回有效权重"""
        result = self.service.optimize_weights(self.factor_returns, method="min_variance")

        assert "weights" in result
        weights = result["weights"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.05)

    def test_max_sharpe_single_factor_should_return_weight_one(self):
        """最大夏普单因子应返回权重1"""
        single_factor = self.factor_returns[["factor_0"]]
        result = self.service.optimize_weights(single_factor, method="max_sharpe")

        assert result["weights"]["factor_0"] == pytest.approx(1.0, abs=1e-6)
        assert result.get("optimization_status") == "skipped: only one factor"

    def test_min_variance_single_factor_should_return_weight_one(self):
        """最小方差单因子应返回权重1"""
        single_factor = self.factor_returns[["factor_0"]]
        result = self.service.optimize_weights(single_factor, method="min_variance")

        assert result["weights"]["factor_0"] == pytest.approx(1.0, abs=1e-6)

    def test_max_sharpe_small_data_should_fallback_to_equal(self):
        """最大夏普小数据集失败时应回退到等权重"""
        # 极少数据可能导致 pypfopt 失败
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        tiny_returns = pd.DataFrame(
            {
                "f1": [0.01, -0.01, 0.005],
                "f2": [0.02, -0.005, 0.01],
            },
            index=dates,
        )

        result = self.service.optimize_weights(tiny_returns, method="max_sharpe")
        # 不应崩溃，可能回退到等权重
        assert "weights" in result
        n = len(tiny_returns.columns)
        if result.get("optimization_status", "").startswith("fallback"):
            for w in result["weights"].values():
                assert w == pytest.approx(1.0 / n, abs=1e-6)

    def test_empty_factor_returns_should_return_error(self):
        """空因子收益率应返回错误"""
        empty_returns = pd.DataFrame()
        result = self.service.optimize_weights(empty_returns, method="equal_weight")

        assert "error" in result
        assert result["weights"] == {}

    def test_unsupported_method_should_return_error(self):
        """不支持的方法应返回错误"""
        result = self.service.optimize_weights(self.factor_returns, method="invalid_method")

        assert "error" in result
        assert "不支持" in result["error"]

    def test_weights_should_be_normalized(self):
        """权重应归一化（和为1）"""
        result = self.service.optimize_weights(self.factor_returns, method="ic_weight")
        weight_sum = sum(result["weights"].values())
        assert weight_sum == pytest.approx(1.0, abs=1e-6)

    def test_inf_in_factor_returns_should_be_handled(self):
        """因子收益率含 Inf 应被处理"""
        factor_returns = self.factor_returns.copy()
        factor_returns.iloc[0, 0] = np.inf
        factor_returns.iloc[1, 1] = -np.inf

        result = self.service.optimize_weights(factor_returns, method="equal_weight")
        assert "weights" in result
        # 不应崩溃

    def test_nan_in_factor_returns_should_be_handled(self):
        """因子收益率含 NaN 应被处理"""
        factor_returns = self.factor_returns.copy()
        factor_returns.iloc[0, 0] = np.nan

        result = self.service.optimize_weights(factor_returns, method="equal_weight")
        assert "weights" in result


# ============================================================
# calculate_combined_factor_score 测试
# ============================================================


class TestCalculateCombinedFactorScore:
    """综合因子得分计算测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()

    def test_normal_factors_should_return_combined_score(self):
        """正常因子数据应返回综合得分"""
        index = ["A", "B", "C"]
        factor_data = {
            "momentum": pd.Series([1.0, 2.0, 3.0], index=index),
            "value": pd.Series([3.0, 2.0, 1.0], index=index),
        }
        weights = {"momentum": 0.6, "value": 0.4}

        result = self.service.calculate_combined_factor_score(factor_data, weights)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_weighted_combination_should_be_correct(self):
        """加权组合应正确计算"""
        index = ["A", "B"]
        factor_data = {
            "f1": pd.Series([1.0, 1.0], index=index),
            "f2": pd.Series([2.0, 2.0], index=index),
        }
        weights = {"f1": 0.5, "f2": 0.5}

        result = self.service.calculate_combined_factor_score(factor_data, weights, normalize=False)
        # 不标准化时: 0.5*1.0 + 0.5*2.0 = 1.5
        assert result.iloc[0] == pytest.approx(1.5, abs=1e-6)
        assert result.iloc[1] == pytest.approx(1.5, abs=1e-6)

    def test_normalize_true_should_zscore_factors(self):
        """normalize=True 应对因子做 Z-score 标准化"""
        index = ["A", "B", "C"]
        factor_data = {
            "f1": pd.Series([10.0, 20.0, 30.0], index=index),
        }
        weights = {"f1": 1.0}

        result = self.service.calculate_combined_factor_score(factor_data, weights, normalize=True)
        # 标准化后均值为0
        assert result.mean() == pytest.approx(0.0, abs=1e-6)

    def test_missing_weight_factor_should_be_ignored(self):
        """权重中不存在的因子应被忽略"""
        index = ["A", "B"]
        factor_data = {
            "f1": pd.Series([1.0, 2.0], index=index),
        }
        weights = {"f1": 0.5, "f_nonexistent": 0.5}

        result = self.service.calculate_combined_factor_score(factor_data, weights, normalize=False)
        # 只有 f1 贡献
        assert result.iloc[0] == pytest.approx(0.5 * 1.0, abs=1e-6)

    def test_empty_factor_data_should_return_empty_series(self):
        """空因子数据应返回空 Series"""
        result = self.service.calculate_combined_factor_score({}, {"f1": 1.0})
        assert len(result) == 0

    def test_nan_in_result_should_be_preserved(self):
        """结果中的 NaN 应被保留（Z-score空间中0.0有语义，不应填充）"""
        index = ["A", "B", "C"]
        factor_data = {
            "f1": pd.Series([1.0, np.nan, 3.0], index=index),
        }
        weights = {"f1": 1.0}

        result = self.service.calculate_combined_factor_score(factor_data, weights, normalize=False)
        # NaN 应被保留，Inf 应被替换为 NaN
        assert result.isna().sum() >= 1  # B位置应有NaN

    def test_inf_in_result_should_be_replaced(self):
        """结果中的 Inf 应被替换为 0"""
        index = ["A"]
        factor_data = {
            "f1": pd.Series([np.inf], index=index),
        }
        weights = {"f1": 1.0}

        result = self.service.calculate_combined_factor_score(factor_data, weights, normalize=False)
        assert not np.isinf(result).any()

    def test_different_indices_should_use_intersection(self):
        """不同索引的因子应取交集"""
        factor_data = {
            "f1": pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"]),
            "f2": pd.Series([4.0, 5.0], index=["B", "C"]),
        }
        weights = {"f1": 0.5, "f2": 0.5}

        result = self.service.calculate_combined_factor_score(factor_data, weights, normalize=False)
        # 交集为 ["B", "C"]
        assert len(result) == 2
        assert set(result.index) == {"B", "C"}


# ============================================================
# compare_weight_methods 测试
# ============================================================


class TestCompareWeightMethods:
    """多方法比较测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()
        self.factor_returns = _make_factor_returns(n_factors=3, n_periods=120)

    def test_default_methods_should_compare_four_methods(self):
        """默认应比较4种方法"""
        result = self.service.compare_weight_methods(self.factor_returns)

        assert "equal_weight" in result
        assert "ic_weight" in result
        assert "risk_parity" in result
        assert "max_sharpe" in result

    def test_custom_methods_should_compare_specified_only(self):
        """自定义方法列表应只比较指定方法"""
        result = self.service.compare_weight_methods(self.factor_returns, methods=["equal_weight", "ic_weight"])

        assert "equal_weight" in result
        assert "ic_weight" in result
        assert "risk_parity" not in result

    def test_each_method_result_should_have_required_keys(self):
        """每种方法结果应包含必要键"""
        result = self.service.compare_weight_methods(self.factor_returns, methods=["equal_weight"])

        method_result = result["equal_weight"]
        assert "annual_return" in method_result
        assert "volatility" in method_result
        assert "sharpe_ratio" in method_result

    def test_error_method_should_be_skipped(self):
        """出错的方法应被跳过不出现在结果中"""
        result = self.service.compare_weight_methods(self.factor_returns, methods=["invalid_method"])
        assert "invalid_method" not in result

    def test_min_variance_method_should_work(self):
        """min_variance 方法应正常工作"""
        result = self.service.compare_weight_methods(self.factor_returns, methods=["min_variance"])
        assert "min_variance" in result

    def test_empty_returns_should_return_empty_dict(self):
        """空因子收益率应返回空字典"""
        empty_returns = pd.DataFrame()
        result = self.service.compare_weight_methods(empty_returns)
        assert result == {}


# ============================================================
# 边界条件综合测试
# ============================================================


class TestEdgeCases:
    """边界条件综合测试"""

    def setup_method(self):
        self.service = PortfolioAnalysisService()

    def test_single_stock_industry_exposure(self):
        """单只股票的行业暴露度"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000"],
                "industry": ["银行"],
                "weight": [1.0],
            }
        )
        result = self.service.calculate_industry_exposure(positions)
        assert result["industry_exposure"]["银行"] == pytest.approx(1.0)

    def test_single_stock_concentration(self):
        """单只股票的集中度"""
        positions = pd.DataFrame(
            {
                "stock_code": ["600000"],
                "weight": [1.0],
            }
        )
        result = self.service.calculate_concentration(positions)
        assert result["top10_concentration"] == pytest.approx(1.0)
        assert result["herfindahl_index"] == pytest.approx(1.0)

    def test_large_number_of_stocks_concentration(self):
        """大量股票的集中度计算不应崩溃"""
        n = 500
        weights = np.random.dirichlet(np.ones(n))
        positions = pd.DataFrame(
            {
                "stock_code": [f"S{i}" for i in range(n)],
                "weight": weights,
            }
        )
        result = self.service.calculate_concentration(positions)
        assert 0.0 <= result["gini_coefficient"] <= 1.0
        assert 0.0 <= result["herfindahl_index"] <= 1.0

    def test_optimize_weights_with_many_factors(self):
        """多因子权重优化不应崩溃"""
        factor_returns = _make_factor_returns(n_factors=10, n_periods=120)
        result = self.service.optimize_weights(factor_returns, method="equal_weight")
        assert len(result["weights"]) == 10
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_combined_score_with_many_factors(self):
        """多因子综合得分不应崩溃"""
        n = 10
        index = [f"S{i}" for i in range(50)]
        factor_data = {f"f{i}": pd.Series(np.random.randn(50), index=index) for i in range(n)}
        weights = {f"f{i}": 1.0 / n for i in range(n)}

        result = self.service.calculate_combined_factor_score(factor_data, weights)
        assert len(result) == 50
        assert not result.isna().any()

    def test_industry_exposure_with_nan_weights(self):
        """行业暴露度含 NaN 权重"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B", "C"],
                "industry": ["银行", "科技", "银行"],
                "weight": [0.3, np.nan, 0.4],
            }
        )
        result = self.service.calculate_industry_exposure(positions)
        # NaN 权重在 groupby sum 时被跳过
        assert isinstance(result, dict)
        assert "industry_exposure" in result

    def test_factor_exposure_with_nan_factor_values(self):
        """因子暴露度含 NaN 因子值"""
        positions = pd.DataFrame(
            {
                "stock_code": ["A", "B", "C"],
                "weight": [0.4, 0.3, 0.3],
            }
        )
        factor_data = {
            "f1": pd.Series([1.0, np.nan, 3.0], index=["A", "B", "C"]),
        }
        result = self.service.calculate_factor_exposure(positions, factor_data)
        # NaN 因子值应被过滤
        assert "factor_exposures" in result
        assert isinstance(result["factor_exposures"]["f1"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
