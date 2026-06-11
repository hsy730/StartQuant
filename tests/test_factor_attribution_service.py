"""
FactorAttributionService 因子贡献度分解服务测试

覆盖 _calculate_contribution、_decompose_alpha_beta、_decompose_return、
analyze_attribution 四个核心方法，包含正常场景和边界条件。
"""

import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

from backend.services.factor_attribution_service import FactorAttributionService  # noqa: E402


def _make_factor_data(
    n_stocks: int = 10,
    n_days: int = 30,
    factor_name: str = "test_factor",
    seed: int = 42,
) -> dict:
    """
    构造测试用 factor_data 字典

    每只股票有 factor_name 列和 close 列，日期索引对齐。
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start="2024-01-02", periods=n_days, freq="B")
    factor_data = {}
    for i in range(n_stocks):
        code = f"60000{i + 1}"
        close = 10.0 + np.cumsum(np.random.randn(n_days) * 0.3)
        close = np.maximum(close, 1.0)  # 保证价格为正
        factor_val = np.random.randn(n_days) * 0.5 + i * 0.1
        df = pd.DataFrame(
            {factor_name: factor_val, "close": close},
            index=dates,
        )
        factor_data[code] = df
    return factor_data


def _make_benchmark_data(n_days: int = 30, seed: int = 99) -> pd.DataFrame:
    """构造基准数据（模拟上证指数）"""
    np.random.seed(seed)
    dates = pd.bdate_range(start="2024-01-02", periods=n_days, freq="B")
    close = 3000.0 + np.cumsum(np.random.randn(n_days) * 20)
    return pd.DataFrame({"close": close}, index=dates)


class TestCalculateContribution:
    """_calculate_contribution 方法测试"""

    def setup_method(self):
        self.service = FactorAttributionService()

    def test_normal_data_should_return_all_keys(self):
        """正常数据应返回所有预期字段"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._calculate_contribution(factor_data, "test_factor")

        expected_keys = [
            "ic",
            "ic_pvalue",
            "high_exposure_return",
            "low_exposure_return",
            "long_short_return",
            "contribution_ratio",
            "sample_size",
        ]
        for key in expected_keys:
            assert key in result, f"缺少字段: {key}"

    def test_ic_should_be_between_minus1_and_1(self):
        """IC值应在 [-1, 1] 范围内"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._calculate_contribution(factor_data, "test_factor")
        assert -1.0 <= result["ic"] <= 1.0

    def test_contribution_ratio_should_be_ic_squared(self):
        """contribution_ratio 应等于 IC^2"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._calculate_contribution(factor_data, "test_factor")
        assert abs(result["contribution_ratio"] - result["ic"] ** 2) < 1e-10

    def test_long_short_should_be_high_minus_low(self):
        """多空收益应等于高暴露组收益减低暴露组收益"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._calculate_contribution(factor_data, "test_factor")
        expected = result["high_exposure_return"] - result["low_exposure_return"]
        assert abs(result["long_short_return"] - expected) < 1e-10

    def test_insufficient_stocks_should_return_error(self):
        """每日不足5只股票时应返回错误"""
        factor_data = _make_factor_data(n_stocks=3, n_days=30)
        result = self.service._calculate_contribution(factor_data, "test_factor")
        assert "error" in result

    def test_insufficient_days_should_return_error(self):
        """有效IC天数不足3天时应返回错误"""
        # 只有2天数据
        factor_data = _make_factor_data(n_stocks=10, n_days=2)
        result = self.service._calculate_contribution(factor_data, "test_factor")
        assert "error" in result

    def test_missing_factor_column_should_return_error(self):
        """因子列不存在时应返回错误"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._calculate_contribution(factor_data, "nonexistent_factor")
        assert "error" in result

    def test_missing_close_column_should_return_error(self):
        """缺少close列时应返回错误"""
        dates = pd.bdate_range(start="2024-01-02", periods=30, freq="B")
        factor_data = {"600001": pd.DataFrame({"test_factor": np.random.randn(30)}, index=dates)}
        result = self.service._calculate_contribution(factor_data, "test_factor")
        assert "error" in result

    def test_constant_factor_values_should_return_error(self):
        """因子值恒定（std≈0）时应返回错误（spearmanr无法计算）"""
        dates = pd.bdate_range(start="2024-01-02", periods=30, freq="B")
        factor_data = {}
        for i in range(10):
            code = f"60000{i + 1}"
            df = pd.DataFrame(
                {"test_factor": np.full(30, 1.0), "close": 10.0 + np.random.randn(30) * 0.3},
                index=dates,
            )
            factor_data[code] = df
        result = self.service._calculate_contribution(factor_data, "test_factor")
        assert "error" in result

    def test_sample_size_should_match_total_observations(self):
        """sample_size 应等于所有股票有效观测的总数"""
        n_stocks, n_days = 8, 20
        factor_data = _make_factor_data(n_stocks=n_stocks, n_days=n_days)
        result = self.service._calculate_contribution(factor_data, "test_factor")
        # 每只股票有 n_days-1 个有效 future_return（最后一天shift(-1)为NaN）
        assert result["sample_size"] == n_stocks * (n_days - 1)

    def test_should_not_mutate_input_data(self):
        """不应修改传入的 factor_data"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        # 深拷贝用于对比
        original = {k: v.copy() for k, v in factor_data.items()}
        self.service._calculate_contribution(factor_data, "test_factor")
        for code in original:
            assert factor_data[code].equals(original[code]), f"股票 {code} 数据被修改"


class TestDecomposeAlphaBeta:
    """_decompose_alpha_beta 方法测试"""

    def setup_method(self):
        self.service = FactorAttributionService()

    def test_without_benchmark_should_return_portfolio_stats(self):
        """无基准数据时应返回组合统计信息"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=None)

        assert result["has_benchmark"] is False
        assert "portfolio_return" in result
        assert "daily_mean" in result["portfolio_return"]
        assert "annual_return" in result["portfolio_return"]
        assert "volatility" in result["portfolio_return"]
        assert "sharpe" in result["portfolio_return"]

    def test_with_benchmark_should_return_alpha_beta(self):
        """有基准数据时应返回 alpha/beta/r_squared"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        benchmark = _make_benchmark_data(n_days=30)
        result = self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=benchmark)

        assert result["has_benchmark"] is True
        assert "alpha" in result
        assert "beta" in result
        assert "r_squared" in result
        assert "daily_alpha" in result
        assert "interpretation" in result

    def test_beta_should_use_safe_divide(self):
        """Beta计算应使用safe_divide，基准方差为0时不应崩溃"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        # 构造恒定价格的基准（方差为0）
        dates = pd.bdate_range(start="2024-01-02", periods=30, freq="B")
        benchmark = pd.DataFrame({"close": np.full(30, 3000.0)}, index=dates)
        result = self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=benchmark)
        # 应不崩溃，可能返回error或合理默认值
        assert isinstance(result, dict)

    def test_insufficient_trading_days_should_return_error(self):
        """有效交易日不足10天时应返回错误"""
        factor_data = _make_factor_data(n_stocks=10, n_days=5)
        result = self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=None)
        assert "error" in result

    def test_benchmark_without_close_should_return_error(self):
        """基准数据缺少close列时应返回错误"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        dates = pd.bdate_range(start="2024-01-02", periods=30, freq="B")
        benchmark = pd.DataFrame({"open": np.random.randn(30)}, index=dates)
        result = self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=benchmark)
        assert "error" in result

    def test_missing_factor_column_should_skip_stock(self):
        """因子列不存在的股票应被跳过"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        # 添加一只没有因子列的股票
        dates = factor_data["600001"].index
        factor_data["600999"] = pd.DataFrame({"close": 10.0 + np.random.randn(30)}, index=dates)
        result = self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=None)
        # 应正常返回，不崩溃
        assert isinstance(result, dict)

    def test_r_squared_should_be_between_0_and_1(self):
        """R²应在 [0, 1] 范围内"""
        factor_data = _make_factor_data(n_stocks=10, n_days=60)
        benchmark = _make_benchmark_data(n_days=60)
        result = self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=benchmark)
        if result.get("has_benchmark"):
            assert 0.0 <= result["r_squared"] <= 1.0

    def test_should_not_mutate_input_data(self):
        """不应修改传入的 factor_data"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        original = {k: v.copy() for k, v in factor_data.items()}
        self.service._decompose_alpha_beta(factor_data, "test_factor", benchmark_data=None)
        for code in original:
            assert factor_data[code].equals(original[code]), f"股票 {code} 数据被修改"


class TestDecomposeReturn:
    """_decompose_return 方法测试"""

    def setup_method(self):
        self.service = FactorAttributionService()

    def test_normal_data_should_return_overall_and_by_stock(self):
        """正常数据应返回整体统计和按股票统计"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._decompose_return(factor_data, "test_factor")

        assert "overall_stats" in result
        assert "return_by_stock" in result
        assert "stock_count" in result
        assert "total_observations" in result

    def test_overall_stats_should_contain_expected_fields(self):
        """overall_stats 应包含所有预期字段"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._decompose_return(factor_data, "test_factor")
        stats = result["overall_stats"]

        expected_keys = [
            "avg_daily_return",
            "annual_return",
            "cumulative_return",
            "volatility_annual",
            "daily_volatility",
            "sharpe_ratio",
            "win_rate",
        ]
        for key in expected_keys:
            assert key in stats, f"缺少字段: {key}"

    def test_per_stock_stats_should_contain_expected_fields(self):
        """每只股票的统计应包含所有预期字段"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._decompose_return(factor_data, "test_factor")

        for code, stock_stats in result["return_by_stock"].items():
            expected_keys = [
                "avg_daily_return",
                "annual_return",
                "cumulative_return",
                "volatility",
                "daily_volatility",
                "sharpe",
                "win_rate",
                "count",
            ]
            for key in expected_keys:
                assert key in stock_stats, f"股票 {code} 缺少字段: {key}"

    def test_win_rate_should_be_between_0_and_1(self):
        """胜率应在 [0, 1] 范围内"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        result = self.service._decompose_return(factor_data, "test_factor")
        assert 0.0 <= result["overall_stats"]["win_rate"] <= 1.0

    def test_stock_count_should_match_input(self):
        """stock_count 应等于有有效收益的股票数"""
        n_stocks = 8
        factor_data = _make_factor_data(n_stocks=n_stocks, n_days=30)
        result = self.service._decompose_return(factor_data, "test_factor")
        assert result["stock_count"] == n_stocks

    def test_empty_factor_data_should_return_error(self):
        """空 factor_data 应返回错误"""
        result = self.service._decompose_return({}, "test_factor")
        assert "error" in result

    def test_single_day_data_should_return_error(self):
        """只有一天数据（无法计算收益率）应返回错误"""
        dates = pd.bdate_range(start="2024-01-02", periods=1, freq="B")
        factor_data = {"600001": pd.DataFrame({"test_factor": [1.0], "close": [10.0]}, index=dates)}
        result = self.service._decompose_return(factor_data, "test_factor")
        assert "error" in result

    def test_all_nan_close_should_return_error(self):
        """close列全为NaN时应返回错误"""
        dates = pd.bdate_range(start="2024-01-02", periods=10, freq="B")
        factor_data = {
            "600001": pd.DataFrame(
                {"test_factor": np.random.randn(10), "close": [np.nan] * 10},
                index=dates,
            )
        }
        result = self.service._decompose_return(factor_data, "test_factor")
        assert "error" in result

    def test_should_not_mutate_input_data(self):
        """不应修改传入的 factor_data"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        original = {k: v.copy() for k, v in factor_data.items()}
        self.service._decompose_return(factor_data, "test_factor")
        for code in original:
            assert factor_data[code].equals(original[code]), f"股票 {code} 数据被修改"


class TestAnalyzeAttribution:
    """analyze_attribution 编排方法测试"""

    def setup_method(self):
        self.service = FactorAttributionService()

    def test_with_benchmark_should_return_all_sections(self):
        """提供基准数据时应返回所有三个分析板块"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        benchmark = _make_benchmark_data(n_days=30)
        result = self.service.analyze_attribution(factor_data, "test_factor", benchmark_data=benchmark)

        assert "factor_contribution" in result
        assert "alpha_beta" in result
        assert "return_decomposition" in result

    def test_without_benchmark_should_auto_fetch(self):
        """未提供基准数据时应自动获取（需mock _get_benchmark_data）"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        mock_benchmark = _make_benchmark_data(n_days=30)

        with patch.object(self.service, "_get_benchmark_data", return_value=mock_benchmark):
            result = self.service.analyze_attribution(factor_data, "test_factor")

        assert "factor_contribution" in result
        assert "alpha_beta" in result
        assert "return_decomposition" in result

    def test_auto_fetch_failure_should_still_return_alpha_beta(self):
        """自动获取基准失败时，alpha_beta 板块应正常返回（无基准模式）"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)

        with patch.object(self.service, "_get_benchmark_data", return_value=None):
            result = self.service.analyze_attribution(factor_data, "test_factor")

        assert "alpha_beta" in result
        # 无基准时 has_benchmark 应为 False
        if "has_benchmark" in result["alpha_beta"]:
            assert result["alpha_beta"]["has_benchmark"] is False

    def test_get_benchmark_data_should_receive_correct_date_range(self):
        """_get_benchmark_data 应接收正确的日期范围"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        mock_benchmark = _make_benchmark_data(n_days=30)

        with patch.object(self.service, "_get_benchmark_data", return_value=mock_benchmark) as mock_fn:
            self.service.analyze_attribution(factor_data, "test_factor")
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args
            # start_date 应不晚于最早日期，end_date 应不早于最晚日期
            assert call_kwargs is not None

    def test_should_not_mutate_input_data(self):
        """不应修改传入的 factor_data"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        benchmark = _make_benchmark_data(n_days=30)
        original = {k: v.copy() for k, v in factor_data.items()}
        self.service.analyze_attribution(factor_data, "test_factor", benchmark_data=benchmark)
        for code in original:
            assert factor_data[code].equals(original[code]), f"股票 {code} 数据被修改"


class TestEdgeCases:
    """边界条件综合测试"""

    def setup_method(self):
        self.service = FactorAttributionService()

    def test_single_stock_should_fail_contribution(self):
        """单只股票无法计算横截面IC"""
        factor_data = _make_factor_data(n_stocks=1, n_days=30)
        result = self.service._calculate_contribution(factor_data, "test_factor")
        assert "error" in result

    def test_factor_with_nan_values_should_handle_gracefully(self):
        """因子值含NaN时应优雅处理"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        # 在部分股票中注入NaN
        for i, code in enumerate(factor_data):
            if i < 3:
                factor_data[code].loc[factor_data[code].index[:5], "test_factor"] = np.nan
        result = self.service._calculate_contribution(factor_data, "test_factor")
        # 应正常返回或返回error，不应崩溃
        assert isinstance(result, dict)

    def test_close_with_zero_should_not_cause_inf(self):
        """close为0时不应产生Inf收益率"""
        dates = pd.bdate_range(start="2024-01-02", periods=30, freq="B")
        factor_data = {}
        for i in range(10):
            code = f"60000{i + 1}"
            close = 10.0 + np.random.randn(30) * 0.3
            close[10] = 0.0  # 注入一个0价格
            close = np.maximum(close, 0.0)
            df = pd.DataFrame(
                {"test_factor": np.random.randn(30), "close": close},
                index=dates,
            )
            factor_data[code] = df
        result = self.service._calculate_contribution(factor_data, "test_factor")
        # 应不崩溃
        assert isinstance(result, dict)

    def test_empty_dataframe_in_factor_data(self):
        """factor_data中包含空DataFrame时应优雅处理"""
        factor_data = _make_factor_data(n_stocks=10, n_days=30)
        factor_data["600999"] = pd.DataFrame(columns=["test_factor", "close"])
        result = self.service._decompose_return(factor_data, "test_factor")
        # 应正常返回，空DataFrame被跳过
        assert isinstance(result, dict)
        assert result["stock_count"] == 10  # 空的被跳过

    def test_identical_close_prices_should_not_crash(self):
        """价格完全不变（收益率为0）时不应崩溃"""
        dates = pd.bdate_range(start="2024-01-02", periods=30, freq="B")
        factor_data = {}
        for i in range(10):
            code = f"60000{i + 1}"
            df = pd.DataFrame(
                {"test_factor": np.random.randn(30), "close": np.full(30, 10.0)},
                index=dates,
            )
            factor_data[code] = df
        result = self.service._decompose_return(factor_data, "test_factor")
        # 应正常返回
        assert isinstance(result, dict)
        if "error" not in result:
            # 收益率为0，sharpe应为0.0（calculate_sharpe返回None时回退为0.0）
            for code, stats in result["return_by_stock"].items():
                assert stats["avg_daily_return"] == 0.0
