"""
量化算法审查修复验证 - 单元测试

覆盖修复点：
1. Sharpe Ratio 无风险利率扣减（3处）
2. 等权重策略 1/N 分配
3. 市值加权策略向量化
4. 回测日收益不再截断
5. 分块回测 rf 参数化
"""
import pytest
import numpy as np
import pandas as pd
from backend.services.statistics_service import StatisticsService
from backend.services.portfolio_analysis_service import PortfolioAnalysisService
from backend.strategies.equal_weight_strategy import EqualWeightStrategy
from backend.strategies.market_cap_strategy import MarketCapStrategy
from backend.strategies.base_strategy import BaseStrategy


# ============ 1. Sharpe Ratio 无风险利率扣减测试 ============

class TestSharpeRatioWithRiskFreeRate:
    """验证 Sharpe Ratio 正确扣减无风险利率"""

    def setup_method(self):
        np.random.seed(42)
        # 生成正收益场景：日均0.1%，标准差1%
        n_days = 252
        self.positive_returns = pd.Series(
            np.random.randn(n_days) * 0.01 + 0.001,
            index=pd.date_range("2023-01-01", periods=n_days, freq="B")
        )
        # 生成零收益场景：日均0%，标准差1%
        self.zero_returns = pd.Series(
            np.random.randn(n_days) * 0.01,
            index=pd.date_range("2023-01-01", periods=n_days, freq="B")
        )

    # --- 1a. StatisticsService.analyze_quantile_returns ---

    def test_quantile_sharpe_lower_with_higher_rf(self):
        """rf越高，Sharpe应该越低"""
        svc = StatisticsService()
        quantile_returns = {"Q1": self.positive_returns}

        r_low_rf = svc.analyze_quantile_returns(
            quantile_returns, risk_free_rate=0.01
        )
        r_high_rf = svc.analyze_quantile_returns(
            quantile_returns, risk_free_rate=0.05
        )

        assert r_low_rf["Q1"]["sharpe"] > r_high_rf["Q1"]["sharpe"], \
            f"rf=0.01 sharpe={r_low_rf['Q1']['sharpe']:.4f} 应 > rf=0.05 sharpe={r_high_rf['Q1']['sharpe']:.4f}"

    def test_quantile_sharpe_negative_when_below_rf(self):
        """收益低于无风险利率时，Sharpe应为负"""
        svc = StatisticsService()
        # 构造日均收益0.0001(年化2.5%)，rf=5%时Sharpe应为负
        # 必须使用非恒定序列（std>0），否则calculate_risk_metrics返回None
        np.random.seed(42)
        low_returns = pd.Series(np.random.randn(252) * 0.001 + 0.0001)
        quantile_returns = {"Q1": low_returns}

        r = svc.analyze_quantile_returns(
            quantile_returns, risk_free_rate=0.05
        )
        assert r["Q1"]["sharpe"] is not None and r["Q1"]["sharpe"] < 0, \
            f"日收益0.0001 vs rf=0.05，Sharpe应为负，实际={r['Q1']['sharpe']}"

    def test_quantile_sharpe_zero_rf_equals_old_behavior(self):
        """rf=0时，Sharpe = mean/std * sqrt(252)，与旧行为一致"""
        svc = StatisticsService()
        quantile_returns = {"Q1": self.positive_returns}

        r = svc.analyze_quantile_returns(
            quantile_returns, risk_free_rate=0.0
        )
        mean = self.positive_returns.mean()
        std = self.positive_returns.std()
        expected = (mean / std) * np.sqrt(252)
        assert abs(r["Q1"]["sharpe"] - expected) < 1e-10, \
            f"rf=0时Sharpe应与mean/std*sqrt(252)一致：{r['Q1']['sharpe']:.6f} vs {expected:.6f}"

    def test_quantile_empty_returns_returns_none(self):
        """空收益率序列应返回None（不可计算）"""
        svc = StatisticsService()
        r = svc.analyze_quantile_returns(
            {"Q1": pd.Series([], dtype=float)}, risk_free_rate=0.03
        )
        assert r["Q1"]["sharpe"] is None

    # --- 1b. FactorReturnAnalysisService._calculate_sharpe_ratio ---

    def test_factor_return_sharpe_uses_rf(self):
        """验证因子收益率Sharpe扣减了rf"""
        try:
            from backend.services.factor_return_analysis_service import (
                FactorReturnAnalysisService,
            )
        except Exception:
            pytest.skip("FactorReturnAnalysisService 依赖不可用（empyrical兼容性问题）")

        svc = FactorReturnAnalysisService()

        sharpe_no_rf = svc._calculate_sharpe_ratio(
            self.positive_returns, risk_free_rate=0.0
        )
        sharpe_with_rf = svc._calculate_sharpe_ratio(
            self.positive_returns, risk_free_rate=0.03
        )

        assert sharpe_with_rf < sharpe_no_rf, \
            f"扣减rf后Sharpe应更低：{sharpe_with_rf:.4f} < {sharpe_no_rf:.4f}"

    def test_factor_return_sharpe_short_series(self):
        """短序列应返回0"""
        try:
            from backend.services.factor_return_analysis_service import (
                FactorReturnAnalysisService,
            )
        except Exception:
            pytest.skip("FactorReturnAnalysisService 依赖不可用（empyrical兼容性问题）")

        svc = FactorReturnAnalysisService()
        r = svc._calculate_sharpe_ratio(pd.Series([0.01]), risk_free_rate=0.03)
        assert r is None  # 规则6：不可计算返回None

    def test_factor_return_sharpe_zero_std(self):
        """零标准差应返回None（规则6）"""
        try:
            from backend.services.factor_return_analysis_service import (
                FactorReturnAnalysisService,
            )
        except Exception:
            pytest.skip("FactorReturnAnalysisService 依赖不可用（empyrical兼容性问题）")

        svc = FactorReturnAnalysisService()
        # 使用全零序列确保标准差精确为0
        r = svc._calculate_sharpe_ratio(
            pd.Series([0.0] * 100), risk_free_rate=0.03
        )
        assert r is None  # 规则6：不可计算返回None

    # --- 1c. PortfolioAnalysisService.optimize_weights (max_sharpe) ---

    def test_max_sharpe_weights_change_with_rf(self):
        """max_sharpe方法中rf不同，权重应不同"""
        svc = PortfolioAnalysisService()
        np.random.seed(42)

        # 两个因子：一个高收益，一个低收益
        n = 252
        factor_returns = pd.DataFrame({
            "high_return": np.random.randn(n) * 0.02 + 0.002,  # 年化约50%
            "low_return": np.random.randn(n) * 0.02 + 0.0001,  # 年化约2.5%
        })

        r_low_rf = svc.optimize_weights(
            factor_returns, method="max_sharpe", risk_free_rate=0.01
        )
        r_high_rf = svc.optimize_weights(
            factor_returns, method="max_sharpe", risk_free_rate=0.05
        )

        # rf=5%时，低收益因子Sharpe可能为负，权重应为0
        w_low = r_low_rf["weights"].get("low_return", 0)
        w_high = r_high_rf["weights"].get("low_return", 0)
        # 高rf下低收益因子权重应更低或为0
        assert w_high <= w_low, \
            f"高rf下低收益因子权重应≤低rf下权重：{w_high:.4f} vs {w_low:.4f}"


# ============ 2. 等权重策略 1/N 测试 ============

class TestEqualWeightStrategy:
    """验证等权重策略正确使用 1/N"""

    def setup_method(self):
        self.strategy = EqualWeightStrategy()

    def _make_multi_stock_df(self, n_stocks: int, n_dates: int = 10):
        """创建多股票DataFrame"""
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        records = []
        for d in dates:
            for i in range(n_stocks):
                records.append({
                    "date": d,
                    "stock_code": f"{i:06d}",
                    "close": 100 + np.random.randn() * 5,
                })
        df = pd.DataFrame(records)
        df.set_index(["date", "stock_code"], inplace=True)
        return df

    def test_single_stock_weight_is_one(self):
        """单股票单日期时权重应为1.0（1/1）"""
        df = self._make_multi_stock_df(n_stocks=1, n_dates=1)
        signals = pd.Series(1, index=df.index)
        weights = self.strategy.calculate_weights(df, signals)

        assert (weights == 1.0).all(), f"单股票权重应为1.0，实际: {weights.unique()}"

    def test_multi_stock_weights_sum_to_one_per_date(self):
        """多股票时：每个日期内权重和为1.0"""
        n_stocks = 5
        n_dates = 3
        df = self._make_multi_stock_df(n_stocks=n_stocks, n_dates=n_dates)
        signals = pd.Series(1, index=df.index)
        weights = self.strategy.calculate_weights(df, signals)

        # 每个日期内权重和应为1.0
        for date in df.index.get_level_values(0).unique():
            date_weights = weights.xs(date, level=0)
            assert abs(date_weights.sum() - 1.0) < 1e-10, \
                f"日期{date}权重和应为1.0，实际: {date_weights.sum():.6f}"

        # 每只股票权重应为 1/n_stocks
        expected_weight = 1.0 / n_stocks
        assert (abs(weights - expected_weight) < 1e-10).all(), \
            f"每只权重应为1/{n_stocks}={expected_weight:.4f}"

    def test_multi_stock_equal_weights(self):
        """多股票时每只权重相等 = 1/N_stocks"""
        n_stocks = 4
        n_dates = 2
        df = self._make_multi_stock_df(n_stocks=n_stocks, n_dates=n_dates)
        signals = pd.Series(1, index=df.index)
        weights = self.strategy.calculate_weights(df, signals)

        expected_weight = 1.0 / n_stocks
        assert (abs(weights - expected_weight) < 1e-10).all(), \
            f"每只权重应为1/{n_stocks}={expected_weight:.4f}，实际: {weights.unique()}"

    def test_no_signal_zero_weights(self):
        """无信号时所有权重为0"""
        df = self._make_multi_stock_df(n_stocks=5, n_dates=1)
        signals = pd.Series(0, index=df.index)
        weights = self.strategy.calculate_weights(df, signals)

        assert (weights == 0.0).all(), "无信号时权重应全部为0"

    def test_mixed_signals_correct_weights(self):
        """部分信号为1时，只有信号=1的股票有权重"""
        df = self._make_multi_stock_df(n_stocks=4, n_dates=1)
        signals = pd.Series([1, 0, 1, 0], index=df.index)
        weights = self.strategy.calculate_weights(df, signals)

        # 只有2只信号=1，每只权重=0.5
        assert weights.iloc[0] == 0.5
        assert weights.iloc[1] == 0.0
        assert weights.iloc[2] == 0.5
        assert weights.iloc[3] == 0.0
        assert abs(weights.sum() - 1.0) < 1e-10


# ============ 3. 市值加权策略向量化测试 ============

class TestMarketCapStrategy:
    """验证市值加权策略向量化正确性"""

    def setup_method(self):
        self.strategy = MarketCapStrategy()

    def _make_df_with_mcap(self, n_stocks: int, n_dates: int = 3):
        """创建带市值的多股票DataFrame（单级索引，date为索引名）"""
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        records = []
        np.random.seed(42)
        for d in dates:
            mcaps = np.random.lognormal(mean=10, sigma=1, size=n_stocks)
            for i in range(n_stocks):
                records.append({
                    "date": d,
                    "stock_code": f"{i:06d}",
                    "close": 100 + np.random.randn() * 5,
                    "market_cap": mcaps[i],
                })
        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        return df

    def _make_multi_index_df(self, n_stocks: int, n_dates: int = 3):
        """创建多级索引（date, stock_code）DataFrame"""
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        records = []
        np.random.seed(42)
        for d in dates:
            mcaps = np.random.lognormal(mean=10, sigma=1, size=n_stocks)
            for i in range(n_stocks):
                records.append({
                    "date": d,
                    "stock_code": f"{i:06d}",
                    "close": 100 + np.random.randn() * 5,
                    "market_cap": mcaps[i],
                })
        df = pd.DataFrame(records)
        df.set_index(["date", "stock_code"], inplace=True)
        return df

    def test_weights_sum_to_one_per_date_single_index(self):
        """单级索引：每个日期权重和应为1.0"""
        df = self._make_df_with_mcap(n_stocks=5, n_dates=3)
        signals = self.strategy.generate_signals(df)
        weights = self.strategy.calculate_weights(df, signals)

        for date in df.index.unique():
            date_weights = weights.loc[date]
            assert abs(date_weights.sum() - 1.0) < 1e-10, \
                f"日期{date}权重和应为1.0，实际: {date_weights.sum():.6f}"

    def test_weights_sum_to_one_per_date_multi_index(self):
        """多级索引：每个日期权重和应为1.0"""
        df = self._make_multi_index_df(n_stocks=5, n_dates=3)
        signals = self.strategy.generate_signals(df)
        weights = self.strategy.calculate_weights(df, signals)

        for date in df.index.get_level_values("date").unique():
            date_weights = weights.xs(date, level="date")
            assert abs(date_weights.sum() - 1.0) < 1e-10, \
                f"日期{date}权重和应为1.0，实际: {date_weights.sum():.6f}"

    def test_larger_mcap_gets_larger_weight(self):
        """市值大的股票权重应更大"""
        df = self._make_df_with_mcap(n_stocks=3, n_dates=1)
        signals = self.strategy.generate_signals(df)
        weights = self.strategy.calculate_weights(df, signals)

        mcaps = df["market_cap"]
        w = weights.values
        # 排序：市值最大的股票权重也应最大
        mcap_order = np.argsort(mcaps.values)
        weight_order = np.argsort(w)
        assert mcap_order[-1] == weight_order[-1], \
            "市值最大的股票应有最大权重"

    def test_no_market_cap_fallback_to_equal_weight(self):
        """无市值列时退化为等权重"""
        df = self._make_df_with_mcap(n_stocks=4, n_dates=1)
        df = df.drop(columns=["market_cap"])
        signals = pd.Series(1, index=df.index)
        weights = self.strategy.calculate_weights(df, signals)

        assert abs(weights.sum() - 1.0) < 1e-10
        assert (weights == 0.25).all(), "应退化为1/N等权重"

    def test_zero_mcap_gets_zero_weight(self):
        """市值为0或NaN的股票权重为0"""
        df = self._make_df_with_mcap(n_stocks=3, n_dates=1)
        # 使用 iloc 按位置修改，避免 label-based indexing 选中所有同行日期
        df.iloc[0, df.columns.get_loc("market_cap")] = 0
        signals = self.strategy.generate_signals(df)
        weights = self.strategy.calculate_weights(df, signals)

        assert weights.iloc[0] == 0.0, "市值为0的股票权重应为0"
        assert abs(weights.iloc[1:].sum() - 1.0) < 1e-10, "其余股票权重和应为1.0"


# ============ 4. 回测不再截断收益测试 ============

class TestBaseStrategyNoClipping:
    """验证回测不再对日收益进行 ±50% 截断"""

    def test_extreme_returns_not_clipped(self):
        """极端收益不应被截断"""
        class TestStrategy(BaseStrategy):
            def generate_signals(self, df):
                return pd.Series(1, index=df.index)
            def calculate_weights(self, df, signals):
                return pd.Series(1.0, index=df.index)

        strategy = TestStrategy()

        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "close": [100, 200, 50, 500, 10],
        }, index=dates)
        df.index.name = "date"

        result = strategy.backtest(df)
        returns = result["portfolio_returns"].dropna()

        # forward_return = pct_change(1).shift(-1) = [1.0, -0.75, 9.0, -0.98]
        # 扣除首日佣金后: [~1.0, -0.75, 9.0, -0.98]
        assert abs(returns.iloc[1] - (-0.75)) < 0.01, \
            f"-75%日收益不应被截断，实际: {returns.iloc[1]:.4f}"
        assert abs(returns.iloc[2] - 9.0) < 0.01, \
            f"+900%日收益不应被截断，实际: {returns.iloc[2]:.4f}"
        assert not ((returns == 0.5) | (returns == -0.5)).any(), \
            "不应有任何收益被截断到±0.5"

    def test_normal_returns_unchanged(self):
        """正常收益（±10%以内）不受影响"""
        class TestStrategy(BaseStrategy):
            def generate_signals(self, df):
                return pd.Series(1, index=df.index)
            def calculate_weights(self, df, signals):
                return pd.Series(1.0, index=df.index)

        strategy = TestStrategy()

        dates = pd.date_range("2023-01-01", periods=6, freq="B")
        prices = [100, 105, 98, 103, 110, 108]
        df = pd.DataFrame({"close": prices}, index=dates)
        df.index.name = "date"

        result = strategy.backtest(df)
        returns = result["portfolio_returns"].dropna()

        # forward_return = pct_change(1).shift(-1) = [0.05, -0.0667, 0.0510, 0.0680, -0.0182]
        expected_returns = pd.Series([
            (prices[1] - prices[0]) / prices[0],
            (prices[2] - prices[1]) / prices[1],
            (prices[3] - prices[2]) / prices[2],
            (prices[4] - prices[3]) / prices[3],
            (prices[5] - prices[4]) / prices[4],
        ], index=returns.index)

        for i in range(len(returns)):
            assert abs(returns.iloc[i] - expected_returns.iloc[i]) < 0.001, \
                f"正常收益不应被修改：期望{expected_returns.iloc[i]:.4f}，实际{returns.iloc[i]:.4f}"


# ============ 5. 分块回测 rf 参数化测试 ============

class TestVectorBTChunkedRfParam:
    """验证分块回测接受并使用 risk_free_rate 参数"""

    def test_chunked_backtest_accepts_rf_param(self):
        """验证 chunked_single_factor_backtest 接受 risk_free_rate 参数"""
        pytest.importorskip("vectorbt", reason="vectorbt 未安装")

        from backend.services.vectorbt_backtest_service import VectorBTBacktestService

        svc = VectorBTBacktestService()

        np.random.seed(42)
        n = 1000
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        close = np.maximum(close, 1)
        factor = np.random.randn(n) * 3

        df = pd.DataFrame({
            "close": close,
            "factor_test": factor,
            "tradable_mask": True,
            "is_limit_up": False,
            "is_limit_down": False,
        }, index=dates)

        r1 = svc.chunked_single_factor_backtest(
            df=df, factor_name="factor_test", risk_free_rate=0.03,
            chunk_size=500, overlap_size=50
        )
        r2 = svc.chunked_single_factor_backtest(
            df=df, factor_name="factor_test", risk_free_rate=0.05,
            chunk_size=500, overlap_size=50
        )

        assert "sharpe_ratio" in r1
        assert "sharpe_ratio" in r2
        if r1["sharpe_ratio"] is not None and r2["sharpe_ratio"] is not None:
            assert r1["sharpe_ratio"] != r2["sharpe_ratio"], \
                f"不同rf应产生不同Sharpe：{r1['sharpe_ratio']:.4f} vs {r2['sharpe_ratio']:.4f}"


# ============ 6. BaseStrategy 指标计算完整性测试 ============

class TestBaseStrategyMetrics:
    """验证 BaseStrategy.calculate_metrics 各项指标计算正确"""

    def setup_method(self):
        class TestStrategy(BaseStrategy):
            def generate_signals(self, df):
                return pd.Series(0, index=df.index)

            def calculate_weights(self, df, signals):
                return pd.Series(0.0, index=df.index)

        self.strategy = TestStrategy()

    def test_sharpe_matches_manual_calculation(self):
        """夏普比率与手动计算一致"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        metrics = self.strategy.calculate_metrics(returns, risk_free_rate=0.03)

        daily_rf = 0.03 / 252
        excess = returns - daily_rf
        expected_sharpe = excess.mean() * 252 / (returns.std() * np.sqrt(252))

        assert abs(metrics["sharpe_ratio"] - expected_sharpe) < 0.01, \
            f"Sharpe {metrics['sharpe_ratio']:.6f} vs 期望 {expected_sharpe:.6f}"

    def test_sortino_uses_downside_deviation(self):
        """Sortino比率使用下行偏差"""
        # 构造正收益为主但有少量负收益的序列
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.002)

        metrics = self.strategy.calculate_metrics(returns, risk_free_rate=0.03)

        # Sortino应该大于Sharpe（因为Sortino只考虑下行风险）
        assert metrics["sortino_ratio"] > metrics["sharpe_ratio"], \
            "正收益为主的序列，Sortino应 > Sharpe"

    def test_calmar_ratio_positive(self):
        """卡玛比率公式正确"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)

        metrics = self.strategy.calculate_metrics(returns, risk_free_rate=0.03)

        total_return = (1 + returns).prod() - 1
        n = len(returns)
        annual_return = (1 + total_return) ** (252 / n) - 1
        equity = (1 + returns).cumprod()
        max_dd = ((equity.cummax() - equity) / equity.cummax()).max()

        expected_calmar = annual_return / max_dd if max_dd > 0 else 0.0
        assert abs(metrics["calmar_ratio"] - expected_calmar) < 1e-10, \
            f"Calmar {metrics['calmar_ratio']:.6f} vs 期望 {expected_calmar:.6f}"

    def test_empty_returns_returns_zero_metrics(self):
        """空收益率序列返回None指标（不可计算）"""
        metrics = self.strategy.calculate_metrics(pd.Series([], dtype=float))
        assert metrics["sharpe_ratio"] is None
        assert metrics["total_return"] is None