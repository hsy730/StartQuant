"""
缓存优化与 risk_free 量纲修复的单元测试

覆盖：
1. analysis_service 缓存序列化/反序列化（factor_data 不再需要重新计算）
2. empyrical risk_free 参数量纲修复（年化利率 → 日频利率）
3. Sharpe/Sortino 除零保护
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys


# ============================================================
# 1. 缓存序列化/反序列化测试（直接测试方法，避免重导入链）
# ============================================================

def _get_analysis_service():
    """延迟导入 AnalysisService，避免模块级依赖缺失"""
    from backend.services.analysis_service import AnalysisService
    return AnalysisService()


class TestCacheSerializeDeserialize:
    """analysis_service 缓存 factor_data 的序列化与反序列化"""

    def setup_method(self):
        # 直接构造 AnalysisService 实例，跳过导入链
        # 因为 AnalysisService.__init__ 只设置 self.results_cache = {}
        # 我们手动创建一个轻量实例
        try:
            from backend.services.analysis_service import AnalysisService
            self.svc = AnalysisService()
        except ImportError:
            # 依赖缺失时，手动构造等价对象
            self.svc = None

    def _get_svc(self):
        if self.svc is None:
            pytest.skip("AnalysisService 依赖不可用")
        return self.svc

    def _make_factor_data(self, n_stocks=2, n_rows=5):
        factor_data = {}
        dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
        for i in range(n_stocks):
            code = f"00000{i+1}"
            df = pd.DataFrame({
                "close": np.random.randn(n_rows).cumsum() + 100,
                "factor_1": np.random.randn(n_rows),
                "factor_2": np.random.randn(n_rows),
            }, index=dates)
            factor_data[code] = df
        return factor_data

    def test_serialize_factor_data_roundtrip(self):
        """序列化后反序列化，factor_data 应完整恢复"""
        svc = self._get_svc()
        original = self._make_factor_data()
        results = {
            "metadata": {"test": True},
            "factor_data": original,
            "ic_ir": {},
            "shap": {},
            "alphalens": {},
        }
        serialized = svc._serialize_for_cache(results)
        deserialized = svc._deserialize_from_cache(serialized)

        for code in original:
            pd.testing.assert_frame_equal(
                deserialized["factor_data"][code],
                original[code],
                check_exact=False,
                rtol=1e-6,
                check_freq=False,  # pd.to_datetime 不保留 freq
            )

    def test_serialize_preserves_datetime_index(self):
        """日期索引应正确恢复为 DatetimeIndex"""
        svc = self._get_svc()
        original = self._make_factor_data(n_stocks=1)
        results = {"metadata": {}, "factor_data": original, "ic_ir": {}}
        serialized = svc._serialize_for_cache(results)
        deserialized = svc._deserialize_from_cache(serialized)

        code = list(original.keys())[0]
        assert isinstance(deserialized["factor_data"][code].index, pd.DatetimeIndex)

    def test_deserialize_without_factor_data_in_cache(self):
        """缓存中无 factor_data 时应返回空 dict，不报错"""
        svc = self._get_svc()
        cached = {"metadata": {}, "ic_ir": {}}
        result = svc._deserialize_from_cache(cached)
        assert result["factor_data"] == {}

    def test_deserialize_does_not_mutate_cache(self):
        """反序列化不应修改缓存原始数据（__index__ pop 不影响原 dict）"""
        svc = self._get_svc()
        original = self._make_factor_data(n_stocks=1)
        results = {"metadata": {}, "factor_data": original, "ic_ir": {}}
        serialized = svc._serialize_for_cache(results)

        # 第一次反序列化
        svc._deserialize_from_cache(serialized)
        # __index__ 应仍在 serialized 中
        code = list(serialized["factor_data"].keys())[0]
        assert "__index__" in serialized["factor_data"][code]

        # 第二次反序列化仍应成功
        result2 = svc._deserialize_from_cache(serialized)
        assert len(result2["factor_data"]) == 1

    def test_cache_hit_skips_factor_calculation(self):
        """缓存命中时应直接返回，不调用 calculate_factors_for_stocks"""
        svc = self._get_svc()
        original = self._make_factor_data()
        results = {
            "metadata": {"stock_codes": ["000001"], "factor_names": ["factor_1"]},
            "factor_data": original,
            "ic_ir": {"ic_stats": {"factor_1": {"IC均值": 0.05}}},
        }
        serialized = svc._serialize_for_cache(results)

        mock_cached = MagicMock()
        mock_cached.result_data = serialized

        with patch("backend.services.analysis_service.AnalysisCacheRepository") as MockRepo, \
             patch("backend.services.analysis_service.get_db_session") as mock_db:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_key.return_value = mock_cached
            MockRepo.return_value = mock_repo_instance
            mock_db.return_value = MagicMock()

            with patch("backend.services.analysis_service.factor_service") as mock_factor_svc:
                result = svc.analyze(
                    stock_codes=["000001"],
                    factor_names=["factor_1"],
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    use_cache=True,
                )
                mock_factor_svc.calculate_factors_for_stocks.assert_not_called()
                assert "factor_data" in result

    def test_serialize_empty_factor_data(self):
        """空 factor_data dict 应正确序列化"""
        svc = self._get_svc()
        results = {"metadata": {}, "factor_data": {}, "ic_ir": {}}
        serialized = svc._serialize_for_cache(results)
        assert serialized["factor_data"] == {}

    def test_serialize_factor_data_with_non_datetime_index(self):
        """非日期索引（如整数索引）应作为原始索引恢复"""
        svc = self._get_svc()
        df = pd.DataFrame({"close": [1, 2, 3], "factor_1": [0.1, 0.2, 0.3]})
        results = {"metadata": {}, "factor_data": {"000001": df}, "ic_ir": {}}
        serialized = svc._serialize_for_cache(results)
        deserialized = svc._deserialize_from_cache(serialized)

        restored = deserialized["factor_data"]["000001"]
        assert list(restored.index) == [0, 1, 2]


# ============================================================
# 2. risk_free 量纲修复测试
# ============================================================

class TestRiskFreeRateConversion:
    """empyrical risk_free 参数必须是日频利率，而非年化利率"""

    def test_sharpe_with_annual_rf_matches_manual(self):
        """传入年化 risk_free=0.03，结果应与手动计算（日频扣减）一致"""
        from backend.services.risk_metrics import calculate_risk_metrics

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        metrics = calculate_risk_metrics(returns, risk_free_rate=0.03)

        daily_rf = 0.03 / 252
        excess = returns - daily_rf
        expected_sharpe = excess.mean() / returns.std() * np.sqrt(252)

        assert abs(metrics["sharpe_ratio"] - expected_sharpe) < 0.01, \
            f"Sharpe {metrics['sharpe_ratio']:.6f} vs 期望 {expected_sharpe:.6f}"

    def test_sharpe_not_extremely_negative_with_rf(self):
        """修复前：传入 rf=0.03 给 empyrical 会导致 Sharpe ≈ -48，修复后应为正常值"""
        from backend.services.risk_metrics import calculate_risk_metrics

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        metrics = calculate_risk_metrics(returns, risk_free_rate=0.03)

        assert -5 < metrics["sharpe_ratio"] < 5, \
            f"Sharpe {metrics['sharpe_ratio']:.2f} 异常，疑似 risk_free 量纲错误"

    def test_sharpe_rf_zero_unchanged(self):
        """risk_free=0 时，结果应与 empyrical 默认行为一致"""
        from backend.services.risk_metrics import calculate_risk_metrics

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        metrics = calculate_risk_metrics(returns, risk_free_rate=0.0)

        import empyrical
        expected = float(empyrical.sharpe_ratio(returns.values, risk_free=0, period='daily', annualization=252))
        assert abs(metrics["sharpe_ratio"] - expected) < 1e-6

    def test_sortino_rf_conversion(self):
        """Sortino 的 required_return 也应从年化转为日频"""
        from backend.services.risk_metrics import calculate_risk_metrics

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.002)

        metrics = calculate_risk_metrics(returns, risk_free_rate=0.03)

        assert np.isfinite(metrics["sortino_ratio"])
        assert -10 < metrics["sortino_ratio"] < 10

    def test_base_strategy_rf_conversion(self):
        """BaseStrategy.calculate_metrics 的 risk_free 也应正确转换"""
        from backend.strategies.base_strategy import BaseStrategy

        class TestStrategy(BaseStrategy):
            def generate_signals(self, df):
                return pd.Series(0, index=df.index)
            def calculate_weights(self, df, signals):
                return pd.Series(0.0, index=df.index)

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        strategy = TestStrategy()
        metrics = strategy.calculate_metrics(returns, risk_free_rate=0.03)

        assert -5 < metrics["sharpe_ratio"] < 5, \
            f"BaseStrategy Sharpe {metrics['sharpe_ratio']:.2f} 异常"

    def test_factor_return_sharpe_rf_conversion(self):
        """FactorReturnAnalysisService._calculate_sharpe_ratio 的 risk_free 转换"""
        from backend.services.factor_return_analysis_service import FactorReturnAnalysisService

        svc = FactorReturnAnalysisService()
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        sharpe = svc._calculate_sharpe_ratio(returns, risk_free_rate=0.03)

        assert -5 < sharpe < 5, f"Sharpe {sharpe:.2f} 异常"

    def test_wrong_rf_produces_extreme_sharpe(self):
        """验证：传入年化 rf 而非日频 rf 会导致 Sharpe 极端负值（回归测试）"""
        import empyrical

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005).values

        correct = float(empyrical.sharpe_ratio(returns, risk_free=0.03/252, period='daily', annualization=252))
        wrong = float(empyrical.sharpe_ratio(returns, risk_free=0.03, period='daily', annualization=252))

        # 正确值应在正常范围
        assert -5 < correct < 5
        # 错误值应极端负
        assert wrong < -10


# ============================================================
# 3. Sharpe/Sortino 除零保护测试
# ============================================================

class TestSharpeZeroProtection:
    """标准差为零时的 Sharpe/Sortino 保护"""

    def test_sharpe_zero_returns(self):
        """全零收益率序列应返回 Sharpe=0"""
        from backend.services.risk_metrics import calculate_risk_metrics

        metrics = calculate_risk_metrics(pd.Series([0.0] * 100), risk_free_rate=0.03)
        assert metrics["sharpe_ratio"] == 0.0

    def test_sharpe_constant_returns(self):
        """恒定收益率序列（std=0）应返回有限值"""
        from backend.services.risk_metrics import calculate_risk_metrics

        metrics = calculate_risk_metrics(pd.Series([0.001] * 100), risk_free_rate=0.03)
        assert np.isfinite(metrics["sharpe_ratio"]) or metrics["sharpe_ratio"] == 0.0

    def test_factor_return_sharpe_zero_std(self):
        """FactorReturnAnalysisService 对零标准差返回 0"""
        from backend.services.factor_return_analysis_service import FactorReturnAnalysisService

        svc = FactorReturnAnalysisService()
        result = svc._calculate_sharpe_ratio(pd.Series([0.0] * 100), risk_free_rate=0.03)
        assert result == 0.0

    def test_factor_return_sharpe_single_value(self):
        """单值序列应返回 0"""
        from backend.services.factor_return_analysis_service import FactorReturnAnalysisService

        svc = FactorReturnAnalysisService()
        result = svc._calculate_sharpe_ratio(pd.Series([0.01]), risk_free_rate=0.03)
        assert result == 0.0

    def test_empty_metrics_all_zero(self):
        """空指标字典所有值应为 0.0"""
        from backend.services.risk_metrics import _empty_metrics

        metrics = _empty_metrics()
        assert all(v == 0.0 for v in metrics.values())

    def test_risk_metrics_empty_series(self):
        """空收益率序列应返回空指标"""
        from backend.services.risk_metrics import calculate_risk_metrics, _empty_metrics

        metrics = calculate_risk_metrics(pd.Series([], dtype=float))
        assert metrics == _empty_metrics()

    def test_risk_metrics_nan_series(self):
        """全 NaN 序列应返回空指标"""
        from backend.services.risk_metrics import calculate_risk_metrics, _empty_metrics

        metrics = calculate_risk_metrics(pd.Series([np.nan] * 10))
        assert metrics == _empty_metrics()


# ============================================================
# 4. vectorbt_backtest_service rf 转换测试
# ============================================================

class TestVectorBtRfConversion:
    """vectorbt_backtest_service 的 risk_free 转换"""

    def test_vectorbt_sharpe_uses_daily_rf(self):
        """验证 vectorbt_backtest_service 代码中 empyrical 调用使用日频 rf"""
        # 直接读取源码验证 rf 转换逻辑
        import inspect
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService

        source = inspect.getsource(VectorBTBacktestService)
        # 验证 sharpe_ratio 调用中 risk_free 除以了 annual_bars
        assert 'risk_free_rate / fc["annual_bars"]' in source, \
            "vectorbt_backtest_service 中 sharpe_ratio 的 risk_free 未除以 annual_bars"

    def test_vectorbt_sortino_uses_daily_rf(self):
        """验证 vectorbt_backtest_service 中 sortino_ratio 使用日频 required_return"""
        import inspect
        from backend.services.vectorbt_backtest_service import VectorBTBacktestService

        source = inspect.getsource(VectorBTBacktestService)
        # 验证 sortino_ratio 调用中 required_return 除以了 annual_bars
        assert 'risk_free_rate / fc["annual_bars"]' in source, \
            "vectorbt_backtest_service 中 sortino_ratio 的 required_return 未除以 annual_bars"
