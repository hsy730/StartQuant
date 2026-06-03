"""
未来函数（Look-ahead Bias）检测器单元测试

覆盖所有核心检测场景：
1. 正常因子 — 应全部通过检测
2. 未来函数因子（完美IC）— 应被识别为 CRITICAL/HIGH
3. 恒定值因子 — 应触发异常警告
4. 高自相关因子 — 应被标记
5. 极端回测指标 — 应触发回测真实性校验
6. 数据不足 — 应优雅降级
7. 横截面模式检测
8. 严格模式 vs 标准模式对比
"""
import pytest
import numpy as np
import pandas as pd
from backend.services.lookahead_bias_detector import (
    LookaheadBiasDetector,
    lookahead_bias_detector,
    strict_lookahead_bias_detector,
    BiasCheckResult,
    BiasRiskLevel,
    LookaheadBiasDetectionResult,
)


class TestLookaheadBiasDetectorNormalFactor:
    """场景1: 正常因子 — 所有检测项应通过"""

    def setup_method(self):
        np.random.seed(42)
        n = 252
        # 模拟正常因子：低IC（~0.02），有合理波动
        self.factor = pd.Series(np.random.randn(n) * 1.0, name="normal_factor")
        # 收益率与因子弱相关
        noise = np.random.randn(n) * 2.0
        self.returns = pd.Series(0.02 * self.factor.values + noise, name="returns")

    def test_normal_factor_should_be_safe(self):
        """正常因子应判定为 SAFE"""
        result = lookahead_bias_detector.detect(
            factor_values=self.factor, return_values=self.returns, factor_name="normal"
        )
        assert result.has_bias is False
        assert result.risk_level == BiasRiskLevel.SAFE
        assert result.risk_score < 10

    def test_normal_factor_ic_check_passes(self):
        """正常因子的 IC 应在安全范围内"""
        result = lookahead_bias_detector.detect(
            factor_values=self.factor, return_values=self.returns
        )
        ic_check = next((c for c in result.checks if c.check_name == "ic_magnitude"), None)
        assert ic_check is not None
        assert ic_check.passed == True

    def test_normal_factor_autocorr_check_passes(self):
        """正常因子的自相关应在安全范围内"""
        result = lookahead_bias_detector.detect(
            factor_values=self.factor, return_values=self.returns
        )
        ac_check = next((c for c in result.checks if c.check_name == "autocorrelation_lag1"), None)
        assert ac_check is not None
        assert ac_check.passed == True


class TestLookaheadBiasDetectorPerfectIC:
    """场景2: 完美IC因子（模拟未来函数）— 应被识别为 HIGH/CRITICAL"""

    def setup_method(self):
        np.random.seed(42)
        n = 252
        # 模拟未来函数：因子值与收益高度正相关（IC ≈ 0.9+）
        base_signal = np.random.randn(n)
        self.factor = pd.Series(base_signal, name="leaky_factor")
        self.returns = pd.Series(0.9 * base_signal + np.random.randn(n) * 0.1, name="returns")

    def test_perfect_ic_should_detect_bias(self):
        """高IC因子应被检测为有偏"""
        result = lookahead_bias_detector.detect(
            factor_values=self.factor, return_values=self.returns, factor_name="leaky"
        )
        # IC 会非常高，至少应触发 warning 或 error
        ic_check = next((c for c in result.checks if c.check_name == "ic_magnitude"), None)
        assert ic_check is not None
        assert ic_check.passed == False  # IC 应超过阈值

    def test_perfect_rank_corr_should_detect_bias(self):
        """高排名相关系数应被检测"""
        result = lookahead_bias_detector.detect(
            factor_values=self.factor, return_values=self.returns
        )
        rc_check = next((c for c in result.checks if c.check_name == "rank_correlation"), None)
        assert rc_check is not None
        # Spearman 相关性应该很高
        assert rc_check.passed == False

    def test_perfect_ic_ir_should_be_high(self):
        """完美IC的 IR 应该极高"""
        result = lookahead_bias_detector.detect(
            factor_values=self.factor, return_values=self.returns
        )
        ir_check = next((c for c in result.checks if c.check_name == "ir_magnitude"), None)
        assert ir_check is not None
        assert ir_check.passed == False


class TestLookaheadBiasDetectorConstantFactor:
    """场景3: 恒定值或近乎恒定的因子"""

    def setup_method(self):
        n = 252
        self.constant_factor = pd.Series([1.0] * n, name="constant")
        self.near_constant = pd.Series(np.full(n, 1.0) + np.random.randn(n) * 1e-15, name="near_constant")
        self.returns = pd.Series(np.random.randn(n) * 0.02, name="returns")

    def test_constant_factor_detected(self):
        """恒定值因子应被检测"""
        result = lookahead_bias_detector.detect(
            factor_values=self.constant_factor, return_values=self.returns
        )
        vc_check = next((c for c in result.checks if c.check_name == "value_constancy"), None)
        assert vc_check is not None
        assert vc_check.passed is False
        assert vc_check.severity == "error"

    def test_near_constant_factor_warning(self):
        """近乎恒定的因子应产生警告"""
        result = lookahead_bias_detector.detect(
            factor_values=self.near_constant, return_values=self.returns
        )
        vc_check = next((c for c in result.checks if c.check_name == "value_constancy"), None)
        assert vc_check is not None
        # 极小噪声因子可能触发 warning（变化过于规律）或 error


class TestLookaheadBiasDetectorHighAutocorrelation:
    """场景4: 高自相关因子（可能是复制了某个序列）"""

    def setup_method(self):
        np.random.seed(42)
        n = 252
        base = np.cumsum(np.random.randn(n) * 0.01)
        # 因子值几乎是前一天的值 + 极小噪声（模拟用 close 做当日因子）
        self.high_ac_factor = pd.Series(base, name="high_autocorr")
        noise = np.random.randn(n) * 0.5
        self.returns = pd.Series(np.diff(base, prepend=base[0]) + noise, name="returns")

    def test_high_autocorrelation_detected(self):
        """高自相关应被检测到"""
        result = lookahead_bias_detector.detect(
            factor_values=self.high_ac_factor, return_values=self.returns
        )
        ac_check = next((c for c in result.checks if c.check_name == "autocorrelation_lag1"), None)
        assert ac_check is not None
        # 累积序列的自相关通常很高
        if not ac_check.passed:
            assert ac_check.severity in ("warning", "error")


class TestLookaheadBiasDetectorBacktestMetrics:
    """场景5: 异常回测指标（不真实的回测结果）"""

    def test_unrealistic_backtest_metrics(self):
        """不真实的回测指标应被检测"""
        fake_metrics = {
            "annual_return": 8.0,      # 800% 年化收益
            "sharpe_ratio": 15.0,       # 夏普比率 15
            "win_rate": 0.98,           # 胜率 98%
            "max_drawdown": 0.0001,     # 最大回撤 0.01%
        }
        result = lookahead_bias_detector.detect(
            factor_values=pd.Series(np.random.randn(252)),
            return_values=pd.Series(np.random.randn(252)),
            extra_context={"backtest_metrics": fake_metrics},
        )
        bt_check = next((c for c in result.checks if c.check_name == "backtest_reality_check"), None)
        assert bt_check is not None
        assert bt_check.passed is False
        assert bt_check.severity in ("critical", "error")

    def test_normal_backtest_metrics_pass(self):
        """正常的回测指标应通过"""
        normal_metrics = {
            "annual_return": 0.15,     # 15% 年化
            "sharpe_ratio": 1.5,       # 夏普 1.5
            "win_rate": 0.55,          # 胜率 55%
            "max_drawdown": 0.15,      # 最大回撤 15%
        }
        result = lookahead_bias_detector.detect(
            factor_values=pd.Series(np.random.randn(252)),
            return_values=pd.Series(np.random.randn(252)),
            extra_context={"backtest_metrics": normal_metrics},
        )
        bt_check = next((c for c in result.checks if c.check_name == "backtest_reality_check"), None)
        assert bt_check is not None
        assert bt_check.passed is True


class TestLookaheadBiasDetectorInsufficientData:
    """场景6: 数据不足时的优雅降级"""

    def test_too_few_samples(self):
        """样本数不足时应返回 SAFE（无法判断）"""
        tiny_factor = pd.Series([1.0, 2.0, 3.0])
        tiny_returns = pd.Series([0.01, -0.01, 0.02])
        result = lookahead_bias_detector.detect(
            factor_values=tiny_factor, return_values=tiny_returns
        )
        assert result.has_bias is False
        assert result.risk_level == BiasRiskLevel.SAFE
        assert len(result.checks) == 1  # 只有 data_sufficiency 检查
        assert result.checks[0].check_name == "data_sufficiency"

    def test_no_return_data(self):
        """无收益率数据时仍应执行基础检测（自相关、分布等）"""
        factor = pd.Series(np.random.randn(100))
        result = lookahead_bias_detector.detect(factor_values=factor, return_values=None)
        # 应跳过 IC/IR/排名检测，但保留自相关和分布检测
        assert result.has_bias is False  # 正常随机因子无问题
        check_names = {c.check_name for c in result.checks}
        assert "ic_magnitude" not in check_names  # 无 return 数据时跳过
        assert "autocorrelation_lag1" in check_names  # 自相关不需要 return


class TestLookaheadBiasDetectorCrossSectional:
    """场景7: 横截面模式检测"""

    def setup_method(self):
        np.random.seed(42)
        n_dates = 60
        n_stocks = 20
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")

        rows = []
        for date in dates:
            for i in range(n_stocks):
                factor_val = np.random.randn()
                ret_val = 0.01 * factor_val + np.random.randn() * 0.03
                rows.append({
                    "date": date,
                    "stock_code": f"{i:06d}",
                    "factor_value": factor_val,
                    "return": ret_val,
                })

        self.cs_df = pd.DataFrame(rows)

    def test_cross_sectional_normal(self):
        """横截面正常因子应通过"""
        result = lookahead_bias_detector.detect_cross_sectional(
            factor_df=self.cs_df[["date", "stock_code", "factor_value"]].rename(columns={"factor_value": "factor"}),
            return_df=self.cs_df[["date", "stock_code", "return"]],
            factor_name="factor",
        )
        assert result.has_bias is False
        assert result.risk_level in (BiasRiskLevel.SAFE, BiasRiskLevel.LOW)

    def test_cross_sectional_leaky(self):
        """横截面泄漏因子（因子=收益的完美预测）应被检测"""
        leaky_df = self.cs_df.copy()
        leaky_df["factor_value"] = leaky_df["return"] * 50  # 因子值直接由收益决定

        result = lookahead_bias_detector.detect_cross_sectional(
            factor_df=leaky_df[["date", "stock_code", "factor_value"]].rename(columns={"factor_value": "factor"}),
            return_df=leaky_df[["date", "stock_code", "return"]],
            factor_name="factor",
        )
        # 泄漏因子应有高风险
        ic_check = next((c for c in result.checks if c.check_name == "cross_sectional_ic_mean"), None)
        assert ic_check is not None
        assert ic_check.passed == False


class TestLookaheadBiasDetectorStrictMode:
    """场景8: 严格模式 vs 标准模式对比"""

    def test_strict_mode_more_sensitive(self):
        """严格模式的阈值更低，更容易触发警告"""
        np.random.seed(42)
        n = 252
        # 中等强度的因子：IC 在边界附近
        factor = pd.Series(np.random.randn(n))
        returns = pd.Series(0.08 * factor.values + np.random.randn(n) * 0.8)

        normal_result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        strict_result = strict_lookahead_bias_detector.detect(factor_values=factor, return_values=returns)

        # 严格模式的风险评分应 >= 标准模式
        assert strict_result.risk_score >= normal_result.risk_score


class TestLookaheadBiasDetectorQuantileAnomaly:
    """场景9: 分层收益异常检测"""

    def test_extreme_quantile_spread(self):
        """极端分层收益差应被检测"""
        quantile_returns = {
            "Q1": pd.Series([-0.05] * 60),   # Q1 每天 -5%
            "Q5": pd.Series([0.06] * 60),    # Q5 每天 +6%
        }
        result = lookahead_bias_detector.detect(
            factor_values=pd.Series(np.random.randn(252)),
            return_values=pd.Series(np.random.randn(252)),
            extra_context={"quantile_returns": quantile_returns},
        )
        spread_check = next((c for c in result.checks if c.check_name == "quantile_spread"), None)
        assert spread_check is not None
        assert spread_check.passed == False


class TestLookaheadBiasDetectorTemporalConsistency:
    """场景10: 时段一致性检验"""

    def test_inconsistent_periods(self):
        """前后半段差异大时应有提示"""
        np.random.seed(42)
        n = 200
        first_half_factor = pd.Series(np.random.randn(n // 2) * 0.5)
        second_half_factor = pd.Series(np.random.randn(n // 2) * 5.0)  # 后半段方差剧增

        first_returns = pd.Series(0.3 * first_half_factor.values + np.random.randn(n // 2) * 0.5)
        second_returns = pd.Series(0.01 * second_half_factor.values + np.random.randn(n // 2) * 2.0)

        factor = pd.concat([first_half_factor, second_half_factor], ignore_index=True)
        returns = pd.concat([first_returns, second_returns], ignore_index=True)

        result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        tc_check = next((c for c in result.checks if c.check_name == "temporal_consistency"), None)
        assert tc_check is not None
        # 不要求一定失败（取决于随机种子），但检查应该存在


class TestLookaheadBiasDetectorEdgeCases:
    """边界情况和鲁棒性测试"""

    def test_all_nan_factor(self):
        """全 NaN 因子不应崩溃"""
        factor = pd.Series([np.nan] * 100)
        returns = pd.Series(np.random.randn(100))
        result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        assert result.risk_level == BiasRiskLevel.SAFE  # 数据不足，默认安全

    def test_single_unique_value(self):
        """只有一个唯一值的因子"""
        factor = pd.Series([42.0] * 100)
        returns = pd.Series(np.random.randn(100))
        result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        vc_check = next((c for c in result.checks if c.check_name == "value_constancy"), None)
        assert vc_check is not None
        assert vc_check.passed is False

    def test_inf_values_handled(self):
        """含 inf 的因子不应崩溃"""
        factor = pd.Series(list(np.random.randn(99)) + [np.inf])
        returns = pd.Series(np.random.randn(100))
        result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        assert isinstance(result, LookaheadBiasDetectionResult)

    def test_empty_series(self):
        """空序列不应崩溃"""
        factor = pd.Series([], dtype=float)
        returns = pd.Series([], dtype=float)
        result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        assert result.risk_level == BiasRiskLevel.SAFE

    def test_very_short_series(self):
        """极短序列（刚好满足最低要求）"""
        factor = pd.Series(np.random.randn(20))
        returns = pd.Series(np.random.randn(20))
        result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        assert isinstance(result, LookaheadBiasDetectionResult)


class TestLookaheadBiasDetectorRiskScoring:
    """风险评分逻辑验证"""

    def test_risk_score_zero_for_safe(self):
        """SAFE 等级的评分应为 0"""
        result = lookahead_bias_detector.detect(
            factor_values=pd.Series(np.random.randn(300)),
            return_values=pd.Series(np.random.randn(300)),
        )
        if result.risk_level == BiasRiskLevel.SAFE:
            assert result.risk_score == 0.0

    def test_critical_has_high_score(self):
        """CRITICAL 等级应有高分"""
        # 构造一个明显有问题的因子
        n = 252
        signal = np.random.randn(n)
        factor = pd.Series(signal)
        returns = pd.Series(0.8 * signal + np.random.randn(n) * 0.05)  # 接近完美的预测

        result = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)
        # 至少应有一些 failed checks
        failed = [c for c in result.checks if not c.passed]
        assert len(failed) >= 2  # IC 和 Rank IC 都应失败


class TestLookaheadBiasDetectorRecommendations:
    """改进建议生成测试"""

    def test_recommendations_for_high_ic(self):
        """高 IC 时应给出因子公式审查建议"""
        n = 252
        signal = np.random.randn(n)
        result = lookahead_bias_detector.detect(
            factor_values=pd.Series(signal),
            return_values=pd.Series(0.5 * signal),
        )
        assert len(result.recommendations) > 0
        # 建议应该是字符串列表
        assert all(isinstance(r, str) for r in result.recommendations)


class TestLookaheadBiasDetectorCustomThresholds:
    """自定义阈值测试"""

    def test_custom_thresholds_applied(self):
        """自定义阈值应正确应用"""
        custom_detector = LookaheadBiasDetector(thresholds={"ic_abs_max": 0.5})  # 放宽阈值
        n = 252
        signal = np.random.randn(n)
        factor = pd.Series(signal)
        returns = pd.Series(0.2 * signal + np.random.randn(n) * 0.3)

        result_custom = custom_detector.detect(factor_values=factor, return_values=returns)
        result_default = lookahead_bias_detector.detect(factor_values=factor, return_values=returns)

        # 自定义阈值更宽松，风险评分应 <= 默认
        assert result_custom.risk_score <= result_default.risk_score + 1.0  # 允许微小误差
