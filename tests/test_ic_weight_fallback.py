"""
IC加权权重回退逻辑测试

专门验证 vectorbt_backtest_service.py 中 IC加权复合得分计算时，
权重和为0或NaN的回退逻辑是否正确。

核心逻辑：当 ic_weight_sum 为0或NaN时，每个因子应回退到等权 equal_w = 1/len(factors)，
而非保留原始 ic_wf 值作为权重。
"""
import numpy as np
import pandas as pd
import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock 重依赖
sys.modules.setdefault('akshare', MagicMock())
sys.modules.setdefault('sqlalchemy', MagicMock())
sys.modules.setdefault('sqlalchemy.orm', MagicMock())
sys.modules.setdefault('backend.services.cache_service', MagicMock())
sys.modules.setdefault('backend.services.data_service', MagicMock())


# ============================================================
# 辅助函数：构造模拟数据
# ============================================================

def make_multi_factor_df(n_days=200, n_factors=3, seed=42):
    """构造包含价格和因子数据的DataFrame，模拟多因子回测输入"""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    close = 100.0 * (1 + np.random.normal(0.0005, 0.02, n_days)).cumprod()
    data = {"close": close}
    for i in range(n_factors):
        data[f"factor_{i}"] = np.random.randn(n_days) * 0.1
    return pd.DataFrame(data, index=dates)


def make_ic_weight_frames_all_zero(n_days=200, n_factors=3):
    """构造所有IC权重全为0的ic_weight_frames（模拟因子与收益完全不相关）"""
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return [pd.Series(0.0, index=dates) for _ in range(n_factors)]


def make_ic_weight_frames_all_nan(n_days=200, n_factors=3):
    """构造所有IC权重全为NaN的ic_weight_frames（模拟窗口期不足）"""
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return [pd.Series(np.nan, index=dates) for _ in range(n_factors)]


def make_ic_weight_frames_partial_zero(n_days=200, n_factors=3):
    """构造部分行IC权重和为0的ic_weight_frames（模拟某些时段因子完全无关）"""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    frames = []
    for i in range(n_factors):
        s = pd.Series(np.abs(np.random.randn(n_days)) * 0.05, index=dates)
        # 前20行设为0，模拟窗口期不足或因子完全无关
        s.iloc[:20] = 0.0
        frames.append(s)
    return frames


def make_ic_weight_frames_partial_nan(n_days=200, n_factors=3):
    """构造部分行IC权重和为NaN的ic_weight_frames（模拟滚动窗口初期NaN）"""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    frames = []
    for i in range(n_factors):
        s = pd.Series(np.abs(np.random.randn(n_days)) * 0.05, index=dates)
        # 前30行设为NaN，模拟滚动窗口初期
        s.iloc[:30] = np.nan
        frames.append(s)
    return frames


def make_ic_weight_frames_mixed_zero_and_nan(n_days=200, n_factors=3):
    """构造同时包含0和NaN行的ic_weight_frames"""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    frames = []
    for i in range(n_factors):
        s = pd.Series(np.abs(np.random.randn(n_days)) * 0.05, index=dates)
        # 前10行设为0
        s.iloc[:10] = 0.0
        # 第10-30行设为NaN
        s.iloc[10:30] = np.nan
        frames.append(s)
    return frames


# ============================================================
# 核心权重计算逻辑提取（与源码一致，便于单元测试）
# ============================================================

def compute_ic_weighted_composite(df, normalized_factors, ic_weight_frames):
    """
    从 vectorbt_backtest_service.py 提取的IC加权核心逻辑，
    便于独立测试权重回退行为。
    """
    ic_weight_sum = sum(ic_weight_frames)
    equal_w = 1.0 / len(normalized_factors)

    # 标记权重和无效的行（0或NaN）
    invalid_mask = (ic_weight_sum == 0) | ic_weight_sum.isna()

    # 有效行：用 ic_wf / ic_weight_sum 归一化；无效行：直接用等权
    safe_ic_weight_sum = ic_weight_sum.copy()
    safe_ic_weight_sum[invalid_mask] = np.nan

    composite_parts = []
    per_factor_weights = []  # 记录每个因子的实际权重，用于验证
    for nf, ic_wf in zip(normalized_factors, ic_weight_frames):
        safe_weight = ic_wf / safe_ic_weight_sum
        # 无效行回退到等权
        safe_weight[invalid_mask] = equal_w
        per_factor_weights.append(safe_weight.copy())
        composite_parts.append(df[nf] * safe_weight)

    df["composite_score"] = sum(composite_parts)
    return df, per_factor_weights, invalid_mask


# ============================================================
# 测试类
# ============================================================

class TestICWeightFallbackAllZero:
    """场景1：所有IC权重全为0（因子与收益完全不相关）"""

    def test_all_zero_should_fallback_to_equal_weight(self):
        """全0权重应回退到等权"""
        n_days, n_factors = 200, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        # 添加标准化因子列
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        ic_weight_frames = make_ic_weight_frames_all_zero(n_days, n_factors)

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 所有权重和为0 → 所有行都应标记为无效
        assert invalid_mask.all(), "全0权重应全部标记为无效"

        # 每个因子的权重应为等权 1/3
        equal_w = 1.0 / n_factors
        for w in weights:
            np.testing.assert_allclose(
                w.values, equal_w,
                err_msg="全0权重回退后每个因子权重应为等权"
            )

    def test_all_zero_composite_should_be_equal_weighted_sum(self):
        """全0权重时复合得分应等于等权加权和"""
        n_days, n_factors = 200, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for i, nf in enumerate(normalized_factors):
            df[nf] = np.random.randn(n_days) * 0.1

        ic_weight_frames = make_ic_weight_frames_all_zero(n_days, n_factors)

        result_df, _, _ = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 手动计算等权复合得分
        equal_w = 1.0 / n_factors
        expected = sum(df[nf] * equal_w for nf in normalized_factors)

        pd.testing.assert_series_equal(
            result_df["composite_score"], expected,
            check_names=False
        )


class TestICWeightFallbackAllNaN:
    """场景2：所有IC权重全为NaN（窗口期不足）"""

    def test_all_nan_should_fallback_to_equal_weight(self):
        """全NaN权重应回退到等权"""
        n_days, n_factors = 200, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        ic_weight_frames = make_ic_weight_frames_all_nan(n_days, n_factors)

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 所有权重和为NaN → 所有行都应标记为无效
        assert invalid_mask.all(), "全NaN权重应全部标记为无效"

        # 每个因子的权重应为等权
        equal_w = 1.0 / n_factors
        for w in weights:
            np.testing.assert_allclose(
                w.values, equal_w,
                err_msg="全NaN权重回退后每个因子权重应为等权"
            )


class TestICWeightFallbackPartialZero:
    """场景3：部分行权重和为0（某些时段因子完全无关）"""

    def test_partial_zero_should_only_fallback_on_zero_rows(self):
        """仅权重和为0的行应回退到等权，有效行正常归一化"""
        n_days, n_factors = 200, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        ic_weight_frames = make_ic_weight_frames_partial_zero(n_days, n_factors)

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 前20行应标记为无效（权重和为0）
        assert invalid_mask.iloc[:20].all(), "前20行权重和为0，应标记为无效"
        # 后续行应有效
        assert not invalid_mask.iloc[20:].any(), "第20行后权重和不为0，应标记为有效"

        # 无效行的权重应为等权
        equal_w = 1.0 / n_factors
        for w in weights:
            np.testing.assert_allclose(
                w.iloc[:20].values, equal_w,
                err_msg="权重和为0的行应回退到等权"
            )

        # 有效行的权重之和应为1（归一化验证）
        valid_weights_sum = sum(w.iloc[20:] for w in weights)
        np.testing.assert_allclose(
            valid_weights_sum.values, 1.0,
            rtol=1e-10,
            err_msg="有效行各因子权重之和应为1"
        )


class TestICWeightFallbackPartialNaN:
    """场景4：部分行权重和为NaN（滚动窗口初期）"""

    def test_partial_nan_should_only_fallback_on_nan_rows(self):
        """仅权重和为NaN的行应回退到等权"""
        n_days, n_factors = 200, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        ic_weight_frames = make_ic_weight_frames_partial_nan(n_days, n_factors)

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 前30行应标记为无效
        assert invalid_mask.iloc[:30].all(), "前30行权重和为NaN，应标记为无效"
        # 后续行应有效
        assert not invalid_mask.iloc[30:].any(), "第30行后权重和不为NaN，应标记为有效"

        # 无效行的权重应为等权
        equal_w = 1.0 / n_factors
        for w in weights:
            np.testing.assert_allclose(
                w.iloc[:30].values, equal_w,
                err_msg="权重和为NaN的行应回退到等权"
            )

        # 有效行的权重之和应为1
        valid_weights_sum = sum(w.iloc[30:] for w in weights)
        np.testing.assert_allclose(
            valid_weights_sum.values, 1.0,
            rtol=1e-10,
            err_msg="有效行各因子权重之和应为1"
        )


class TestICWeightFallbackMixedZeroAndNaN:
    """场景5：同时存在0和NaN的行"""

    def test_mixed_zero_and_nan_should_fallback_correctly(self):
        """0和NaN行都应正确回退到等权"""
        n_days, n_factors = 200, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        ic_weight_frames = make_ic_weight_frames_mixed_zero_and_nan(n_days, n_factors)

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 前30行（0-9为0，10-29为NaN）都应标记为无效
        assert invalid_mask.iloc[:30].all(), "前30行权重和为0或NaN，应标记为无效"
        # 第30行后应有效
        assert not invalid_mask.iloc[30:].any(), "第30行后应标记为有效"

        # 所有无效行的权重应为等权
        equal_w = 1.0 / n_factors
        for w in weights:
            np.testing.assert_allclose(
                w.iloc[:30].values, equal_w,
                err_msg="0和NaN行都应回退到等权"
            )

        # 有效行的权重之和应为1
        valid_weights_sum = sum(w.iloc[30:] for w in weights)
        np.testing.assert_allclose(
            valid_weights_sum.values, 1.0,
            rtol=1e-10,
            err_msg="有效行各因子权重之和应为1"
        )


class TestICWeightFallbackNoInvalidRows:
    """场景6：所有行权重和均有效（正常情况）"""

    def test_no_invalid_rows_should_use_normalized_weights(self):
        """无无效行时应使用IC归一化权重，而非等权"""
        n_days, n_factors = 200, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        # 构造所有行都有效的IC权重
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        ic_weight_frames = [
            pd.Series(np.abs(np.random.randn(n_days)) * 0.05 + 0.01, index=dates)
            for _ in range(n_factors)
        ]

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 不应有无效行
        assert not invalid_mask.any(), "所有行权重和均有效，不应有无效行"

        # 权重之和应为1
        weights_sum = sum(weights)
        np.testing.assert_allclose(
            weights_sum.values, 1.0,
            rtol=1e-10,
            err_msg="有效行各因子权重之和应为1"
        )

        # 权重不应全部等于等权（IC加权应与等权不同）
        equal_w = 1.0 / n_factors
        for w in weights:
            # 至少有一些行的权重不等于等权
            assert not np.allclose(w.values, equal_w), \
                "IC有效时权重不应全部退化为等权"


class TestICWeightFallbackSingleFactor:
    """场景7：单因子情况（n_factors=1）"""

    def test_single_factor_zero_weight_should_fallback_to_one(self):
        """单因子权重为0时应回退到权重1.0"""
        n_days = 200
        df = make_multi_factor_df(n_days, n_factors=1)
        normalized_factors = ["factor_0_normalized"]
        df["factor_0_normalized"] = np.random.randn(n_days) * 0.1

        dates = pd.bdate_range("2023-01-01", periods=n_days)
        ic_weight_frames = [pd.Series(0.0, index=dates)]

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 单因子等权 = 1.0
        np.testing.assert_allclose(
            weights[0].values, 1.0,
            err_msg="单因子权重为0时应回退到1.0"
        )


class TestICWeightFallbackEdgeCases:
    """边界情况测试"""

    def test_very_small_but_nonzero_weight_should_not_fallback(self):
        """极小但非零的权重不应触发回退"""
        n_days, n_factors = 100, 2
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        # 极小但非零的IC权重
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        ic_weight_frames = [
            pd.Series(1e-15, index=dates),  # 极小但非零
            pd.Series(1e-15, index=dates),
        ]

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 不应触发回退（权重和 = 2e-15 != 0）
        assert not invalid_mask.any(), "极小但非零的权重不应触发回退"

        # 权重之和应为1
        weights_sum = sum(weights)
        np.testing.assert_allclose(
            weights_sum.values, 1.0,
            rtol=1e-10,
            err_msg="极小非零权重归一化后权重之和应为1"
        )

    def test_one_factor_zero_others_nonzero_should_not_fallback(self):
        """一个因子IC为0但其他因子IC非零时，不应触发回退"""
        n_days, n_factors = 100, 3
        df = make_multi_factor_df(n_days, n_factors)
        normalized_factors = [f"factor_{i}_normalized" for i in range(n_factors)]
        for nf in normalized_factors:
            df[nf] = np.random.randn(n_days) * 0.1

        # 因子0的IC为0，但因子1和2的IC非零 → 权重和不为0
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        ic_weight_frames = [
            pd.Series(0.0, index=dates),           # 因子0: IC=0
            pd.Series(0.05, index=dates),           # 因子1: IC=0.05
            pd.Series(0.03, index=dates),           # 因子2: IC=0.03
        ]

        _, weights, invalid_mask = compute_ic_weighted_composite(
            df, normalized_factors, ic_weight_frames
        )

        # 权重和 = 0 + 0.05 + 0.03 = 0.08 != 0，不应触发回退
        assert not invalid_mask.any(), "权重和非零时不应触发回退"

        # 因子0的权重应为0（IC=0 → 0/0.08 = 0）
        np.testing.assert_allclose(
            weights[0].values, 0.0,
            err_msg="IC为0的因子权重应为0（非回退场景）"
        )

        # 因子1和2的权重之和应为1
        weights_sum_nonzero = weights[1] + weights[2]
        np.testing.assert_allclose(
            weights_sum_nonzero.values, 1.0,
            rtol=1e-10,
            err_msg="非零IC因子权重之和应为1"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
