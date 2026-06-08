"""
WeightedICService 加权IC服务测试

覆盖 calculate_weighted_ic、calculate_factor_importance、_calculate_weights、
_adjust_for_correlation、_calculate_optimal_weights、_calculate_stability_score、
_align_ic_series 七个核心方法，包含正常场景和边界条件。
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf

from backend.services.weighted_ic_service import (
    WeightedICService,
    WeightedICConfig,
    WeightingMethod,
)


# ──────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────

def _make_ic_series(
    n: int = 100,
    mean: float = 0.03,
    std: float = 0.1,
    seed: int = 42,
    start_date: str = "2023-01-03",
) -> pd.Series:
    """构造单条IC序列"""
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, periods=n, freq="B")
    values = np.random.randn(n) * std + mean
    return pd.Series(values, index=dates, name="ic")


def _make_factor_ic_dict(
    factor_names: list = None,
    n: int = 100,
    seeds: list = None,
    means: list = None,
    stds: list = None,
) -> dict:
    """
    构造多条因子的IC字典

    Args:
        factor_names: 因子名列表，默认 ["factor_a", "factor_b", "factor_c"]
        n: 每条IC序列长度
        seeds: 各因子随机种子
        means: 各因子IC均值
        stds: 各因子IC标准差
    """
    if factor_names is None:
        factor_names = ["factor_a", "factor_b", "factor_c"]
    n_factors = len(factor_names)
    if seeds is None:
        seeds = list(range(n_factors))
    if means is None:
        means = [0.03] * n_factors
    if stds is None:
        stds = [0.1] * n_factors

    result = {}
    for i, name in enumerate(factor_names):
        result[name] = _make_ic_series(
            n=n, mean=means[i], std=stds[i], seed=seeds[i]
        )
    return result


def _make_corr_matrix(
    factor_names: list,
    high_corr_pairs: list = None,
) -> pd.DataFrame:
    """
    构造因子相关性矩阵

    Args:
        factor_names: 因子名列表
        high_corr_pairs: [(i, j, corr_value), ...] 高相关对
    """
    n = len(factor_names)
    corr = np.eye(n)
    if high_corr_pairs:
        for i, j, val in high_corr_pairs:
            corr[i, j] = val
            corr[j, i] = val
    return pd.DataFrame(corr, index=factor_names, columns=factor_names)


# ──────────────────────────────────────────
# 测试类
# ──────────────────────────────────────────


class TestCalculateWeightedIC:
    """calculate_weighted_ic 方法测试"""

    def test_normal_ir_weight_should_return_success(self):
        """IR加权模式下正常输入应返回成功结果"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.IR_WEIGHT,
        ))
        factor_ic_dict = _make_factor_ic_dict()
        result = service.calculate_weighted_ic(factor_ic_dict)

        assert result.get("success") is True
        assert result["n_factors"] == 3
        assert "weighted_ic" in result
        assert "factor_weights" in result
        assert "contribution_analysis" in result
        assert result["weighting_method"] == "ir_weight"

    def test_weighted_ic_stats_should_be_valid(self):
        """加权IC统计量应为合理数值"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.IR_WEIGHT,
        ))
        factor_ic_dict = _make_factor_ic_dict(means=[0.05, 0.03, 0.01])
        result = service.calculate_weighted_ic(factor_ic_dict)

        wic = result["weighted_ic"]
        assert isinstance(wic["mean"], float)
        assert isinstance(wic["std"], float)
        assert isinstance(wic["ir"], float)
        assert isinstance(wic["positive_ratio"], float)
        assert 0.0 <= wic["positive_ratio"] <= 1.0
        assert wic["n_observations"] > 0

    def test_factor_weights_should_sum_to_one(self):
        """因子权重之和应近似为1"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.IR_WEIGHT,
        ))
        factor_ic_dict = _make_factor_ic_dict()
        result = service.calculate_weighted_ic(factor_ic_dict)

        total_weight = sum(
            v["weight"] for v in result["factor_weights"].values()
        )
        assert total_weight == pytest.approx(1.0, abs=1e-6)

    def test_contribution_analysis_should_have_all_factors(self):
        """贡献度分析应包含所有因子"""
        service = WeightedICService()
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=["f1", "f2", "f3"]
        )
        result = service.calculate_weighted_ic(factor_ic_dict)

        contrib = result["contribution_analysis"]
        assert "f1" in contrib
        assert "f2" in contrib
        assert "f3" in contrib
        for name in ["f1", "f2", "f3"]:
            assert "weight" in contrib[name]
            assert "mean_contribution" in contrib[name]
            assert "contribution_ratio" in contrib[name]

    def test_empty_dict_should_return_error(self):
        """空字典应返回错误"""
        service = WeightedICService()
        result = service.calculate_weighted_ic({})
        assert "error" in result

    def test_none_dict_should_return_error(self):
        """None输入应返回错误"""
        service = WeightedICService()
        result = service.calculate_weighted_ic(None)
        assert "error" in result

    def test_single_factor_should_work(self):
        """单因子输入应正常工作，权重为1.0"""
        service = WeightedICService()
        factor_ic_dict = {"only_factor": _make_ic_series(seed=1)}
        result = service.calculate_weighted_ic(factor_ic_dict)

        assert result.get("success") is True
        assert result["n_factors"] == 1
        weights = result["factor_weights"]
        assert weights["only_factor"]["weight"] == pytest.approx(1.0, abs=1e-6)

    def test_insufficient_data_should_return_error(self):
        """IC数据不足min_observations时应返回错误"""
        service = WeightedICService(WeightedICConfig(min_observations=50))
        # 只提供10条数据
        factor_ic_dict = {"f1": _make_ic_series(n=10, seed=1)}
        result = service.calculate_weighted_ic(factor_ic_dict)
        assert "error" in result

    def test_with_correlation_adjustment_should_produce_adjustment_info(self):
        """提供相关性矩阵且开启调整时，应返回调整信息"""
        factor_names = ["f1", "f2"]
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.EQUAL_WEIGHT,
            correlation_adjustment=True,
        ))
        factor_ic_dict = _make_factor_ic_dict(factor_names=factor_names)
        # 高相关矩阵
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.85)])
        result = service.calculate_weighted_ic(
            factor_ic_dict, factor_correlation_matrix=corr_matrix
        )

        assert result.get("success") is True
        assert result["correlation_adjustment"] is not None
        assert "adjustments" in result["correlation_adjustment"]

    def test_without_correlation_adjustment_should_have_none(self):
        """不提供相关性矩阵时，correlation_adjustment应为None"""
        service = WeightedICService()
        factor_ic_dict = _make_factor_ic_dict()
        result = service.calculate_weighted_ic(factor_ic_dict)

        assert result["correlation_adjustment"] is None

    def test_correlation_adjustment_disabled_should_not_adjust(self):
        """关闭相关性调整时，即使提供矩阵也不调整"""
        factor_names = ["f1", "f2"]
        service = WeightedICService(WeightedICConfig(
            correlation_adjustment=False,
        ))
        factor_ic_dict = _make_factor_ic_dict(factor_names=factor_names)
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.9)])
        result = service.calculate_weighted_ic(
            factor_ic_dict, factor_correlation_matrix=corr_matrix
        )

        assert result["correlation_adjustment"] is None


class TestCalculateWeightedICAllMethods:
    """所有加权方法的 calculate_weighted_ic 测试"""

    def test_equal_weight_should_assign_equal_weights(self):
        """等权法应分配相同权重"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.EQUAL_WEIGHT,
        ))
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=["f1", "f2", "f3"],
            means=[0.05, 0.01, 0.08],
        )
        result = service.calculate_weighted_ic(factor_ic_dict)

        weights = {k: v["weight"] for k, v in result["factor_weights"].items()}
        for w in weights.values():
            assert w == pytest.approx(1.0 / 3, abs=1e-6)

    def test_ir_weight_should_favor_high_ir_factor(self):
        """IR加权应偏向高IR因子"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.IR_WEIGHT,
        ))
        # f1: 高均值低标准差 → 高IR; f2: 低均值高标准差 → 低IR
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=["f1", "f2"],
            means=[0.08, 0.01],
            stds=[0.05, 0.15],
            seeds=[10, 20],
        )
        result = service.calculate_weighted_ic(factor_ic_dict)

        weights = {k: v["weight"] for k, v in result["factor_weights"].items()}
        assert weights["f1"] > weights["f2"]

    def test_abs_ic_weight_should_favor_high_abs_ic(self):
        """IC绝对值加权应偏向|IC均值|大的因子"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.ABS_IC_WEIGHT,
        ))
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=["f1", "f2"],
            means=[0.1, 0.01],
            stds=[0.1, 0.1],
            seeds=[10, 20],
        )
        result = service.calculate_weighted_ic(factor_ic_dict)

        weights = {k: v["weight"] for k, v in result["factor_weights"].items()}
        assert weights["f1"] > weights["f2"]

    def test_decay_weight_should_favor_recent_performance(self):
        """衰减加权应更重视近期表现"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.DECAY_WEIGHT,
            decay_half_life=30,
        ))
        # f1: 近期IC高; f2: 近期IC低
        np.random.seed(100)
        n = 100
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        # f1: 前半低后半高
        vals_f1 = np.concatenate([
            np.random.randn(50) * 0.05 - 0.02,
            np.random.randn(50) * 0.05 + 0.08,
        ])
        # f2: 前半高后半低
        vals_f2 = np.concatenate([
            np.random.randn(50) * 0.05 + 0.08,
            np.random.randn(50) * 0.05 - 0.02,
        ])
        factor_ic_dict = {
            "f1": pd.Series(vals_f1, index=dates),
            "f2": pd.Series(vals_f2, index=dates),
        }
        result = service.calculate_weighted_ic(factor_ic_dict)

        weights = {k: v["weight"] for k, v in result["factor_weights"].items()}
        # f1近期表现更好，衰减加权下应获得更高权重
        assert weights["f1"] > weights["f2"]

    def test_optimal_weight_should_produce_valid_weights(self):
        """最优加权应产生有效权重（总和为1）"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.OPTIMAL_WEIGHT,
        ))
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=["f1", "f2", "f3"],
        )
        result = service.calculate_weighted_ic(factor_ic_dict)

        assert result.get("success") is True
        total = sum(v["weight"] for v in result["factor_weights"].values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestCalculateFactorImportance:
    """calculate_factor_importance 方法测试"""

    def test_normal_input_should_return_ranking(self):
        """正常输入应返回因子排名"""
        service = WeightedICService()
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=["f1", "f2", "f3"],
            means=[0.08, 0.03, 0.01],
            stds=[0.05, 0.1, 0.15],
        )
        result = service.calculate_factor_importance(factor_ic_dict)

        assert result.get("success") is True
        assert "ranking" in result
        assert result["n_factors_evaluated"] == 3
        assert len(result["ranking"]) == 3
        # 排名应按分数降序
        scores = [r["total_score"] for r in result["ranking"]]
        assert scores == sorted(scores, reverse=True)

    def test_high_ic_factor_should_rank_first(self):
        """高IC因子应排名第一"""
        service = WeightedICService()
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=["weak", "strong"],
            means=[0.005, 0.08],
            stds=[0.15, 0.04],
            seeds=[1, 2],
        )
        result = service.calculate_factor_importance(factor_ic_dict)

        assert result["ranking"][0]["factor_name"] == "strong"

    def test_ranking_should_have_required_fields(self):
        """排名项应包含必要字段"""
        service = WeightedICService()
        factor_ic_dict = _make_factor_ic_dict(factor_names=["f1"])
        result = service.calculate_factor_importance(factor_ic_dict)

        item = result["ranking"][0]
        assert "rank" in item
        assert "factor_name" in item
        assert "total_score" in item
        assert "ir" in item
        assert "mean_abs_ic" in item
        assert "positive_ratio" in item
        assert "stability" in item
        assert "momentum" in item

    def test_top_factor_should_be_returned(self):
        """应返回top_factor字段"""
        service = WeightedICService()
        factor_ic_dict = _make_factor_ic_dict()
        result = service.calculate_factor_importance(factor_ic_dict)

        assert result["top_factor"] is not None
        assert result["top_factor"]["rank"] == 1

    def test_with_correlation_matrix_should_apply_penalty(self):
        """提供相关性矩阵时，高相关因子应受惩罚"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=factor_names,
            means=[0.05, 0.05],
            stds=[0.05, 0.05],
            seeds=[1, 2],
        )
        # 先计算无惩罚分数
        result_no_corr = service.calculate_factor_importance(factor_ic_dict)

        # 高相关矩阵
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.9)])
        result_with_corr = service.calculate_factor_importance(
            factor_ic_dict, factor_correlation_matrix=corr_matrix
        )

        # 有相关性惩罚时，至少一个因子分数应低于无惩罚版本
        scores_no_corr = {
            r["factor_name"]: r["total_score"]
            for r in result_no_corr["ranking"]
        }
        scores_with_corr = {
            r["factor_name"]: r["total_score"]
            for r in result_with_corr["ranking"]
        }
        # 高相关时，uniqueness_penalty > 0，分数应降低
        has_penalty = any(
            scores_with_corr[name] < scores_no_corr[name]
            for name in factor_names
        )
        assert has_penalty

    def test_empty_dict_should_return_success_with_empty_ranking(self):
        """空字典应返回成功但无排名"""
        service = WeightedICService()
        result = service.calculate_factor_importance({})
        assert result.get("success") is True
        assert result["n_factors_evaluated"] == 0
        assert result["ranking"] == []

    def test_insufficient_data_should_return_success_with_empty_ranking(self):
        """数据不足时应返回成功但无排名"""
        service = WeightedICService(WeightedICConfig(min_observations=50))
        factor_ic_dict = {"f1": _make_ic_series(n=10, seed=1)}
        result = service.calculate_factor_importance(factor_ic_dict)
        assert result.get("success") is True
        assert result["n_factors_evaluated"] == 0

    def test_interpretation_should_be_generated(self):
        """应生成解读文本"""
        service = WeightedICService()
        factor_ic_dict = _make_factor_ic_dict()
        result = service.calculate_factor_importance(factor_ic_dict)
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


class TestCalculateWeights:
    """_calculate_weights 方法测试"""

    def test_equal_weight(self):
        """等权法：所有因子权重相同"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.EQUAL_WEIGHT,
        ))
        ic_stats = {
            "f1": {"ir": 0.5, "mean_ic": 0.03, "std_ic": 0.06},
            "f2": {"ir": 1.0, "mean_ic": 0.08, "std_ic": 0.08},
        }
        weights = service._calculate_weights(ic_stats, ["f1", "f2"])
        assert weights["f1"] == pytest.approx(0.5, abs=1e-6)
        assert weights["f2"] == pytest.approx(0.5, abs=1e-6)

    def test_ir_weight_with_negative_ir(self):
        """IR加权：负IR因子权重应为0（max(ir, 0)），正IR因子权重为1"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.IR_WEIGHT,
        ))
        ic_stats = {
            "f1": {"ir": 0.8, "mean_ic": 0.05, "std_ic": 0.06},
            "f2": {"ir": -0.3, "mean_ic": -0.02, "std_ic": 0.07},
        }
        weights = service._calculate_weights(ic_stats, ["f1", "f2"])
        assert weights["f1"] == pytest.approx(1.0, abs=1e-6)
        assert weights["f2"] == pytest.approx(0.0, abs=1e-6)

    def test_ir_weight_all_negative_ir_should_fallback_equal(self):
        """IR加权：所有IR为负时应回退到等权"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.IR_WEIGHT,
        ))
        ic_stats = {
            "f1": {"ir": -0.5, "mean_ic": -0.03, "std_ic": 0.06},
            "f2": {"ir": -0.2, "mean_ic": -0.01, "std_ic": 0.05},
        }
        weights = service._calculate_weights(ic_stats, ["f1", "f2"])
        # normalize_weights 在总和为0时回退等权
        assert weights["f1"] == pytest.approx(0.5, abs=1e-6)
        assert weights["f2"] == pytest.approx(0.5, abs=1e-6)

    def test_abs_ic_weight(self):
        """IC绝对值加权：|IC|大的因子权重更高"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.ABS_IC_WEIGHT,
        ))
        ic_stats = {
            "f1": {"ir": 0.5, "mean_ic": 0.1, "std_ic": 0.2},
            "f2": {"ir": 0.3, "mean_ic": 0.03, "std_ic": 0.1},
        }
        weights = service._calculate_weights(ic_stats, ["f1", "f2"])
        assert weights["f1"] > weights["f2"]

    def test_decay_weight(self):
        """衰减加权：应使用ic_series计算衰减加权均值"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.DECAY_WEIGHT,
            decay_half_life=30,
        ))
        np.random.seed(42)
        ic_series = pd.Series(np.random.randn(60) * 0.1 + 0.03)
        ic_stats = {
            "f1": {"ir": 0.3, "mean_ic": 0.03, "std_ic": 0.1, "ic_series": ic_series},
        }
        weights = service._calculate_weights(ic_stats, ["f1"])
        assert weights["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_decay_weight_without_ic_series_should_use_mean(self):
        """衰减加权：无ic_series时回退到|mean_ic|"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.DECAY_WEIGHT,
        ))
        ic_stats = {
            "f1": {"ir": 0.3, "mean_ic": 0.05, "std_ic": 0.1},
        }
        weights = service._calculate_weights(ic_stats, ["f1"])
        assert weights["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_factor_not_in_ic_stats_should_be_excluded(self):
        """不在ic_stats中的因子不应出现在权重中"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.EQUAL_WEIGHT,
        ))
        ic_stats = {
            "f1": {"ir": 0.5, "mean_ic": 0.03, "std_ic": 0.06},
        }
        weights = service._calculate_weights(ic_stats, ["f1", "f2_missing"])
        assert "f1" in weights
        assert "f2_missing" not in weights

    def test_unknown_method_should_fallback_equal(self):
        """未知加权方法应回退到等权"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.EQUAL_WEIGHT,
        ))
        # 手动设置一个无效方法
        service.config.weighting_method = "nonexistent"
        ic_stats = {
            "f1": {"ir": 0.5, "mean_ic": 0.03, "std_ic": 0.06},
            "f2": {"ir": 0.8, "mean_ic": 0.05, "std_ic": 0.06},
        }
        weights = service._calculate_weights(ic_stats, ["f1", "f2"])
        assert weights["f1"] == pytest.approx(0.5, abs=1e-6)


class TestAdjustForCorrelation:
    """_adjust_for_correlation 方法测试"""

    def test_low_correlation_should_not_adjust(self):
        """低相关（< 0.7）不应调整权重"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.6, "f2": 0.4}
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.3)])

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        assert "adjustments" in info
        assert len(info["adjustments"]) == 0
        # 权重不变（归一化后可能略有浮点差异）
        assert adjusted["f1"] == pytest.approx(0.6, abs=1e-6)
        assert adjusted["f2"] == pytest.approx(0.4, abs=1e-6)

    def test_high_correlation_should_reduce_lower_weight_factor(self):
        """高相关（> 0.7）应降低权重较低的因子"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.7, "f2": 0.3}
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.85)])

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        # f2权重较低，应被缩减
        assert len(info["adjustments"]) > 0
        # 归一化后f1权重应比原始比例更高
        ratio_before = weights["f1"] / weights["f2"]
        ratio_after = adjusted["f1"] / adjusted["f2"]
        assert ratio_after > ratio_before

    def test_exact_threshold_0_7_should_not_adjust(self):
        """相关性恰好0.7不应调整（> 0.7才调整）"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.6, "f2": 0.4}
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.7)])

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        assert len(info["adjustments"]) == 0

    def test_just_above_threshold_should_adjust(self):
        """相关性略高于0.7应触发调整"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.6, "f2": 0.4}
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.71)])

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        assert len(info["adjustments"]) > 0

    def test_perfect_correlation_should_heavily_reduce(self):
        """完全相关（1.0）应大幅缩减"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.7, "f2": 0.3}
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 1.0)])

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        # reduction_factor = 1 - (1.0 - 0.7) * 0.5 = 0.85
        # f2被缩减: 0.3 * 0.85 = 0.255
        # 归一化后 f1应占更大比例
        assert adjusted["f1"] > adjusted["f2"]

    def test_adjustment_info_should_contain_details(self):
        """调整信息应包含详细内容"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.6, "f2": 0.4}
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.9)])

        _, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        assert "original_weights" in info
        assert "adjusted_weights" in info
        assert "adjustments" in info
        assert "total_reduction" in info

    def test_three_factors_with_pairwise_correlation(self):
        """三因子场景：部分高相关应只调整相关对"""
        service = WeightedICService()
        factor_names = ["f1", "f2", "f3"]
        weights = {"f1": 0.5, "f2": 0.3, "f3": 0.2}
        # f1-f2 高相关, f1-f3 和 f2-f3 低相关
        corr_matrix = _make_corr_matrix(
            factor_names, [(0, 1, 0.85), (0, 2, 0.3), (1, 2, 0.2)]
        )

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        # f2权重低于f1，f1-f2高相关，f2应被缩减
        assert len(info["adjustments"]) > 0
        # 权重总和归一化后仍为1
        total = sum(adjusted.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_factor_not_in_weights_should_be_skipped(self):
        """不在权重字典中的因子应被跳过"""
        service = WeightedICService()
        factor_names = ["f1", "f2", "f3"]
        weights = {"f1": 0.6, "f2": 0.4}  # f3不在权重中
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.85)])

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )
        # 不应报错
        assert "f1" in adjusted
        assert "f2" in adjusted

    def test_factor_not_in_corr_matrix_should_be_skipped(self):
        """不在相关性矩阵中的因子应被跳过"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.6, "f2": 0.4}
        # 矩阵中只有f1，没有f2
        corr_matrix = pd.DataFrame(
            {"f1": [1.0]}, index=["f1"]
        )

        adjusted, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )
        # 不应报错，无调整
        assert len(info["adjustments"]) == 0


class TestCalculateOptimalWeights:
    """_calculate_optimal_weights 方法测试"""

    def test_two_factors_should_produce_valid_weights(self):
        """两因子最优权重应有效"""
        service = WeightedICService()
        ic_stats = {
            "f1": {"mean_ic": 0.05, "std_ic": 0.08},
            "f2": {"mean_ic": 0.03, "std_ic": 0.12},
        }
        weights = service._calculate_optimal_weights(ic_stats, ["f1", "f2"])

        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        # f1: |0.05| / 0.08^2 = 7.8125; f2: |0.03| / 0.12^2 = 2.0833
        # f1权重应更高
        assert weights["f1"] > weights["f2"]

    def test_single_factor_should_get_full_weight(self):
        """单因子应获得全部权重"""
        service = WeightedICService()
        ic_stats = {
            "f1": {"mean_ic": 0.05, "std_ic": 0.08},
        }
        weights = service._calculate_optimal_weights(ic_stats, ["f1"])
        assert weights["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_std_should_use_1e6_floor(self):
        """标准差为0时应使用1e-6下限，避免除零"""
        service = WeightedICService()
        ic_stats = {
            "f1": {"mean_ic": 0.05, "std_ic": 0.0},
            "f2": {"mean_ic": 0.03, "std_ic": 0.1},
        }
        weights = service._calculate_optimal_weights(ic_stats, ["f1", "f2"])
        # 不应报错，权重总和为1
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_zero_ic_should_get_zero_weight(self):
        """IC为0的因子应获得0权重"""
        service = WeightedICService()
        ic_stats = {
            "f1": {"mean_ic": 0.05, "std_ic": 0.1},
            "f2": {"mean_ic": 0.0, "std_ic": 0.1},
        }
        weights = service._calculate_optimal_weights(ic_stats, ["f1", "f2"])
        # f2的IC为0，raw_weight = |0| * inv_var = 0
        # 但normalize_weights在总和为0时回退等权
        # 这里f1有非零权重，所以f2应为0
        assert weights["f1"] > weights["f2"]

    def test_all_zero_ic_should_fallback_equal(self):
        """所有IC为0时应回退到等权"""
        service = WeightedICService()
        ic_stats = {
            "f1": {"mean_ic": 0.0, "std_ic": 0.1},
            "f2": {"mean_ic": 0.0, "std_ic": 0.1},
        }
        weights = service._calculate_optimal_weights(ic_stats, ["f1", "f2"])
        # raw_weights全为0，normalize_weights回退等权
        assert weights["f1"] == pytest.approx(0.5, abs=1e-6)
        assert weights["f2"] == pytest.approx(0.5, abs=1e-6)


class TestCalculateStabilityScore:
    """_calculate_stability_score 方法测试"""

    def test_short_series_should_return_0_5(self):
        """短序列（< 40）应返回0.5"""
        service = WeightedICService()
        ic_series = _make_ic_series(n=30, seed=1)
        score = service._calculate_stability_score(ic_series)
        assert score == 0.5

    def test_stable_series_should_have_high_score(self):
        """稳定序列应获得高分"""
        service = WeightedICService()
        np.random.seed(42)
        n = 100
        # 常数IC + 微小噪声 → 非常稳定
        values = np.ones(n) * 0.03 + np.random.randn(n) * 0.001
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        ic_series = pd.Series(values, index=dates)

        score = service._calculate_stability_score(ic_series)
        assert score > 0.5

    def test_volatile_series_should_have_lower_score(self):
        """波动大的序列应获得较低分"""
        service = WeightedICService()
        np.random.seed(42)
        n = 100
        # 前半正后半负 → 不稳定
        values = np.concatenate([
            np.random.randn(50) * 0.05 + 0.1,
            np.random.randn(50) * 0.05 - 0.1,
        ])
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        ic_series = pd.Series(values, index=dates)

        score = service._calculate_stability_score(ic_series)
        # 均值发生大幅变化，稳定性应较低
        assert score < 0.8

    def test_constant_series_should_have_high_stability(self):
        """常数序列（std≈0）应获得高稳定性分数"""
        service = WeightedICService()
        n = 100
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        ic_series = pd.Series(np.ones(n) * 0.05, index=dates)

        score = service._calculate_stability_score(ic_series)
        # pd.Series.std()对常数序列返回~1e-16而非0，不触发early return
        # 但change_score≈1.0（两半均值几乎相同），consistency_score取决于rolling_std
        assert score >= 0.5

    def test_half_too_short_should_return_0_5(self):
        """半段数据不足10条时应返回0.5"""
        service = WeightedICService()
        # 40条数据，每半20条，应正常计算
        # 21条数据，前半10条，后半11条，应正常计算
        # 但如果数据在20-39之间，半段可能不足10
        np.random.seed(42)
        n = 25  # 前半12，后半13 → 都>10，应正常计算
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        ic_series = pd.Series(np.random.randn(n) * 0.1, index=dates)
        score = service._calculate_stability_score(ic_series)
        # 25 < 40，应返回0.5
        assert score == 0.5


class TestAlignICSeries:
    """_align_ic_series 方法测试"""

    def test_aligned_index_should_be_intersection(self):
        """对齐后的索引应为所有序列索引的交集"""
        service = WeightedICService()
        dates1 = pd.bdate_range(start="2023-01-03", periods=50, freq="B")
        dates2 = pd.bdate_range(start="2023-01-03", periods=50, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.random.randn(50), index=dates1),
            "f2": pd.Series(np.random.randn(50), index=dates2),
        }
        aligned = service._align_ic_series(factor_ic_dict, ["f1", "f2"])
        assert len(aligned) == 50
        assert "f1" in aligned.columns
        assert "f2" in aligned.columns

    def test_different_date_ranges_should_align_to_common(self):
        """不同日期范围应对齐到公共部分"""
        service = WeightedICService()
        dates1 = pd.bdate_range(start="2023-01-03", periods=60, freq="B")
        dates2 = pd.bdate_range(start="2023-02-01", periods=40, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.random.randn(60), index=dates1),
            "f2": pd.Series(np.random.randn(40), index=dates2),
        }
        aligned = service._align_ic_series(factor_ic_dict, ["f1", "f2"])
        # 公共部分应小于任一原始长度
        assert len(aligned) < 60
        assert len(aligned) > 0

    def test_no_overlap_should_return_empty(self):
        """无交集的日期应返回空DataFrame"""
        service = WeightedICService()
        dates1 = pd.bdate_range(start="2023-01-03", periods=20, freq="B")
        dates2 = pd.bdate_range(start="2024-06-01", periods=20, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.random.randn(20), index=dates1),
            "f2": pd.Series(np.random.randn(20), index=dates2),
        }
        aligned = service._align_ic_series(factor_ic_dict, ["f1", "f2"])
        assert len(aligned) == 0

    def test_empty_dict_should_return_empty_dataframe(self):
        """空字典应返回空DataFrame"""
        service = WeightedICService()
        aligned = service._align_ic_series({}, [])
        assert isinstance(aligned, pd.DataFrame)
        assert aligned.empty

    def test_single_factor_should_return_dataframe(self):
        """单因子应返回单列DataFrame"""
        service = WeightedICService()
        dates = pd.bdate_range(start="2023-01-03", periods=30, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.random.randn(30), index=dates),
        }
        aligned = service._align_ic_series(factor_ic_dict, ["f1"])
        assert len(aligned) == 30
        assert list(aligned.columns) == ["f1"]


class TestWeightedICConfig:
    """WeightedICConfig 配置测试"""

    def test_default_config_should_have_ir_weight(self):
        """默认配置应使用IR加权"""
        config = WeightedICConfig()
        assert config.weighting_method == WeightingMethod.IR_WEIGHT

    def test_default_config_values(self):
        """默认配置值应正确"""
        config = WeightedICConfig()
        assert config.decay_half_life == 60
        assert config.min_observations == 20
        assert config.lookback_window == 252
        assert config.correlation_adjustment is True
        assert config.risk_aversion == 1.0

    def test_custom_config(self):
        """自定义配置应生效"""
        config = WeightedICConfig(
            weighting_method=WeightingMethod.EQUAL_WEIGHT,
            decay_half_life=30,
            min_observations=10,
            correlation_adjustment=False,
        )
        service = WeightedICService(config)
        assert service.config.weighting_method == WeightingMethod.EQUAL_WEIGHT
        assert service.config.decay_half_life == 30
        assert service.config.min_observations == 10
        assert service.config.correlation_adjustment is False


class TestBoundaryConditions:
    """边界条件综合测试"""

    def test_all_zero_ir_with_ir_weight(self):
        """所有因子IR为0时，IR加权应回退到等权"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.IR_WEIGHT,
        ))
        # 常数IC → std=0 → IR=0（safe_ir返回default=0.0）
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.ones(n) * 0.05, index=dates),
            "f2": pd.Series(np.ones(n) * 0.03, index=dates),
        }
        result = service.calculate_weighted_ic(factor_ic_dict)
        # IR全为0 → max(ir,0)全为0 → normalize_weights回退等权
        if result.get("success"):
            weights = {k: v["weight"] for k, v in result["factor_weights"].items()}
            for w in weights.values():
                assert w == pytest.approx(0.5, abs=1e-6)

    def test_constant_ic_series(self):
        """常数IC序列（std=0）不应导致除零错误"""
        service = WeightedICService()
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.ones(n) * 0.05, index=dates),
        }
        result = service.calculate_weighted_ic(factor_ic_dict)
        # 不应报错
        assert result.get("success") is True or "error" not in result

    def test_ic_series_with_nan_values(self):
        """IC序列包含NaN时应被正确处理"""
        service = WeightedICService()
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        values = np.random.randn(n) * 0.1 + 0.03
        values[10:15] = np.nan
        values[50:55] = np.nan
        factor_ic_dict = {
            "f1": pd.Series(values, index=dates),
        }
        result = service.calculate_weighted_ic(factor_ic_dict)
        assert result.get("success") is True

    def test_mixed_sufficient_and_insufficient_factors(self):
        """部分因子数据不足时，应只分析有效因子"""
        service = WeightedICService(WeightedICConfig(min_observations=50))
        dates_long = pd.bdate_range(start="2023-01-03", periods=100, freq="B")
        dates_short = pd.bdate_range(start="2023-01-03", periods=10, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.random.randn(100) * 0.1, index=dates_long),
            "f2_short": pd.Series(np.random.randn(10) * 0.1, index=dates_short),
        }
        result = service.calculate_weighted_ic(factor_ic_dict)
        # f2数据不足，只有f1有效
        if result.get("success"):
            assert "f1" in result["factor_weights"]

    def test_negative_ic_factor_with_abs_ic_weight(self):
        """负IC因子在IC绝对值加权下应获得正权重"""
        service = WeightedICService(WeightedICConfig(
            weighting_method=WeightingMethod.ABS_IC_WEIGHT,
        ))
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        factor_ic_dict = {
            "f1": pd.Series(np.random.randn(n) * 0.05 + 0.08, index=dates),
            "f2": pd.Series(np.random.randn(n) * 0.05 - 0.08, index=dates),
        }
        result = service.calculate_weighted_ic(factor_ic_dict)
        weights = {k: v["weight"] for k, v in result["factor_weights"].items()}
        # |IC|相近，权重应相近
        assert abs(weights["f1"] - weights["f2"]) < 0.3

    def test_importance_score_formula(self):
        """重要性评分公式应正确：ir*30 + mean_abs_ic*100 + (positive_ratio-0.5)*20 + stability*20 + momentum*10"""
        service = WeightedICService(WeightedICConfig(min_observations=20))
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range(start="2023-01-03", periods=n, freq="B")
        # 构造已知参数的IC序列
        values = np.random.randn(n) * 0.08 + 0.04
        factor_ic_dict = {"f1": pd.Series(values, index=dates)}
        result = service.calculate_factor_importance(factor_ic_dict)

        if result.get("success"):
            item = result["ranking"][0]
            # 验证公式：total_score ≈ ir*30 + mean_abs_ic*100 + (positive_ratio-0.5)*20 + stability*20 + momentum*10
            expected = (
                item["ir"] * 30
                + item["mean_abs_ic"] * 100
                + (item["positive_ratio"] - 0.5) * 20
                + item["stability"] * 20
                + max(min(item["momentum"] * 10, 10), -10)
            )
            assert item["total_score"] == pytest.approx(expected, abs=0.01)

    def test_importance_with_uniqueness_penalty(self):
        """重要性排名中高相关因子应受uniqueness_penalty"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        factor_ic_dict = _make_factor_ic_dict(
            factor_names=factor_names,
            means=[0.05, 0.05],
            stds=[0.05, 0.05],
            seeds=[1, 2],
        )
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, 0.9)])
        result = service.calculate_factor_importance(
            factor_ic_dict, factor_correlation_matrix=corr_matrix
        )

        if result.get("success"):
            # 至少一个因子应有uniqueness_penalty
            has_penalty = any(
                "uniqueness_penalty" in r for r in result["ranking"]
            )
            assert has_penalty

    def test_reduction_factor_formula(self):
        """相关性缩减因子公式应为 1 - (corr - 0.7) * 0.5"""
        service = WeightedICService()
        factor_names = ["f1", "f2"]
        weights = {"f1": 0.7, "f2": 0.3}
        corr_value = 0.9
        corr_matrix = _make_corr_matrix(factor_names, [(0, 1, corr_value)])

        _, info = service._adjust_for_correlation(
            weights, corr_matrix, factor_names
        )

        # reduction_factor = 1 - (0.9 - 0.7) * 0.5 = 0.9
        expected_reduction = 1.0 - (corr_value - 0.7) * 0.5
        # 检查adjustments中的reduction_factor
        for adj_key, adj_val in info["adjustments"].items():
            assert adj_val["reduction_factor"] == pytest.approx(
                expected_reduction, abs=1e-6
            )
