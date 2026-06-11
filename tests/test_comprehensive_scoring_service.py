"""
comprehensive_scoring_service.py 综合评分服务单元测试

验证 ComprehensiveScoringService 的因子评分、策略评分、组合评分、
排名比较、评级映射、滑点敏感性分析等功能正确性。
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

from backend.services.comprehensive_scoring_service import ComprehensiveScoringService  # noqa: E402


class TestScoreFactor:
    """因子评分测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_score_factor_with_default_weights_should_return_all_fields(self):
        """默认权重下因子评分应返回完整字段"""
        metrics = {"ic_mean": 0.05, "ir": 1.2, "stability_score": 0.8, "turnover": 0.3}
        result = self.service.score_factor(metrics)

        assert "total_score" in result
        assert "grade" in result
        assert "details" in result
        assert "weights" in result
        assert isinstance(result["total_score"], float)
        assert isinstance(result["grade"], str)

    def test_score_factor_ic_score_calculation(self):
        """IC得分 = min(abs(ic_mean)*400, 100)"""
        # ic_mean=0.25 → 0.25*400=100 → 满分
        metrics = {"ic_mean": 0.25, "ir": 0, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["ic_score"] == 100.0

        # ic_mean=0.05 → 0.05*400=20
        metrics = {"ic_mean": 0.05, "ir": 0, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["ic_score"] == 20.0

    def test_score_factor_ic_score_capped_at_100(self):
        """IC得分上限为100"""
        metrics = {"ic_mean": 0.5, "ir": 0, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["ic_score"] == 100.0

    def test_score_factor_ic_score_uses_absolute_value(self):
        """IC得分使用绝对值"""
        metrics_pos = {"ic_mean": 0.05, "ir": 0, "stability_score": 0, "turnover": 0}
        metrics_neg = {"ic_mean": -0.05, "ir": 0, "stability_score": 0, "turnover": 0}
        result_pos = self.service.score_factor(metrics_pos)
        result_neg = self.service.score_factor(metrics_neg)
        assert result_pos["details"]["ic_score"] == result_neg["details"]["ic_score"]

    def test_score_factor_ir_score_calculation(self):
        """IR得分 = min(abs(ir)*40, 100)"""
        # ir=2.5 → 2.5*40=100 → 满分
        metrics = {"ic_mean": 0, "ir": 2.5, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["ir_score"] == 100.0

        # ir=1.0 → 1.0*40=40
        metrics = {"ic_mean": 0, "ir": 1.0, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["ir_score"] == 40.0

    def test_score_factor_ir_score_capped_at_100(self):
        """IR得分上限为100"""
        metrics = {"ic_mean": 0, "ir": 5.0, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["ir_score"] == 100.0

    def test_score_factor_stability_score_calculation(self):
        """稳定性得分 = stability_score * 100"""
        metrics = {"ic_mean": 0, "ir": 0, "stability_score": 0.85, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["stability_score"] == 85.0

    def test_score_factor_turnover_score_calculation(self):
        """换手率得分 = max(100 - turnover*200, 0)"""
        # turnover=0.3 → 100-60=40
        metrics = {"ic_mean": 0, "ir": 0, "stability_score": 0, "turnover": 0.3}
        result = self.service.score_factor(metrics)
        assert result["details"]["turnover_score"] == 40.0

        # turnover=0 → 100-0=100
        metrics = {"ic_mean": 0, "ir": 0, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)
        assert result["details"]["turnover_score"] == 100.0

    def test_score_factor_turnover_score_floored_at_0(self):
        """换手率得分下限为0"""
        # turnover=1.0 → 100-200=-100 → max(-100,0)=0
        metrics = {"ic_mean": 0, "ir": 0, "stability_score": 0, "turnover": 1.0}
        result = self.service.score_factor(metrics)
        assert result["details"]["turnover_score"] == 0.0

    def test_score_factor_custom_weights(self):
        """自定义权重应正确应用"""
        metrics = {"ic_mean": 0.1, "ir": 1.0, "stability_score": 0.8, "turnover": 0.2}
        custom_weights = {"ic": 1.0, "ir": 0.0, "stability": 0.0, "turnover": 0.0}
        result = self.service.score_factor(metrics, weights=custom_weights)

        # ic_score = 0.1*400 = 40, total = 1.0*40 = 40
        assert result["total_score"] == 40.0
        assert result["weights"] == custom_weights

    def test_score_factor_missing_optional_fields_use_defaults(self):
        """缺失可选字段应使用默认值"""
        # 只提供ic_mean和ir，stability和turnover使用默认值
        metrics = {"ic_mean": 0.05, "ir": 1.0}
        result = self.service.score_factor(metrics)

        # stability默认0.8 → 80分, turnover默认0.3 → 40分
        assert result["details"]["stability_score"] == 80.0
        assert result["details"]["turnover_score"] == 40.0

    def test_score_factor_empty_metrics_use_defaults(self):
        """空指标字典应使用所有默认值"""
        result = self.service.score_factor({})
        # ic_mean默认0→0分, ir默认0→0分, stability默认0.8→80分, turnover默认0.3→40分
        assert result["details"]["ic_score"] == 0.0
        assert result["details"]["ir_score"] == 0.0
        assert result["details"]["stability_score"] == 80.0
        assert result["details"]["turnover_score"] == 40.0

    def test_score_factor_total_score_is_weighted_sum(self):
        """总分应为加权求和"""
        metrics = {"ic_mean": 0.1, "ir": 1.0, "stability_score": 0.8, "turnover": 0.2}
        weights = {"ic": 0.35, "ir": 0.30, "stability": 0.20, "turnover": 0.15}
        result = self.service.score_factor(metrics, weights=weights)

        min(0.1 * 400, 100)  # 40
        min(1.0 * 40, 100)  # 40
        max(100 - 0.2 * 200, 0)  # 60

        expected = 0.35 * 40 + 0.30 * 40 + 0.20 * 80 + 0.15 * 60
        assert abs(result["total_score"] - round(expected, 2)) < 0.01


class TestScoreStrategy:
    """策略评分测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_score_strategy_with_default_weights_should_return_all_fields(self):
        """默认权重下策略评分应返回完整字段"""
        metrics = {
            "annual_return": 0.15,
            "max_drawdown": -0.1,
            "sharpe_ratio": 1.5,
            "win_rate": 0.55,
            "turnover": 0.5,
        }
        result = self.service.score_strategy(metrics)

        assert "total_score" in result
        assert "grade" in result
        assert "details" in result
        assert "weights" in result
        assert "return_score" in result["details"]
        assert "risk_score" in result["details"]
        assert "efficiency_score" in result["details"]
        assert "stability_score" in result["details"]
        assert "cost_score" in result["details"]

    def test_score_strategy_return_score_calculation(self):
        """收益率得分 = min(max(annual_return/0.2*100, 0), 100)"""
        # annual_return=0.2 → 100分
        metrics = {"annual_return": 0.2, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["return_score"] == 100.0

        # annual_return=0.1 → 50分
        metrics = {"annual_return": 0.1, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["return_score"] == 50.0

    def test_score_strategy_return_score_capped_at_100(self):
        """收益率得分上限为100"""
        metrics = {"annual_return": 0.5, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["return_score"] == 100.0

    def test_score_strategy_return_score_floored_at_0(self):
        """负收益率得分下限为0"""
        metrics = {"annual_return": -0.1, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["return_score"] == 0.0

    def test_score_strategy_risk_score_calculation(self):
        """风险得分 = max(100 - abs(max_drawdown)/0.1*100, 0)"""
        # max_drawdown=-0.1 → abs=0.1 → 100-100=0
        metrics = {"annual_return": 0, "max_drawdown": -0.1, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["risk_score"] == 0.0

        # max_drawdown=-0.05 → abs=0.05 → 100-50=50
        metrics = {"annual_return": 0, "max_drawdown": -0.05, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["risk_score"] == 50.0

    def test_score_strategy_risk_score_floored_at_0(self):
        """极大回撤时风险得分下限为0"""
        metrics = {"annual_return": 0, "max_drawdown": -0.5, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["risk_score"] == 0.0

    def test_score_strategy_efficiency_score_calculation(self):
        """效率得分 = max(min(sharpe_ratio/2.0*100, 100), 0)"""
        # sharpe=2.0 → 100分
        metrics = {"annual_return": 0, "max_drawdown": 0, "sharpe_ratio": 2.0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["efficiency_score"] == 100.0

        # sharpe=1.0 → 50分
        metrics = {"annual_return": 0, "max_drawdown": 0, "sharpe_ratio": 1.0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["efficiency_score"] == 50.0

    def test_score_strategy_efficiency_score_negative_sharpe_floored(self):
        """负夏普比率效率得分下限为0"""
        metrics = {"annual_return": 0, "max_drawdown": 0, "sharpe_ratio": -1.0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["efficiency_score"] == 0.0

    def test_score_strategy_stability_score_calculation(self):
        """稳定性得分 = win_rate * 100"""
        metrics = {"annual_return": 0, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0.6, "turnover": 0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["stability_score"] == 60.0

    def test_score_strategy_cost_score_calculation(self):
        """成本得分 = max(100 - turnover*100, 0)"""
        # turnover=0.5 → 100-50=50
        metrics = {"annual_return": 0, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0.5}
        result = self.service.score_strategy(metrics)
        assert result["details"]["cost_score"] == 50.0

    def test_score_strategy_cost_score_floored_at_0(self):
        """高换手率成本得分下限为0"""
        metrics = {"annual_return": 0, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 2.0}
        result = self.service.score_strategy(metrics)
        assert result["details"]["cost_score"] == 0.0

    def test_score_strategy_custom_weights(self):
        """自定义权重应正确应用"""
        metrics = {"annual_return": 0.2, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        custom_weights = {"return": 1.0, "risk": 0.0, "efficiency": 0.0, "stability": 0.0, "cost": 0.0}
        result = self.service.score_strategy(metrics, weights=custom_weights)

        assert result["total_score"] == 100.0
        assert result["weights"] == custom_weights

    def test_score_strategy_missing_fields_use_defaults(self):
        """缺失字段应使用默认值"""
        metrics = {"annual_return": 0.1}
        result = self.service.score_strategy(metrics)

        # max_drawdown默认0.2 → 100-200=-100→0, sharpe默认0→0, win_rate默认0.5→50, turnover默认0.5→50
        assert result["details"]["risk_score"] == 0.0
        assert result["details"]["efficiency_score"] == 0.0
        assert result["details"]["stability_score"] == 50.0
        assert result["details"]["cost_score"] == 50.0


class TestScorePortfolio:
    """组合评分测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_score_portfolio_without_benchmark(self):
        """无基准时组合评分应使用绝对收益"""
        metrics = {
            "annual_return": 0.15,
            "volatility": 0.15,
            "max_drawdown": 0.1,
            "herfindahl_index": 0.1,
            "sharpe_ratio": 1.0,
        }
        result = self.service.score_portfolio(metrics)

        assert "total_score" in result
        assert "grade" in result
        assert "details" in result
        # 无基准时 return_score = min(max(0.15/0.15*100, 0), 100) = 100
        assert result["details"]["return_score"] == 100.0

    def test_score_portfolio_with_benchmark(self):
        """有基准时应使用超额收益"""
        portfolio_metrics = {
            "annual_return": 0.12,
            "volatility": 0.15,
            "max_drawdown": 0.1,
            "herfindahl_index": 0.1,
            "sharpe_ratio": 1.0,
        }
        benchmark_metrics = {"annual_return": 0.08}
        result = self.service.score_portfolio(portfolio_metrics, benchmark_metrics=benchmark_metrics)

        # 超额收益 = 0.12 - 0.08 = 0.04 → 0.04/0.05*100 = 80
        assert abs(result["details"]["return_score"] - 80.0) < 0.01

    def test_score_portfolio_with_benchmark_negative_excess(self):
        """超额收益为负时得分应为0"""
        portfolio_metrics = {
            "annual_return": 0.05,
            "volatility": 0.15,
            "max_drawdown": 0.1,
            "herfindahl_index": 0.1,
            "sharpe_ratio": 1.0,
        }
        benchmark_metrics = {"annual_return": 0.10}
        result = self.service.score_portfolio(portfolio_metrics, benchmark_metrics=benchmark_metrics)

        assert result["details"]["return_score"] == 0.0

    def test_score_portfolio_risk_score_calculation(self):
        """风险得分 = max(100 - (volatility/0.2*50 + max_drawdown/0.15*50), 0)"""
        metrics = {
            "annual_return": 0,
            "volatility": 0.1,
            "max_drawdown": 0.075,
            "herfindahl_index": 0,
            "sharpe_ratio": 0,
        }
        result = self.service.score_portfolio(metrics)

        # volatility/0.2*50 = 0.1/0.2*50 = 25
        # max_drawdown/0.15*50 = 0.075/0.15*50 = 25
        # risk_score = 100 - 25 - 25 = 50
        assert result["details"]["risk_score"] == 50.0

    def test_score_portfolio_risk_score_floored_at_0(self):
        """高风险时风险得分下限为0"""
        metrics = {"annual_return": 0, "volatility": 0.5, "max_drawdown": 0.5, "herfindahl_index": 0, "sharpe_ratio": 0}
        result = self.service.score_portfolio(metrics)
        assert result["details"]["risk_score"] == 0.0

    def test_score_portfolio_diversification_score(self):
        """分散化得分 = max(100 - herfindahl_index*100, 0)"""
        # herfindahl_index=0.1 → 100-10=90
        metrics = {"annual_return": 0, "volatility": 0, "max_drawdown": 0, "herfindahl_index": 0.1, "sharpe_ratio": 0}
        result = self.service.score_portfolio(metrics)
        assert result["details"]["diversification_score"] == 90.0

    def test_score_portfolio_efficiency_score(self):
        """效率得分 = min(sharpe_ratio/2.0*100, 100)"""
        metrics = {"annual_return": 0, "volatility": 0, "max_drawdown": 0, "herfindahl_index": 0, "sharpe_ratio": 1.5}
        result = self.service.score_portfolio(metrics)
        assert result["details"]["efficiency_score"] == 75.0

    def test_score_portfolio_custom_weights(self):
        """自定义权重应正确应用"""
        metrics = {"annual_return": 0.15, "volatility": 0, "max_drawdown": 0, "herfindahl_index": 0, "sharpe_ratio": 0}
        custom_weights = {"return": 1.0, "risk": 0.0, "diversification": 0.0, "efficiency": 0.0}
        result = self.service.score_portfolio(metrics, weights=custom_weights)

        assert result["total_score"] == 100.0
        assert result["weights"] == custom_weights

    def test_score_portfolio_default_weights(self):
        """默认权重应包含return/risk/diversification/efficiency"""
        metrics = {"annual_return": 0, "volatility": 0, "max_drawdown": 0, "herfindahl_index": 0, "sharpe_ratio": 0}
        result = self.service.score_portfolio(metrics)

        expected_weights = {"return": 0.35, "risk": 0.30, "diversification": 0.2, "efficiency": 0.15}
        assert result["weights"] == expected_weights


class TestCompareAndRank:
    """排名比较测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_compare_and_rank_strategy_type(self):
        """策略类型排名应按得分降序"""
        items = [
            {
                "name": "策略A",
                "metrics": {
                    "annual_return": 0.3,
                    "max_drawdown": -0.05,
                    "sharpe_ratio": 2.0,
                    "win_rate": 0.6,
                    "turnover": 0.3,
                },
            },
            {
                "name": "策略B",
                "metrics": {
                    "annual_return": 0.1,
                    "max_drawdown": -0.15,
                    "sharpe_ratio": 0.5,
                    "win_rate": 0.45,
                    "turnover": 0.8,
                },
            },
            {
                "name": "策略C",
                "metrics": {
                    "annual_return": 0.2,
                    "max_drawdown": -0.08,
                    "sharpe_ratio": 1.2,
                    "win_rate": 0.55,
                    "turnover": 0.5,
                },
            },
        ]
        result = self.service.compare_and_rank(items, scoring_type="strategy")

        assert len(result) == 3
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2
        assert result[2]["rank"] == 3
        # 降序排列
        assert result[0]["score"] >= result[1]["score"] >= result[2]["score"]

    def test_compare_and_rank_factor_type(self):
        """因子类型排名应正确调用score_factor"""
        items = [
            {"name": "因子A", "metrics": {"ic_mean": 0.1, "ir": 2.0, "stability_score": 0.9, "turnover": 0.1}},
            {"name": "因子B", "metrics": {"ic_mean": 0.02, "ir": 0.5, "stability_score": 0.6, "turnover": 0.5}},
        ]
        result = self.service.compare_and_rank(items, scoring_type="factor")

        assert len(result) == 2
        assert result[0]["score"] >= result[1]["score"]
        assert result[0]["rank"] == 1

    def test_compare_and_rank_portfolio_type(self):
        """组合类型排名应正确调用score_portfolio"""
        items = [
            {
                "name": "组合A",
                "metrics": {
                    "annual_return": 0.2,
                    "volatility": 0.1,
                    "max_drawdown": 0.05,
                    "herfindahl_index": 0.05,
                    "sharpe_ratio": 2.0,
                },
            },
            {
                "name": "组合B",
                "metrics": {
                    "annual_return": 0.05,
                    "volatility": 0.3,
                    "max_drawdown": 0.2,
                    "herfindahl_index": 0.3,
                    "sharpe_ratio": 0.3,
                },
            },
        ]
        result = self.service.compare_and_rank(items, scoring_type="portfolio")

        assert len(result) == 2
        assert result[0]["score"] >= result[1]["score"]

    def test_compare_and_rank_unknown_type_raises(self):
        """未知评分类型应抛出ValueError"""
        items = [{"name": "test", "metrics": {}}]
        with pytest.raises(ValueError, match="未知的评分类型"):
            self.service.compare_and_rank(items, scoring_type="unknown")

    def test_compare_and_rank_empty_list(self):
        """空列表应返回空结果"""
        result = self.service.compare_and_rank([], scoring_type="strategy")
        assert result == []

    def test_compare_and_rank_single_item(self):
        """单项应排名为1"""
        items = [{"name": "唯一策略", "metrics": {"annual_return": 0.15}}]
        result = self.service.compare_and_rank(items, scoring_type="strategy")

        assert len(result) == 1
        assert result[0]["rank"] == 1

    def test_compare_and_rank_result_has_required_fields(self):
        """结果应包含name/score/grade/details/rank"""
        items = [{"name": "策略A", "metrics": {"annual_return": 0.15}}]
        result = self.service.compare_and_rank(items, scoring_type="strategy")

        for key in ["name", "score", "grade", "details", "rank"]:
            assert key in result[0], f"缺少字段: {key}"

    def test_compare_and_rank_equal_scores(self):
        """相同得分的项目都应正确排名"""
        items = [
            {"name": "A", "metrics": {"annual_return": 0.1}},
            {"name": "B", "metrics": {"annual_return": 0.1}},
        ]
        result = self.service.compare_and_rank(items, scoring_type="strategy")
        assert len(result) == 2
        assert result[0]["score"] == result[1]["score"]


class TestGetGrade:
    """评级映射测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_grade_s_plus(self):
        """得分>=95应为S+"""
        assert self.service._get_grade(95) == "S+"
        assert self.service._get_grade(100) == "S+"
        assert self.service._get_grade(99.9) == "S+"

    def test_grade_a_plus(self):
        """90<=得分<95应为A+"""
        assert self.service._get_grade(90) == "A+"
        assert self.service._get_grade(94.9) == "A+"

    def test_grade_a(self):
        """85<=得分<90应为A"""
        assert self.service._get_grade(85) == "A"
        assert self.service._get_grade(89.9) == "A"

    def test_grade_a_minus(self):
        """80<=得分<85应为A-"""
        assert self.service._get_grade(80) == "A-"
        assert self.service._get_grade(84.9) == "A-"

    def test_grade_b_plus(self):
        """75<=得分<80应为B+"""
        assert self.service._get_grade(75) == "B+"
        assert self.service._get_grade(79.9) == "B+"

    def test_grade_b(self):
        """70<=得分<75应为B"""
        assert self.service._get_grade(70) == "B"
        assert self.service._get_grade(74.9) == "B"

    def test_grade_b_minus(self):
        """65<=得分<70应为B-"""
        assert self.service._get_grade(65) == "B-"
        assert self.service._get_grade(69.9) == "B-"

    def test_grade_c_plus(self):
        """60<=得分<65应为C+"""
        assert self.service._get_grade(60) == "C+"
        assert self.service._get_grade(64.9) == "C+"

    def test_grade_c(self):
        """55<=得分<60应为C"""
        assert self.service._get_grade(55) == "C"
        assert self.service._get_grade(59.9) == "C"

    def test_grade_c_minus(self):
        """50<=得分<55应为C-"""
        assert self.service._get_grade(50) == "C-"
        assert self.service._get_grade(54.9) == "C-"

    def test_grade_d(self):
        """得分<50应为D"""
        assert self.service._get_grade(0) == "D"
        assert self.service._get_grade(49.9) == "D"

    def test_grade_boundary_completeness(self):
        """所有评级边界应无遗漏"""
        boundaries = [
            (95, "S+"),
            (90, "A+"),
            (85, "A"),
            (80, "A-"),
            (75, "B+"),
            (70, "B"),
            (65, "B-"),
            (60, "C+"),
            (55, "C"),
            (50, "C-"),
            (0, "D"),
        ]
        for score, expected_grade in boundaries:
            assert self.service._get_grade(score) == expected_grade, f"得分{score}应为{expected_grade}"


class TestAnalyzeSlippageSensitivity:
    """滑点敏感性分析测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_analyze_slippage_sensitivity_basic(self):
        """基本滑点敏感性分析应返回完整结果"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics)

        assert "base_slippage" in result
        assert "sensitivity_level" in result
        assert "scenarios" in result
        assert "recommendations" in result
        assert "cost_impact_ratio" in result

    def test_analyze_slippage_sensitivity_annual_cost_calculation(self):
        """年化滑点成本 = slippage * turnover * 2"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        # 找到base_slippage对应的scenario
        base_scenario = None
        for s in result["scenarios"]:
            if s["is_recommended"]:
                base_scenario = s
                break

        assert base_scenario is not None
        # annual_cost = 0.002 * 12 * 2 = 0.048 → 4.8%
        assert base_scenario["annual_cost_pct"] == 4.8

    def test_analyze_slippage_sensitivity_net_return(self):
        """净收益 = 年化收益 - 年化滑点成本"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        base_scenario = None
        for s in result["scenarios"]:
            if s["is_recommended"]:
                base_scenario = s
                break

        # net_return = 0.15 - 0.048 = 0.102 → 10.2%
        assert base_scenario["net_annual_return"] == 10.2

    def test_analyze_slippage_sensitivity_low_sensitivity(self):
        """低敏感性：滑点成本占收益<10%"""
        # 高收益低换手 → 低敏感
        metrics = {"annual_return": 0.5, "turnover": 2.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.001)

        # base_cost = 0.001 * 2 * 2 = 0.004, ratio = 0.004/0.5 = 0.008 < 0.1
        assert result["sensitivity_level"] == "low"

    def test_analyze_slippage_sensitivity_medium_sensitivity(self):
        """中敏感性：滑点成本占收益10-25%"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.001)

        # base_cost = 0.001 * 12 * 2 = 0.024, ratio = 0.024/0.15 = 0.16 → medium
        assert result["sensitivity_level"] == "medium"

    def test_analyze_slippage_sensitivity_high_sensitivity(self):
        """高敏感性：滑点成本占收益25-50%"""
        metrics = {"annual_return": 0.10, "turnover": 20.0}
        self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        # base_cost = 0.002 * 20 * 2 = 0.08, ratio = 0.08/0.10 = 0.8 → very_high
        # 需要调整参数使得ratio在0.25-0.5之间
        metrics2 = {"annual_return": 0.15, "turnover": 20.0}
        self.service.analyze_slippage_sensitivity(metrics2, base_slippage=0.002)
        # base_cost = 0.002 * 20 * 2 = 0.08, ratio = 0.08/0.15 ≈ 0.533 → very_high
        # 再调整
        metrics3 = {"annual_return": 0.20, "turnover": 15.0}
        result3 = self.service.analyze_slippage_sensitivity(metrics3, base_slippage=0.002)
        # base_cost = 0.002 * 15 * 2 = 0.06, ratio = 0.06/0.20 = 0.3 → high
        assert result3["sensitivity_level"] == "high"

    def test_analyze_slippage_sensitivity_very_high_sensitivity(self):
        """极高敏感性：滑点成本占收益>50%"""
        metrics = {"annual_return": 0.05, "turnover": 30.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        # base_cost = 0.002 * 30 * 2 = 0.12, ratio = 0.12/0.05 = 2.4 → very_high
        assert result["sensitivity_level"] == "very_high"

    def test_analyze_slippage_sensitivity_custom_test_slippages(self):
        """自定义滑点列表应被使用"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        test_slippages = [0.001, 0.005, 0.01]
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002, test_slippages=test_slippages)

        # 应包含自定义的滑点值 + base_slippage
        slippage_rates = [s["slippage_rate"] for s in result["scenarios"]]
        for ts in test_slippages:
            assert ts in slippage_rates

    def test_analyze_slippage_sensitivity_scenarios_sorted(self):
        """场景应按滑点率升序排列"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        test_slippages = [0.01, 0.001, 0.005]
        result = self.service.analyze_slippage_sensitivity(metrics, test_slippages=test_slippages)

        slippage_rates = [s["slippage_rate"] for s in result["scenarios"]]
        assert slippage_rates == sorted(slippage_rates)

    def test_analyze_slippage_sensitivity_zero_return_default_inf(self):
        """零收益时敏感性比率应为inf"""
        metrics = {"annual_return": 0.0, "turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        # sensitivity_ratio = abs(base_cost) / 0.0 → inf → very_high
        assert result["sensitivity_level"] == "very_high"

    def test_analyze_slippage_sensitivity_with_smart_detector(self):
        """提供股票代码时应调用智能检测器"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        stock_codes = ["600036", "000001"]

        mock_recommendation = MagicMock()
        mock_recommendation.recommended_slippage = 0.0015
        mock_recommendation.conservative_slippage = 0.002
        mock_recommendation.aggressive_slippage = 0.001
        mock_recommendation.confidence = 0.85
        mock_recommendation.reasoning = "主板大盘股"
        mock_recommendation.sensitivity_analysis = {}
        mock_recommendation.warnings = []
        mock_recommendation.tips = ["选择流动性好的时段"]

        with patch("backend.services.comprehensive_scoring_service.smart_slippage_detector") as mock_detector:
            mock_detector.recommend_slippage.return_value = mock_recommendation
            result = self.service.analyze_slippage_sensitivity(metrics, stock_codes=stock_codes)

        # 应使用智能推荐的滑点
        assert result["base_slippage"] == 0.0015
        assert "smart_recommendation" in result
        assert result["smart_recommendation"]["recommended_slippage"] == 0.0015
        assert result["smart_recommendation"]["confidence"] == 0.85

    def test_analyze_slippage_sensitivity_smart_detector_failure_fallback(self):
        """智能检测器失败时应回退到默认值"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        stock_codes = ["600036"]

        with patch("backend.services.comprehensive_scoring_service.smart_slippage_detector") as mock_detector:
            mock_detector.recommend_slippage.side_effect = Exception("检测失败")
            result = self.service.analyze_slippage_sensitivity(metrics, stock_codes=stock_codes, base_slippage=0.002)

        # 应回退到默认base_slippage
        assert result["base_slippage"] == 0.002
        assert "smart_recommendation" not in result

    def test_analyze_slippage_sensitivity_no_stock_codes(self):
        """不提供股票代码时不应调用智能检测器"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}

        with patch("backend.services.comprehensive_scoring_service.smart_slippage_detector") as mock_detector:
            result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        mock_detector.recommend_slippage.assert_not_called()
        assert "smart_recommendation" not in result

    def test_analyze_slippage_sensitivity_empty_stock_codes(self):
        """空股票代码列表时不应调用智能检测器"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}

        with patch("backend.services.comprehensive_scoring_service.smart_slippage_detector") as mock_detector:
            self.service.analyze_slippage_sensitivity(metrics, stock_codes=[], base_slippage=0.002)

        mock_detector.recommend_slippage.assert_not_called()

    def test_analyze_slippage_sensitivity_recommendations_high_sensitivity(self):
        """高敏感性应有critical优先级建议"""
        metrics = {"annual_return": 0.10, "turnover": 30.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.003)

        # base_cost = 0.003*30*2 = 0.18, ratio = 0.18/0.10 = 1.8 → very_high
        priorities = [r["priority"] for r in result["recommendations"]]
        assert "critical" in priorities

    def test_analyze_slippage_sensitivity_recommendations_low_sensitivity(self):
        """低敏感性应有low优先级建议"""
        metrics = {"annual_return": 0.50, "turnover": 2.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.001)

        # base_cost = 0.001*2*2 = 0.004, ratio = 0.004/0.50 = 0.008 → low
        priorities = [r["priority"] for r in result["recommendations"]]
        assert "low" in priorities

    def test_analyze_slippage_sensitivity_high_turnover_recommendation(self):
        """高换手率(>24)且非低敏感应有换手频率建议"""
        metrics = {"annual_return": 0.15, "turnover": 30.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        categories = [r["category"] for r in result["recommendations"]]
        assert "strategy" in categories or "cost_control" in categories

    def test_analyze_slippage_sensitivity_default_slippages(self):
        """未指定test_slippages时应使用默认列表"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        slippage_rates = [s["slippage_rate"] for s in result["scenarios"]]
        for ds in [0.0, 0.001, 0.003, 0.005, 0.01]:
            assert ds in slippage_rates

    def test_analyze_slippage_sensitivity_return_decay_calculation(self):
        """收益衰减百分比应正确计算"""
        metrics = {"annual_return": 0.15, "turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        # 找slippage=0.002的scenario
        for s in result["scenarios"]:
            if abs(s["slippage_rate"] - 0.002) < 0.0001:
                # annual_cost = 0.002*12*2 = 0.048
                # return_decay = 0.048/0.15 * 100 = 32%
                assert abs(s["return_decay_pct"] - 32.0) < 0.1
                break


class TestGenerateScoringReport:
    """评分报告生成测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_generate_report_contains_name(self):
        """报告应包含项目名称"""
        score_result = self.service.score_strategy({"annual_return": 0.15})
        report = self.service.generate_scoring_report(score_result, "测试策略")

        assert "测试策略" in report

    def test_generate_report_contains_score_and_grade(self):
        """报告应包含得分和评级"""
        score_result = self.service.score_strategy({"annual_return": 0.15})
        report = self.service.generate_scoring_report(score_result, "测试策略")

        assert str(score_result["total_score"]) in report
        assert score_result["grade"] in report

    def test_generate_report_contains_details(self):
        """报告应包含分项得分"""
        score_result = self.service.score_strategy({"annual_return": 0.15})
        report = self.service.generate_scoring_report(score_result, "测试策略")

        # 默认权重包含 return/risk/efficiency/stability/cost
        assert "RETURN" in report
        assert "RISK" in report


class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        self.service = ComprehensiveScoringService()

    def test_score_factor_all_zeros(self):
        """所有因子指标为零"""
        metrics = {"ic_mean": 0, "ir": 0, "stability_score": 0, "turnover": 0}
        result = self.service.score_factor(metrics)

        assert result["details"]["ic_score"] == 0.0
        assert result["details"]["ir_score"] == 0.0
        assert result["details"]["stability_score"] == 0.0
        assert result["details"]["turnover_score"] == 100.0  # turnover=0 → 100

    def test_score_factor_all_max(self):
        """所有因子指标取最大值"""
        metrics = {"ic_mean": 1.0, "ir": 10.0, "stability_score": 1.0, "turnover": 0}
        result = self.service.score_factor(metrics)

        assert result["details"]["ic_score"] == 100.0
        assert result["details"]["ir_score"] == 100.0
        assert result["details"]["stability_score"] == 100.0
        assert result["details"]["turnover_score"] == 100.0
        assert result["grade"] == "S+"

    def test_score_strategy_all_zeros(self):
        """所有策略指标为零或默认"""
        metrics = {"annual_return": 0, "max_drawdown": 0, "sharpe_ratio": 0, "win_rate": 0, "turnover": 0}
        result = self.service.score_strategy(metrics)

        assert result["details"]["return_score"] == 0.0
        assert result["details"]["risk_score"] == 100.0  # drawdown=0 → 100
        assert result["details"]["efficiency_score"] == 0.0
        assert result["details"]["stability_score"] == 0.0
        assert result["details"]["cost_score"] == 100.0  # turnover=0 → 100

    def test_score_portfolio_zero_volatility_zero_drawdown(self):
        """零波动零回撤的组合"""
        metrics = {"annual_return": 0.15, "volatility": 0, "max_drawdown": 0, "herfindahl_index": 0, "sharpe_ratio": 0}
        result = self.service.score_portfolio(metrics)

        # risk_score = 100 - (0 + 0) = 100
        assert result["details"]["risk_score"] == 100.0

    def test_score_factor_extreme_turnover(self):
        """极端换手率"""
        metrics = {"ic_mean": 0.05, "ir": 1.0, "stability_score": 0.8, "turnover": 100.0}
        result = self.service.score_factor(metrics)
        assert result["details"]["turnover_score"] == 0.0

    def test_score_strategy_negative_sharpe(self):
        """负夏普比率"""
        metrics = {"annual_return": -0.05, "max_drawdown": -0.3, "sharpe_ratio": -1.5, "win_rate": 0.3, "turnover": 1.5}
        result = self.service.score_strategy(metrics)

        assert result["details"]["return_score"] == 0.0
        assert result["details"]["efficiency_score"] == 0.0
        assert result["details"]["cost_score"] == 0.0

    def test_compare_and_rank_preserves_all_items(self):
        """排名比较应保留所有输入项"""
        items = [{"name": f"策略{i}", "metrics": {"annual_return": 0.1 * i}} for i in range(10)]
        result = self.service.compare_and_rank(items, scoring_type="strategy")
        assert len(result) == 10

    def test_analyze_slippage_sensitivity_default_turnover(self):
        """未提供turnover时应使用默认值12"""
        metrics = {"annual_return": 0.15}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        assert result["strategy_turnover"] == 12.0

    def test_analyze_slippage_sensitivity_default_annual_return(self):
        """未提供annual_return时应使用默认值0.15"""
        metrics = {"turnover": 12.0}
        result = self.service.analyze_slippage_sensitivity(metrics, base_slippage=0.002)

        assert result["original_annual_return"] == 15.0  # 0.15 * 100

    def test_score_factor_negative_ir_uses_absolute(self):
        """负IR应使用绝对值计算得分"""
        metrics_pos = {"ic_mean": 0, "ir": 1.5, "stability_score": 0, "turnover": 0}
        metrics_neg = {"ic_mean": 0, "ir": -1.5, "stability_score": 0, "turnover": 0}
        result_pos = self.service.score_factor(metrics_pos)
        result_neg = self.service.score_factor(metrics_neg)
        assert result_pos["details"]["ir_score"] == result_neg["details"]["ir_score"]
