"""
新功能单元测试 - 验证因子分析增强功能

覆盖范围：
1. 因子收益分析服务（Quantile Returns, Cumulative Returns, Turnover）
2. 行业市值联合回归中性化
3. 因子加权IC服务
4. Tear Sheet全貌报告生成器

运行方式：
    pytest tests/test_new_factor_features.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFactorReturnAnalysisService:
    """测试因子收益分析服务"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """设置测试数据"""
        from backend.services.factor_return_analysis_service import (
            FactorReturnAnalysisService,
            FactorReturnAnalysisConfig,
        )

        self.config = FactorReturnAnalysisConfig(
            n_quantiles=5,
            enable_bootstrap=False,  # 加速测试
        )
        self.service = FactorReturnAnalysisService(config=self.config)

        self.sample_data = self._generate_sample_data()

    def _generate_sample_data(self):
        """生成模拟的因子数据"""
        np.random.seed(42)
        n_days = 100

        data = {}

        for stock_id in range(10):
            dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

            factor_values = np.random.normal(0, 1, n_days).cumsum()
            prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, n_days))

            df = pd.DataFrame(
                {
                    "test_factor": factor_values,
                    "close": prices,
                    "market_cap": np.random.uniform(1e9, 1e12, n_days),
                },
                index=dates,
            )

            data[f"stock_{stock_id}"] = df

        return data

    def test_quantile_returns_basic(self):
        """测试基本的分组收益计算"""
        result = self.service.calculate_quantile_returns(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        assert result["success"]
        assert "quantile_returns" in result
        assert len(result["quantile_returns"]) == 5

        for q in result["quantile_returns"]:
            assert "avg_return" in q
            assert "group" in q
            assert q["group"].startswith("Q")

    def test_quantile_returns_spread_calculation(self):
        """测试多空利差计算"""
        result = self.service.calculate_quantile_returns(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        assert "spread" in result
        spread = result["spread"]

        assert "long_short_spread" in spread
        assert "is_significant" in spread
        assert isinstance(spread["long_short_spread"], float)

    def test_quantile_returns_monotonicity_test(self):
        """测试单调性检验"""
        result = self.service.calculate_quantile_returns(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        assert "monotonicity_test" in result
        mono = result["monotonicity_test"]

        assert "is_monotonic" in mono
        assert "monotonicity_ratio" in mono
        assert 0 <= mono["monotonicity_ratio"] <= 1

    def test_cumulative_returns_basic(self):
        """测试累计收益曲线计算"""
        result = self.service.calculate_cumulative_returns(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        assert "error" not in result
        assert "dates" in result
        assert "long_short_cumulative" in result
        assert len(result["dates"]) > 0

    def test_cumulative_returns_statistics(self):
        """测试累计收益统计指标"""
        result = self.service.calculate_cumulative_returns(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        if "summary_statistics" in result:
            stats = result["summary_statistics"]

            assert "final_cumulative_return" in stats
            assert "max_drawdown" in stats
            assert "sharpe_ratio" in stats

            assert isinstance(stats["max_drawdown"], float)

    def test_turnover_analysis_basic(self):
        """测试换手率分析基本功能"""
        result = self.service.calculate_turnover_analysis(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        assert result["success"]
        assert "turnover_stats" in result
        assert "autocorrelation" in result
        assert "stability_analysis" in result

    def test_turnover_interpretation(self):
        """测试换手率解读生成"""
        result = self.service.calculate_turnover_analysis(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        turnover_stats = result["turnover_stats"]

        assert "interpretation" in turnover_stats
        assert isinstance(turnover_stats["interpretation"], str)
        assert len(turnover_stats["interpretation"]) > 0


class TestJointNeutralization:
    """测试联合回归中性化"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """设置测试数据"""
        from backend.services.factor_preprocessing_pipeline import (
            FactorPreprocessingPipeline,
            PreprocessingConfig,
        )

        self.config = PreprocessingConfig(
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=True,
        )

        self.pipeline = FactorPreprocessingPipeline(config=self.config)

        self.test_data = self._generate_test_data()

    def _generate_test_data(self):
        """生成测试数据"""
        np.random.seed(123)
        n = 100

        return pd.DataFrame(
            {
                "factor": np.random.normal(0, 1, n),
                "market_cap": np.random.uniform(1e9, 1e11, n),
                "industry": np.random.choice(["tech", "finance", "healthcare", "energy"], n),
            }
        )

    def test_joint_neutralization_exists(self):
        """测试联合回归方法存在"""
        assert hasattr(self.pipeline, "_neutralize_joint")
        assert callable(self.pipeline._neutralize_joint)

    def test_joint_neutralization_execution(self):
        """测试联合回归执行"""
        result = self.pipeline._neutralize_joint(
            factor_values=self.test_data["factor"],
            market_cap=self.test_data["market_cap"],
            industry=self.test_data["industry"],
        )

        assert len(result) == len(self.test_data)
        assert isinstance(result, pd.Series)

    def test_cross_sectional_with_joint(self):
        """测试横截面处理使用联合回归"""
        df = self.test_data.copy()
        df["date"] = pd.date_range(start="2024-01-01", periods=len(df))

        processed_df, stats = self.pipeline.process_factor_dataframe(
            df=df,
            factor_columns=["factor"],
            market_cap_column="market_cap",
            industry_column="industry",
            date_column="date",
        )

        assert "factor" in processed_df.columns
        assert len(processed_df) == len(df)


class TestWeightedICService:
    """测试因子加权IC服务"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """设置测试数据"""
        from backend.services.weighted_ic_service import (
            WeightedICService,
            WeightedICConfig,
        )

        self.config = WeightedICConfig()
        self.service = WeightedICService(config=self.config)

        self.factor_ic_dict = {
            "factor_a": pd.Series(np.random.normal(0.05, 0.1, 100)),
            "factor_b": pd.Series(np.random.normal(0.03, 0.08, 100)),
            "factor_c": pd.Series(np.random.normal(-0.02, 0.12, 100)),
        }

    def test_weighted_ic_equal_weight(self):
        """测试等权加权"""
        from backend.services.weighted_ic_service import WeightedICService, WeightingMethod

        self.config.weighting_method = WeightingMethod.EQUAL_WEIGHT
        service = WeightedICService(config=self.config)

        result = service.calculate_weighted_ic(
            factor_ic_dict=self.factor_ic_dict,
        )

        assert result["success"]
        assert "weighted_ic" in result
        assert "factor_weights" in result

        weights = result["factor_weights"]
        total_weight = sum(w["weight"] for w in weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_weighted_ic_ir_weight(self):
        """测试IR加权"""
        from backend.services.weighted_ic_service import WeightedICService, WeightingMethod

        self.config.weighting_method = WeightingMethod.IR_WEIGHT
        service = WeightedICService(config=self.config)

        result = service.calculate_weighted_ic(
            factor_ic_dict=self.factor_ic_dict,
        )

        assert result["success"]
        assert result["weighting_method"] == "ir_weight"

    def test_factor_importance_ranking(self):
        """测试因子重要性排名"""
        result = self.service.calculate_factor_importance(
            factor_ic_dict=self.factor_ic_dict,
        )

        assert result["success"]
        assert "ranking" in result
        assert len(result["ranking"]) == 3

        ranking = result["ranking"]
        scores = [r["total_score"] for r in ranking]

        assert sorted(scores, reverse=True) == scores

    def test_contribution_analysis(self):
        """测试贡献度归因分析"""
        result = self.service.calculate_weighted_ic(
            factor_ic_dict=self.factor_ic_dict,
        )

        assert "contribution_analysis" in result

        contributions = result["contribution_analysis"]
        for name, contrib in contributions.items():
            assert "weight" in contrib
            assert "mean_contribution" in contrib


class TestTearSheetService:
    """测试Tear Sheet全貌报告生成器"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """设置测试数据"""
        from backend.services.tear_sheet_service import (
            TearSheetService,
            TearSheetConfig,
        )

        self.config = TearSheetConfig(include_bootstrap=False)
        self.service = TearSheetService(config=self.config)

        self.sample_data = self._generate_sample_data()

    def _generate_sample_data(self):
        """生成模拟数据"""
        np.random.seed(456)
        n_days = 80

        data = {}
        for stock_id in range(8):
            dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

            factor_values = np.random.normal(0, 1, n_days).cumsum()
            prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, n_days))

            df = pd.DataFrame(
                {
                    "test_factor": factor_values,
                    "close": prices,
                },
                index=dates,
            )

            data[f"stock_{stock_id}"] = df

        return data

    def test_tear_sheet_generation(self):
        """测试Tear Sheet基本生成"""
        result = self.service.create_full_tear_sheet(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        assert result["success"]
        assert "tear_sheet" in result

        tear_sheet = result["tear_sheet"]
        assert "metadata" in tear_sheet
        assert "summary" in tear_sheet
        assert "sections" in tear_sheet

    def test_tear_sheet_scoring(self):
        """测试综合评分系统"""
        result = self.service.create_full_tear_sheet(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        summary = result["tear_sheet"]["summary"]

        assert "overall_score" in summary
        assert "grade" in summary
        assert "score_breakdown" in summary

        score = summary["overall_score"]
        assert 0 <= score <= 100
        assert isinstance(summary["grade"], str)

    def test_tear_sheet_sections(self):
        """测试各分析板块是否包含"""
        result = self.service.create_full_tear_sheet(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        result["tear_sheet"]["sections"]

        completed = result["tear_sheet"]["summary"]["sections_completed"]

        assert len(completed) > 0

    def test_tear_sheet_recommendations(self):
        """测试改进建议生成"""
        result = self.service.create_full_tear_sheet(
            factor_data=self.sample_data,
            factor_name="test_factor",
        )

        assert "recommendations" in result["tear_sheet"]

        recommendations = result["tear_sheet"]["recommendations"]
        assert len(recommendations) > 0

        for rec in recommendations:
            assert "priority" in rec
            assert "category" in rec
            assert "suggestion" in rec


class TestEdgeCases:
    """边界情况和错误处理测试"""

    def test_empty_data_handling(self):
        """测试空数据处理"""
        from backend.services.factor_return_analysis_service import factor_return_analysis_service

        result = factor_return_analysis_service.calculate_quantile_returns(
            factor_data={},
            factor_name="nonexistent",
        )

        assert "error" in result

    def test_insufficient_data(self):
        """测试数据不足情况"""
        from backend.services.factor_return_analysis_service import (
            factor_return_analysis_service,
            FactorReturnAnalysisConfig,
        )

        config = FactorReturnAnalysisConfig(n_quantiles=10)
        service = factor_return_analysis_service.__class__(config)

        small_data = {
            "stock_1": pd.DataFrame(
                {
                    "factor": [1, 2, 3],
                    "close": [10, 11, 12],
                }
            )
        }

        result = service.calculate_quantile_returns(
            factor_data=small_data,
            factor_name="factor",
        )

        assert "error" in result or result.get("total_observations", 0) < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
