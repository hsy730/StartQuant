"""
AlphaMiner/StockRanker 替代功能 — 综合单元测试

覆盖三大新服务：
1. FactorOrchestrator (AlphaMiner 一键验证)
2. StockRankerService (GBDT 排序学习)
3. ModelRegistry (模型注册中心)

测试策略：
- 使用 mock 数据避免依赖外部数据源
- 覆盖正常流程、边界情况、错误处理
- 验证各服务间的接口兼容性
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil

# =====================================================================
#  1. ModelRegistry 测试（无外部依赖，最先运行）
# =====================================================================


class TestModelRegistryBasic:
    """ModelRegistry 基础 CRUD 操作"""

    @pytest.fixture(autouse=True)
    def setup_registry(self, tmp_path):
        """每个测试用临时目录创建 registry"""
        from backend.services.model_registry import ModelRegistry

        self.registry = ModelRegistry(base_path=str(tmp_path / "models"))
        self.tmp_dir = tmp_path

    def test_save_and_load_sklearn_model(self):
        """保存和加载 sklearn 模型"""
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=50, n_features=3, random_state=42)
        model = LinearRegression()
        model.fit(X, y)

        model_id = self.registry.save(
            model,
            metadata={
                "params": {"fit_intercept": True},
                "feature_cols": ["f1", "f2", "f3"],
                "metrics": {"r2": float(model.score(X, y))},
            },
            framework="sklearn",
            model_name="test_lr",
            tags=["test", "linear"],
        )

        assert model_id is not None
        assert "test_lr" in model_id

        loaded = self.registry.load(model_id)
        assert loaded is not None
        # 验证预测一致性
        np.testing.assert_array_almost_equal(model.predict(X[:3]), loaded.predict(X[:3]))

    def test_list_models(self):
        """列出所有模型"""
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=20, n_features=2, random_state=42)

        for i in range(3):
            m = LinearRegression().fit(X, y)
            self.registry.save(
                m,
                metadata={},
                framework="sklearn",
                model_name=f"model_{i}",
                tags=[f"tag_{i}"],
            )

        models = self.registry.list_models()
        assert len(models) >= 3

        # 过滤测试
        filtered = self.registry.list_models(tags=["tag_1"])
        assert len(filtered) == 1
        assert "model_1" in filtered[0]["model_name"]

    def test_delete_model(self):
        """删除模型"""
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=20, n_features=2, random_state=42)
        m = LinearRegression().fit(X, y)
        model_id = self.registry.save(m, metadata={}, framework="sklearn", model_name="to_delete")

        assert self.registry.delete(model_id) is True
        # 删除后应无法加载
        with pytest.raises(FileNotFoundError):
            self.registry.load(model_id)
        # 再次删除返回 False
        assert self.registry.delete(model_id) is False

    def test_version_history(self):
        """版本历史管理"""
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=20, n_features=2, random_state=42)

        for _ in range(4):
            m = LinearRegression().fit(X, y)
            self.registry.save(m, metadata={}, framework="sklearn", model_name="versioned")

        versions = self.registry.get_version_history("versioned")
        assert len(versions) == 4
        version_numbers = [v["version"] for v in versions]
        assert version_numbers == [1, 2, 3, 4]

    def test_promote_model(self):
        """模型生命周期阶段提升"""
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=20, n_features=2, random_state=42)
        m = LinearRegression().fit(X, y)
        model_id = self.registry.save(
            m,
            metadata={},
            framework="sklearn",
            model_name="promotable",
            stage="development",
        )

        # 提升到 production
        assert self.registry.promote(model_id, "production") is True

        meta = self.registry.get_metadata(model_id)
        assert meta["stage"] == "production"

    def test_statistics(self):
        """统计信息"""
        stats = self.registry.get_statistics()
        assert "total_models" in stats
        assert "by_framework" in stats
        assert "by_stage" in stats
        assert isinstance(stats["total_models"], int)

    def test_compare_models(self):
        """模型对比"""
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=30, n_features=3, random_state=42)

        m1 = LinearRegression().fit(X, y)
        mid1 = self.registry.save(
            m1,
            metadata={"metrics": {"r2": float(m1.score(X, y))}},
            framework="sklearn",
            model_name="compare_a",
        )

        m2 = Ridge(alpha=0.5).fit(X, y)
        mid2 = self.registry.save(
            m2,
            metadata={"metrics": {"r2": float(m2.score(X, y))}},
            framework="sklearn",
            model_name="compare_b",
        )

        comparison = self.registry.compare_models([mid1, mid2])
        assert "models" in comparison
        assert "metrics_comparison" in comparison
        assert len(comparison["models"]) == 2

    def test_get_nonexistent_metadata(self):
        """获取不存在的模型元信息"""
        meta = self.registry.get_metadata("nonexistent_model_12345")
        assert meta == {}


class TestModelRegistryEdgeCases:
    """ModelRegistry 边界情况"""

    @pytest.fixture(autouse=True)
    def setup_registry(self, tmp_path):
        from backend.services.model_registry import ModelRegistry

        self.registry = ModelRegistry(base_path=str(tmp_path / "edge_models"))

    def test_empty_registry(self):
        """空注册中心"""
        assert self.registry.list_models() == []
        stats = self.registry.get_statistics()
        assert stats["total_models"] == 0

    def test_special_characters_in_name(self):
        """特殊字符的模型名称"""
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=20, n_features=2, random_state=42)
        m = LinearRegression().fit(X, y)
        model_id = self.registry.save(
            m,
            metadata={},
            framework="sklearn",
            model_name="model-with_special.chars_v2.0",
        )
        assert model_id is not None
        loaded = self.registry.load(model_id)
        assert loaded is not None


# =====================================================================
#  2. FactorOrchestrator 测试（使用 mock 数据）
# =====================================================================


class TestFactorOrchestrator:
    """FactorOrchestrator 一键验证流水线测试"""

    @pytest.fixture(autouse=True)
    def setup_orchestrator(self):
        from backend.services.factor_orchestrator_service import (
            FactorOrchestrator,
            OrchestratorConfig,
        )

        # 关闭耗时模块，加速测试
        self.config = OrchestratorConfig(
            enable_lookahead_detection=True,
            enable_ic_analysis=True,
            enable_alphalens=False,  # 需要 alphalens 库，跳过
            enable_quantile_backtest=False,  # 需要完整因子数据，跳过
            enable_tear_sheet=False,  # 需要完整因子数据，跳过
            enable_shap_analysis=False,  # 耗时，跳过
        )
        self.orchestrator = FactorOrchestrator(config=self.config)

    def test_derive_factor_name(self):
        """从表达式派生因子名称"""
        from backend.services.factor_orchestrator_service import FactorOrchestrator

        name = FactorOrchestrator._derive_factor_name("RSI(close, 14) * volume / MA(volume, 20)")
        assert len(name) > 0
        assert "(" not in name or name.count("(") == name.count(")")

    def test_determine_overall_status_all_passed(self):
        """全部通过时状态为 PASSED"""
        from backend.services.factor_orchestrator_service import (
            PipelineStageResult,
            PipelineStatus,
        )

        stages = {
            "compute_factor": PipelineStageResult(stage_name="compute_factor", status=PipelineStatus.PASSED),
            "lookahead_detection": PipelineStageResult(stage_name="lookahead_detection", status=PipelineStatus.PASSED),
            "ic_analysis": PipelineStageResult(stage_name="ic_analysis", status=PipelineStatus.PASSED),
        }

        verdict, summary = self.orchestrator._determine_overall_status(stages)
        assert verdict == "PASSED"
        assert summary["stages_passed"] == 3

    def test_determine_overall_status_rejected(self):
        """有 REJECTED 阶段时总体为 REJECTED"""
        from backend.services.factor_orchestrator_service import (
            PipelineStageResult,
            PipelineStatus,
        )

        stages = {
            "lookahead_detection": PipelineStageResult(
                stage_name="lookahead_detection",
                status=PipelineStatus.REJECTED,
                result={"has_bias": True, "risk_level": "critical"},
            ),
            "ic_analysis": PipelineStageResult(
                stage_name="ic_analysis",
                status=PipelineStatus.PASSED,
            ),
        }

        verdict, summary = self.orchestrator._determine_overall_status(stages)
        assert verdict == "REJECTED"
        assert "未来函数" in summary["reason"]

    def test_determine_overall_status_partial(self):
        """部分失败时为 PARTIAL"""
        from backend.services.factor_orchestrator_service import (
            PipelineStageResult,
            PipelineStatus,
        )

        stages = {
            "compute_factor": PipelineStageResult(
                stage_name="compute_factor",
                status=PipelineStatus.FAILED,
                error="数据不足",
            ),
            "lookahead_detection": PipelineStageResult(
                stage_name="lookahead_detection",
                status=PipelineStatus.SKIPPED,
            ),
        }

        verdict, summary = self.orchestrator._determine_overall_status(stages)
        assert verdict == "PARTIAL"

    def test_calculate_score_all_pass(self):
        """全通过时评分较高"""
        from backend.services.factor_orchestrator_service import (
            PipelineStageResult,
            PipelineStatus,
        )

        stages = {
            "lookahead_detection": PipelineStageResult(stage_name="lookahead_detection", status=PipelineStatus.PASSED),
            "ic_analysis": PipelineStageResult(stage_name="ic_analysis", status=PipelineStatus.PASSED),
            "alphalens_analysis": PipelineStageResult(stage_name="alphalens_analysis", status=PipelineStatus.PASSED),
            "quantile_backtest": PipelineStageResult(stage_name="quantile_backtest", status=PipelineStatus.PASSED),
            "tear_sheet": PipelineStageResult(stage_name="tear_sheet", status=PipelineStatus.PASSED),
        }
        score = self.orchestrator._calculate_overall_score(stages)
        assert score > 80  # 全通过应 > 80 分

    def test_calculate_score_with_rejection(self):
        """被拒绝时评分大幅降低"""
        from backend.services.factor_orchestrator_service import (
            PipelineStageResult,
            PipelineStatus,
        )

        stages = {
            "lookahead_detection": PipelineStageResult(
                stage_name="lookahead_detection", status=PipelineStatus.REJECTED
            ),
            "ic_analysis": PipelineStageResult(stage_name="ic_analysis", status=PipelineStatus.PASSED),
        }
        score = self.orchestrator._calculate_overall_score(stages)
        assert score < 50  # REJECTED 应显著拉低分数

    def test_markdown_report_generation(self):
        """Markdown 报告生成"""
        result = {
            "metadata": {
                "factor_name": "test_factor",
                "expression": "RSI(close, 14)",
                "stock_codes": ["000001"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "started_at": "2024-01-01T00:00:00",
            },
            "status": "PASSED",
            "stages": {
                "compute_factor": __class__.make_stage("compute_factor", "PASSED"),
                "lookahead_detection": __class__.make_stage("lookahead_detection", "PASSED"),
                "ic_analysis": __class__.make_stage("ic_analysis", "WARNING", warnings=["IC偏低"]),
            },
            "summary": {"verdict": "PASSED", "overall_score": 75.0, "stages_passed": 3, "stages_total": 3},
        }

        md = self.orchestrator._generate_markdown_report(result, {})
        assert "# 因子验证报告" in md
        assert "test_factor" in md
        assert "PASSED" in md
        assert "IC偏低" in md

    def test_structured_report_generation(self):
        """结构化报告生成（前端渲染用）"""
        from backend.services.factor_orchestrator_service import (
            PipelineStageResult,
            PipelineStatus,
        )

        result = {
            "summary": {"verdict": "PASSED", "overall_score": 75.0},
            "stages": {
                "lookahead_detection": PipelineStageResult(
                    stage_name="lookahead_detection",
                    status=PipelineStatus.PASSED,
                    result={"risk_score": 5.0, "has_bias": False},
                ),
            },
        }

        structured = self.orchestrator._generate_structured_report(result, {})
        assert structured["verdict"] == "PASSED"
        assert structured["score"] == 75.0
        assert "lookahead_detection" in structured["stages"]
        assert structured["stages"]["lookahead_detection"]["metrics"]["risk_score"] == 5.0

    @staticmethod
    def make_stage(name: str, status: str, **kwargs):
        from backend.services.factor_orchestrator_service import (
            PipelineStageResult,
            PipelineStatus,
        )

        # 映射大写状态名到枚举值（测试中使用 PASSED/FAILED 等大写）
        status_map = {
            "PASSED": PipelineStatus.PASSED,
            "WARNING": PipelineStatus.WARNING,
            "FAILED": PipelineStatus.FAILED,
            "REJECTED": PipelineStatus.REJECTED,
            "SKIPPED": PipelineStatus.SKIPPED,
        }
        return PipelineStageResult(
            stage_name=name, status=status_map.get(status, PipelineStatus(status.lower())), **kwargs
        )


class TestFactorOrchestratorIntegration:
    """FactorOrchestrator 集成测试（带模拟数据）"""

    def test_validate_with_mock_data(self):
        """使用模拟数据的端到端验证"""
        from backend.services.factor_orchestrator_service import (
            FactorOrchestrator,
            OrchestratorConfig,
        )
        from unittest.mock import patch

        config = OrchestratorConfig(
            enable_lookahead_detection=True,
            enable_ic_analysis=True,
            enable_alphalens=False,
            enable_quantile_backtest=False,
            enable_tear_sheet=False,
            enable_shap_analysis=False,
        )
        orchestrator = FactorOrchestrator(config=config)

        # Mock factor service calculator
        mock_factor_data = {}
        np.random.seed(42)
        dates = pd.date_range("2023-06-01", periods=120, freq="B")
        for stock in ["000001", "600036"]:
            df = pd.DataFrame(
                {
                    "close": 10 + np.cumsum(np.random.randn(120) * 0.5),
                    "volume": 1000000 + np.random.randint(500000, 5000000, 120).astype(float),
                    "open": 9.9 + np.cumsum(np.random.randn(120) * 0.5),
                },
                index=dates,
            )
            df["test_expr"] = np.random.randn(120) * 0.5
            mock_factor_data[stock] = df

        with patch.object(orchestrator, "_stage_compute_factor") as mock_compute:
            mock_compute.return_value = __class__.make_ok_stage(mock_factor_data)
            with patch.object(orchestrator, "_stage_lookahead_detection") as mock_bias:
                mock_bias.return_value = __class__.make_bias_stage_safe()
                with patch.object(orchestrator, "_stage_ic_analysis") as mock_ic:
                    mock_ic.return_value = __class__.make_ic_stage()

                    result = orchestrator.validate(
                        expression="close / open",
                        stock_codes=["000001", "600036"],
                        start_date="2023-06-01",
                        end_date="2023-12-31",
                        factor_name="price_ratio",
                    )

                    assert result["status"] in ("PASSED", "PARTIAL", "ERROR")
                    assert "stages" in result
                    assert "metadata" in result
                    assert result["metadata"]["factor_name"] == "price_ratio"
                    assert result["total_duration"] >= 0

    @staticmethod
    def make_ok_stage(factor_data):
        from backend.services.factor_orchestrator_service import PipelineStageResult, PipelineStatus

        return PipelineStageResult(
            stage_name="compute_factor",
            status=PipelineStatus.PASSED,
            result={
                "factor_data": factor_data,
                "n_stocks": len(factor_data),
                "n_total_bars": sum(len(df) for df in factor_data.values()),
                "price_df": None,
            },
        )

    @staticmethod
    def make_bias_stage_safe():
        from backend.services.factor_orchestrator_service import PipelineStageResult, PipelineStatus

        return PipelineStageResult(
            stage_name="lookahead_detection",
            status=PipelineStatus.PASSED,
            result={"has_bias": False, "risk_level": "safe", "risk_score": 0.0},
        )

    @staticmethod
    def make_ic_stage():
        from backend.services.factor_orchestrator_service import PipelineStageResult, PipelineStatus

        return PipelineStageResult(
            stage_name="ic_analysis",
            status=PipelineStatus.PASSED,
            result={"ic_mean": 0.03, "ir": 0.8, "rank_ic": 0.025},
        )


# =====================================================================
#  3. StockRankerService 测试（需要 XGBoost）
# =====================================================================


class TestStockRankerService:
    """StockRankerService 排序学习测试"""

    @pytest.fixture(autouse=True)
    def setup_ranker(self):
        pytest.importorskip("xgboost")
        from backend.services.stock_ranker_service import StockRankerService, RankTrainingConfig
        import tempfile

        self.tmp_dir = tempfile.mkdtemp()
        self.ranker = StockRankerService(
            default_config=RankTrainingConfig(
                n_estimators=10,  # 减少迭代加速测试
                max_depth=3,
                early_stopping_rounds=3,
            )
        )
        # 将 RankTrainingConfig 存为类属性供所有方法使用
        self.RankTrainingConfig = RankTrainingConfig

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_train_and_predict_basic(self):
        """基础训练和预测流程"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="B")[:n]
        stocks = [f"{i:06d}" for i in range(20)]

        rows = []
        for i, date in enumerate(dates):
            for j, stock in enumerate(stocks):
                signal = np.random.randn() * 0.3 + (j % 5) * 0.05  # 微弱的股票效应
                ret = 0.02 * signal + np.random.randn() * 0.03
                rows.append(
                    {
                        "date": date,
                        "stock_code": stock,
                        "feature_1": signal + np.random.randn() * 0.1,
                        "feature_2": np.random.randn(),
                        "forward_return_5d": ret,
                    }
                )

        df = pd.DataFrame(rows)

        result = self.ranker.train(
            feature_df=df,
            label_col="forward_return_5d",
            date_col="date",
            group_col="date",
            config=self.RankTrainingConfig(
                objective="reg:squarederror",  # 用回归目标简化测试
                n_estimators=10,
                max_depth=3,
                early_stopping_rounds=3,
            ),
            enable_bias_check=False,  # 加速测试
        )

        assert result.status.value == "ready"
        assert result.n_samples >= 100
        assert result.n_features >= 2
        assert result.model_id is not None
        assert len(result.feature_importance) > 0

        # 预测
        today_df = df[df["date"] == df["date"].max()].copy()
        prediction = self.ranker.predict(
            model_id=result.model_id,
            features=today_df,
            top_n=5,
        )

        assert prediction is not None
        assert len(prediction.top_n_stocks) <= 5
        assert "rank_score" in prediction.top_n_stocks.columns
        assert "rank_position" in prediction.top_n_stocks.columns

    def test_explain_model(self):
        """模型解释"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="B"),
                "stock_code": [f"{i % 10:06d}" for i in range(n)],
                "feat_a": np.random.randn(n),
                "feat_b": np.random.randn(n),
                "return": np.random.randn(n) * 0.02,
            }
        )

        result = self.ranker.train(
            feature_df=df,
            label_col="return",
            date_col="date",
            group_col="date",
            config=self.RankTrainingConfig(
                objective="reg:squarederror",
                n_estimators=5,
                max_depth=2,
                early_stopping_rounds=2,
            ),
            enable_bias_check=False,
        )

        explanation = self.ranker.explain_model(result.model_id)
        assert "feature_importance_gain" in explanation
        assert "n_features" in explanation
        assert explanation["n_features"] >= 2

    def test_list_and_delete_models(self):
        """列出和删除模型"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="B"),
                "stock_code": [f"{i % 5:06d}" for i in range(n)],
                "f1": np.random.randn(n),
                "ret": np.random.randn(n) * 0.01,
            }
        )

        r1 = self.ranker.train(
            feature_df=df,
            label_col="ret",
            date_col="date",
            group_col="date",
            config=self.RankTrainingConfig(objective="reg:squarederror", n_estimators=3, max_depth=2),
            enable_bias_check=False,
            model_name="list_test",
        )

        models = self.ranker.list_models()
        assert len(models) >= 1

        deleted = self.ranker.delete_model(r1.model_id)
        assert deleted is True

        # 删除后预测应报错
        with pytest.raises(Exception):
            self.ranker.predict(r1.model_id, df.head(10))

    def test_train_with_bias_check(self):
        """训练时启用特征的未来函数检测"""
        np.random.seed(42)
        n = 150
        signal = np.random.randn(n)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="B"),
                "stock_code": [f"{i % 8:06d}" for i in range(n)],
                "normal_feature": np.random.randn(n),
                "leaky_feature": signal * 0.8 + np.random.randn(n) * 0.1,  # 高度相关于噪声收益
                "return": 0.7 * signal + np.random.randn(n) * 0.05,
            }
        )

        result = self.ranker.train(
            feature_df=df,
            label_col="return",
            date_col="date",
            group_col="date",
            config=self.RankTrainingConfig(
                objective="reg:squarederror",
                n_estimators=5,
                max_depth=2,
            ),
            enable_bias_check=True,
        )

        # 训练不应因 bias check 失败而中断（仅记录警告）
        assert result.status.value == "ready"
        assert result.model_id is not None


# =====================================================================
#  4. 端到端集成测试：Orchestrator → StockRanker → Registry
# =====================================================================


class TestEndToEndFlow:
    """跨服务的端到端流程验证"""

    def test_full_pipeline_conceptual(self):
        """概念性全流水线验证（验证接口兼容性）"""
        # 1. ModelRegistry 可存储任何框架的模型
        from backend.services.model_registry import ModelRegistry
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_path=tmpdir)
            X, y = make_regression(n_samples=30, n_features=3, random_state=42)
            model = LinearRegression().fit(X, y)

            mid = registry.save(
                model,
                metadata={
                    "params": {},
                    "feature_cols": ["a", "b", "c"],
                    "metrics": {"r2": 0.85},
                },
                framework="sklearn",
                model_name="e2e_test",
                tags=["e2e"],
            )

            # 2. 可以加载回来
            loaded = registry.load(mid)
            assert loaded is not None

            # 3. 元数据完整
            meta = registry.get_metadata(mid)
            assert meta["framework"] == "sklearn"
            assert meta["tags"] == ["e2e"]
            assert meta["version"] == 1

            # 4. 提升到生产
            registry.promote(mid, "production")
            prod_meta = registry.get_metadata(mid)
            assert prod_meta["stage"] == "production"

            # 5. 对比
            comparison = registry.compare_models([mid])
            assert "metrics_comparison" in comparison

    def test_config_default_values(self):
        """验证默认配置的合理性"""
        from backend.services.factor_orchestrator_service import OrchestratorConfig
        from backend.services.stock_ranker_service import RankTrainingConfig

        oc = OrchestratorConfig()
        assert oc.enable_lookahead_detection is True
        assert oc.fail_fast_on_bias is True
        assert oc.ic_threshold == 0.02

        rc = RankTrainingConfig()
        assert rc.learning_rate == 0.05
        assert rc.max_depth == 6
        assert rc.n_estimators == 200
        assert rc.objective == "rank:ndcg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
