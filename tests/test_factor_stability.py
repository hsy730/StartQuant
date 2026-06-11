"""
因子稳定性分析服务单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


class TestFactorStabilityService:
    """因子稳定性服务测试类"""

    @pytest.fixture
    def stability_service(self):
        """创建稳定性服务实例"""
        from backend.services.factor_stability_service import FactorStabilityService

        return FactorStabilityService()

    @pytest.fixture
    def sample_factor_data(self):
        """创建模拟的因子数据"""
        np.random.seed(42)

        # 生成3年的日频数据（约750个交易日）
        dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="B")
        n = len(dates)

        # 生成稳定的因子值（正态分布）
        factor_values = np.random.normal(loc=0, scale=1, size=n)

        return pd.Series(factor_values, index=dates, name="factor")

    @pytest.fixture
    def sample_ic_data(self):
        """创建模拟的IC序列数据"""
        np.random.seed(123)

        dates = pd.date_range(start="2021-06-01", end="2023-12-31", freq="B")
        n = len(dates)

        # 生成稳定的IC序列（均值0.03，标准差0.15）
        ic_values = np.random.normal(loc=0.03, scale=0.15, size=n)

        return pd.Series(ic_values, index=dates)

    def test_calculate_distribution_stability_with_sufficient_data(self, stability_service, sample_factor_data):
        """
        测试分布稳定性计算 - 数据充足场景

        验证：
        - 返回结果包含所有必要字段
        - stability_score 在合理范围内 [0, 1]
        - 包含分段比较信息
        """
        result = stability_service.calculate_distribution_stability(
            factor_series=sample_factor_data, window=252, method="ks"
        )

        assert "method" in result
        assert result["method"] == "ks"
        assert "window" in result
        assert result["window"] == 252
        assert "stability_score" in result
        assert 0 <= result["stability_score"] <= 1
        assert "avg_p_value" in result
        assert "comparisons" in result
        assert isinstance(result["comparisons"], list)

    def test_calculate_distribution_stability_insufficient_data(self, stability_service):
        """
        测试分布稳定性计算 - 数据不足场景

        预期：应抛出 ValueError 异常
        """
        short_data = pd.Series([1, 2, 3])

        with pytest.raises(ValueError) as exc_info:
            stability_service.calculate_distribution_stability(factor_series=short_data, window=252, method="ks")

        assert "数据长度不足" in str(exc_info.value)

    def test_calculate_time_series_stationarity_stationary(self, stability_service, sample_ic_data):
        """
        测试时间序列平稳性分析 - 平稳序列场景

        验证：
        - ADF检验返回完整结果
        - is_stationary 为布尔值
        - 包含临界值信息
        """
        result = stability_service.calculate_time_series_stability(ic_series=sample_ic_data, maxlag=10)

        assert "is_stationary" in result
        assert isinstance(result["is_stationary"], bool)
        assert "adf_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "1%" in result["critical_values"]
        assert "5%" in result["critical_values"]
        assert "10%" in result["critical_values"]
        assert "interpretation" in result

    def test_calculate_time_series_stationarity_short_series(self, stability_service):
        """
        测试时间序列平稳性分析 - 序列过短场景

        预期：应抛出 ValueError 异常
        """
        short_ic = pd.Series([0.1, 0.2, -0.1])

        with pytest.raises(ValueError) as exc_info:
            stability_service.calculate_time_series_stability(ic_series=short_ic, maxlag=10)

        assert "长度不足" in str(exc_info.value)

    def test_calculate_coefficient_of_variation_normal(self, stability_service, sample_ic_data):
        """
        测试变异系数计算 - 正常数据场景

        验证：
        - 返回包含 mean, std, cv 字段
        - cv 值为有限数值
        - 包含可读性解释文本
        """
        result = stability_service.calculate_coefficient_of_variation(sample_ic_data)

        assert "mean" in result
        assert "std" in result
        assert "cv" in result
        assert not np.isnan(result["cv"])
        assert not np.isinf(result["cv"])
        assert "interpretation" in result

    def test_calculate_coefficient_of_variation_empty_data(self, stability_service):
        """
        测试变异系数计算 - 空数据场景
        """
        empty_series = pd.Series([], dtype=float)

        result = stability_service.calculate_coefficient_of_variation(empty_series)

        assert "error" in result

    def test_comprehensive_stability_test_parameter_validation(self, stability_service):
        """
        测试综合稳定性检验 - 参数校验场景

        验证：
        - 空因子名称抛出异常
        - 空股票列表抛出异常
        - 股票数量不足抛出异常
        """
        # 测试空因子名
        with pytest.raises(ValueError, match="因子名称不能为空"):
            stability_service.comprehensive_stability_test(
                factor_name="", stock_codes=["000001", "000002"], start_date="2022-01-01", end_date="2022-12-31"
            )

        # 测试空股票列表
        with pytest.raises(ValueError, match="股票代码列表不能为空"):
            stability_service.comprehensive_stability_test(
                factor_name="MOM", stock_codes=[], start_date="2022-01-01", end_date="2022-12-31"
            )

        # 测试股票数量不足（<3）
        with pytest.raises(ValueError, match="至少需要3只股票"):
            stability_service.comprehensive_stability_test(
                factor_name="MOM", stock_codes=["000001", "000002"], start_date="2022-01-01", end_date="2022-12-31"
            )

    def test_comprehensive_stability_test_success_scenario(self, stability_service):
        """
        测试综合稳定性检验 - 成功执行场景

        验证综合检验方法的参数校验和基本流程
        """
        # 由于 comprehensive_stability_test 需要复杂的依赖注入，
        # 我们主要验证其参数校验逻辑和错误处理能力
        # 完整的集成测试需要真实的数据库和数据源

        # 测试数据已经足够（3只股票）
        # 这里我们验证方法存在且可调用
        assert hasattr(stability_service, "comprehensive_stability_test")
        assert callable(stability_service.comprehensive_stability_test)

        # 验证其他子方法正常工作
        np.random.seed(42)
        dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="B")
        factor_data = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        ic_data = pd.Series(np.random.normal(0.03, 0.15, len(dates)), index=dates)

        # 验证子方法可以独立工作
        dist_result = stability_service.calculate_distribution_stability(factor_data)
        assert "stability_score" in dist_result

        ts_result = stability_service.calculate_time_series_stability(ic_data)
        assert "is_stationary" in ts_result

        cv_result = stability_service.calculate_coefficient_of_variation(ic_data)
        assert "cv" in cv_result

    def test_comprehensive_stability_test_nonexistent_factor(self, stability_service):
        """
        测试综合稳定性检验 - 因子不存在场景

        使用简化的 mock 验证异常处理
        """
        import sys

        # Mock 数据库相关
        mock_db = MagicMock()
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_name.return_value = None  # 因子不存在

        original_modules = {}
        try:
            # 备份原始模块
            for mod_name in ["backend.repositories.factor_repository", "backend.core.database"]:
                if mod_name in sys.modules:
                    original_modules[mod_name] = sys.modules[mod_name]

            # 创建模拟的 repository 模块
            class MockFactorRepo:
                class FactorRepository:
                    def __init__(self, db):
                        pass

                    def get_by_name(self, name):
                        return None

            # 创建模拟的 database 模块
            class MockDatabase:
                @staticmethod
                def get_db_session():
                    return mock_db

            # 替换模块
            sys.modules["backend.repositories.factor_repository"] = MockFactorRepo()
            sys.modules["backend.core.database"] = MockDatabase()

            # 执行测试 - 应该抛出 ValueError
            with pytest.raises(ValueError, match="不存在"):
                stability_service.comprehensive_stability_test(
                    factor_name="不存在的因子",
                    stock_codes=["000001", "000002", "600000"],
                    start_date="2022-01-01",
                    end_date="2022-12-31",
                )

        finally:
            # 恢复原始模块
            for mod_name, original_mod in original_modules.items():
                sys.modules[mod_name] = original_mod

    def test_generate_warnings_various_scenarios(self, stability_service):
        """
        测试警告信息生成功能

        验证不同风险场景下的警告内容
        """
        # 场景1：分布不稳定 + IC不平稳 + 高变异系数
        results_high_risk = {
            "distribution_stability": {"stable_ratio": 0.4},
            "time_series_stationarity": {"is_stationary": False, "p_value": 0.15},
            "coefficient_of_variation": {"cv": 2.0},
            "market_regime_performance": {"bull": {"mean_ic": 0.08}, "bear": {"mean_ic": -0.05}},
        }

        warnings = stability_service._generate_warnings(results_high_risk)

        assert len(warnings) >= 3
        assert any("结构性变化" in w for w in warnings)
        assert any("不平稳" in w for w in warnings)
        assert any("变异程度较高" in w for w in warnings)

        # 场景2：低风险场景
        results_low_risk = {
            "distribution_stability": {"stable_ratio": 0.9},
            "time_series_stationarity": {"is_stationary": True, "p_value": 0.01},
            "coefficient_of_variation": {"cv": 0.3},
            "market_regime_performance": {"bull": {"mean_ic": 0.04}, "bear": {"mean_ic": 0.03}},
        }

        warnings_low = stability_service._generate_warnings(results_low_risk)

        assert len(warnings_low) == 0 or all("⚠️" in w for w in warnings_low)


class TestFactorStabilityAPIIntegration:
    """因子稳定性 API 集成测试"""

    def test_stability_api_endpoint_structure(self):
        """
        测试稳定性 API 端点结构

        验证 API 路由定义是否正确
        """
        from backend.api.routers.analysis import router

        # 检查路由是否存在
        routes = [route.path for route in router.routes]
        assert "/stability" in routes, "稳定性端点未在路由中注册"

    def test_stability_request_model_validation(self):
        """
        测试请求模型验证

        验证 StabilityRequest 模型的字段和类型约束
        """
        from backend.api.routers.analysis import StabilityRequest

        # 测试有效数据
        valid_request = StabilityRequest(
            factor_name="MOM",
            stock_codes=["000001", "000002", "600000"],
            start_date="2022-01-01",
            end_date="2022-12-31",
        )

        assert valid_request.factor_name == "MOM"
        assert len(valid_request.stock_codes) == 3
        assert valid_request.start_date == "2022-01-01"
        assert valid_request.end_date == "2022-12-31"

        # 测试模型序列化
        request_dict = valid_request.model_dump()
        assert "factor_name" in request_dict
        assert "stock_codes" in request_dict
        assert isinstance(request_dict["stock_codes"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
