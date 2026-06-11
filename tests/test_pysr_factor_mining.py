"""
PySR 因子挖掘服务单元测试

覆盖之前发现并修复的核心问题：
1. IC/IR 字段提取：analyze_ic() 返回嵌套格式 {pearson_ic: {1D: {mean_ic, ir}}}
2. IR 上限保护：min(ir_val, 5.0) 防止异常值爆炸
3. best_fv/best_ret 初始化位置：确保 alphalens 回退路径可用
4. 横截面 vs 单股票模式切换
5. _route_fitness 正确路由 fitness 目标
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from backend.services.pysr_factor_mining_service import PySRFactorMiningService  # noqa: E402

# ============================================================================
# 测试数据工厂
# ============================================================================


def create_mock_stock_data(n_days=100, seed=42):
    """创建模拟股票 OHLCV 数据"""
    np.random.seed(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)

    close = 10 + np.cumsum(np.random.randn(n_days) * 0.5)
    open_price = close + np.random.randn(n_days) * 0.3
    high = np.maximum(close, open_price) + abs(np.random.randn(n_days) * 0.2)
    low = np.minimum(close, open_price) - abs(np.random.randn(n_days) * 0.2)
    volume = 1000000 + np.random.randint(0, 500000, n_days).astype(float)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "return": np.concatenate([[np.nan], np.diff(close) / close[:-1]]),
        }
    ).set_index("date")


def create_base_factor_values(pricing_df):
    """创建基础因子值（模拟 PySR 输入）"""
    return {
        "close": {"values": pricing_df["close"]},
        "open": {"values": pricing_df["open"]},
        "high": {"values": pricing_df["high"]},
        "low": {"values": pricing_df["low"]},
        "volume": {"values": pricing_df["volume"]},
    }


def create_stock_factor_map(stock_codes, base_factor_values_template):
    """为多只股票创建因子映射"""
    stock_factor_map = {}
    for code in stock_codes:
        stock_data = pd.DataFrame(index=base_factor_values_template["close"]["values"].index)
        for name, info in base_factor_values_template.items():
            noise = np.random.randn(len(info["values"])) * 0.01
            stock_data[name] = info["values"] + noise
        stock_factor_map[code] = stock_data
    return stock_factor_map


def create_alphalens_ic_results(mean_ic=0.05, std_ic=0.1, periods=["1D"]):
    """
    创建模拟的 analyze_ic() 返回格式

    这是修复 Bug #1 的关键：返回格式必须是嵌套的：
    {
        "pearson_ic": {"1D": {"mean_ic": ..., "std_ic": ..., "ir": ...}},
        "spearman_ic": {"1D": {"mean_ic": ..., "std_ic": ..., "ir": ...}},
        "stability": ...
    }
    """
    results = {
        "stability": 0.8,
    }

    for ic_type in ["pearson_ic", "spearman_ic"]:
        results[ic_type] = {}
        for period in periods:
            if std_ic > 1e-10:
                ir = mean_ic / std_ic
            else:
                ir = 0.0

            results[ic_type][period] = {
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ir": ir,
                "ic_count": len(periods) * 20,
                "risk_adjusted_ic": mean_ic / (std_ic + 1e-10),
            }

    return results


def create_pysr_service_with_mocks():
    """
    创建一个配置好 mock 状态的 PySR 服务实例

    使用正确的构造函数签名：
    PySRFactorMiningService(base_factors, data, ...)

    注意：data_service 在 base_mining_service.py 中导入，
    factor_service 在 _precompute_base_factors 中延迟导入。
    通过提供 factor_calculator mock 避免延迟导入，
    通过 patch base_mining_service.data_service 避免数据库调用。
    """
    # 创建基础数据
    df = create_mock_stock_data(n_days=120)
    base_factors = ["close", "open", "high", "low", "volume"]

    # 创建 mock factor_calculator，避免 _precompute_base_factors 延迟导入 factor_service
    mock_calculator = MagicMock()
    # 让 calculate 返回合理的因子值序列
    for factor_name in base_factors:
        mock_calculator.calculate.side_effect = lambda data, code, _fn=factor_name: data.get(
            _fn, data.get("close", pd.Series(np.random.randn(len(data))))
        )

    with patch("backend.services.base_mining_service.data_service"):
        service = PySRFactorMiningService(
            base_factors=base_factors,
            data=df,
            factor_calculator=mock_calculator,
        )

        # 手动设置内部状态（绕过 set_stock_pool 的数据库依赖）
        service.base_factor_values = create_base_factor_values(df)
        service.stock_pool_data = {}
        service.stock_pool_return_values = {}
        service.use_cross_sectional = True

        return service


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_pricing_df():
    """单个股票的价格数据"""
    return create_mock_stock_data(n_days=120)


@pytest.fixture
def pysr_service():
    """创建配置好的 PySR 服务实例（使用正确的构造函数）"""
    return create_pysr_service_with_mocks()


@pytest.fixture
def multi_stock_setup(pysr_service):
    """多只股票测试设置"""
    codes = ["000001.SZ", "000002.SZ", "600000.SH"]

    for code in codes:
        df = create_mock_stock_data(n_days=120, seed=hash(code) % 10000)
        pysr_service.stock_pool_data[code] = df
        pysr_service.stock_pool_return_values[code] = df["close"].pct_change()

    base_factors = create_base_factor_values(create_mock_stock_data(120))
    stock_factor_map = create_stock_factor_map(codes, base_factors)

    return pysr_service, codes, stock_factor_map


# ============================================================================
# 测试类：IC/IR 字段提取 (Bug #1 修复验证)
# ============================================================================


class TestICIRFieldExtraction:
    """
    验证修复 Bug #1: IC/IR 字段名不匹配

    问题：analyze_ic() 返回 {pearson_ic: {1D: {mean_ic, ir}}}
    但代码错误地使用 ic_results.get("mean_ic", 0) → 总是返回 0

    修复后应正确从嵌套格式中提取
    """

    def test_extract_ic_from_nested_format(self, multi_stock_setup):
        """测试从正确的嵌套格式中提取 IC 值"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return x[:, 0] / x[:, 1]  # close/open

        expected_ic = 0.03
        expected_ir = 0.6

        with patch("backend.services.pysr_factor_mining_service.alphalens_analysis_service") as mock_alphalens:
            mock_factor_data = pd.DataFrame(
                {
                    "factor": np.random.randn(200),
                    "return": np.random.randn(200) * 0.02,
                }
            )
            mock_alphalens.prepare_factor_data.return_value = mock_factor_data
            mock_alphalens.analyze_ic.return_value = create_alphalens_ic_results(
                mean_ic=expected_ic,
                std_ic=expected_ic / expected_ir if expected_ir > 0 else 0.1,
            )

            fitness, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

            assert validation is not None, "validation 不应为 None"
            assert "ic_validation" in validation, "validation 应包含 ic_validation"

            actual_ic = validation["ic_validation"]["ic"]
            assert abs(actual_ic - expected_ic) < 0.001, f"IC 提取错误: 期望 {expected_ic}, 实际 {actual_ic}"

            print(f"✅ IC 正确提取: {actual_ic:.4f} (期望: {expected_ic:.4f})")

    def test_extract_ir_from_nested_format(self, multi_stock_setup):
        """测试从正确的嵌套格式中提取 IR 值"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return x[:, 0] / x[:, 1]

        test_cases = [
            (0.03, 0.15, 0.2),
            (0.05, 0.1, 0.5),
            (0.08, 0.04, 2.0),
        ]

        for mean_ic, std_ic, expected_ir in test_cases:
            with patch("backend.services.pysr_factor_mining_service.alphalens_analysis_service") as mock_alphalens:
                mock_factor_data = pd.DataFrame(
                    {
                        "factor": np.random.randn(200),
                        "return": np.random.randn(200) * 0.02,
                    }
                )
                mock_alphalens.prepare_factor_data.return_value = mock_factor_data
                mock_alphalens.analyze_ic.return_value = create_alphalens_ic_results(mean_ic=mean_ic, std_ic=std_ic)

                _, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

                raw_ir = validation.get("_raw_ir", 0)
                assert abs(raw_ir - expected_ir) < 0.01, f"IR 提取错误: 期望 {expected_ir}, 实际 {raw_ir}"

                print(f"✅ IR={raw_ir:.2f} 正确提取 (mean_ic={mean_ic}, std_ic={std_ic})")


# ============================================================================
# 测试类：IR 上限保护 (Bug #2 修复验证)
# ============================================================================


class TestIRCapping:
    """
    验证修复 Bug #2: IR 上限保护

    问题：单股票 rolling IC 极稳定时 std→0 导致 IR=mean/std 爆炸到 96000+
    修复后 IR 应被限制在 max 5.0
    """

    def test_ir_capped_at_5_for_extreme_values(self, multi_stock_setup):
        """测试极端 IR 值被限制在 5.0 以内"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return x[:, 0]

        extreme_test_cases = [
            (0.9, 0.00001),  # IR ≈ 90000
            (0.95, 0.0001),  # IR ≈ 9500
            (0.8, 0.001),  # IR ≈ 800
            (0.5, 0.05),  # IR = 10
        ]

        for mean_ic, std_ic in extreme_test_cases:
            with patch("backend.services.pysr_factor_mining_service.alphalens_analysis_service") as mock_alphalens:
                mock_factor_data = pd.DataFrame(
                    {
                        "factor": np.random.randn(200),
                        "return": np.random.randn(200) * 0.02,
                    }
                )
                mock_alphalens.prepare_factor_data.return_value = mock_factor_data
                mock_alphalens.analyze_ic.return_value = create_alphalens_ic_results(mean_ic=mean_ic, std_ic=std_ic)

                _, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

                capped_ir = validation["ir_validation"]["ir"]
                assert capped_ir <= 5.0, f"IR 应被限制在 5.0 以内，实际: {capped_ir} (原始 IR≈{mean_ic/std_ic})"

                raw_ir = validation.get("_raw_ir", 0)
                expected_raw = mean_ic / std_ic if std_ic > 0 else 0
                assert abs(raw_ir - expected_raw) < expected_raw * 0.01

                print(f"✅ IR 上限保护生效: raw={raw_ir:.1f} → capped={capped_ir:.1f}")

    def test_normal_ir_not_affected(self, multi_stock_setup):
        """测试正常范围内的 IR 值不受影响"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return x[:, 0]

        normal_cases = [
            (0.02, 0.2, 0.1),
            (0.04, 0.1, 0.4),
            (0.06, 0.06, 1.0),
            (0.08, 0.04, 2.0),
        ]

        for mean_ic, std_ic, expected_ir in normal_cases:
            with patch("backend.services.pysr_factor_mining_service.alphalens_analysis_service") as mock_alphalens:
                mock_factor_data = pd.DataFrame(
                    {
                        "factor": np.random.randn(200),
                        "return": np.random.randn(200) * 0.02,
                    }
                )
                mock_alphalens.prepare_factor_data.return_value = mock_factor_data
                mock_alphalens.analyze_ic.return_value = create_alphalens_ic_results(mean_ic=mean_ic, std_ic=std_ic)

                _, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

                actual_ir = validation["ir_validation"]["ir"]
                assert abs(actual_ir - expected_ir) < 0.01, f"正常 IR 不应被改变: 期望 {expected_ir}, 实际 {actual_ir}"


# ============================================================================
# 测试类：best_fv/best_ret 初始化和回退路径 (Bug #3 & #4 修复验证)
# ============================================================================


class TestFallbackPath:
    """
    验证修复 Bug #3 & #4: best_fv/best_ret 初始化位置和 if-else 逻辑

    问题：
    - best_fv/best_ret 在 alphalens try 块内赋值，导致失败时无法回退
    - if-else 缩进错误导致即使 alphalens 成功也进入错误分支

    修复后：
    - best_fv/best_ret 在 try 块前初始化
    - alphalens 成功时直接返回，失败时才尝试回退
    """

    def test_best_fv_best_ret_initialized_before_alphalens(self, multi_stock_setup):
        """验证 best_fv/best_ret 在 alphalens 尝试之前已初始化，且回退路径工作"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return np.log(x[:, 0] / x[:, 1])

        with patch("backend.services.pysr_factor_mining_service.alphalens_analysis_service") as mock_alphalens:
            from alphalens.utils import MaxLossExceededError

            mock_alphalens.prepare_factor_data.side_effect = MaxLossExceededError("max_loss (35.0%) exceeded 100.0%")

            with patch("backend.services.pysr_factor_mining_service.factor_validation_service") as mock_validator:
                mock_validator.validate_factor.return_value = {
                    "ic_validation": {"ic": 0.91, "passed": True},
                    "ir_validation": {"ir": 5.0, "passed": True},
                    "score": 90.4,
                }

                fitness, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

                assert validation is not None, "回退路径应产生有效的 validation"
                assert "ic_validation" in validation, "回退结果应包含 ic_validation"

                mock_validator.validate_factor.assert_called_once()

                print(f"✅ 回退路径正常工作: IC={validation['ic_validation']['ic']}")

    def test_fallback_path_when_alphalens_returns_error(self, multi_stock_setup):
        """当 alphalens analyze_ic 返回 error 时应触发回退"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return x[:, 0]

        with patch("backend.services.pysr_factor_mining_service.alphalens_analysis_service") as mock_alphalens:
            mock_factor_data = pd.DataFrame(
                {
                    "factor": np.random.randn(200),
                    "return": np.random.randn(200) * 0.02,
                }
            )
            mock_alphalens.prepare_factor_data.return_value = mock_factor_data
            mock_alphalens.analyze_ic.return_value = {"error": "Insufficient data"}

            with patch("backend.services.pysr_factor_mining_service.factor_validation_service") as mock_validator:
                mock_validator.validate_factor.return_value = {
                    "ic_validation": {"ic": 0.05, "passed": True},
                    "ir_validation": {"ir": 0.5, "passed": True},
                    "score": 50.0,
                }

                _, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

                assert validation is not None
                assert validation["ic_validation"]["ic"] == 0.05

                print("✅ alphalens 返回 error 时正确回退到单股票验证")

    def test_no_fallback_when_factor_values_dict_empty(self, pysr_service):
        """当 factor_values_dict 为空时不应尝试回退"""

        def expr_callable(x):
            return x[:, 0]

        empty_stock_factor_map = {}

        fitness, validation = pysr_service._evaluate_pysr_factor(expr_callable, empty_stock_factor_map)

        assert fitness == 0.0
        assert validation == {}, "空的因子映射应返回空 validation"

        print("✅ 空 factor_values_dict 正确处理")


# ============================================================================
# 测试类：_route_fitness 方法
# ============================================================================


class TestRouteFitness:
    """测试 _route_fitness 从嵌套 ic_results 中正确计算 fitness"""

    def setup_method(self):
        df = create_mock_stock_data(n_days=120)
        base_factors = ["close", "open", "high", "low", "volume"]

        mock_calculator = MagicMock()
        for factor_name in base_factors:
            mock_calculator.calculate.side_effect = lambda data, code, _fn=factor_name: data.get(
                _fn, data.get("close", pd.Series(np.random.randn(len(data))))
            )

        with patch("backend.services.base_mining_service.data_service"):
            self.service = PySRFactorMiningService(
                base_factors=base_factors,
                data=df,
                factor_calculator=mock_calculator,
            )

    def test_route_fitness_with_pearson_ic(self):
        """测试从 pearson_ic 提取 fitness"""
        ic_results = create_alphalens_ic_results(mean_ic=0.05, std_ic=0.1)

        fitness = self.service._route_fitness(ic_results)

        assert fitness > 0, "fitness 应大于 0"
        assert isinstance(fitness, float), "fitness 应是浮点数"

    def test_route_fitness_with_ir_objective(self):
        """测试 ir_ratio 目标下的 fitness 计算"""
        self.service.fitness_objective = "ir_ratio"

        ic_results = create_alphalens_ic_results(mean_ic=0.04, std_ic=0.08)
        fitness = self.service._route_fitness(ic_results)

        expected_ir = 0.04 / 0.08
        assert abs(fitness - expected_ir) < 0.01, f"IR objective: 期望 {expected_ir}, 实际 {fitness}"

    def test_route_fitness_handles_zero_std(self):
        """处理 std_ic 接近零的情况（防止除零）"""
        ic_results = create_alphalens_ic_results(mean_ic=0.05, std_ic=1e-12)

        fitness = self.service._route_fitness(ic_results)

        assert not np.isinf(fitness), "fitness 不应是无穷大"
        assert not np.isnan(fitness), "fitness 不应是 NaN"


# ============================================================================
# 测试类：横截面 vs 单股票模式
# ============================================================================


class TestCrossSectionalVsSingleStock:
    """测试不同模式下的行为差异"""

    def test_cross_sectional_mode_uses_alphalens(self, multi_stock_setup):
        """横截面模式应优先使用 alphalens 分析"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return x[:, 0]

        with patch("backend.services.pysr_factor_mining_service.alphalens_analysis_service") as mock_alphalens:
            mock_factor_data = pd.DataFrame(
                {
                    "factor": np.random.randn(200),
                    "return": np.random.randn(200) * 0.02,
                }
            )
            mock_alphalens.prepare_factor_data.return_value = mock_factor_data
            mock_alphalens.analyze_ic.return_value = create_alphalens_ic_results(mean_ic=0.03, std_ic=0.1)

            _, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

            assert validation.get("_alphalens"), "横截面模式应标记使用 alphalens"
            mock_alphalens.prepare_factor_data.assert_called_once()
            mock_alphalens.analyze_ic.assert_called_once()

    def test_insufficient_stocks_returns_empty(self, pysr_service):
        """少于 2 只股票时应返回空结果"""

        def expr_callable(x):
            return x[:, 0]

        single_stock_map = create_stock_factor_map(
            ["000001.SZ"], create_base_factor_values(create_mock_stock_data(120))
        )

        df = create_mock_stock_data(120)
        pysr_service.stock_pool_data["000001.SZ"] = df
        pysr_service.stock_pool_return_values["000001.SZ"] = df["close"].pct_change()

        fitness, validation = pysr_service._evaluate_pysr_factor(expr_callable, single_stock_map)

        assert fitness == 0.0
        assert validation == {}, "不足 2 只股票应返回空结果"


# ============================================================================
# 测试类：边界条件和异常处理
# ============================================================================


class TestEdgeCasesAndExceptions:
    """边界条件和异常情况测试"""

    def test_expr_callable_produces_nan(self, multi_stock_setup):
        """处理 expr_callable 产生大量 NaN 的情况"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return np.full(len(x), np.nan)

        fitness, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)

        assert not np.isnan(fitness), "fitness 不应是 NaN"

    def test_expr_callable_produces_inf(self, multi_stock_setup):
        """处理 expr_callable 产生 inf 的情况"""
        service, codes, stock_factor_map = multi_stock_setup

        def expr_callable(x):
            return x[:, 0] / (x[:, 1] - x[:, 1] + 1e-15)

        fitness, validation = service._evaluate_pysr_factor(expr_callable, stock_factor_map)
        assert isinstance(fitness, float)

    def test_single_stock_evaluate(self):
        """测试单股票评估方法 _evaluate_pysr_factor_single"""
        df = create_mock_stock_data(120)
        base_factors = ["close", "open", "high", "low", "volume"]

        mock_calculator = MagicMock()
        for factor_name in base_factors:
            mock_calculator.calculate.side_effect = lambda data, code, _fn=factor_name: data.get(
                _fn, data.get("close", pd.Series(np.random.randn(len(data))))
            )

        with patch("backend.services.base_mining_service.data_service"):
            service = PySRFactorMiningService(
                base_factors=base_factors,
                data=df,
                factor_calculator=mock_calculator,
            )

            service.base_factor_values = create_base_factor_values(df)
            service.return_values = df["close"].pct_change()

            def expr_callable(x):
                return x[:, 0]

            fitness, validation = service._evaluate_pysr_factor_single(expr_callable)

            assert isinstance(fitness, float)
            assert isinstance(validation, dict)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
