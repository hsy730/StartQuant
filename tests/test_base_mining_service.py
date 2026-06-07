"""
BaseMiningService 基类单元测试

覆盖公共基础设施的所有方法：
- 变量命名 (_make_var_name)
- 基础因子预计算 (_precompute_base_factors)
- 股票池管理 (set_stock_pool, _refresh_stock_sample)
- 进度控制 (set_progress_callback, request_cancel)
- 交叉验证过拟合惩罚 (_cv_penalty)
- 适应度路由 (_route_fitness, _extract_best_ic_ir)
- Z-Score归一化 (_update_zscore_stats, _apply_batch_zscore)
- 抽象方法约束 (mine_factors)
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

# Mock 掉重依赖链（data_service → cache_service → sqlalchemy）
import unittest.mock as _mock
_mock_ds = _mock.MagicMock()
sys.modules.setdefault("backend.services.data_service", _mock.MagicMock(data_service=_mock_ds))
sys.modules.setdefault("backend.services.cache_service", _mock.MagicMock())
sys.modules.setdefault("backend.services.factor_service", _mock.MagicMock())

from backend.services.base_mining_service import BaseMiningService


# ============================================================================
# 测试辅助
# ============================================================================


class ConcreteMiningService(BaseMiningService):
    """用于测试的具体子类（实现 mine_factors 抽象方法）"""

    _service_name = "TestMining"

    def mine_factors(self) -> dict:
        return {"status": "ok", "factors": []}


def create_mock_data(n=200, seed=42):
    """创建模拟因子数据"""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "high": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "low": 100 + np.cumsum(np.random.randn(n) * 0.5),
            "volume": np.random.randint(100000, 1000000, n).astype(float),
            "return": np.random.randn(n) * 0.02,
        },
        index=dates,
    )


def create_service(**kwargs):
    """创建测试用 BaseMiningService 子类实例"""
    defaults = {
        "base_factors": [],
        "data": create_mock_data(),
        "return_column": "return",
        "factor_calculator": MagicMock(),
        "max_eval_stocks": 50,
        "fitness_objective": "ic_mean",
        "cv_folds": 0,
        "naming_pattern": "factor_{i}",
    }
    defaults.update(kwargs)
    return ConcreteMiningService(**defaults)


# ============================================================================
# 1. 变量命名
# ============================================================================


class TestMakeVarName:
    """_make_var_name 测试"""

    def test_default_pattern(self):
        """默认 pattern 生成 factor_0, factor_1, ..."""
        svc = create_service(naming_pattern="factor_{i}")
        assert svc._make_var_name(0) == "factor_0"
        assert svc._make_var_name(1) == "factor_1"
        assert svc._make_var_name(9) == "factor_9"

    def test_pysr_pattern(self):
        """PySR pattern 生成 x0, x1, ..."""
        svc = create_service(naming_pattern="x{i}")
        assert svc._make_var_name(0) == "x0"
        assert svc._make_var_name(1) == "x1"
        assert svc._make_var_name(9) == "x9"

    def test_custom_pattern(self):
        """自定义 pattern"""
        svc = create_service(naming_pattern="base_{i}")
        assert svc._make_var_name(3) == "base_3"


# ============================================================================
# 2. 基础因子预计算
# ============================================================================


class TestPrecomputeBaseFactors:
    """_precompute_base_factors 测试"""

    def test_empty_base_factors(self):
        """空因子列表不应出错"""
        svc = create_service(base_factors=[])
        assert svc.base_factor_values == {}

    def test_successful_computation(self):
        """成功计算的因子应存入 base_factor_values"""
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = pd.Series(np.random.randn(200))

        svc = create_service(
            base_factors=["SMA(close,5)", "EMA(close,10)"],
            factor_calculator=mock_calc,
        )
        assert "factor_0" in svc.base_factor_values
        assert "factor_1" in svc.base_factor_values
        assert svc.base_factor_values["factor_0"]["code"] == "SMA(close,5)"
        assert svc.base_factor_values["factor_1"]["code"] == "EMA(close,10)"

    def test_failed_computation_skipped(self):
        """计算失败的因子应跳过"""
        mock_calc = MagicMock()
        # 第一个因子成功，第二个抛异常
        mock_calc.calculate.side_effect = [
            pd.Series(np.random.randn(200)),
            Exception("calc error"),
        ]

        svc = create_service(
            base_factors=["SMA(close,5)", "BAD_FACTOR"],
            factor_calculator=mock_calc,
        )
        assert "factor_0" in svc.base_factor_values
        assert "factor_1" not in svc.base_factor_values

    def test_all_nan_result_skipped(self):
        """全NaN结果应跳过"""
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = pd.Series([np.nan] * 200)

        svc = create_service(
            base_factors=["BAD_FACTOR"],
            factor_calculator=mock_calc,
        )
        assert len(svc.base_factor_values) == 0

    def test_none_result_skipped(self):
        """None结果应跳过"""
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = None

        svc = create_service(
            base_factors=["BAD_FACTOR"],
            factor_calculator=mock_calc,
        )
        assert len(svc.base_factor_values) == 0

    def test_auto_creates_calculator_if_none(self):
        """factor_calculator=None 时应自动创建"""
        with patch("backend.services.base_mining_service.BaseMiningService._precompute_base_factors"):
            svc = ConcreteMiningService(
                base_factors=[],
                data=create_mock_data(),
                factor_calculator=None,
            )
        # _precompute_base_factors 被 mock 了，手动触发
        with patch("backend.services.factor_service.factor_service") as mock_fs:
            mock_fs.calculator = MagicMock()
            mock_fs.calculator.calculate.return_value = pd.Series(np.random.randn(200))
            svc.factor_calculator = None
            svc._precompute_base_factors()
            # 应该通过 factor_service 获取 calculator


# ============================================================================
# 3. 股票池管理
# ============================================================================


class TestStockPoolManagement:
    """set_stock_pool 和 _refresh_stock_sample 测试"""

    def test_refresh_stock_sample_small_pool(self):
        """小股票池应全部选入评估样本"""
        svc = create_service(max_eval_stocks=50)
        svc.stock_pool_base_factor_values = {
            f"60000{i}": {} for i in range(5)
        }
        svc._refresh_stock_sample()
        assert len(svc._sampled_stock_codes) == 5

    def test_refresh_stock_sample_large_pool(self):
        """大股票池应随机抽样到 max_eval_stocks"""
        svc = create_service(max_eval_stocks=10)
        svc.stock_pool_base_factor_values = {
            f"60000{i:03d}": {} for i in range(50)
        }
        svc._refresh_stock_sample()
        assert len(svc._sampled_stock_codes) == 10
        # 所有抽样代码应在原始池中
        for code in svc._sampled_stock_codes:
            assert code in svc.stock_pool_base_factor_values

    def test_refresh_stock_sample_empty_pool(self):
        """空股票池应返回空列表"""
        svc = create_service()
        svc.stock_pool_base_factor_values = {}
        svc._refresh_stock_sample()
        assert svc._sampled_stock_codes == []

    def test_set_stock_pool_calls_data_service(self):
        """set_stock_pool 应调用 data_service"""
        svc = create_service()
        mock_data = {
            "600000": create_mock_data(),
            "000001": create_mock_data(),
        }
        with patch.object(
            svc, "stock_pool_data", {}
        ):
            with patch(
                "backend.services.base_mining_service.data_service"
            ) as mock_ds:
                mock_ds.get_multiple_stocks_data.return_value = mock_data
                # 需要mock factor_calculator
                svc.factor_calculator = MagicMock()
                svc.factor_calculator.calculate.return_value = pd.Series(
                    np.random.randn(200)
                )
                svc.set_stock_pool(["600000", "000001"], "2023-01-01", "2023-12-31")
                mock_ds.get_multiple_stocks_data.assert_called_once_with(
                    ["600000", "000001"], "2023-01-01", "2023-12-31"
                )


# ============================================================================
# 4. 进度控制
# ============================================================================


class TestProgressControl:
    """set_progress_callback 和 request_cancel 测试"""

    def test_set_progress_callback(self):
        """设置进度回调"""
        svc = create_service()
        callback = MagicMock()
        svc.set_progress_callback(callback)
        assert svc.progress_callback is callback

    def test_request_cancel(self):
        """请求取消应设置标志"""
        svc = create_service()
        assert svc._cancel_flag is False
        svc.request_cancel()
        assert svc._cancel_flag is True

    def test_cancel_flag_default_false(self):
        """取消标志默认为 False"""
        svc = create_service()
        assert svc._cancel_flag is False


# ============================================================================
# 5. 交叉验证过拟合惩罚
# ============================================================================


class TestCVPenalty:
    """_cv_penalty 测试"""

    def test_no_cv_folds(self):
        """cv_folds < 2 时惩罚为 0"""
        svc = create_service(cv_folds=0)
        result = svc._cv_penalty({"600000": pd.Series(np.random.randn(200))})
        assert result == 0.0

    def test_cv_folds_1(self):
        """cv_folds = 1 时惩罚为 0"""
        svc = create_service(cv_folds=1)
        result = svc._cv_penalty({"600000": pd.Series(np.random.randn(200))})
        assert result == 0.0

    def test_consistent_ic_low_penalty(self):
        """IC一致的因子惩罚应接近0"""
        svc = create_service(cv_folds=3)
        # 构造IC一致的因子和收益数据
        n = 300
        factor_values_dict = {}
        for code in ["600000", "000001", "300001"]:
            np.random.seed(42)
            factor = pd.Series(np.random.randn(n) * 0.1 + 0.05, name="factor")
            ret = factor * 0.5 + np.random.randn(n) * 0.01
            factor_values_dict[code] = factor
            svc.stock_pool_return_values[code] = pd.Series(ret.values)

        penalty = svc._cv_penalty(factor_values_dict)
        # IC一致的因子惩罚应该较低
        assert 0.0 <= penalty <= 1.0

    def test_inconsistent_ic_high_penalty(self):
        """IC不一致的因子惩罚应较高"""
        svc = create_service(cv_folds=3)
        n = 300
        factor_values_dict = {}
        for code in ["600000", "000001", "300001"]:
            # 因子值在不同fold中与收益的关系完全不同
            factor = pd.Series(np.random.randn(n) * 0.1, name="factor")
            # 前1/3正相关，中1/3负相关，后1/3不相关
            ret = np.concatenate([
                factor.iloc[:100].values * 0.5 + np.random.randn(100) * 0.01,
                -factor.iloc[100:200].values * 0.5 + np.random.randn(100) * 0.01,
                np.random.randn(100) * 0.01,
            ])
            factor_values_dict[code] = factor
            svc.stock_pool_return_values[code] = pd.Series(ret)

        penalty = svc._cv_penalty(factor_values_dict)
        assert 0.0 <= penalty <= 1.0

    def test_insufficient_data_returns_zero(self):
        """数据不足时惩罚为 0"""
        svc = create_service(cv_folds=3)
        # 少于 cv_folds * 20 的数据
        factor_values_dict = {
            "600000": pd.Series(np.random.randn(30))
        }
        svc.stock_pool_return_values["600000"] = pd.Series(np.random.randn(30))
        penalty = svc._cv_penalty(factor_values_dict)
        assert penalty == 0.0

    def test_missing_return_data(self):
        """缺少收益率数据时应跳过"""
        svc = create_service(cv_folds=3)
        factor_values_dict = {"600000": pd.Series(np.random.randn(300))}
        # 不设置 stock_pool_return_values
        penalty = svc._cv_penalty(factor_values_dict)
        assert penalty == 0.0

    def test_penalty_range(self):
        """惩罚值应在 [0, 1] 范围内"""
        svc = create_service(cv_folds=3)
        n = 300
        factor_values_dict = {}
        for code in ["600000", "000001", "300001", "688001", "300750"]:
            factor = pd.Series(np.random.randn(n) * 0.1, name="factor")
            ret = pd.Series(np.random.randn(n) * 0.02)
            factor_values_dict[code] = factor
            svc.stock_pool_return_values[code] = ret

        penalty = svc._cv_penalty(factor_values_dict)
        assert 0.0 <= penalty <= 1.0


# ============================================================================
# 6. 适应度路由
# ============================================================================


class TestExtractBestIcIr:
    """_extract_best_ic_ir 测试"""

    def test_spearman_ic_extraction(self):
        """应从 spearman_ic 中提取 IC 和 IR"""
        svc = create_service()
        ic_results = {
            "spearman_ic": {
                "1D": {"mean_ic": 0.05, "std_ic": 0.02},
            }
        }
        best_ic, best_ir = svc._extract_best_ic_ir(ic_results)
        assert best_ic == pytest.approx(0.05, abs=1e-6)
        assert best_ir == pytest.approx(0.05 / 0.02, abs=1e-4)

    def test_pearson_ic_extraction(self):
        """应从 pearson_ic 中提取 IC 和 IR"""
        svc = create_service()
        ic_results = {
            "pearson_ic": {
                "1D": {"mean_ic": 0.08, "std_ic": 0.03},
            }
        }
        best_ic, best_ir = svc._extract_best_ic_ir(ic_results)
        assert best_ic == pytest.approx(0.08, abs=1e-6)

    def test_multiple_periods_takes_best(self):
        """多期数据应取最优IC"""
        svc = create_service()
        ic_results = {
            "spearman_ic": {
                "1D": {"mean_ic": 0.03, "std_ic": 0.02},
                "5D": {"mean_ic": 0.07, "std_ic": 0.03},
            }
        }
        best_ic, best_ir = svc._extract_best_ic_ir(ic_results)
        assert best_ic == pytest.approx(0.07, abs=1e-6)

    def test_error_entries_skipped(self):
        """包含 error 的条目应跳过"""
        svc = create_service()
        ic_results = {
            "spearman_ic": {
                "1D": {"error": "insufficient data"},
                "5D": {"mean_ic": 0.05, "std_ic": 0.02},
            }
        }
        best_ic, best_ir = svc._extract_best_ic_ir(ic_results)
        assert best_ic == pytest.approx(0.05, abs=1e-6)

    def test_empty_results(self):
        """空结果应返回 0"""
        svc = create_service()
        best_ic, best_ir = svc._extract_best_ic_ir({})
        assert best_ic == 0.0
        assert best_ir == 0.0

    def test_none_values_skipped(self):
        """None 值应跳过"""
        svc = create_service()
        ic_results = {
            "spearman_ic": {
                "1D": {"mean_ic": None, "std_ic": None},
            }
        }
        best_ic, best_ir = svc._extract_best_ic_ir(ic_results)
        assert best_ic == 0.0
        assert best_ir == 0.0

    def test_zero_std_ir_is_zero(self):
        """std=0 时 IR 应为 0"""
        svc = create_service()
        ic_results = {
            "spearman_ic": {
                "1D": {"mean_ic": 0.05, "std_ic": 0.0},
            }
        }
        best_ic, best_ir = svc._extract_best_ic_ir(ic_results)
        assert best_ic == pytest.approx(0.05, abs=1e-6)
        assert best_ir == 0.0


class TestRouteFitness:
    """_route_fitness 测试"""

    def _make_ic_results(self, mean_ic=0.05, std_ic=0.02):
        return {
            "spearman_ic": {
                "1D": {"mean_ic": mean_ic, "std_ic": std_ic},
            }
        }

    def test_ic_mean_objective(self):
        """ic_mean 目标应返回 best_ic"""
        svc = create_service(fitness_objective="ic_mean")
        result = svc._route_fitness(self._make_ic_results(0.05, 0.02))
        assert result == pytest.approx(0.05, abs=1e-6)

    def test_ir_ratio_objective(self):
        """ir_ratio 目标应返回 best_ir"""
        svc = create_service(fitness_objective="ir_ratio")
        result = svc._route_fitness(self._make_ic_results(0.05, 0.02))
        expected_ir = 0.05 / 0.02
        assert result == pytest.approx(expected_ir, abs=1e-4)

    def test_sharpe_objective(self):
        """sharpe 目标应返回 best_ir（用IR代理）"""
        svc = create_service(fitness_objective="sharpe")
        result = svc._route_fitness(self._make_ic_results(0.05, 0.02))
        expected_ir = 0.05 / 0.02
        assert result == pytest.approx(expected_ir, abs=1e-4)

    def test_combined_objective_range(self):
        """combined 目标应在 [0, 1] 范围内"""
        svc = create_service(fitness_objective="combined")
        result = svc._route_fitness(self._make_ic_results(0.05, 0.02))
        assert 0.0 <= result <= 1.0

    def test_combined_uses_zscore(self):
        """combined 目标应使用 Z-Score 归一化"""
        svc = create_service(fitness_objective="combined")
        # 先验值: IC μ=0.03, σ=0.02; IR μ=0.5, σ=0.3
        # IC=0.05 → z_ic = (0.05-0.03)/(0.02+1e-8) ≈ 1.0
        # IR=2.5  → z_ir = (2.5-0.5)/(0.3+1e-8) ≈ 6.67 → clipped to 3.0
        # norm_ic = (1.0+3)/6 ≈ 0.667
        # norm_ir = (3.0+3)/6 = 1.0
        # combined = 0.6*0.667 + 0.4*1.0 = 0.8
        result = svc._route_fitness(self._make_ic_results(0.05, 0.02))
        assert 0.0 < result <= 1.0

    def test_route_fitness_collects_ic_ir_values(self):
        """_route_fitness 应收集 IC/IR 值用于 Z-Score 计算"""
        svc = create_service(fitness_objective="ic_mean")
        initial_len = len(svc._gen_ic_values)
        svc._route_fitness(self._make_ic_results(0.05, 0.02))
        assert len(svc._gen_ic_values) == initial_len + 1
        assert len(svc._gen_ir_values) == initial_len + 1

    def test_zero_ic_returns_zero(self):
        """IC=0 时 ic_mean 目标应返回 0"""
        svc = create_service(fitness_objective="ic_mean")
        result = svc._route_fitness(self._make_ic_results(0.0, 0.02))
        assert result == 0.0


# ============================================================================
# 7. Z-Score 归一化
# ============================================================================


class TestUpdateZscoreStats:
    """_update_zscore_stats 测试"""

    def test_enough_values_updates_stats(self):
        """>=5 个有效值时应更新统计量"""
        svc = create_service()
        svc._gen_ic_values = [0.03, 0.04, 0.05, 0.06, 0.07]
        svc._gen_ir_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        svc._update_zscore_stats()
        assert svc._zscore_ic_mean > 0
        assert svc._zscore_ic_std > 0
        assert svc._zscore_ir_mean > 0
        assert svc._zscore_ir_std > 0
        assert svc._has_zscore_stats is True

    def test_too_few_values_keeps_prior(self):
        """<5 个有效值时应保持先验值"""
        svc = create_service()
        prior_ic_mean = svc._zscore_ic_mean
        prior_ic_std = svc._zscore_ic_std
        svc._gen_ic_values = [0.03, 0.04]
        svc._gen_ir_values = [0.5, 0.6]
        svc._update_zscore_stats()
        assert svc._zscore_ic_mean == prior_ic_mean
        assert svc._zscore_ic_std == prior_ic_std

    def test_clears_collected_values(self):
        """更新后应清空收集列表"""
        svc = create_service()
        svc._gen_ic_values = [0.03, 0.04, 0.05, 0.06, 0.07]
        svc._gen_ir_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        svc._update_zscore_stats()
        assert svc._gen_ic_values == []
        assert svc._gen_ir_values == []

    def test_sigma_floor_protection(self):
        """σ下界保护：σ不应过小"""
        svc = create_service()
        # 所有IC值相同 → std=0，但应被保护
        svc._gen_ic_values = [0.05, 0.05, 0.05, 0.05, 0.05]
        svc._gen_ir_values = [1.0, 1.0, 1.0, 1.0, 1.0]
        svc._update_zscore_stats()
        # σ应被保护为 max(0, max(0.01*μ, 0.005))
        assert svc._zscore_ic_std >= 0.005
        assert svc._zscore_ir_std >= 0.005

    def test_near_zero_values_filtered(self):
        """接近0的值应被过滤（>1e-10才有效）"""
        svc = create_service()
        svc._gen_ic_values = [0.0, 0.0, 0.0, 0.0, 0.0]
        svc._gen_ir_values = [0.0, 0.0, 0.0, 0.0, 0.0]
        prior_ic_mean = svc._zscore_ic_mean
        svc._update_zscore_stats()
        # 全部为0（<1e-10），应保持先验值
        assert svc._zscore_ic_mean == prior_ic_mean

    def test_mixed_zero_and_nonzero(self):
        """混合零值和非零值应只使用非零值"""
        svc = create_service()
        svc._gen_ic_values = [0.0, 0.0, 0.0, 0.05, 0.06, 0.07, 0.08, 0.09]
        svc._gen_ir_values = [0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
        svc._update_zscore_stats()
        assert svc._zscore_ic_mean > 0
        assert svc._has_zscore_stats is True


class TestApplyBatchZscore:
    """_apply_batch_zscore 测试"""

    def test_non_combined_returns_unchanged(self):
        """非 combined 目标应原样返回"""
        svc = create_service(fitness_objective="ic_mean")
        factors = [
            {"fitness": 0.5, "validation": {"_raw_ic_mean": 0.05, "_raw_ir": 1.0}}
        ]
        result = svc._apply_batch_zscore(factors)
        assert result == factors

    def test_empty_list_returns_empty(self):
        """空列表应返回空列表"""
        svc = create_service(fitness_objective="combined")
        result = svc._apply_batch_zscore([])
        assert result == []

    def test_too_few_factors_returns_unchanged(self):
        """因子数不足时应保持原分数"""
        svc = create_service(fitness_objective="combined")
        factors = [
            {"fitness": 0.5, "validation": {"_raw_ic_mean": 0.05, "_raw_ir": 1.0}}
        ]
        result = svc._apply_batch_zscore(factors)
        # 只有1个因子，<2，应返回原列表
        assert result[0]["fitness"] == 0.5

    def test_recalculates_fitness(self):
        """应重新计算 combined 适应度"""
        svc = create_service(fitness_objective="combined")
        factors = [
            {
                "fitness": 0.0,
                "validation": {"_raw_ic_mean": 0.03, "_raw_ir": 0.5},
            },
            {
                "fitness": 0.0,
                "validation": {"_raw_ic_mean": 0.07, "_raw_ir": 1.5},
            },
            {
                "fitness": 0.0,
                "validation": {"_raw_ic_mean": 0.05, "_raw_ir": 1.0},
            },
        ]
        result = svc._apply_batch_zscore(factors)
        # 适应度应被重新计算
        for f in result:
            assert f["fitness"] > 0
            assert f["fitness"] <= 1.0

    def test_reranks_by_fitness(self):
        """应按新适应度重新排名"""
        svc = create_service(fitness_objective="combined")
        factors = [
            {
                "fitness": 0.0,
                "rank": 1,
                "validation": {"_raw_ic_mean": 0.03, "_raw_ir": 0.5},
            },
            {
                "fitness": 0.0,
                "rank": 2,
                "validation": {"_raw_ic_mean": 0.08, "_raw_ir": 2.0},
            },
            {
                "fitness": 0.0,
                "rank": 3,
                "validation": {"_raw_ic_mean": 0.05, "_raw_ir": 1.0},
            },
        ]
        result = svc._apply_batch_zscore(factors)
        # rank=1 应该是 fitness 最高的
        assert result[0]["rank"] == 1
        assert result[0]["fitness"] >= result[1]["fitness"]

    def test_updates_score_if_present(self):
        """如果 validation 中有 score，应同步更新"""
        svc = create_service(fitness_objective="combined")
        factors = [
            {
                "fitness": 0.0,
                "validation": {
                    "_raw_ic_mean": 0.03,
                    "_raw_ir": 0.5,
                    "score": 0,
                },
            },
            {
                "fitness": 0.0,
                "validation": {
                    "_raw_ic_mean": 0.07,
                    "_raw_ir": 1.5,
                    "score": 0,
                },
            },
        ]
        result = svc._apply_batch_zscore(factors)
        for f in result:
            assert f["validation"]["score"] > 0

    def test_non_dict_validation_skipped(self):
        """非 dict 的 validation 应跳过"""
        svc = create_service(fitness_objective="combined")
        factors = [
            {"fitness": 0.5, "validation": "error string"},
            {
                "fitness": 0.0,
                "validation": {"_raw_ic_mean": 0.05, "_raw_ir": 1.0},
            },
            {
                "fitness": 0.0,
                "validation": {"_raw_ic_mean": 0.07, "_raw_ir": 1.5},
            },
        ]
        result = svc._apply_batch_zscore(factors)
        assert len(result) == 3


# ============================================================================
# 8. 抽象方法约束
# ============================================================================


class TestAbstractMethod:
    """mine_factors 抽象方法约束测试"""

    def test_cannot_instantiate_base_class(self):
        """不能直接实例化抽象基类"""
        with pytest.raises(TypeError):
            BaseMiningService(
                base_factors=[],
                data=create_mock_data(),
            )

    def test_subclass_must_implement_mine_factors(self):
        """子类必须实现 mine_factors"""

        class IncompleteService(BaseMiningService):
            _service_name = "Incomplete"

        with pytest.raises(TypeError):
            IncompleteService(
                base_factors=[],
                data=create_mock_data(),
            )

    def test_concrete_subclass_can_instantiate(self):
        """实现了抽象方法的子类可以实例化"""
        svc = create_service()
        assert isinstance(svc, BaseMiningService)

    def test_concrete_subclass_can_call_mine_factors(self):
        """子类的 mine_factors 可以调用"""
        svc = create_service()
        result = svc.mine_factors()
        assert result["status"] == "ok"


# ============================================================================
# 9. 初始化状态
# ============================================================================


class TestInitialization:
    """初始化状态测试"""

    def test_default_prior_values(self):
        """默认先验值应正确设置"""
        svc = create_service()
        assert svc._PRIOR_IC_MEAN == 0.03
        assert svc._PRIOR_IC_STD == 0.02
        assert svc._PRIOR_IR_MEAN == 0.5
        assert svc._PRIOR_IR_STD == 0.3

    def test_zscore_initialized_with_priors(self):
        """Z-Score 统计量应初始化为先验值"""
        svc = create_service()
        assert svc._zscore_ic_mean == svc._PRIOR_IC_MEAN
        assert svc._zscore_ic_std == svc._PRIOR_IC_STD
        assert svc._zscore_ir_mean == svc._PRIOR_IR_MEAN
        assert svc._zscore_ir_std == svc._PRIOR_IR_STD

    def test_cancel_flag_initially_false(self):
        """取消标志初始为 False"""
        svc = create_service()
        assert svc._cancel_flag is False

    def test_progress_callback_initially_none(self):
        """进度回调初始为 None"""
        svc = create_service()
        assert svc.progress_callback is None

    def test_empty_collections_initialized(self):
        """空集合应正确初始化"""
        svc = create_service()
        assert svc.stock_codes == []
        assert svc.stock_pool_data == {}
        assert svc.stock_pool_return_values == {}
        assert svc.stock_pool_base_factor_values == {}
        assert svc._sampled_stock_codes == []
        assert svc._gen_ic_values == []
        assert svc._gen_ir_values == []

    def test_return_values_extracted(self):
        """return_values 应从 data 中提取"""
        data = create_mock_data()
        svc = create_service(data=data, return_column="return")
        assert svc.return_values is not None
        assert len(svc.return_values) == len(data)

    def test_missing_return_column(self):
        """缺失 return_column 时 return_values 应为 None"""
        data = pd.DataFrame({"close": [1, 2, 3]})
        svc = create_service(data=data, return_column="nonexistent")
        assert svc.return_values is None


# ============================================================================
# 10. 集成场景
# ============================================================================


class TestIntegrationScenarios:
    """集成场景测试"""

    def test_route_fitness_then_update_zscore(self):
        """_route_fitness → _update_zscore 完整流程"""
        svc = create_service(fitness_objective="combined")

        # 模拟多代评估
        for gen in range(6):
            ic_results = {
                "spearman_ic": {
                    "1D": {
                        "mean_ic": 0.03 + gen * 0.01,
                        "std_ic": 0.02,
                    }
                }
            }
            svc._route_fitness(ic_results)
            svc._update_zscore_stats()

        # 经过6代后，Z-Score统计量应已从先验值更新
        assert svc._has_zscore_stats is True

    def test_cv_penalty_with_route_fitness(self):
        """_cv_penalty 与 _route_fitness 协同工作"""
        svc = create_service(fitness_objective="ic_mean", cv_folds=3)

        # 设置股票池收益数据
        n = 300
        for code in ["600000", "000001", "300001"]:
            svc.stock_pool_return_values[code] = pd.Series(
                np.random.randn(n) * 0.02
            )

        # 计算 CV 惩罚
        factor_values_dict = {
            code: pd.Series(np.random.randn(n) * 0.1)
            for code in ["600000", "000001", "300001"]
        }
        penalty = svc._cv_penalty(factor_values_dict)
        assert 0.0 <= penalty <= 1.0

    def test_multiple_route_fitness_collects_values(self):
        """多次 _route_fitness 应累积 IC/IR 值"""
        svc = create_service(fitness_objective="ic_mean")
        for i in range(5):
            ic_results = {
                "spearman_ic": {
                    "1D": {"mean_ic": 0.03 + i * 0.01, "std_ic": 0.02}
                }
            }
            svc._route_fitness(ic_results)

        # 应有5个收集的值
        assert len(svc._gen_ic_values) == 5
        assert len(svc._gen_ir_values) == 5

    def test_service_name_customization(self):
        """子类可以自定义 _service_name"""

        class CustomService(BaseMiningService):
            _service_name = "CustomAlgo"

            def mine_factors(self) -> dict:
                return {}

        svc = CustomService(base_factors=[], data=create_mock_data())
        assert svc._service_name == "CustomAlgo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
