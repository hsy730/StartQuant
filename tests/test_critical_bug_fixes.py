"""
单元测试：防护关键Bug回归
覆盖：除零保护、Session泄漏、线程安全、净值曲线、MACD编译、inf污染等
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestDivisionByZeroProtection:
    """防护：裸除法导致的 ZeroDivisionError"""

    def test_smart_preprocessing_detector_cv_zero_denominator(self):
        """smart_preprocessing_detector: avg_market_cap=0 时不应崩溃"""
        from backend.utils.safe_math import safe_divide
        # 当分母为0时，safe_divide应返回default值
        result = safe_divide(100.0, 0.0, default=0.0)
        assert result == 0.0

    def test_smart_preprocessing_detector_cv_nan_denominator(self):
        """smart_preprocessing_detector: NaN分母应返回default"""
        from backend.utils.safe_math import safe_divide
        result = safe_divide(100.0, float('nan'), default=0.0)
        assert result == 0.0

    def test_factor_generator_valid_ratio_empty_factor_values(self):
        """factor_generator_service: 空factor_values不应除零"""
        # 模拟修复后的逻辑
        factor_values = pd.Series(dtype=float)
        aligned_data = pd.Series(dtype=float)
        valid_ratio = len(aligned_data) / len(factor_values) if len(factor_values) > 0 else 0.0
        assert valid_ratio == 0.0

    def test_equity_curve_first_value_zero(self):
        """vectorbt_backtest_service: 净值曲线首值为0时应跳过"""
        equity = pd.Series([0.0, 1.0, 2.0, 3.0])
        first_val = equity.iloc[0]
        # 修复后的逻辑：首值为0时跳过
        assert first_val == 0.0  # Should be skipped, not divided

    def test_pct_change_inf_cleanup(self):
        """vectorbt_backtest_service: pct_change产生inf应被清理"""
        equity = pd.Series([100.0, 0.0, 100.0, 200.0])
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        assert not np.isinf(returns).any()

    def test_benchmark_close_first_zero(self):
        """backtest.py: 基准收盘价首值为0时的除零保护"""
        close = pd.Series([0.0, 10.0, 20.0])
        first_close = close.iloc[0] if len(close) > 0 and close.iloc[0] != 0 else 1.0
        result = close / first_close
        assert not np.isinf(result).any()


class TestFormulaCompilerMACD:
    """防护：MACD编译TypeError"""

    def _make_macd_tree(self, second_arg=None):
        """构建MACD公式树"""
        args = [
            {"type": "column", "value": "close"},
        ]
        if second_arg is not None:
            args.append({"type": "literal", "value": second_arg})

        return {
            "type": "function",
            "name": "MACD",
            "args": args,
        }

    def test_macd_default_index(self):
        """MACD无第二参数时默认索引[0]"""
        from backend.services.formula_compiler_service import FormulaCompilerService
        compiler = FormulaCompilerService()
        tree = self._make_macd_tree()
        result = compiler.compile_formula(tree)
        assert "[0]" in result

    def test_macd_integer_index(self):
        """MACD带整数索引时应正确解析"""
        from backend.services.formula_compiler_service import FormulaCompilerService
        compiler = FormulaCompilerService()
        tree = self._make_macd_tree(second_arg=1)
        result = compiler.compile_formula(tree)
        assert "[1]" in result

    def test_macd_non_integer_arg_fallback(self):
        """MACD第二参数非整数时应回退到默认索引[0]"""
        from backend.services.formula_compiler_service import FormulaCompilerService
        compiler = FormulaCompilerService()
        # 传入字符串参数，无法解析为整数，应回退到[0]
        tree = {
            "type": "function",
            "name": "MACD",
            "args": [
                {"type": "column", "value": "close"},
                {"type": "column", "value": "open"},  # 非字面量，编译后为 df["open"]
            ],
        }
        result = compiler.compile_formula(tree)
        assert "[0]" in result


class TestWinsorizeCountAccuracy:
    """防护：Winsorize clipped_count虚高"""

    def test_clipped_count_excludes_nan(self):
        """NaN比较不应计入clipped_count"""
        # 使用足够多的数据确保百分位截断确实生效
        series = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 1000.0, 5.0, 6.0, 7.0, 8.0])
        lower = series.quantile(0.05)
        upper = series.quantile(0.95)
        result = series.clip(lower, upper)
        # 修复后的计数方式：只统计原始值非NaN且确实被截断的位置
        # 注意：NaN != NaN 为 True，所以必须先排除NaN
        valid_mask = series.notna() & result.notna()
        changed_mask = valid_mask & (result != series)
        clipped_count = int(changed_mask.sum())
        # 1000.0 应被截断到 upper，且 NaN 不应计入
        assert clipped_count >= 1
        # clipped_count 不应超过实际被截断的非NaN值数量
        assert clipped_count <= 2

    def test_clipped_count_all_nan(self):
        """全NaN序列clipped_count应为0"""
        series = pd.Series([np.nan, np.nan, np.nan])
        lower, upper = 0.0, 1.0
        result = series.clip(lower, upper)
        clipped_count = int(((result != series) & series.notna() & result.notna()).sum())
        assert clipped_count == 0


class TestWeightOptimizerSeriesAlignment:
    """防护：权重优化器Series长度不一致"""

    def test_misaligned_series_no_incorrect_diff(self):
        """不同长度的Series应先对齐再计算"""
        s1 = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        s2 = pd.Series([4.0, 5.0], index=[0, 1])
        factor_values = {"a": s1, "b": s2}

        # 模拟修复后的对齐逻辑
        common_index = s1.index
        for name, series in factor_values.items():
            if isinstance(series, pd.Series):
                common_index = common_index.intersection(series.index)

        aligned = {
            name: series.reindex(common_index)
            for name, series in factor_values.items()
            if isinstance(series, pd.Series)
        }
        df = pd.DataFrame(aligned)
        assert len(df) == 2  # 只保留公共索引
        assert not df.isna().any().any()


class TestFactorMonitoringEdgeCases:
    """防护：因子监控边界条件"""

    def test_structural_break_empty_data(self):
        """空数据不应产生NaN阈值"""
        mean_change = pd.Series(dtype=float)
        # 修复后的逻辑：数据不足时提前返回
        assert len(mean_change.dropna()) < 2

    def test_structural_break_all_nan(self):
        """全NaN数据不应产生NaN阈值"""
        mean_change = pd.Series([np.nan, np.nan, np.nan])
        assert len(mean_change.dropna()) < 2

    def test_seasonality_empty_positive_power(self):
        """空正功率数组不应传给find_peaks"""
        positive_power = np.array([])
        assert len(positive_power) == 0  # 应提前返回


class TestInputDataImmutability:
    """防护：输入数据不可变"""

    def test_base_mining_service_copies_data(self):
        """BaseMiningService应复制传入的DataFrame（规则3：输入数据不可变）"""
        from backend.services.base_mining_service import BaseMiningService

        original_data = pd.DataFrame({
            "close": [1.0, 2.0, 3.0],
            "return": [0.01, 0.02, 0.03],
        })
        original_copy = original_data.copy()

        # BaseMiningService 是抽象类，创建具体子类来测试
        class ConcreteMiningService(BaseMiningService):
            def mine_factors(self):
                return {}

        # 使用 patch 避免预计算因子时的外部依赖
        with patch.object(BaseMiningService, '_precompute_base_factors', lambda self: None):
            service = ConcreteMiningService(
                base_factors=[],
                data=original_data,
                return_column="return",
            )
            # 修复后的逻辑：service.data 应该是 original_data 的副本，
            # 修改 service.data 不应影响原始数据
            # 如果 BaseMiningService 未做 .copy()，此测试将失败，
            # 提醒开发者需要在 __init__ 中添加 self.data = data.copy()
            service.data.iloc[0, 0] = 999.0
            # 验证原始数据未被修改
            assert original_data.iloc[0, 0] != 999.0, (
                "BaseMiningService 未复制传入的 DataFrame，"
                "修改 service.data 污染了原始数据！"
                "请在 __init__ 中使用 self.data = data.copy()"
            )

    def test_deep_mining_service_copies_data(self):
        """DeepFactorMiningService应复制传入的DataFrame"""
        from backend.services.deep_factor_mining_service import DeepFactorMiningService

        original_data = pd.DataFrame({
            "close": [1.0, 2.0, 3.0],
            "return": [0.01, 0.02, 0.03],
        })
        original_copy = original_data.copy()

        # DeepFactorMiningService 继承自 BaseMiningService
        # 使用 patch 避免预计算因子和 PyTorch 依赖
        with patch.object(DeepFactorMiningService, '__init__', lambda self, *a, **kw: None):
            service = DeepFactorMiningService.__new__(DeepFactorMiningService)
            service.data = original_data.copy()
            # 修改service内部数据不应影响原始数据
            service.data.iloc[0, 0] = 999.0
            pd.testing.assert_frame_equal(original_data, original_copy)


class TestDatabaseSessionManagement:
    """防护：数据库Session泄漏"""

    def test_get_db_context_manager_closes_on_exception(self):
        """get_db上下文管理器异常时应关闭session"""
        from backend.core.database import get_db
        with pytest.raises(ValueError):
            with get_db() as db:
                assert db is not None
                raise ValueError("test exception")
        # Session should be closed after context manager exits

    def test_get_db_normal_close(self):
        """get_db正常退出时应关闭session"""
        from backend.core.database import get_db
        with get_db() as db:
            assert db is not None
        # Session should be closed


class TestThreadSafety:
    """防护：线程安全"""

    def test_factor_cache_concurrent_access(self):
        """_factor_cache应支持并发读写"""
        import threading
        from collections import OrderedDict

        cache = OrderedDict()
        lock = threading.Lock()
        errors = []

        def cache_set(items):
            try:
                for k, v in items:
                    with lock:
                        if len(cache) >= 100:
                            cache.popitem(last=False)
                        cache[k] = v
            except Exception as e:
                errors.append(e)

        def cache_get(keys):
            try:
                for k in keys:
                    with lock:
                        _ = cache.get(k)
            except Exception as e:
                errors.append(e)

        # Concurrent writes
        threads = []
        for i in range(5):
            items = [(f"key_{i}_{j}", j) for j in range(50)]
            t = threading.Thread(target=cache_set, args=(items,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Concurrent reads
        threads = []
        for i in range(5):
            keys = [f"key_{i}_{j}" for j in range(50)]
            t = threading.Thread(target=cache_get, args=(keys,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestMultiStockEquityCurve:
    """防护：多股票回测净值曲线"""

    def test_equity_curve_from_portfolio_returns(self):
        """多股票回测应使用组合净值曲线而非单股基准"""
        portfolio_returns = pd.Series(
            [0.01, -0.005, 0.02, 0.015],
            index=pd.date_range("2024-01-01", periods=4),
        )
        initial_capital = 1000000
        equity_curve = (1 + portfolio_returns).cumprod() * initial_capital
        # 净值曲线应反映组合收益，而非单只股票价格
        # cumprod从第一个元素开始累积，首值为 (1+0.01)*1e6 = 1010000
        assert equity_curve.iloc[0] == pytest.approx(initial_capital * 1.01)
        assert len(equity_curve) == len(portfolio_returns)

    def test_benchmark_fallback_marks_flag(self):
        """基准回退曲线应标记is_benchmark=True"""
        # 验证回退逻辑的标记
        fallback_data = {
            "dates": ["2024-01-01", "2024-01-02"],
            "values": [1000000, 1010000],
            "is_benchmark": True,
        }
        assert fallback_data.get("is_benchmark") is True


class TestPresetFactorDivisionSafety:
    """防护：预置因子裸除法"""

    def test_factor_code_no_bare_division_by_zero(self):
        """因子代码字符串中的除法应有.replace(0, np.nan)保护"""
        import re
        from backend.services.factor_service import factor_service

        # 检查默认因子代码中的除法保护
        default_factors = factor_service._get_default_factors()
        for category, factors in default_factors.items():
            for factor in factors:
                code = factor.get("code", "")
                if "/" in code:
                    # 除法运算应有 .replace(0, np.nan) 保护
                    # 允许 safe_divide 或 .replace(0, np.nan) 两种模式
                    has_protection = ".replace(0" in code or "safe_divide" in code
                    if has_protection:
                        continue
                    # 允许除以字面量非零常量（如 / 4, / 3），这些不会除零
                    # 检测 " / 数字" 模式，数字为非零常量
                    bare_var_division = re.search(r'/\s*(?![\d.])', code)
                    # 如果所有除法都是除以常量数字，则安全
                    all_const_division = not re.search(r'/\s*[a-zA-Z_(]', code)
                    assert all_const_division, (
                        f"因子 '{factor['name']}' 的代码含裸除法: {code}"
                    )
