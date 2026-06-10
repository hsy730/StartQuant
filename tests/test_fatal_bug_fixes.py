"""
致命Bug修复验证测试

F1: factor_preprocessing_pipeline.py - 联合中性化else分支 y 未定义 NameError
F2: formula_compiler_service.py - 统计函数双重包装 Bug
"""
import pytest
import numpy as np
import pandas as pd

from backend.services.factor_preprocessing_pipeline import (
    FactorPreprocessingPipeline,
    PreprocessingConfig,
    WinsorizeMethod,
    StandardizeMethod,
)
from backend.services.formula_compiler_service import FormulaCompilerService


# ============================================================
# F1: 联合中性化 else 分支 y 未定义 NameError
# ============================================================

class TestJointNeutralizationElseBranchFix:
    """
    验证修复：当 use_joint_neutralization=True, has_market_cap=True,
    has_industry=True 但行业数 < 2 时，代码不再引用未定义的 y 变量，
    而是正确回退到仅市值中性化。
    """

    def _build_single_industry_cross_sectional_data(self):
        """
        构建只有1个行业的横截面数据，触发联合中性化的 else 分支。

        条件：
        - use_joint_neutralization=True
        - has_market_cap=True
        - has_industry=True
        - unique_inds < 2（只有1个行业）
        """
        np.random.seed(123)
        n_dates = 5
        n_stocks = 30

        dates = pd.date_range(start="2024-01-01", periods=n_dates, freq="B")
        stock_codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]

        rows = []
        for date in dates:
            for stock in stock_codes:
                rows.append({
                    "date": date,
                    "stock_code": stock,
                    "factor_1": np.random.randn() * 10 + 5,
                    "market_cap": np.random.lognormal(mean=10, sigma=1),
                    "industry": "OnlyOne",  # 只有一个行业值
                })

        return pd.DataFrame(rows)

    def test_single_industry_no_nameerror(self):
        """只有1个行业时联合中性化不抛出 NameError"""
        df = self._build_single_industry_cross_sectional_data()
        config = PreprocessingConfig(
            use_joint_neutralization=True,
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=True,
            cross_sectional=True,
            min_samples=10,
        )
        pipeline = FactorPreprocessingPipeline(config)

        # 修复前：此处会抛出 NameError: name 'y' is not defined
        # 修复后：应正常执行，回退到仅市值中性化
        result_df, stats = pipeline.process_factor_dataframe(
            df,
            factor_columns=["factor_1"],
            market_cap_column="market_cap",
            industry_column="industry",
            date_column="date",
            parallel=False,
        )

        assert "factor_1" in result_df.columns
        assert len(result_df) == len(df)

    def test_single_industry_fallback_to_market_cap_only(self):
        """只有1个行业时，联合中性化回退到仅市值中性化，结果应与直接市值中性化一致"""
        df = self._build_single_industry_cross_sectional_data()

        # 联合中性化配置（但行业不足，应回退到市值中性化）
        config_joint = PreprocessingConfig(
            use_joint_neutralization=True,
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=True,
            cross_sectional=True,
            min_samples=10,
        )

        # 仅市值中性化配置
        config_mc_only = PreprocessingConfig(
            use_joint_neutralization=False,
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=False,
            cross_sectional=True,
            min_samples=10,
        )

        pipeline_joint = FactorPreprocessingPipeline(config_joint)
        pipeline_mc = FactorPreprocessingPipeline(config_mc_only)

        result_joint, _ = pipeline_joint.process_factor_dataframe(
            df.copy(), ["factor_1"], "market_cap", "industry", "date", parallel=False,
        )
        result_mc, _ = pipeline_mc.process_factor_dataframe(
            df.copy(), ["factor_1"], "market_cap", "industry", "date", parallel=False,
        )

        # 两种方式的结果应该近似（都是市值中性化）
        np.testing.assert_allclose(
            result_joint["factor_1"].values,
            result_mc["factor_1"].values,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_single_industry_result_not_all_nan(self):
        """只有1个行业时，中性化结果不应全为 NaN"""
        df = self._build_single_industry_cross_sectional_data()
        config = PreprocessingConfig(
            use_joint_neutralization=True,
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=True,
            cross_sectional=True,
            min_samples=10,
        )
        pipeline = FactorPreprocessingPipeline(config)

        result_df, _ = pipeline.process_factor_dataframe(
            df, ["factor_1"], "market_cap", "industry", "date", parallel=False,
        )

        # 结果不应全为 NaN
        assert not result_df["factor_1"].isna().all(), "中性化结果不应全为 NaN"

    def test_single_industry_residuals_uncorrelated_with_market_cap(self):
        """只有1个行业时，回退到市值中性化后残差应与市值不相关"""
        df = self._build_single_industry_cross_sectional_data()
        config = PreprocessingConfig(
            use_joint_neutralization=True,
            enable_market_cap_neutralization=True,
            enable_industry_neutralization=True,
            cross_sectional=True,
            min_samples=10,
        )
        pipeline = FactorPreprocessingPipeline(config)

        result_df, _ = pipeline.process_factor_dataframe(
            df, ["factor_1"], "market_cap", "industry", "date", parallel=False,
        )

        # 残差与log(市值)的相关系数应接近0
        log_mc = np.log(df["market_cap"])
        valid = result_df["factor_1"].notna() & log_mc.notna()
        correlation = result_df.loc[valid, "factor_1"].corr(log_mc[valid])
        assert abs(correlation) < 0.15, f"残差与市值相关性过高: {correlation:.4f}"


# ============================================================
# F2: 统计函数双重包装 Bug
# ============================================================

class TestStatisticalFunctionDoubleWrappingFix:
    """
    验证修复：mean, std, max, min, rank, zscore 编译后不再产生
    df["df["close"]"].mean() 这样的双重包装无效代码。

    修复前：df["{compiled_args[0]}"].{func_name}()
      → compiled_args[0] 已经是 df["close"]，再包一层变成 df["df["close"]"]

    修复后：{compiled_args[0]}.{func_name}()
      → 直接调用 df["close"].mean()
    """

    def setup_method(self):
        self.compiler = FormulaCompilerService()
        self.df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        })

    def _compile_and_eval(self, formula_tree):
        """编译公式树并执行，返回结果"""
        code = self.compiler.compile_formula(formula_tree)
        local_vars = {"df": self.df, "np": np}
        result = eval(code, {"__builtins__": {}}, local_vars)
        return code, result

    def test_mean_compiles_correctly(self):
        """mean(close) 编译为 df["close"].mean()，可执行且结果正确"""
        tree = {
            "type": "function",
            "name": "mean",
            "args": [{"type": "column", "value": "close"}],
        }
        code, result = self._compile_and_eval(tree)

        # 编译结果应为 df["close"].mean()，而非 df["df["close"]"].mean()
        assert code == 'df["close"].mean()', f"编译结果不正确: {code}"
        assert result == pytest.approx(self.df["close"].mean())

    def test_std_compiles_correctly(self):
        """std(close) 编译为 df["close"].std()，可执行且结果正确"""
        tree = {
            "type": "function",
            "name": "std",
            "args": [{"type": "column", "value": "close"}],
        }
        code, result = self._compile_and_eval(tree)

        assert code == 'df["close"].std()', f"编译结果不正确: {code}"
        assert result == pytest.approx(self.df["close"].std())

    def test_max_compiles_correctly(self):
        """max(close) 编译为 df["close"].max()，可执行且结果正确"""
        tree = {
            "type": "function",
            "name": "max",
            "args": [{"type": "column", "value": "close"}],
        }
        code, result = self._compile_and_eval(tree)

        assert code == 'df["close"].max()', f"编译结果不正确: {code}"
        assert result == pytest.approx(self.df["close"].max())

    def test_min_compiles_correctly(self):
        """min(close) 编译为 df["close"].min()，可执行且结果正确"""
        tree = {
            "type": "function",
            "name": "min",
            "args": [{"type": "column", "value": "close"}],
        }
        code, result = self._compile_and_eval(tree)

        assert code == 'df["close"].min()', f"编译结果不正确: {code}"
        assert result == pytest.approx(self.df["close"].min())

    def test_rank_compiles_correctly(self):
        """rank(close) 编译为 df["close"].rank()，可执行且结果正确"""
        tree = {
            "type": "function",
            "name": "rank",
            "args": [{"type": "column", "value": "close"}],
        }
        code, result = self._compile_and_eval(tree)

        assert code == 'df["close"].rank()', f"编译结果不正确: {code}"
        pd.testing.assert_series_equal(result, self.df["close"].rank())

    def test_zscore_compiles_correctly(self):
        """zscore(close) 编译为滚动zscore表达式，可执行且结果正确"""
        tree = {
            "type": "function",
            "name": "zscore",
            "args": [{"type": "column", "value": "close"}],
        }
        code = self.compiler.compile_formula(tree)

        # 编译结果不应包含双重包装
        assert 'df["df["' not in code, f"检测到双重包装: {code}"

        # 应该能成功执行
        from backend.utils.safe_math import safe_series_divide
        local_vars = {"df": self.df, "np": np, "safe_series_divide": safe_series_divide}
        result = eval(code, {"__builtins__": {}}, local_vars)

        # 验证zscore计算逻辑：滚动窗口默认20，数据只有10行，前19个应为NaN
        assert result.isna().sum() >= 9  # 窗口20 > 数据10，大部分为NaN

    def test_no_double_wrapping_in_any_stat_function(self):
        """所有统计函数编译结果都不应包含双重包装"""
        stat_funcs = ["mean", "std", "max", "min", "rank", "zscore"]
        for func_name in stat_funcs:
            tree = {
                "type": "function",
                "name": func_name,
                "args": [{"type": "column", "value": "close"}],
            }
            code = self.compiler.compile_formula(tree)
            assert 'df["df["' not in code, (
                f"{func_name} 编译结果包含双重包装: {code}"
            )
