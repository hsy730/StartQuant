"""
factor_neutralization_service.py 因子中性化服务测试

测试覆盖：
- neutralize_market_cap: 市值中性化（线性回归残差法）
- neutralize_industry: 行业中性化（行业哑变量回归残差法）
- neutralize_both: 行业+市值联合中性化
- _validate_columns: 列校验
- _build_result_series: 结果Series构建
- add_industry_classification: 添加行业分类
"""
import sys
import os
import logging
import warnings
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf

from backend.services.factor_neutralization_service import FactorNeutralizationService


# ---------------------------------------------------------------------------
# 辅助函数：生成测试数据
# ---------------------------------------------------------------------------

def _make_market_cap_df(n=30, seed=42):
    """生成含因子值和市值的横截面测试数据（n >= 10 满足 MIN_SAMPLES）"""
    np.random.seed(seed)
    return pd.DataFrame({
        "factor_value": np.random.randn(n) * 0.1,
        "market_cap": np.random.lognormal(mean=10, sigma=1, size=n),
    })


def _make_industry_df(n=30, n_industries=3, seed=42):
    """生成含因子值和行业分类的横截面测试数据"""
    np.random.seed(seed)
    industries = [f"ind_{i}" for i in range(n_industries)]
    return pd.DataFrame({
        "factor_value": np.random.randn(n) * 0.1,
        "industry": np.random.choice(industries, size=n),
    })


def _make_full_df(n=30, n_industries=3, seed=42):
    """生成含因子值、市值和行业分类的完整横截面测试数据"""
    np.random.seed(seed)
    industries = [f"ind_{i}" for i in range(n_industries)]
    return pd.DataFrame({
        "factor_value": np.random.randn(n) * 0.1,
        "market_cap": np.random.lognormal(mean=10, sigma=1, size=n),
        "industry": np.random.choice(industries, size=n),
    })


# ===========================================================================
# TestValidateColumns
# ===========================================================================

class TestValidateColumns:
    """_validate_columns 列校验测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_validate_columns_all_present_should_not_raise(self):
        """所有列都存在时不应抛出异常"""
        df = pd.DataFrame({"a": [1], "b": [2]})
        self.service._validate_columns(df, "a", "b")  # 不抛异常即通过

    def test_validate_columns_missing_should_raise_value_error(self):
        """缺少列时应抛出 ValueError"""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="缺少列"):
            self.service._validate_columns(df, "a", "b")

    def test_validate_columns_single_missing_should_raise_value_error(self):
        """单个缺失列应抛出 ValueError"""
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="缺少列"):
            self.service._validate_columns(df, "y")

    def test_validate_columns_empty_df_with_columns_should_not_raise(self):
        """空 DataFrame 但列名存在时不应抛出异常"""
        df = pd.DataFrame(columns=["a", "b"])
        self.service._validate_columns(df, "a", "b")

    def test_validate_columns_no_args_should_not_raise(self):
        """不传列名参数时不应抛出异常"""
        df = pd.DataFrame({"a": [1]})
        self.service._validate_columns(df)


# ===========================================================================
# TestBuildResultSeries
# ===========================================================================

class TestBuildResultSeries:
    """_build_result_series 结果Series构建测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_build_result_series_valid_indices_should_have_values(self):
        """有效索引位置应有残差值"""
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0]})
        valid_index = pd.Index([1, 3])
        residuals = np.array([0.5, -0.5])
        result = self.service._build_result_series(df, "f", valid_index, residuals)
        assert result.iloc[1] == pytest.approx(0.5)
        assert result.iloc[3] == pytest.approx(-0.5)

    def test_build_result_series_invalid_indices_should_be_nan(self):
        """非有效索引位置应为 NaN"""
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0]})
        valid_index = pd.Index([1, 3])
        residuals = np.array([0.5, -0.5])
        result = self.service._build_result_series(df, "f", valid_index, residuals)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[4])

    def test_build_result_series_all_valid_should_have_no_nan(self):
        """全部有效索引时不应有 NaN"""
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
        valid_index = df.index
        residuals = np.array([0.1, 0.2, 0.3])
        result = self.service._build_result_series(df, "f", valid_index, residuals)
        assert result.notna().all()

    def test_build_result_series_preserves_original_index(self):
        """结果Series应保留原始 DataFrame 的索引"""
        idx = pd.Index([10, 20, 30])
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=idx)
        valid_index = pd.Index([20])
        residuals = np.array([0.5])
        result = self.service._build_result_series(df, "f", valid_index, residuals)
        assert list(result.index) == [10, 20, 30]
        assert result.loc[20] == pytest.approx(0.5)

    def test_build_result_series_dtype_should_be_float(self):
        """结果Series的dtype应为float"""
        df = pd.DataFrame({"f": [1.0, 2.0]})
        valid_index = df.index[:1]
        residuals = np.array([0.5])
        result = self.service._build_result_series(df, "f", valid_index, residuals)
        assert result.dtype == float


# ===========================================================================
# TestNeutralizeMarketCap
# ===========================================================================

class TestNeutralizeMarketCap:
    """neutralize_market_cap 市值中性化测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_normal_data_should_return_residuals(self):
        """正常数据应返回回归残差"""
        df = _make_market_cap_df(n=30, seed=42)
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        # 有效位置的残差不应全为 NaN
        assert result.notna().sum() >= 10

    def test_residuals_should_be_uncorrelated_with_log_market_cap(self):
        """残差应与 log(市值) 不相关（中性化目的）"""
        np.random.seed(42)
        n = 200
        log_mc = np.random.randn(n) * 2 + 10
        market_cap = np.exp(log_mc)
        # 构造与市值强相关的因子
        factor = 0.5 * log_mc + np.random.randn(n) * 0.01
        df = pd.DataFrame({"factor_value": factor, "market_cap": market_cap})

        result = self.service.neutralize_market_cap(df, "factor_value")
        valid = result.dropna()
        valid_log_mc = np.log(df.loc[valid.index, "market_cap"])

        corr = np.corrcoef(valid.values, valid_log_mc.values)[0, 1]
        assert abs(corr) < 0.1, f"残差与log(市值)相关性过高: {corr}"

    def test_insufficient_data_should_raise_value_error(self):
        """有效数据不足时应抛出 ValueError"""
        df = _make_market_cap_df(n=5, seed=42)
        with pytest.raises(ValueError, match="有效数据不足"):
            self.service.neutralize_market_cap(df, "factor_value")

    def test_missing_factor_column_should_raise_value_error(self):
        """缺少因子列时应抛出 ValueError"""
        df = pd.DataFrame({"market_cap": [1e8] * 20})
        with pytest.raises(ValueError, match="缺少列"):
            self.service.neutralize_market_cap(df, "nonexistent_factor")

    def test_missing_market_cap_column_should_raise_value_error(self):
        """缺少市值列时应抛出 ValueError"""
        df = pd.DataFrame({"factor_value": np.random.randn(20)})
        with pytest.raises(ValueError, match="缺少列"):
            self.service.neutralize_market_cap(df, "factor_value")

    def test_zero_market_cap_should_be_excluded(self):
        """市值为0的记录应被排除"""
        df = _make_market_cap_df(n=20, seed=42)
        # 将部分市值设为0，剩余有效数据 >= 10
        df.loc[df.index[:6], "market_cap"] = 0
        result = self.service.neutralize_market_cap(df, "factor_value")
        # 市值为0的行应为 NaN
        assert pd.isna(result.iloc[0])

    def test_negative_market_cap_should_be_excluded(self):
        """市值为负的记录应被排除"""
        df = _make_market_cap_df(n=20, seed=42)
        df.loc[df.index[:6], "market_cap"] = -1e8
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert pd.isna(result.iloc[0])

    def test_all_zero_market_cap_should_raise_value_error(self):
        """所有市值为0时应抛出 ValueError"""
        df = pd.DataFrame({
            "factor_value": np.random.randn(15),
            "market_cap": 0.0,
        })
        with pytest.raises(ValueError, match="市值>0"):
            self.service.neutralize_market_cap(df, "factor_value")

    def test_nan_in_factor_should_be_excluded(self):
        """因子值含 NaN 的行应被排除"""
        df = _make_market_cap_df(n=20, seed=42)
        df.loc[df.index[:5], "factor_value"] = np.nan
        result = self.service.neutralize_market_cap(df, "factor_value")
        # NaN 行应为 NaN
        assert pd.isna(result.iloc[0])

    def test_nan_in_market_cap_should_be_excluded(self):
        """市值含 NaN 的行应被排除"""
        df = _make_market_cap_df(n=20, seed=42)
        df.loc[df.index[:5], "market_cap"] = np.nan
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert pd.isna(result.iloc[0])

    def test_all_nan_should_raise_value_error(self):
        """全部 NaN 数据应抛出 ValueError"""
        df = pd.DataFrame({
            "factor_value": [np.nan] * 15,
            "market_cap": [np.nan] * 15,
        })
        with pytest.raises(ValueError, match="有效数据不足"):
            self.service.neutralize_market_cap(df, "factor_value")

    def test_custom_market_cap_column_should_work(self):
        """自定义市值列名应正常工作"""
        df = _make_market_cap_df(n=20, seed=42)
        df = df.rename(columns={"market_cap": "mkt_cap"})
        result = self.service.neutralize_market_cap(df, "factor_value", market_cap_column="mkt_cap")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_result_length_should_match_input(self):
        """结果Series长度应与输入DataFrame一致"""
        df = _make_market_cap_df(n=25, seed=42)
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert len(result) == len(df)

    def test_residuals_mean_should_be_near_zero(self):
        """残差均值应接近0（线性回归性质）"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=n),
        })
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert abs(result.dropna().mean()) < 1e-10


# ===========================================================================
# TestNeutralizeIndustry
# ===========================================================================

class TestNeutralizeIndustry:
    """neutralize_industry 行业中性化测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_normal_data_should_return_residuals(self):
        """正常多行业数据应返回回归残差"""
        df = _make_industry_df(n=30, n_industries=3, seed=42)
        result = self.service.neutralize_industry(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.notna().sum() >= 10

    def test_residuals_should_remove_industry_effect(self):
        """残差应消除行业间系统性差异"""
        np.random.seed(42)
        n = 150
        industries = ["Tech", "Finance", "Health"]
        industry_effect = {"Tech": 5.0, "Finance": -3.0, "Health": 1.0}
        ind_choices = np.random.choice(industries, size=n)
        factor = np.array([industry_effect[i] for i in ind_choices]) + np.random.randn(n) * 0.01
        df = pd.DataFrame({"factor_value": factor, "industry": ind_choices})

        result = self.service.neutralize_industry(df, "factor_value")
        valid = result.dropna()
        # 中性化后，各行业残差均值应接近0
        for ind in industries:
            ind_mask = df.loc[valid.index, "industry"] == ind
            if ind_mask.sum() >= 5:
                assert abs(valid[ind_mask].mean()) < 0.5, f"行业 {ind} 残差均值偏移过大"

    def test_single_industry_should_return_original_factor(self):
        """只有1个行业时应返回原始因子值（跳过中性化）"""
        df = pd.DataFrame({
            "factor_value": np.random.randn(20),
            "industry": "Tech",
        })
        result = self.service.neutralize_industry(df, "factor_value")
        pd.testing.assert_series_equal(result, df["factor_value"])

    def test_insufficient_data_should_raise_value_error(self):
        """有效数据不足时应抛出 ValueError"""
        df = _make_industry_df(n=5, n_industries=2, seed=42)
        with pytest.raises(ValueError, match="有效数据不足"):
            self.service.neutralize_industry(df, "factor_value")

    def test_missing_factor_column_should_raise_value_error(self):
        """缺少因子列时应抛出 ValueError"""
        df = pd.DataFrame({"industry": ["A"] * 20})
        with pytest.raises(ValueError, match="缺少列"):
            self.service.neutralize_industry(df, "nonexistent")

    def test_missing_industry_column_should_raise_value_error(self):
        """缺少行业列时应抛出 ValueError"""
        df = pd.DataFrame({"factor_value": np.random.randn(20)})
        with pytest.raises(ValueError, match="缺少列"):
            self.service.neutralize_industry(df, "factor_value")

    def test_small_industries_should_be_filtered_with_warning(self, caplog):
        """小行业（<5样本）应被过滤并发出警告日志"""
        np.random.seed(42)
        n = 30
        # 主行业有足够样本，小行业只有2个样本
        industries = ["Big"] * 20 + ["Small1"] * 2 + ["Medium"] * 8
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "industry": industries,
        })
        with caplog.at_level(logging.WARNING, logger="backend.services.factor_neutralization_service"):
            result = self.service.neutralize_industry(df, "factor_value")
            # 应有关于小行业的警告日志
            assert any("样本量" in record.message for record in caplog.records)

    def test_all_small_industries_should_return_original_factor(self):
        """过滤后不足2个行业时应返回原始因子值"""
        np.random.seed(42)
        # 每个行业只有2个样本，全部 < 5
        industries = ["A"] * 2 + ["B"] * 2 + ["C"] * 2 + ["D"] * 10
        df = pd.DataFrame({
            "factor_value": np.random.randn(16),
            "industry": industries,
        })
        # D行业有10个样本，但过滤掉A/B/C后只剩1个行业
        # 实际上D有10个，所以过滤后只有1个有效行业，应跳过
        result = self.service.neutralize_industry(df, "factor_value")
        pd.testing.assert_series_equal(result, df["factor_value"])

    def test_nan_in_factor_should_be_excluded(self):
        """因子值含 NaN 的行应被排除"""
        df = _make_industry_df(n=20, n_industries=3, seed=42)
        df.loc[df.index[:5], "factor_value"] = np.nan
        result = self.service.neutralize_industry(df, "factor_value")
        assert pd.isna(result.iloc[0])

    def test_nan_in_industry_should_be_excluded(self):
        """行业分类含 NaN 的行应被排除"""
        df = _make_industry_df(n=20, n_industries=3, seed=42)
        df.loc[df.index[:5], "industry"] = np.nan
        result = self.service.neutralize_industry(df, "factor_value")
        assert pd.isna(result.iloc[0])

    def test_custom_industry_column_should_work(self):
        """自定义行业列名应正常工作"""
        df = _make_industry_df(n=20, n_industries=3, seed=42)
        df = df.rename(columns={"industry": "sector"})
        result = self.service.neutralize_industry(df, "factor_value", industry_column="sector")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_result_length_should_match_input(self):
        """结果Series长度应与输入DataFrame一致"""
        df = _make_industry_df(n=25, n_industries=3, seed=42)
        result = self.service.neutralize_industry(df, "factor_value")
        assert len(result) == len(df)

    def test_two_industries_should_work(self):
        """恰好2个行业应正常计算"""
        np.random.seed(42)
        n = 20
        industries = ["A"] * 10 + ["B"] * 10
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "industry": industries,
        })
        result = self.service.neutralize_industry(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10


# ===========================================================================
# TestNeutralizeBoth
# ===========================================================================

class TestNeutralizeBoth:
    """neutralize_both 行业+市值联合中性化测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_full_data_should_return_residuals(self):
        """完整数据（市值+行业）应返回联合回归残差"""
        df = _make_full_df(n=30, n_industries=3, seed=42)
        result = self.service.neutralize_both(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.notna().sum() >= 10

    def test_residuals_should_remove_both_effects(self):
        """联合中性化应同时消除市值和行业影响"""
        np.random.seed(42)
        n = 300
        industries = np.random.choice(["Tech", "Finance", "Health"], size=n)
        industry_effect = {"Tech": 5.0, "Finance": -3.0, "Health": 1.0}
        log_mc = np.random.randn(n) * 2 + 10
        market_cap = np.exp(log_mc)

        # 构造同时受市值和行业影响的因子
        factor = (
            np.array([industry_effect[i] for i in industries])
            + 0.5 * log_mc
            + np.random.randn(n) * 0.01
        )
        df = pd.DataFrame({
            "factor_value": factor,
            "market_cap": market_cap,
            "industry": industries,
        })

        result = self.service.neutralize_both(df, "factor_value")
        valid = result.dropna()

        # 残差与 log(市值) 的相关性应很低
        valid_log_mc = np.log(df.loc[valid.index, "market_cap"])
        corr_mc = np.corrcoef(valid.values, valid_log_mc.values)[0, 1]
        assert abs(corr_mc) < 0.15, f"残差与log(市值)相关性过高: {corr_mc}"

    def test_only_market_cap_should_fallback_to_market_cap_neutralization(self):
        """仅有市值列时应退化为市值中性化"""
        df = _make_market_cap_df(n=20, seed=42)
        result = self.service.neutralize_both(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_only_industry_should_fallback_to_industry_neutralization(self):
        """仅有行业列时应退化为行业中性化"""
        df = _make_industry_df(n=20, n_industries=3, seed=42)
        result = self.service.neutralize_both(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_neither_column_should_return_original_factor(self):
        """既无市值也无行业列时应返回原始因子值"""
        df = pd.DataFrame({
            "factor_value": np.random.randn(20),
        })
        result = self.service.neutralize_both(df, "factor_value")
        pd.testing.assert_series_equal(result, df["factor_value"])

    def test_insufficient_data_should_raise_value_error(self):
        """有效数据不足时应抛出 ValueError"""
        df = _make_full_df(n=5, n_industries=2, seed=42)
        with pytest.raises(ValueError, match="有效数据不足"):
            self.service.neutralize_both(df, "factor_value")

    def test_missing_factor_column_should_raise_value_error(self):
        """缺少因子列时应抛出 ValueError"""
        df = pd.DataFrame({
            "market_cap": [1e8] * 20,
            "industry": ["A"] * 10 + ["B"] * 10,
        })
        with pytest.raises(ValueError, match="缺少列"):
            self.service.neutralize_both(df, "nonexistent")

    def test_zero_market_cap_should_be_excluded(self):
        """市值为0的记录应被排除"""
        df = _make_full_df(n=20, n_industries=3, seed=42)
        df.loc[df.index[:6], "market_cap"] = 0
        result = self.service.neutralize_both(df, "factor_value")
        assert pd.isna(result.iloc[0])

    def test_all_zero_market_cap_should_raise_value_error(self):
        """所有市值为0时应抛出 ValueError"""
        df = pd.DataFrame({
            "factor_value": np.random.randn(15),
            "market_cap": 0.0,
            "industry": ["A"] * 8 + ["B"] * 7,
        })
        with pytest.raises(ValueError, match="市值>0"):
            self.service.neutralize_both(df, "factor_value")

    def test_single_industry_with_market_cap_should_skip_industry(self):
        """只有1个行业但有市值时，应跳过行业部分，仅做市值中性化"""
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=n),
            "industry": "OnlyOne",
        })
        result = self.service.neutralize_both(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_nan_in_any_column_should_be_excluded(self):
        """任一列含 NaN 的行应被排除"""
        df = _make_full_df(n=20, n_industries=3, seed=42)
        df.loc[df.index[0], "factor_value"] = np.nan
        df.loc[df.index[1], "market_cap"] = np.nan
        df.loc[df.index[2], "industry"] = np.nan
        result = self.service.neutralize_both(df, "factor_value")
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_custom_column_names_should_work(self):
        """自定义列名应正常工作"""
        df = _make_full_df(n=20, n_industries=3, seed=42)
        df = df.rename(columns={"market_cap": "mkt_cap", "industry": "sector"})
        result = self.service.neutralize_both(
            df, "factor_value",
            market_cap_column="mkt_cap",
            industry_column="sector"
        )
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_result_length_should_match_input(self):
        """结果Series长度应与输入DataFrame一致"""
        df = _make_full_df(n=25, n_industries=3, seed=42)
        result = self.service.neutralize_both(df, "factor_value")
        assert len(result) == len(df)

    def test_small_industries_filtered_should_still_work(self):
        """过滤小行业后仍应有足够数据计算"""
        np.random.seed(42)
        n = 30
        industries = ["Big"] * 20 + ["Small"] * 3 + ["Medium"] * 7
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=n),
            "industry": industries,
        })
        result = self.service.neutralize_both(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10


# ===========================================================================
# TestAddIndustryClassification
# ===========================================================================

class TestAddIndustryClassification:
    """add_industry_classification 添加行业分类测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_with_stock_code_column_should_map_industry(self):
        """有 stock_code 列时应根据映射添加 industry 列"""
        df = pd.DataFrame({
            "stock_code": ["600036.SH", "000001.SZ", "300750.SZ"],
            "factor_value": [1.0, 2.0, 3.0],
        })
        industry_map = {"600036": "Finance", "000001": "Finance", "300750": "Manufacturing"}
        with patch.object(self.service, "get_industry_classification", return_value=industry_map):
            result = self.service.add_industry_classification(df, ["600036", "000001", "300750"])
        assert "industry" in result.columns
        assert result.loc[0, "industry"] == "Finance"
        assert result.loc[1, "industry"] == "Finance"
        assert result.loc[2, "industry"] == "Manufacturing"

    def test_without_stock_code_column_should_use_unknown(self):
        """没有 stock_code 列时 industry 应全部为 unknown"""
        df = pd.DataFrame({
            "factor_value": [1.0, 2.0, 3.0],
        })
        with patch.object(self.service, "get_industry_classification", return_value={}):
            result = self.service.add_industry_classification(df, ["600036"])
        assert "industry" in result.columns
        assert (result["industry"] == "unknown").all()

    def test_should_not_modify_original_df(self):
        """不应修改原始 DataFrame（数据不可变原则）"""
        df = pd.DataFrame({
            "stock_code": ["600036.SH"],
            "factor_value": [1.0],
        })
        original_cols = list(df.columns)
        with patch.object(self.service, "get_industry_classification", return_value={"600036": "Finance"}):
            self.service.add_industry_classification(df, ["600036"])
        assert list(df.columns) == original_cols
        assert "industry" not in df.columns

    def test_bj_suffix_should_be_stripped(self):
        """北交所 .BJ 后缀应被正确去除"""
        df = pd.DataFrame({
            "stock_code": ["830799.BJ"],
            "factor_value": [1.0],
        })
        industry_map = {"830799": "Mining"}
        with patch.object(self.service, "get_industry_classification", return_value=industry_map):
            result = self.service.add_industry_classification(df, ["830799"])
        assert result.loc[0, "industry"] == "Mining"

    def test_sh_suffix_should_be_stripped(self):
        """.SH 后缀应被正确去除"""
        df = pd.DataFrame({
            "stock_code": ["601318.SH"],
            "factor_value": [1.0],
        })
        industry_map = {"601318": "Insurance"}
        with patch.object(self.service, "get_industry_classification", return_value=industry_map):
            result = self.service.add_industry_classification(df, ["601318"])
        assert result.loc[0, "industry"] == "Insurance"

    def test_sz_suffix_should_be_stripped(self):
        """.SZ 后缀应被正确去除"""
        df = pd.DataFrame({
            "stock_code": ["000002.SZ"],
            "factor_value": [1.0],
        })
        industry_map = {"000002": "RealEstate"}
        with patch.object(self.service, "get_industry_classification", return_value=industry_map):
            result = self.service.add_industry_classification(df, ["000002"])
        assert result.loc[0, "industry"] == "RealEstate"

    def test_unmapped_stock_code_should_be_nan(self):
        """未在映射中的股票代码 industry 应为 NaN"""
        df = pd.DataFrame({
            "stock_code": ["600036.SH", "999999.SH"],
            "factor_value": [1.0, 2.0],
        })
        industry_map = {"600036": "Finance"}  # 不含 999999
        with patch.object(self.service, "get_industry_classification", return_value=industry_map):
            result = self.service.add_industry_classification(df, ["600036", "999999"])
        assert result.loc[0, "industry"] == "Finance"
        assert pd.isna(result.loc[1, "industry"])

    def test_get_industry_classification_failure_should_use_unknown(self):
        """获取行业分类失败时应使用 unknown 作为默认值"""
        df = pd.DataFrame({
            "stock_code": ["600036.SH"],
            "factor_value": [1.0],
        })
        with patch("backend.services.factor_neutralization_service.data_service") as mock_ds:
            mock_ds.get_industry_classification.side_effect = Exception("service unavailable")
            result = self.service.add_industry_classification(df, ["600036"])
        assert "industry" in result.columns
        assert result.loc[0, "industry"] == "unknown"


# ===========================================================================
# TestGetIndustryClassification
# ===========================================================================

class TestGetIndustryClassification:
    """get_industry_classification 行业分类获取测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_success_should_return_mapping(self):
        """成功获取时应返回行业映射"""
        expected = {"600036": "Finance", "000001": "Banking"}
        with patch("backend.services.factor_neutralization_service.data_service") as mock_ds:
            mock_ds.get_industry_classification.return_value = expected
            result = self.service.get_industry_classification(["600036", "000001"])
        assert result == expected

    def test_failure_should_return_unknown_mapping(self):
        """获取失败时应返回全部 unknown 的映射"""
        with patch("backend.services.factor_neutralization_service.data_service") as mock_ds:
            mock_ds.get_industry_classification.side_effect = Exception("timeout")
            result = self.service.get_industry_classification(["600036", "000001"])
        assert result == {"600036": "unknown", "000001": "unknown"}

    def test_empty_codes_should_return_empty_dict(self):
        """空股票代码列表应返回空字典"""
        with patch("backend.services.factor_neutralization_service.data_service") as mock_ds:
            mock_ds.get_industry_classification.return_value = {}
            result = self.service.get_industry_classification([])
        assert result == {}


# ===========================================================================
# TestIntegrationAndEdgeCases
# ===========================================================================

class TestIntegrationAndEdgeCases:
    """集成与边界情况测试"""

    def setup_method(self):
        self.service = FactorNeutralizationService()

    def test_market_cap_neutralization_should_not_modify_input(self):
        """市值中性化不应修改输入 DataFrame（数据不可变原则）"""
        df = _make_market_cap_df(n=20, seed=42)
        original_values = df["factor_value"].copy()
        self.service.neutralize_market_cap(df, "factor_value")
        pd.testing.assert_series_equal(df["factor_value"], original_values)

    def test_industry_neutralization_should_not_modify_input(self):
        """行业中性化不应修改输入 DataFrame"""
        df = _make_industry_df(n=20, n_industries=3, seed=42)
        original_values = df["factor_value"].copy()
        self.service.neutralize_industry(df, "factor_value")
        pd.testing.assert_series_equal(df["factor_value"], original_values)

    def test_both_neutralization_should_not_modify_input(self):
        """联合中性化不应修改输入 DataFrame"""
        df = _make_full_df(n=20, n_industries=3, seed=42)
        original_values = df["factor_value"].copy()
        self.service.neutralize_both(df, "factor_value")
        pd.testing.assert_series_equal(df["factor_value"], original_values)

    def test_min_samples_boundary_exactly_10_should_work(self):
        """恰好10条有效数据应能正常计算"""
        np.random.seed(42)
        df = pd.DataFrame({
            "factor_value": np.random.randn(10),
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=10),
        })
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() == 10

    def test_min_samples_boundary_9_should_raise(self):
        """9条有效数据应抛出 ValueError"""
        np.random.seed(42)
        df = pd.DataFrame({
            "factor_value": np.random.randn(9),
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=9),
        })
        with pytest.raises(ValueError, match="有效数据不足"):
            self.service.neutralize_market_cap(df, "factor_value")

    def test_large_market_cap_range_should_work(self):
        """市值跨度很大（如从1e6到1e12）时应正常计算"""
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "market_cap": np.logspace(6, 12, n),
        })
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_constant_factor_with_varying_market_cap(self):
        """因子值恒定但市值变化时，残差应接近0"""
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            "factor_value": 1.0,  # 恒定因子
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=n),
        })
        result = self.service.neutralize_market_cap(df, "factor_value")
        valid = result.dropna()
        # 恒定因子完全可被回归解释，残差应接近0
        assert abs(valid.mean()) < 1e-10
        assert valid.std() < 1e-10

    def test_industry_neutralization_with_many_industries(self):
        """多个行业（>=5）应正常计算"""
        np.random.seed(42)
        n = 50
        industries = [f"ind_{i}" for i in range(5)]
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "industry": np.random.choice(industries, size=n),
        })
        result = self.service.neutralize_industry(df, "factor_value")
        assert isinstance(result, pd.Series)
        assert result.notna().sum() >= 10

    def test_both_neutralization_with_mixed_nan(self):
        """联合中性化中混合 NaN 分布应正确处理"""
        np.random.seed(42)
        n = 30
        df = _make_full_df(n=n, n_industries=3, seed=42)
        # 在不同列散布 NaN
        df.loc[df.index[0], "factor_value"] = np.nan
        df.loc[df.index[3], "market_cap"] = np.nan
        df.loc[df.index[7], "industry"] = np.nan
        result = self.service.neutralize_both(df, "factor_value")
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[3])
        assert pd.isna(result.iloc[7])

    def test_market_cap_neutralization_with_string_index(self):
        """使用字符串索引的 DataFrame 应正常工作"""
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            "factor_value": np.random.randn(n),
            "market_cap": np.random.lognormal(mean=10, sigma=1, size=n),
        }, index=[f"stock_{i}" for i in range(n)])
        result = self.service.neutralize_market_cap(df, "factor_value")
        assert list(result.index) == list(df.index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
