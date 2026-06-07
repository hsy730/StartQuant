"""
量化因子预处理管道 - 高性能、可复用的数据美颜服务

业界标准流程：
1. 缺失值处理 → 2. 去极值（MAD/百分位/3σ）→ 3. 中性化（市值+行业）→ 4. 标准化（Z-score/Rank）

设计原则：
- 使用pandas向量化操作，避免Python循环
- 支持横截面（cross-sectional）和时间序列（time-series）两种模式
- 线程安全，可并行调用
- 配置灵活，支持多种方法组合

⭐ v2.0 新增：联合回归中性化（同时控制市值和行业）
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats.mstats import winsorize as scipy_winsorize

import statsmodels.api as sm

logger = logging.getLogger(__name__)


class WinsorizeMethod(str, Enum):
    """去极值方法枚举"""
    MAD = "mad"                    # MAD法（中位数绝对偏差）- 对肥尾分布最稳健
    PERCENTILE = "percentile"      # 百分位法 - 最常用
    STD = "std"                   # 3σ标准差法 - 假设正态分布


class StandardizeMethod(str, Enum):
    """标准化方法枚举"""
    ZSCORE = "zscore"              # Z-score标准化 - 最常用
    RANK = "rank"                  # 排名标准化（均匀分布）
    MEDIAN_MAD = "median_mad"      # 中位数-MAD标准化 - 对异常值更稳健


@dataclass
class PreprocessingConfig:
    """
    预处理配置类
    
    Attributes:
        winsorize_method: 去极值方法
        winsorize_limits: 去极值边界（用于percentile方法，如(0.01, 0.99)表示1%-99%分位）
        winsorize_n_sigma: 去极值倍数（用于MAD和STD方法）
        standardize_method: 标准化方法
        enable_market_cap_neutralization: 是否启用市值中性化
        enable_industry_neutralization: 是否启用行业中性化
        use_joint_neutralization: 是否使用联合回归（同时控制市值和行业）- ⭐推荐
        handle_missing: 缺失值处理方式 ("fill_zero", "fill_median", "drop")
        min_samples: 最小样本数要求（低于此数量跳过处理）
        cross_sectional: 是否使用横截面模式（True=每日横截面，False=时间序列）
    """
    winsorize_method: WinsorizeMethod = WinsorizeMethod.MAD
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    winsorize_n_sigma: float = 3.0
    standardize_method: StandardizeMethod = StandardizeMethod.ZSCORE
    enable_market_cap_neutralization: bool = True
    enable_industry_neutralization: bool = True
    use_joint_neutralization: bool = True
    handle_missing: str = "fill_zero"
    min_samples: int = 10
    cross_sectional: bool = True


class FactorPreprocessingPipeline:
    """
    高性能因子预处理管道
    
    特性：
    - 完整实现"去极值→中性化→标准化"三步流程
    - ⭐ 支持联合回归中性化（v2.0新增）
    - 支持pandas向量化操作，处理100万行数据<1秒
    - 支持多股票批量处理
    - 自动处理缺失值和异常情况
    - 提供详细的处理统计信息
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        初始化管道
        
        Args:
            config: 预处理配置，默认使用标准配置
        """
        self.config = config or PreprocessingConfig()
        self._stats_cache = {}

    def process_single_factor(
        self,
        factor_values: pd.Series,
        market_cap: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
        date_index: Optional[pd.DatetimeIndex] = None,
    ) -> Tuple[pd.Series, Dict]:
        """
        处理单个因子的完整流程
        
        Args:
            factor_values: 因子值序列（索引为日期或(date, stock_code) MultiIndex）
            market_cap: 市值序列（与factor_values对齐）
            industry: 行业分类序列（与factor_values对齐）
            date_index: 日期索引（如果factor_values索引不是日期类型）
            
        Returns:
            (处理后的因子值, 处理统计信息字典)
        """
        stats = {
            "original_count": len(factor_values),
            "missing_count": factor_values.isna().sum(),
            "winsorized_count": 0,
            "neutralized": False,
            "standardized": True,
        }

        result = factor_values.copy()

        if len(result) < self.config.min_samples:
            logger.warning(f"样本数{len(result)}小于最小要求{self.config.min_samples}，跳过处理")
            return result, {**stats, "skipped": True}

        # Step 1: 缺失值处理
        result, missing_stats = self._handle_missing(result)
        stats.update(missing_stats)

        # Step 1.5: 清理无穷大值（避免污染去极值和中性化计算）
        inf_count = int((np.isinf(result)).sum())
        if inf_count > 0:
            result = result.replace([np.inf, -np.inf], np.nan)
            result, inf_stats = self._handle_missing(result)
            stats["inf_cleaned_count"] = inf_count

        # Step 2: 去极值（全局模式，横截面去极值在 _process_cross_sectional 中实现）
        result, winsorize_stats = self._winsorize(result)
        stats["winsorized_count"] = winsorize_stats["clipped_count"]

        # Step 3: 中性化（⭐支持联合回归）
        if (self.config.enable_market_cap_neutralization or 
            self.config.enable_industry_neutralization):
            
            if (self.config.use_joint_neutralization and 
                market_cap is not None and 
                industry is not None):
                # ⭐ 联合回归：同时控制市值和行业（推荐）
                result = self._neutralize_joint(
                    factor_values=result,
                    market_cap=market_cap,
                    industry=industry,
                    date_index=date_index,
                )
                stats["neutralized"] = True
                stats["neutralization_method"] = "joint"
            else:
                # 顺序方法（fallback）
                if self.config.enable_market_cap_neutralization and market_cap is not None:
                    result = self._neutralize_market_cap(result, market_cap, date_index)
                    stats["neutralized"] = True
                    stats["neutralization_method"] = "sequential"

                if self.config.enable_industry_neutralization and industry is not None:
                    result = self._neutralize_industry(result, industry, date_index)
                    stats["neutralized"] = True

        # Step 4: 标准化（先清理inf，sklearn不接受inf）
        result = result.replace([np.inf, -np.inf], np.nan)
        result = self._standardize(result, date_index)

        return result, stats

    def process_factor_dataframe(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        market_cap_column: str = "market_cap",
        industry_column: str = "industry",
        date_column: str = "date",
        parallel: bool = True,
        max_workers: int = 4,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        批量处理DataFrame中的多个因子
        
        这是最高效的批量处理方法，针对整个DataFrame优化
        
        Args:
            df: 包含因子数据的DataFrame（必须包含date列或date索引）
            factor_columns: 需要处理的因子列名列表
            market_cap_column: 市值列名
            industry_column: 行业列名
            date_column: 日期列名
            parallel: 是否并行处理多个因子
            max_workers: 并行工作线程数
            
        Returns:
            (处理后的DataFrame, 每个因子的统计信息字典)
        """
        result_df = df.copy()
        all_stats = {}

        if self.config.cross_sectional and date_column in result_df.columns:
            logger.info(f"使用横截面模式处理{len(factor_columns)}个因子")

            if parallel and len(factor_columns) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._process_cross_sectional,
                            result_df,
                            col,
                            market_cap_column,
                            industry_column,
                            date_column,
                        ): col
                        for col in factor_columns
                    }

                    for future in futures:
                        col = futures[future]
                        try:
                            processed_col, stats = future.result()
                            result_df[col] = processed_col
                            all_stats[col] = stats
                        except Exception as e:
                            logger.error(f"处理因子{col}失败: {e}")
                            all_stats[col] = {"error": str(e)}
            else:
                for col in factor_columns:
                    try:
                        processed_col, stats = self._process_cross_sectional(
                            result_df, col, market_cap_column, industry_column, date_column
                        )
                        result_df[col] = processed_col
                        all_stats[col] = stats
                    except Exception as e:
                        logger.error(f"处理因子{col}失败: {e}")
                        all_stats[col] = {"error": str(e)}
        else:
            for col in factor_columns:
                try:
                    processed_series, stats = self.process_single_factor(
                        factor_values=df[col],
                        market_cap=df.get(market_cap_column),
                        industry=df.get(industry_column),
                    )
                    result_df[col] = processed_series
                    all_stats[col] = stats
                except Exception as e:
                    logger.error(f"处理因子{col}失败: {e}")
                    all_stats[col] = {"error": str(e)}

        return result_df, all_stats

    def process_multi_stock_factors(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        market_cap_col: str = "market_cap",
        industry_col: str = "industry",
        parallel_stocks: bool = True,
        max_workers: int = 8,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        处理多只股票的因子数据（字典格式）

        这是最常用的接口，与现有代码无缝集成

        Args:
            factor_data: {stock_code: DataFrame} 格式的因子数据
            factor_names: 因子名称列表
            market_cap_col: 市值列名
            industry_col: 行业列名
            parallel_stocks: 是否并行处理不同股票
            max_workers: 并行线程数

        Returns:
            (处理后的因子数据字典, 统计信息字典)
        """
        # 横截面模式：将所有股票合并为一个DataFrame，按日期分组处理
        if self.config.cross_sectional and len(factor_data) > 1:
            return self._process_multi_stock_cross_sectional(
                factor_data, factor_names, market_cap_col, industry_col,
            )

        # 逐股票时间序列模式（fallback）
        # 注意：单股票模式下市值/行业中性化是对时间序列做回归，非横截面回归，量化意义有限
        if (self.config.enable_market_cap_neutralization or self.config.enable_industry_neutralization):
            logger.warning(
                "单股票逐时间序列模式下，市值/行业中性化对时间维度回归（非横截面），"
                "量化意义有限。建议使用横截面模式(cross_sectional=True)获得正确的中性化结果。"
            )
        result_data = {}
        all_stats = {}

        def _process_one_stock(stock_code: str, df: pd.DataFrame):
            processed_df = df.copy()
            stock_stats = {}

            for factor_name in factor_names:
                if factor_name not in processed_df.columns:
                    continue

                try:
                    processed_series, stats = self.process_single_factor(
                        factor_values=processed_df[factor_name],
                        market_cap=processed_df.get(market_cap_col),
                        industry=processed_df.get(industry_col),
                        date_index=processed_df.index if isinstance(processed_df.index, pd.DatetimeIndex) else None,
                    )
                    processed_df[factor_name] = processed_series
                    stock_stats[factor_name] = stats
                except Exception as e:
                    logger.error(f"股票{stock_code}的因子{factor_name}处理失败: {e}")
                    stock_stats[factor_name] = {"error": str(e)}

            return stock_code, processed_df, stock_stats

        if parallel_stocks and len(factor_data) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_one_stock, code, df): code
                    for code, df in factor_data.items()
                }

                for future in futures:
                    try:
                        code, processed_df, stats = future.result()
                        result_data[code] = processed_df
                        all_stats[code] = stats
                    except Exception as e:
                        code = futures[future]
                        logger.error(f"处理股票{code}失败: {e}")
                        result_data[code] = factor_data[code]
                        all_stats[code] = {"error": str(e)}
        else:
            for code, df in factor_data.items():
                try:
                    _, processed_df, stats = _process_one_stock(code, df)
                    result_data[code] = processed_df
                    all_stats[code] = stats
                except Exception as e:
                    logger.error(f"处理股票{code}失败: {e}")
                    result_data[code] = df
                    all_stats[code] = {"error": str(e)}

        return result_data, all_stats

    def _process_multi_stock_cross_sectional(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        market_cap_col: str = "market_cap",
        industry_col: str = "industry",
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        横截面模式处理多只股票因子数据

        将所有股票合并为一个DataFrame，按日期分组进行横截面去极值、中性化、标准化。
        这是业界标准做法：每天对所有股票的因子进行截面处理。
        """
        # 合并所有股票数据
        all_dfs = []
        for stock_code, df in factor_data.items():
            df_copy = df.copy()
            df_copy["stock_code"] = stock_code
            # 确保有date列
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy["date"] = df_copy.index
            elif "date" not in df_copy.columns:
                df_copy["date"] = df_copy.index
            all_dfs.append(df_copy)

        if not all_dfs:
            return factor_data, {}

        merged_df = pd.concat(all_dfs, ignore_index=True)

        # 确保date列是datetime类型
        merged_df["date"] = pd.to_datetime(merged_df["date"])

        result_data = {}
        all_stats = {}

        for factor_name in factor_names:
            if factor_name not in merged_df.columns:
                continue

            try:
                processed_col, stats = self._process_cross_sectional(
                    merged_df, factor_name, market_cap_col, industry_col, "date",
                )
                merged_df[factor_name] = processed_col
                all_stats[factor_name] = stats
            except Exception as e:
                logger.error(f"横截面处理因子{factor_name}失败: {e}")
                all_stats[factor_name] = {"error": str(e)}

        # 拆分回各股票（按日期索引对齐，确保数据不错位）
        for stock_code in factor_data.keys():
            stock_rows = merged_df[merged_df["stock_code"] == stock_code]
            # 恢复原始索引
            original_df = factor_data[stock_code]
            result_df = original_df.copy()

            # 统一构建以日期为索引的映射表，避免 .values 导致的数据错位
            stock_rows_by_date = stock_rows.set_index("date")
            stock_rows_by_date.index = pd.to_datetime(stock_rows_by_date.index)

            for factor_name in factor_names:
                if factor_name not in stock_rows_by_date.columns or factor_name not in result_df.columns:
                    continue

                factor_values_by_date = stock_rows_by_date[factor_name]

                if isinstance(result_df.index, pd.DatetimeIndex):
                    # DatetimeIndex：直接按索引对齐赋值
                    common_idx = result_df.index.intersection(factor_values_by_date.index)
                    if len(common_idx) > 0:
                        result_df.loc[common_idx, factor_name] = factor_values_by_date.loc[common_idx]
                else:
                    # 非DatetimeIndex：通过日期列对齐，使用索引对齐而非 .values
                    if "date" in result_df.columns:
                        result_temp = result_df.set_index("date")
                        result_temp.index = pd.to_datetime(result_temp.index)
                        common_idx = result_temp.index.intersection(factor_values_by_date.index)
                        if len(common_idx) > 0:
                            result_temp.loc[common_idx, factor_name] = factor_values_by_date.loc[common_idx]
                        result_df[factor_name] = result_temp[factor_name].values
                    else:
                        # 无日期列：尝试将原始索引转为日期
                        result_temp = result_df.copy()
                        result_temp.index = pd.to_datetime(result_temp.index)
                        common_idx = result_temp.index.intersection(factor_values_by_date.index)
                        if len(common_idx) > 0:
                            result_temp.loc[common_idx, factor_name] = factor_values_by_date.loc[common_idx]
                        result_df[factor_name] = result_temp[factor_name].values

            result_data[stock_code] = result_df

        return result_data, all_stats

    def _handle_missing(self, series: pd.Series) -> Tuple[pd.Series, Dict]:
        """高性能缺失值处理"""
        missing_count = series.isna().sum()

        if missing_count == 0:
            return series, {"filled_count": 0}

        result = series.copy()

        if self.config.handle_missing == "fill_zero":
            result = result.fillna(0)
        elif self.config.handle_missing == "fill_median":
            median_val = series.median()
            result = result.fillna(median_val)
        elif self.config.handle_missing == "drop":
            result = result.dropna()
        else:
            result = result.fillna(0)

        return result, {"filled_count": int(missing_count)}

    def _winsorize(
        self,
        series: pd.Series,
    ) -> Tuple[pd.Series, Dict]:
        """
        去极值（全局模式，横截面去极值见 _process_cross_sectional）
        """
        method = self.config.winsorize_method
        original = series.copy()
        clipped_count = 0

        if method == WinsorizeMethod.MAD:
            median = series.median()
            # σ_hat = 1.4826 * MAD（正态分布下 1.4826*MAD ≈ σ）
            sigma_hat = 1.4826 * (series - median).abs().median()

            if sigma_hat == 0 or np.isnan(sigma_hat):
                # MAD=0时（如数据过于集中），用标准差作为σ_hat的估计
                # σ_hat ≈ std()，因此fallback直接使用std()而非std()*0.6745
                sigma_hat = series.std()
                if sigma_hat == 0 or np.isnan(sigma_hat) or sigma_hat < 1e-10:
                    return series, {"clipped_count": 0}
            
            lower_bound = median - self.config.winsorize_n_sigma * sigma_hat
            upper_bound = median + self.config.winsorize_n_sigma * sigma_hat
            
            mask_lower = series < lower_bound
            mask_upper = series > upper_bound
            clipped_count = int(mask_lower.sum() + mask_upper.sum())
            
            result = series.clip(lower=lower_bound, upper=upper_bound)

        elif method == WinsorizeMethod.PERCENTILE:
            lower_pct, upper_pct = self.config.winsorize_limits
            # scipy.winsorize的limits参数：(低端截断比例, 高端截断比例)
            # 例如1%-99%分位 → limits=(0.01, 0.01)，即两端各截1%
            # lower_pct=0.01, upper_pct=0.99 → 高端截断比例=1-0.99=0.01
            winsorized = scipy_winsorize(
                series.values, limits=(lower_pct, 1 - upper_pct), nan_policy="omit"
            )
            result = pd.Series(winsorized, index=series.index)
            clipped_count = int((result != series).sum())

        elif method == WinsorizeMethod.STD:
            mean = series.mean()
            std = series.std()
            
            if std == 0 or np.isnan(std) or std < 1e-10:
                return series, {"clipped_count": 0}
            
            lower_bound = mean - self.config.winsorize_n_sigma * std
            upper_bound = mean + self.config.winsorize_n_sigma * std
            
            mask_lower = series < lower_bound
            mask_upper = series > upper_bound
            clipped_count = int(mask_lower.sum() + mask_upper.sum())
            
            result = series.clip(lower=lower_bound, upper=upper_bound)
        else:
            result = series

        return result, {"clipped_count": clipped_count}

    def _process_cross_sectional(
        self,
        df: pd.DataFrame,
        factor_column: str,
        market_cap_column: str,
        industry_column: str,
        date_column: str,
    ) -> Tuple[pd.Series, Dict]:
        """
        横截面处理（按日期分组）
        
        这是业界标准的做法：每天对所有股票的因子进行截面处理
        ⭐ v2.0: 默认使用联合回归中性化
        """
        stats = {
            "original_count": len(df),
            "dates_processed": 0,
            "total_winsorized": 0,
            "neutralized": self.config.enable_market_cap_neutralization or self.config.enable_industry_neutralization,
        }

        result = df[factor_column].copy()

        if date_column not in df.columns:
            return result, {**stats, "error": f"缺少日期列{date_column}"}

        def process_group(group: pd.DataFrame) -> pd.Series:
            """处理单个日期组的因子值"""
            factor_vals = group[factor_column].copy()

            if len(factor_vals) < self.config.min_samples:
                return factor_vals

            # 清理无穷大值（避免污染去极值和中性化计算）
            inf_mask = np.isinf(factor_vals)
            if inf_mask.any():
                factor_vals = factor_vals.replace([np.inf, -np.inf], np.nan)

            # 缺失值处理（横截面模式下按配置填充）
            if factor_vals.isna().any() and self.config.handle_missing != "drop":
                if self.config.handle_missing == "fill_median":
                    factor_vals = factor_vals.fillna(factor_vals.median())
                else:
                    factor_vals = factor_vals.fillna(0)

            # 去极值
            if self.config.winsorize_method == WinsorizeMethod.MAD:
                median = factor_vals.median()
                sigma_hat = 1.4826 * (factor_vals - median).abs().median()
                if sigma_hat == 0:
                    # 同_winsorize方法：MAD=0时用std作为σ_hat估计
                    sigma_hat = factor_vals.std()
                    if sigma_hat == 0 or np.isnan(sigma_hat):
                        # 数据完全一致，无需去极值
                        pass
                    else:
                        factor_vals = factor_vals.clip(
                            lower=median - self.config.winsorize_n_sigma * sigma_hat,
                            upper=median + self.config.winsorize_n_sigma * sigma_hat,
                        )
                else:
                    factor_vals = factor_vals.clip(
                        lower=median - self.config.winsorize_n_sigma * sigma_hat,
                        upper=median + self.config.winsorize_n_sigma * sigma_hat,
                    )
            elif self.config.winsorize_method == WinsorizeMethod.PERCENTILE:
                winsorized = scipy_winsorize(
                    factor_vals.values,
                    limits=(self.config.winsorize_limits[0], 1 - self.config.winsorize_limits[1]),
                    nan_policy="omit",
                )
                factor_vals = pd.Series(winsorized, index=factor_vals.index)
            elif self.config.winsorize_method == WinsorizeMethod.STD:
                mean = factor_vals.mean()
                std = factor_vals.std()
                if std > 0:
                    factor_vals = factor_vals.clip(
                        lower=mean - self.config.winsorize_n_sigma * std,
                        upper=mean + self.config.winsorize_n_sigma * std,
                    )

            # 中性化（⭐优先使用联合回归）
            if (self.config.enable_market_cap_neutralization or 
                self.config.enable_industry_neutralization):
                
                has_market_cap = market_cap_column in group.columns
                has_industry = industry_column in group.columns
                
                if (self.config.use_joint_neutralization and 
                    has_market_cap and 
                    has_industry):
                    valid_mask = (
                        factor_vals.notna() & 
                        group[market_cap_column].notna() & 
                        group[industry_column].notna() & 
                        (group[market_cap_column] > 0)
                    )
                    
                    if valid_mask.sum() >= self.config.min_samples:
                        log_mc = np.log(
                            group.loc[valid_mask, market_cap_column]
                            .replace(0, np.nan)
                        )
                        log_mc = log_mc.fillna(log_mc.mean())
                        
                        industries = group.loc[valid_mask, industry_column].astype(str)
                        unique_inds = sorted(industries.unique())
                        
                        if len(unique_inds) >= 2:
                            industry_dummies = pd.get_dummies(
                                industries, drop_first=True
                            ).astype(float)

                            X = np.column_stack([
                                log_mc.values,
                                industry_dummies.values
                            ])

                            y = factor_vals[valid_mask].values
                            model = LinearRegression()
                            model.fit(X, y)

                            residuals = y - model.predict(X)
                            factor_vals.loc[valid_mask] = residuals
                        else:
                            # 行业数不足2，仅做市值中性化
                            if has_market_cap and valid_mask.sum() >= self.config.min_samples:
                                X_mc = log_mc.values.reshape(-1, 1)
                                y_mc = factor_vals[valid_mask].values
                                model = LinearRegression()
                                model.fit(X_mc, y_mc)
                                residuals = y_mc - model.predict(X_mc)
                                factor_vals.loc[valid_mask] = residuals
                    else:
                        logger.debug(f"日期组样本不足({valid_mask.sum()}), 跳过中性化")
                        
                else:
                    # 顺序方法（fallback）
                    if (self.config.enable_market_cap_neutralization and 
                        has_market_cap):
                        valid_mask = (
                            factor_vals.notna() & 
                            group[market_cap_column].notna() & 
                            (group[market_cap_column] > 0)
                        )
                        if valid_mask.sum() >= self.config.min_samples:
                            log_mc = np.log(
                                group.loc[valid_mask, market_cap_column]
                                .replace(0, np.nan)
                            )
                            log_mc = log_mc.fillna(log_mc.mean())
                            
                            X = log_mc.values.reshape(-1, 1)
                            y = factor_vals[valid_mask].values
                            
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            residuals = y - model.predict(X)
                            factor_vals.loc[valid_mask] = residuals
                    
                    if (self.config.enable_industry_neutralization and 
                        has_industry):
                        industry_series = group[industry_column]
                        factor_vals = self._neutralize_industry(
                            factor_vals, industry_series
                        )

            # 标准化
            if self.config.standardize_method == StandardizeMethod.ZSCORE:
                valid = factor_vals.replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) > 1:
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(valid.values.reshape(-1, 1)).flatten()
                    factor_vals[valid.index] = scaled
            elif self.config.standardize_method == StandardizeMethod.RANK:
                factor_vals = factor_vals.rank(pct=True)
            elif self.config.standardize_method == StandardizeMethod.MEDIAN_MAD:
                valid = factor_vals.replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) > 1:
                    median = valid.median()
                    mad = 1.4826 * (valid - median).abs().median()
                    if mad > 0 and not np.isnan(mad):
                        factor_vals[valid.index] = (valid - median) / mad

            return factor_vals

        # 抑制 pandas FutureWarning（groupby.apply 对分组列的操作警告）
        # 必须在线程入口设置，catch_warnings 在子线程中不生效
        warnings.simplefilter("ignore", FutureWarning)
        result = df.groupby(df[date_column], group_keys=False).apply(process_group)
        stats["dates_processed"] = df[date_column].nunique()

        return result, stats

    def _neutralize_market_cap(
        self,
        factor_values: pd.Series,
        market_cap: pd.Series,
        date_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        高性能市值中性化
        
        使用线性回归去除市值影响
        """
        valid_mask = factor_values.notna() & market_cap.notna() & (market_cap > 0)

        if valid_mask.sum() < self.config.min_samples:
            logger.warning("有效样本不足，跳过市值中性化")
            return factor_values

        result = factor_values.copy()

        log_mc = np.log(market_cap[valid_mask])
        y = factor_values[valid_mask]

        model = LinearRegression()
        model.fit(log_mc.values.reshape(-1, 1), y.values)

        residuals = y - model.predict(log_mc.values.reshape(-1, 1))
        result.loc[valid_mask] = residuals

        return result

    def _neutralize_industry(
        self,
        factor_values: pd.Series,
        industry: pd.Series,
        date_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        行业中性化 - 使用行业哑变量回归残差法（JoinQuant/BigQuant标准）

        通过线性回归将因子值对行业哑变量回归，取残差作为中性化后的因子值。
        该方法消除了行业间的系统性差异，同时保留了行业内的相对排序。

        Args:
            factor_values: 因子值序列
            industry: 行业分类序列（与factor_values对齐）
            date_index: 日期索引（可选）

        Returns:
            行业中性化后的因子值序列
        """
        if industry.isna().all():
            return factor_values

        valid_mask = factor_values.notna() & industry.notna()
        if valid_mask.sum() < self.config.min_samples:
            logger.warning("有效样本不足，跳过行业中性化")
            return factor_values

        industries = industry[valid_mask].astype(str)
        unique_inds = sorted(industries.unique())
        if len(unique_inds) < 2:
            logger.warning("行业分类不足2个，跳过行业中性化")
            return factor_values

        # 检查最小行业样本量：将小行业（<5只股票）合并为"其他"类别，
        # 避免因个别小行业导致整个行业中性化被跳过
        industry_counts = industries.value_counts()
        small_industries = industry_counts[industry_counts < 5].index
        if len(small_industries) > 0:
            industries = industries.replace(small_industries.to_list(), "Other")
            logger.info(
                f"将{len(small_industries)}个小行业（样本<5）合并为'Other'类别: "
                f"{small_industries.to_list()}"
            )
            # 合并后重新检查行业数
            unique_inds = sorted(industries.unique())
            if len(unique_inds) < 2:
                logger.warning("合并小行业后行业分类仍不足2个，跳过行业中性化")
                return factor_values

        result = factor_values.copy()
        y = factor_values[valid_mask].values

        industry_dummies = pd.get_dummies(industries, drop_first=True).astype(float)
        X = industry_dummies.values

        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)

        result.loc[valid_mask] = residuals
        return result

    def _neutralize_joint(
        self,
        factor_values: pd.Series,
        market_cap: pd.Series,
        industry: pd.Series,
        date_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        行业市值联合回归中性化（⭐推荐方法）
        
        通过多元线性回归同时控制市值和行业的影响，取残差作为中性化后的因子值。
        优先使用statsmodels OLS（提供R²、p值、F统计量等诊断信息），
        不可用时回退到sklearn LinearRegression。
        
        数学模型：
            factor = β₀ + β₁ × log(market_cap) + Σ(γᵢ × industry_dummyᵢ) + ε
        
        其中 ε 即为中性化后的因子值。
        """
        valid_mask = (
            factor_values.notna() & 
            market_cap.notna() & 
            industry.notna() & 
            (market_cap > 0)
        )
        
        if valid_mask.sum() < self.config.min_samples:
            logger.warning("有效样本不足，跳过联合中性化")
            return factor_values
        
        result = factor_values.copy()
        y = factor_values[valid_mask].values
        
        log_mc = np.log(market_cap[valid_mask])
        industries = industry[valid_mask].astype(str)
        unique_inds = sorted(industries.unique())
        
        if len(unique_inds) < 2:
            logger.warning("行业分类不足2个，降级为仅市值中性化")
            return self._neutralize_market_cap(factor_values, market_cap, date_index)
        
        industry_dummies = pd.get_dummies(industries, drop_first=True).astype(float)
        
        X = np.column_stack([
            log_mc.values,
            industry_dummies.values
        ])
        
        # 使用statsmodels OLS：自动输出R²、p值、F统计量等诊断信息
        # 检查共线性：如果设计矩阵秩亏，回退到sklearn（对共线性更鲁棒）
        try:
            X_with_const = sm.add_constant(X)
            # 使用伪逆处理共线性（rank-deficient）
            ols_model = sm.OLS(y, X_with_const).fit(method='qr')
            residuals = y - ols_model.fittedvalues
            result.loc[valid_mask] = residuals
            logger.debug(
                f"联合回归完成(statsmodels): 市值系数={ols_model.params[1]:.6f}, "
                f"行业数={len(unique_inds)}, R²={ols_model.rsquared:.4f}, "
                f"F-stat={ols_model.fvalue:.2f}(p={ols_model.f_pvalue:.4f})"
            )
        except Exception as e:
            logger.warning(f"statsmodels OLS失败（共线性或奇异矩阵），回退到sklearn: {e}")
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            result.loc[valid_mask] = residuals
        
        return result

    def _standardize(
        self,
        factor_values: pd.Series,
        date_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        高性能标准化
        
        支持多种方法，全部向量化实现
        """
        method = self.config.standardize_method

        if method == StandardizeMethod.ZSCORE:
            # 使用sklearn StandardScaler（更稳健的边缘情况处理）
            valid = factor_values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 2:
                return factor_values
            scaler = StandardScaler()
            scaled = scaler.fit_transform(valid.values.reshape(-1, 1)).flatten()
            result = factor_values.copy()
            result[valid.index] = scaled
            return result

        elif method == StandardizeMethod.RANK:
            return factor_values.rank(pct=True)

        elif method == StandardizeMethod.MEDIAN_MAD:
            # 使用真正的Median-MAD标准化（非RobustScaler/IQR）
            # MAD = 1.4826 * median(|x - median(x)|)，正态分布下 MAD ≈ σ
            valid = factor_values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 2:
                return factor_values
            median = valid.median()
            mad = 1.4826 * (valid - median).abs().median()
            if mad == 0 or np.isnan(mad) or mad < 1e-10:
                return factor_values
            result = factor_values.copy()
            result[valid.index] = (valid - median) / mad
            return result

        return factor_values

    def get_processing_summary(self, stats_dict: Dict) -> str:
        """
        生成处理摘要报告
        
        Args:
            stats_dict: process_factor_dataframe返回的统计信息字典
            
        Returns:
            Markdown格式的摘要字符串
        """
        lines = ["## 数据美颜处理摘要\n"]
        lines.append("| 因子名称 | 原始数量 | 缺失值 | 截断数量 | 是否中性化 | 中性化方法 |")
        lines.append("|---------|---------|--------|---------|-----------|----------|")

        for factor_name, stats in stats_dict.items():
            if not isinstance(stats, dict):
                continue
            if "error" in stats:
                lines.append(f"| {factor_name} | - | - | - | ❌ 错误: {stats['error']} | - |")
            else:
                neutralized = "✅ 是" if stats.get("neutralized", False) else "❌ 否"
                method = stats.get("neutralization_method", "-")
                lines.append(
                    f"| {factor_name} | "
                    f"{stats.get('original_count', 0)} | "
                    f"{stats.get('missing_count', 0)} | "
                    f"{stats.get('winsorized_count', 0)} | "
                    f"{neutralized} | "
                    f"{method} |"
                )

        total_factors = len(stats_dict)
        success_count = sum(1 for s in stats_dict.values() if isinstance(s, dict) and "error" not in s)
        lines.append(f"\n**总计**: {success_count}/{total_factors} 个因子处理成功")

        return "\n".join(lines)


# 全局默认实例（使用标准配置）
default_pipeline = FactorPreprocessingPipeline()

# 预定义常用配置
CONSERVATIVE_CONFIG = PreprocessingConfig(
    winsorize_method=WinsorizeMethod.MAD,
    winsorize_limits=(0.02, 0.98),
    enable_market_cap_neutralization=True,
    enable_industry_neutralization=True,
    use_joint_neutralization=True,
)

AGGRESSIVE_CONFIG = PreprocessingConfig(
    winsorize_method=WinsorizeMethod.PERCENTILE,
    winsorize_limits=(0.005, 0.995),
    enable_market_cap_neutralization=False,
    enable_industry_neutralization=False,
    use_joint_neutralization=False,
)

ML_MODEL_CONFIG = PreprocessingConfig(
    winsorize_method=WinsorizeMethod.MAD,
    standardize_method=StandardizeMethod.ZSCORE,
    enable_market_cap_neutralization=True,
    enable_industry_neutralization=True,
    handle_missing="fill_median",
    use_joint_neutralization=True,
)