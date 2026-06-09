"""
因子中性化服务 - 市值中性化和行业中性化

所有中性化方法统一使用线性回归残差法（JoinQuant/BigQuant标准）：
- 市值中性化：因子值 ~ log(市值) 回归取残差
- 行业中性化：因子值 ~ 行业哑变量 回归取残差
- 联合中性化：因子值 ~ 行业哑变量 + log(市值) 回归取残差

回归残差法的优势：
1. 消除系统性影响的同时保留因子信息
2. 与JoinQuant/BigQuant等业界平台一致
3. 数学上更严谨（控制了所有行业的共同影响）
"""
import logging

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.linear_model import LinearRegression

from backend.services.data_service import data_service

logger = logging.getLogger(__name__)


class FactorNeutralizationService:
    """因子中性化服务类"""

    MIN_SAMPLES: int = 10

    def __init__(self):
        pass

    def _validate_columns(self, df: pd.DataFrame, *columns: str) -> None:
        """校验DataFrame中是否包含指定列"""
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"数据框中缺少列: {col}")

    def _build_result_series(self, df: pd.DataFrame, factor_name: str, valid_index: pd.Index, residuals: np.ndarray) -> pd.Series:
        """构建结果Series，保持原索引，缺失值位置填充NaN"""
        result = pd.Series(index=df.index, dtype=float)
        result.loc[valid_index] = residuals
        return result

    def neutralize_market_cap(
        self,
        df: pd.DataFrame,
        factor_name: str,
        market_cap_column: str = "market_cap"
    ) -> pd.Series:
        """
        市值中性化 - 使用线性回归去除市值影响

        方法：因子值 ~ log(市值) 线性回归，取残差

        Args:
            df: 包含因子值和市值的数据框
            factor_name: 因子列名
            market_cap_column: 市值列名

        Returns:
            中性化后的因子值（回归残差）
        """
        self._validate_columns(df, factor_name, market_cap_column)

        valid_data = df[[factor_name, market_cap_column]].dropna()
        if len(valid_data) < self.MIN_SAMPLES:
            raise ValueError("有效数据不足，无法进行中性化")

        # 排除市值<=0的记录，避免log(0)或log(负数)
        valid_data = valid_data[valid_data[market_cap_column] > 0]
        if len(valid_data) < self.MIN_SAMPLES:
            raise ValueError("有效数据不足（市值>0），无法进行中性化")

        log_market_cap = np.log(valid_data[market_cap_column])

        model = LinearRegression()
        X = log_market_cap.values.reshape(-1, 1)
        y = valid_data[factor_name].values

        model.fit(X, y)
        residuals = y - model.predict(X)

        return self._build_result_series(df, factor_name, valid_data.index, residuals)

    def neutralize_industry(
        self,
        df: pd.DataFrame,
        factor_name: str,
        industry_column: str = "industry"
    ) -> pd.Series:
        """
        行业中性化 - 使用行业哑变量回归残差法（JoinQuant/BigQuant标准）

        方法：因子值 ~ 行业哑变量 线性回归，取残差
        消除行业间的系统性差异，同时保留行业内的相对排序。

        Args:
            df: 包含因子值和行业分类的数据框
            factor_name: 因子列名
            industry_column: 行业分类列名

        Returns:
            行业中性化后的因子值（回归残差）
        """
        self._validate_columns(df, factor_name, industry_column)

        valid_data = df[[factor_name, industry_column]].dropna()
        if len(valid_data) < self.MIN_SAMPLES:
            raise ValueError("有效数据不足，无法进行行业中性化")

        industries = valid_data[industry_column].astype(str)
        unique_industries = sorted(industries.unique())

        if len(unique_industries) < 2:
            logger.warning("行业分类不足2个，跳过行业中性化")
            return df[factor_name].copy()

        # 检查最小行业样本量，样本过小的行业会导致回归不稳定
        MIN_INDUSTRY_SIZE = 5
        industry_counts = industries.value_counts()
        small_industries = industry_counts[industry_counts < MIN_INDUSTRY_SIZE]
        if len(small_industries) > 0:
            for ind_name in small_industries.index:
                logger.warning(f"行业 '{ind_name}' 样本量仅 {industry_counts[ind_name]}，不足{MIN_INDUSTRY_SIZE}，已过滤")
            # 过滤掉小行业（而非合并为Other），避免小行业噪声影响回归
            valid_mask = ~industries.isin(small_industries.index)
            valid_data = valid_data[valid_mask]
            industries = industries[valid_mask]
            df = df[valid_mask]
            unique_industries = sorted(industries.unique())
            if len(unique_industries) < 2:
                logger.warning("过滤小行业后行业分类不足2个，跳过行业中性化")
                return df[factor_name].copy()

        industry_dummies = pd.get_dummies(industries, drop_first=True).astype(float)
        X = industry_dummies.values
        y = valid_data[factor_name].values

        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)

        return self._build_result_series(df, factor_name, valid_data.index, residuals)

    def neutralize_both(
        self,
        df: pd.DataFrame,
        factor_name: str,
        market_cap_column: str = "market_cap",
        industry_column: str = "industry"
    ) -> pd.Series:
        """
        行业+市值联合中性化（JoinQuant/BigQuant标准：一次回归同时剥离）

        方法：因子值 ~ 行业哑变量 + log(市值) 线性回归，取残差
        同时控制市值和行业两个维度的系统性影响。

        Args:
            df: 数据框
            factor_name: 因子列名
            market_cap_column: 市值列名
            industry_column: 行业列名

        Returns:
            双重中性化后的因子值
        """
        has_mc = market_cap_column in df.columns
        has_industry = industry_column in df.columns

        if not has_mc and not has_industry:
            return df[factor_name].copy()

        self._validate_columns(df, factor_name)

        cols = [factor_name]
        if has_mc:
            cols.append(market_cap_column)
        if has_industry:
            cols.append(industry_column)

        valid_data = df[cols].dropna()
        if len(valid_data) < self.MIN_SAMPLES:
            raise ValueError("有效数据不足，无法进行联合中性化")

        # Step 1: 先完成所有过滤，再统一构建特征矩阵
        # 这样避免过滤顺序导致的数据对齐Bug

        # 1a. 过滤市值<=0的记录（必须在构建dummies之前完成）
        if has_mc:
            valid_data = valid_data[valid_data[market_cap_column] > 0]
            if len(valid_data) < self.MIN_SAMPLES:
                raise ValueError("有效数据不足（市值>0），无法进行联合中性化")

        # 1b. 合并小行业（必须在构建dummies之前完成）
        merged_industries = None
        if has_industry:
            industries = valid_data[industry_column].astype(str)
            unique_industries = sorted(industries.unique())
            if len(unique_industries) < 2:
                logger.warning("联合中性化：行业分类不足2个，跳过行业中性化部分")
                has_industry = False
            else:
                MIN_INDUSTRY_SIZE = 5
                industry_counts = industries.value_counts()
                small_industries = industry_counts[industry_counts < MIN_INDUSTRY_SIZE]
                if len(small_industries) > 0:
                    for ind_name in small_industries.index:
                        logger.warning(f"行业 '{ind_name}' 样本量仅 {industry_counts[ind_name]}，不足{MIN_INDUSTRY_SIZE}，已过滤")
                    # 过滤掉小行业（而非合并为Other），避免小行业噪声影响回归
                    valid_mask = ~industries.isin(small_industries.index)
                    valid_data = valid_data[valid_mask]
                    industries = industries[valid_mask]
                    df = df[valid_mask]
                    unique_industries = sorted(industries.unique())
                    if len(unique_industries) < 2:
                        logger.warning("联合中性化：过滤小行业后行业分类不足2个，跳过行业中性化部分")
                        has_industry = False
            if has_industry:
                merged_industries = industries

        # Step 2: 统一构建特征矩阵（此时valid_data已完成所有过滤）
        y = valid_data[factor_name].values
        X_list = []

        if has_industry:
            industry_dummies = pd.get_dummies(merged_industries, drop_first=True).astype(float)
            X_list.append(industry_dummies.values)

        if has_mc:
            log_mc = np.log(valid_data[market_cap_column])
            X_list.append(log_mc.values.reshape(-1, 1))

        if not X_list:
            return df[factor_name].copy()

        X = np.hstack(X_list)
        if X.shape[1] == 0:
            logger.warning("联合中性化：有效特征维度为0，跳过中性化")
            return df[factor_name].copy()

        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)

        return self._build_result_series(df, factor_name, valid_data.index, residuals)

    def get_industry_classification(self, stock_codes: List[str]) -> Dict[str, str]:
        try:
            return data_service.get_industry_classification(stock_codes)
        except Exception as e:
            logger.warning(f"获取行业分类失败: {e}, 使用默认分类")
            return {code: "unknown" for code in stock_codes}

    def add_industry_classification(
        self,
        df: pd.DataFrame,
        stock_codes: List[str]
    ) -> pd.DataFrame:
        industry_map = self.get_industry_classification(stock_codes)

        result = df.copy()

        if "stock_code" in df.columns:
            pure_codes = result["stock_code"].str.replace(r'\.(SH|SZ|BJ)$', '', regex=True)
            result["industry"] = pure_codes.map(industry_map)
        else:
            result["industry"] = "unknown"

        return result


# 全局因子中性化服务实例
factor_neutralization_service = FactorNeutralizationService()
