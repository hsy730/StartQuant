"""
因子中性化服务 - 市值中性化和行业中性化
"""
import logging

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.linear_model import LinearRegression

from backend.services.data_service import data_service

logger = logging.getLogger(__name__)


class FactorNeutralizationService:
    """因子中性化服务类"""

    def __init__(self):
        pass

    def neutralize_market_cap(
        self,
        df: pd.DataFrame,
        factor_name: str,
        market_cap_column: str = "market_cap"
    ) -> pd.Series:
        """
        市值中性化 - 使用线性回归去除市值影响

        Args:
            df: 包含因子值和市值的数据框
            factor_name: 因子列名
            market_cap_column: 市值列名

        Returns:
            中性化后的因子值（回归残差）
        """
        if market_cap_column not in df.columns:
            raise ValueError(f"数据框中缺少市值列: {market_cap_column}")

        if factor_name not in df.columns:
            raise ValueError(f"数据框中缺少因子列: {factor_name}")

        # 移除缺失值
        valid_data = df[[factor_name, market_cap_column]].dropna()

        if len(valid_data) < 10:
            raise ValueError("有效数据不足，无法进行中性化")

        # 对市值取对数
        log_market_cap = np.log(valid_data[market_cap_column].replace(0, np.nan))
        log_market_cap = log_market_cap.fillna(log_market_cap.mean())

        # 线性回归
        model = LinearRegression()
        X = log_market_cap.values.reshape(-1, 1)
        y = valid_data[factor_name].values

        model.fit(X, y)

        # 计算残差（中性化后的因子值）
        residual = y - model.predict(X)

        # 创建返回的Series，保持原索引
        result = pd.Series(index=df.index, dtype=float)
        result.loc[valid_data.index] = residual

        return result

    def neutralize_industry(
        self,
        df: pd.DataFrame,
        factor_name: str,
        industry_column: str = "industry"
    ) -> pd.Series:
        """
        行业中性化 - 在行业内标准化因子

        Args:
            df: 包含因子值和行业分类的数据框
            factor_name: 因子列名
            industry_column: 行业分类列名

        Returns:
            行业中性化后的因子值
        """
        if industry_column not in df.columns:
            raise ValueError(f"数据框中缺少行业列: {industry_column}")

        if factor_name not in df.columns:
            raise ValueError(f"数据框中缺少因子列: {factor_name}")

        # 创建结果Series
        result = pd.Series(index=df.index, dtype=float)

        # 按行业分组，每组内标准化
        for industry, group in df.groupby(industry_column):
            factor_values = group[factor_name]

            # 计算行业内的均值和标准差
            industry_mean = factor_values.mean()
            industry_std = factor_values.std()

            if industry_std > 0:
                # 标准化
                normalized = (factor_values - industry_mean) / industry_std
            else:
                # 如果标准差为0，直接使用原值
                normalized = factor_values - industry_mean

            result.loc[group.index] = normalized

        return result

    def neutralize_both(
        self,
        df: pd.DataFrame,
        factor_name: str,
        market_cap_column: str = "market_cap",
        industry_column: str = "industry"
    ) -> pd.Series:
        """
        同时进行市值和行业中性化

        Args:
            df: 数据框
            factor_name: 因子列名
            market_cap_column: 市值列名
            industry_column: 行业列名

        Returns:
            双重中性化后的因子值
        """
        # 先进行市值中性化
        market_cap_neutralized = self.neutralize_market_cap(
            df, factor_name, market_cap_column
        )

        # 创建临时数据框用于行业中性化
        temp_df = df.copy()
        temp_df[f"{factor_name}_mc_neutral"] = market_cap_neutralized

        # 再进行行业中性化
        industry_neutralized = self.neutralize_industry(
            temp_df, f"{factor_name}_mc_neutral", industry_column
        )

        return industry_neutralized

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
            result["industry"] = result["stock_code"].map(industry_map)
        else:
            result["industry"] = "unknown"

        return result


# 全局因子中性化服务实例
factor_neutralization_service = FactorNeutralizationService()
