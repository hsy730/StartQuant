"""
市值加权策略 - 按市值分配权重
"""
import pandas as pd
from .base_strategy import BaseStrategy
from backend.utils.safe_math import safe_divide


class MarketCapStrategy(BaseStrategy):
    """
    市值加权策略

    逻辑：
    1. 信号为1时，按市值分配权重
    2. 市值大的股票权重高
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        market_cap_column: str = "market_cap",
        **kwargs
    ):
        """
        初始化市值加权策略

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            market_cap_column: 市值列名
        """
        super().__init__(initial_capital, commission_rate, **kwargs)
        self.market_cap_column = market_cap_column

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        逻辑：对所有有市值数据的股票生买入信号

        Args:
            df: 数据

        Returns:
            信号序列
        """
        signals = pd.Series(0, index=df.index)

        # 对有市值数据的股票生买入信号
        if self.market_cap_column in df.columns:
            mask = df[self.market_cap_column].notna() & (df[self.market_cap_column] > 0)
            signals[mask] = 1
        else:
            # 如果没有市值数据，对所有股票生买入信号
            signals = pd.Series(1, index=df.index)

        return signals

    def calculate_weights(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        按市值计算权重（向量化实现）

        Args:
            df: 数据（必须包含market_cap列）
            signals: 信号

        Returns:
            权重序列
        """
        weights = pd.Series(0.0, index=df.index)

        # 检查是否有市值数据
        if self.market_cap_column not in df.columns:
            # 没有市值数据，退化为等权重
            mask = signals == 1
            if isinstance(df.index, pd.MultiIndex):
                # MultiIndex 下按日期分组计算等权
                n_per_date = mask.groupby(level=0).transform("sum")
                weights[mask] = safe_divide(1.0, n_per_date[mask], default=0.0)
            else:
                # 单股票场景：满仓
                weights[mask] = 1.0
            return weights

        # 确定日期级别
        if df.index.nlevels == 1:
            # 识别DatetimeIndex作为日期级别
            if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
                date_level = 0
            else:
                date_level = None
        else:
            level_names = list(df.index.names)
            date_level = level_names.index("date") if "date" in level_names else 0

        # 向量化计算：每个日期的市值权重 = 个股市值 / 该日期总市值
        valid_mask = (signals == 1) & df[self.market_cap_column].notna() & (df[self.market_cap_column] > 0)

        if date_level is not None:
            date_grouper = df.index.get_level_values(date_level) if df.index.nlevels > 1 else df.index
            total_mcap = df.loc[valid_mask, self.market_cap_column].groupby(
                date_grouper[valid_mask.values]
            ).transform("sum")
            weights[valid_mask] = safe_divide(
                df.loc[valid_mask, self.market_cap_column].values,
                total_mcap.values,
                default=0.0,
            )
        else:
            # 无日期级别时，全局市值加权
            total_mcap = df.loc[valid_mask, self.market_cap_column].sum()
            if total_mcap > 0:
                weights[valid_mask] = safe_divide(
                    df.loc[valid_mask, self.market_cap_column].values,
                    total_mcap,
                    default=0.0,
                )

        return weights
