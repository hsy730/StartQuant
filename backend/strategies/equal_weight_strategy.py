"""
等权重策略 - 等权配置所有股票
"""
from typing import Dict
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class EqualWeightStrategy(BaseStrategy):
    """
    等权重策略

    逻辑：
    1. 信号为1时，等权配置所有股票
    2. 信号为-1时，清仓
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        top_percentile: float = 0.2,
        **kwargs
    ):
        """
        初始化等权重策略

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            top_percentile: 选择股票的百分比（默认前20%）
        """
        super().__init__(initial_capital, commission_rate, **kwargs)
        self.top_percentile = top_percentile

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        简化版：假设df已经选好股票池，全部买入

        Args:
            df: 数据

        Returns:
            信号序列
        """
        # 简化实现：全部买入
        signals = pd.Series(1, index=df.index)
        return signals

    def calculate_weights(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        计算等权重

        Args:
            df: 数据
            signals: 信号

        Returns:
            权重序列
        """
        weights = pd.Series(0.0, index=df.index)

        # 有信号时，等权配置（1/N）
        mask = signals == 1
        # 单股票场景：满仓
        weights[mask] = 1.0

        return weights
