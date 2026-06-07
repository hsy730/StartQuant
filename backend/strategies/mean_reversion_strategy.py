"""
均值回归策略 - 价格偏离均值时回归
"""
from typing import Dict
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略

    逻辑：
    1. 计算价格的Z-score（偏离均值的标准差倍数）
    2. Z-score > 2时超买，卖出
    3. Z-score < -2时超卖，买入
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        lookback_window: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        **kwargs
    ):
        """
        初始化均值回归策略

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            lookback_window: 回看窗口（天数）
            entry_threshold: 进场阈值（Z-score）
            exit_threshold: 出场阈值（Z-score）
        """
        super().__init__(initial_capital, commission_rate, **kwargs)
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        基于均值回归生成交易信号

        Args:
            df: 数据（必须包含close列）

        Returns:
            信号序列
        """
        signals = pd.Series(0, index=df.index)

        # 计算移动平均和标准差
        rolling_mean = df["close"].rolling(window=self.lookback_window).mean()
        rolling_std = df["close"].rolling(window=self.lookback_window).std()

        # 计算Z-score
        zscore = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)

        # 带持仓状态记忆的均值回归信号生成
        # entry_threshold: 进场阈值（超买/超卖）
        # exit_threshold: 出场阈值（回归到均值附近时平仓）
        position = 0  # 0: 空仓, 1: 多头, -1: 空头
        for i in range(len(zscore)):
            z = zscore.iloc[i]
            if pd.isna(z):
                continue
            if position == 0:
                if z < -self.entry_threshold:
                    position = 1   # 超卖 → 买入
                elif z > self.entry_threshold:
                    position = -1  # 超买 → 卖出
            elif position == 1:
                if abs(z) < self.exit_threshold:
                    position = 0   # 回归均值 → 平仓
            elif position == -1:
                if abs(z) < self.exit_threshold:
                    position = 0   # 回归均值 → 平仓
            signals.iloc[i] = position

        return signals

    def calculate_weights(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        计算权重

        Args:
            df: 数据
            signals: 信号

        Returns:
            权重序列
        """
        weights = pd.Series(0.0, index=df.index)

        # 信号为1时满仓
        weights[signals == 1] = 1.0

        # 信号为-1时做空（可选）
        # weights[signals == -1] = -1.0

        return weights
