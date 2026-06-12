"""
均值回归策略 - 价格偏离均值时回归
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from backend.utils.safe_math import safe_divide


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
        **kwargs,
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
        # MultiIndex 下必须按资产分组计算，否则跨资产混合统计量
        if isinstance(df.index, pd.MultiIndex):
            rolling_mean = (
                df["close"]
                .groupby(level=1)
                .transform(lambda s: s.rolling(window=self.lookback_window).mean())
            )
            rolling_std = (
                df["close"]
                .groupby(level=1)
                .transform(lambda s: s.rolling(window=self.lookback_window).std())
            )
        else:
            rolling_mean = df["close"].rolling(window=self.lookback_window).mean()
            rolling_std = df["close"].rolling(window=self.lookback_window).std()

        # 计算Z-score（使用safe_divide避免浮点噪声导致Z-score爆炸）
        zscore = safe_divide(df["close"] - rolling_mean, rolling_std, default=np.nan)

        # 带持仓状态记忆的均值回归信号生成
        # MultiIndex 下必须按资产分组维护独立的持仓状态
        if isinstance(df.index, pd.MultiIndex):
            asset_level = 1
            for asset_code in df.index.get_level_values(asset_level).unique():
                asset_mask = df.index.get_level_values(asset_level) == asset_code
                asset_zscore = zscore[asset_mask]
                position = 0
                for i in range(len(asset_zscore)):
                    z = asset_zscore.iloc[i]
                    if pd.isna(z):
                        continue
                    if position == 0:
                        if z < -self.entry_threshold:
                            position = 1
                        elif z > self.entry_threshold:
                            position = -1
                    elif position == 1:
                        if abs(z) < self.exit_threshold:
                            position = 0
                    elif position == -1:
                        if abs(z) < self.exit_threshold:
                            position = 0
                    signals.iloc[df.index.get_loc(asset_zscore.index[i])] = position
        else:
            position = 0
            for i in range(len(zscore)):
                z = zscore.iloc[i]
                if pd.isna(z):
                    continue
                if position == 0:
                    if z < -self.entry_threshold:
                        position = 1
                    elif z > self.entry_threshold:
                        position = -1
                elif position == 1:
                    if abs(z) < self.exit_threshold:
                        position = 0
                elif position == -1:
                    if abs(z) < self.exit_threshold:
                        position = 0
                signals.iloc[i] = position

        return signals

    def calculate_weights(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        计算权重

        Args:
            df: 数据
            signals: 信号

        Returns:
            权重序列
        """
        weights = pd.Series(0.0, index=df.index)

        # 多空信号等权分配（MultiIndex 下按日期分组）
        buy_mask = signals == 1
        sell_mask = signals == -1
        if isinstance(df.index, pd.MultiIndex):
            n_per_date_buy = buy_mask.groupby(level=0).transform("sum")
            n_per_date_sell = sell_mask.groupby(level=0).transform("sum")
            weights[buy_mask] = safe_divide(1.0, n_per_date_buy[buy_mask], default=0.0)
            weights[sell_mask] = -safe_divide(
                1.0, n_per_date_sell[sell_mask], default=0.0
            )
        else:
            # 单股票场景：满仓做多/做空
            weights[buy_mask] = 1.0
            weights[sell_mask] = -1.0

        return weights
