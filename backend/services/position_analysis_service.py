"""
持仓分析服务 - 分析持仓统计信息
"""
from typing import Dict
import pandas as pd
import numpy as np


class PositionAnalysisService:
    """持仓分析服务"""

    def __init__(self):
        pass

    def analyze_positions(
        self,
        positions: pd.Series,
        initial_capital: float = 1000000
    ) -> Dict:
        """
        分析持仓统计信息

        Args:
            positions: 持仓序列（权重）
            initial_capital: 初始资金

        Returns:
            持仓统计信息
        """
        positions_clean = positions.dropna()

        if len(positions_clean) == 0:
            return self._empty_stats()

        # 1. 基础统计
        avg_position = positions_clean.abs().mean()
        max_position = positions_clean.abs().max()
        min_position = positions_clean.abs().min()

        # 2. 持仓分布
        position_zero_ratio = (positions_clean == 0).sum() / len(positions_clean)
        position_full_ratio = (positions_clean.abs() >= 0.9).sum() / len(positions_clean)

        # 3. 持仓变化
        position_changes = positions_clean.diff().abs()
        avg_position_change = position_changes.mean()
        max_position_change = position_changes.max()

        # 4. 持仓时长（假设连续持仓）
        # 找出持仓时段
        is_invested = positions_clean.abs() > 0
        invested_periods = 0
        total_invested_days = 0

        if len(is_invested) > 0:
            current_period = 0
            for invested in is_invested:
                if invested:
                    current_period += 1
                    total_invested_days += 1
                else:
                    if current_period > 0:
                        invested_periods += 1
                    current_period = 0

            if current_period > 0:
                invested_periods += 1

        avg_holding_period = (
            total_invested_days / invested_periods if invested_periods > 0 else 0
        )

        # 5. 换手率（权重变化绝对值之和除以2，买入和卖出是同一笔交易的两面）
        turnover = position_changes.sum() / 2

        # 6. 持仓价值
        position_values = positions_clean * initial_capital
        avg_position_value = position_values.abs().mean()
        max_position_value = position_values.abs().max()

        return {
            "basic_stats": {
                "avg_position": float(avg_position),
                "max_position": float(max_position),
                "min_position": float(min_position),
                "position_zero_ratio": float(position_zero_ratio),
                "position_full_ratio": float(position_full_ratio),
            },
            "position_changes": {
                "avg_position_change": float(avg_position_change),
                "max_position_change": float(max_position_change),
            },
            "holding_stats": {
                "invested_periods": int(invested_periods),
                "total_invested_days": int(total_invested_days),
                "avg_holding_period": float(avg_holding_period),
            },
            "turnover": float(turnover),
            "position_values": {
                "avg_position_value": float(avg_position_value),
                "max_position_value": float(max_position_value),
            },
        }

    def analyze_position_history(
        self,
        positions: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        分析持仓历史（滚动窗口）

        Args:
            positions: 持仓序列
            window: 窗口大小

        Returns:
            持仓历史DataFrame
        """
        df = pd.DataFrame(index=positions.index)
        df["position"] = positions

        # 滚动统计
        df["rolling_avg_position"] = (
            positions.abs().rolling(window=window).mean()
        )
        df["rolling_max_position"] = (
            positions.abs().rolling(window=window).max()
        )
        df["rolling_min_position"] = (
            positions.abs().rolling(window=window).min()
        )

        # 持仓变化
        df["position_change"] = positions.diff().abs()

        return df

    def calculate_position_concentration(
        self,
        positions: pd.Series
    ) -> Dict:
        """
        计算持仓集中度

        Args:
            positions: 持仓序列

        Returns:
            集中度指标
        """
        positions_clean = positions.dropna()

        if len(positions_clean) == 0:
            return {
                "concentration_ratio": 0.0,
                "herfindahl_index": 0.0,
                "gini_coefficient": 0.0,
            }

        # 简化版：单股票，集中度为1
        # 实际应用中应该是多股票组合

        # 1. 集中度（前N大持仓占比）
        # 单股票情况，集中度为100%
        concentration_ratio = 1.0

        # 2. Herfindahl指数（平方和）
        # 单股票情况，H = 1^2 = 1
        herfindahl_index = 1.0

        # 3. 基尼系数
        # 单股票情况，基尼系数为0（完全平等）
        gini_coefficient = 0.0

        return {
            "concentration_ratio": float(concentration_ratio),
            "herfindahl_index": float(herfindahl_index),
            "gini_coefficient": float(gini_coefficient),
        }

    def _empty_stats(self) -> Dict:
        """返回空的统计信息"""
        return {
            "basic_stats": {
                "avg_position": 0.0,
                "max_position": 0.0,
                "min_position": 0.0,
                "position_zero_ratio": 0.0,
                "position_full_ratio": 0.0,
            },
            "position_changes": {
                "avg_position_change": 0.0,
                "max_position_change": 0.0,
            },
            "holding_stats": {
                "invested_periods": 0,
                "total_invested_days": 0,
                "avg_holding_period": 0.0,
            },
            "turnover": 0.0,
            "position_values": {
                "avg_position_value": 0.0,
                "max_position_value": 0.0,
            },
        }


# 全局持仓分析服务实例
position_analysis_service = PositionAnalysisService()
