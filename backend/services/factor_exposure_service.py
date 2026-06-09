"""
因子暴露度分析服务
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any


from backend.utils.safe_math import safe_divide
from backend.utils.factor_data_utils import find_longest_stock

logger = logging.getLogger(__name__)


class FactorExposureService:
    """因子暴露度分析服务类"""

    def __init__(self):
        pass

    def calculate_exposure_metrics(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        window: int = 20
    ) -> Dict[str, Any]:
        """
        计算因子暴露度指标

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_name: 因子名称
            window: 滚动窗口大小

        Returns:
            {
                "current_value": float,          # 当前因子值（第一只股票的最新值）
                "percentile": float,             # 时间序列分位数 (0-100)
                                                   # 含义：当前值在整个所选时间区间内的百分位
                "percentile_meaning": str,       # 分位数含义说明
                "stock_count": int,              # 分析的股票数量
                "stocks_analyzed": list,         # 分析的股票代码列表
                "percentile_time_series": {      # 分位数时间序列
                    "dates": list,
                    "percentiles": list,
                    "values": list
                },
                "rolling_mean": dict,            # 滚动均值
                "rolling_std": dict,             # 滚动标准差
                "cv": float,                     # 变异系数
                "distribution": {                # 分布统计（所有历史数据）
                    "min": float,
                    "max": float,
                    "mean": float,
                    "std": float,
                    "percentiles": {...}
                },
                "histogram": {                   # 直方图数据（所有历史数据）
                    "bins": list,
                    "counts": list
                }
            }
        """
        # 找一个数据完整且时间最长的股票作为主要分析对象
        longest_stock, longest_df = find_longest_stock(factor_data, factor_name)

        if not longest_stock:
            raise ValueError(f"因子 {factor_name} 没有有效数据")

        # 获取该股票的时间序列数据
        time_series = factor_data[longest_stock][factor_name].dropna()

        # 当前值（最新值）
        current_value = float(time_series.iloc[-1])

        # 时间序列分位数：每个时间点的值在整个时间区间内的分位数
        # 使用 rank(pct=True) 替代逐个调用 percentileofscore，提升效率
        percentile_series = time_series.rank(pct=True) * 100.0

        # 当前值的时间序列分位数
        percentile = float(percentile_series.iloc[-1])

        # 分位数时间序列数据（用于前端绘制分位数曲线）
        percentile_time_series = {
            "dates": [str(date) for date in time_series.index],
            "percentiles": [float(p) for p in percentile_series.values],
            "values": [float(v) for v in time_series.values],
            "stock_code": longest_stock
        }

        # 滚动统计
        rolling_mean = time_series.rolling(window=window, min_periods=1).mean()
        rolling_std = time_series.rolling(window=window, min_periods=1).std()

        rolling_mean_dict = {
            str(date): float(val) if pd.notna(val) else None
            for date, val in rolling_mean.items()
        }
        rolling_std_dict = {
            str(date): float(val) if pd.notna(val) else None
            for date, val in rolling_std.items()
        }

        # 使用最新的滚动统计值计算变异系数
        latest_mean = rolling_mean.iloc[-1]
        latest_std = rolling_std.iloc[-1]

        # 变异系数
        cv = safe_divide(float(latest_std), float(latest_mean), default=None)

        # 合并所有股票的所有因子值（用于整体分布统计）
        all_factor_values = []
        for stock_code, df in factor_data.items():
            if factor_name in df.columns:
                all_factor_values.extend(df[factor_name].dropna().tolist())

        factor_series = pd.Series(all_factor_values)

        # 分布统计
        distribution = {
            "min": float(factor_series.min()),
            "max": float(factor_series.max()),
            "mean": float(factor_series.mean()),
            "std": float(factor_series.std()),
            "count": len(factor_series),
            "percentiles": {
                f"p{p}": float(factor_series.quantile(p/100))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
        }

        # 直方图数据（用于前端绘制分布图）
        hist, bins = np.histogram(factor_series.dropna().values, bins=50)
        histogram = {
            "bins": [float(b) for b in bins],
            "counts": [int(c) for c in hist]
        }

        return {
            "current_value": current_value,
            "percentile": percentile,
            "percentile_meaning": "时间序列分位数：当前因子值在整个所选时间区间内的百分位（0-100）",
            "stock_count": len(factor_data),
            "stocks_analyzed": list(factor_data.keys()),
            "percentile_time_series": percentile_time_series,
            "rolling_mean": rolling_mean_dict,
            "rolling_std": rolling_std_dict,
            "cv": cv,
            "latest_mean": float(latest_mean),
            "latest_std": float(latest_std),
            "distribution": distribution,
            "histogram": histogram
        }

    def calculate_exposure_by_stock(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Dict]:
        """
        计算每只股票的因子暴露度

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_name: 因子名称

        Returns:
            {
                "stock_code": {
                    "mean": float,
                    "std": float,
                    "current": float,
                    "min": float,
                    "max": float
                }
            }
        """
        exposure_by_stock = {}

        for stock_code, df in factor_data.items():
            if factor_name in df.columns:
                factor_values = df[factor_name].dropna()
                if len(factor_values) > 0:
                    exposure_by_stock[stock_code] = {
                        "mean": float(factor_values.mean()),
                        "std": float(factor_values.std()),
                        "current": float(factor_values.iloc[-1]),
                        "min": float(factor_values.min()),
                        "max": float(factor_values.max()),
                        "count": len(factor_values)
                    }

        return exposure_by_stock

    def calculate_percentile_distribution(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Any]:
        """
        计算因子值的分位数分布

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_name: 因子名称

        Returns:
            {
                "quintiles": {      # 五分位数
                    "q20": float,
                    "q40": float,
                    "q60": float,
                    "q80": float
                },
                "deciles": {...},   # 十分位数
                "distribution_by_quintile": {  # 各分位数区间内的股票分布
                    "0-20%": [stock_codes],
                    "20-40%": [stock_codes],
                    ...
                }
            }
        """
        # 获取每只股票的最新因子值
        stock_latest_values = {}
        for stock_code, df in factor_data.items():
            if factor_name in df.columns:
                latest_value = df[factor_name].dropna()
                if len(latest_value) > 0:
                    stock_latest_values[stock_code] = float(latest_value.iloc[-1])

        if not stock_latest_values:
            return {"error": "没有可用的因子数据"}

        values = pd.Series(list(stock_latest_values.values()))

        # 计算分位数
        quintiles = {
            "q20": float(values.quantile(0.2)),
            "q40": float(values.quantile(0.4)),
            "q60": float(values.quantile(0.6)),
            "q80": float(values.quantile(0.8))
        }

        deciles = {
            f"d{i*10}": float(values.quantile(i/10))
            for i in range(1, 10)
        }

        # 按五分位数分组
        distribution_by_quintile = {
            "0-20%": [],
            "20-40%": [],
            "40-60%": [],
            "60-80%": [],
            "80-100%": []
        }

        for stock_code, value in stock_latest_values.items():
            if value <= quintiles["q20"]:
                distribution_by_quintile["0-20%"].append(stock_code)
            elif value <= quintiles["q40"]:
                distribution_by_quintile["20-40%"].append(stock_code)
            elif value <= quintiles["q60"]:
                distribution_by_quintile["40-60%"].append(stock_code)
            elif value <= quintiles["q80"]:
                distribution_by_quintile["60-80%"].append(stock_code)
            else:
                distribution_by_quintile["80-100%"].append(stock_code)

        return {
            "quintiles": quintiles,
            "deciles": deciles,
            "distribution_by_quintile": distribution_by_quintile
        }

    def calculate_rolling_exposure(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        window: int = 20
    ) -> Dict[str, Any]:
        """
        计算滚动窗口暴露度

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_name: 因子名称
            window: 滚动窗口大小

        Returns:
            {
                "dates": list,
                "rolling_mean": list,
                "rolling_std": list,
                "rolling_min": list,
                "rolling_max": list,
                "upper_band": list,  # 均值 + 2倍标准差
                "lower_band": list   # 均值 - 2倍标准差
            }
        """
        # 找一个数据完整且时间最长的股票
        longest_stock, longest_df = find_longest_stock(factor_data, factor_name)

        if not longest_stock:
            return {"error": "没有可用的因子数据"}

        time_series = factor_data[longest_stock][factor_name].dropna()

        # 计算滚动统计
        rolling_mean = time_series.rolling(window=window, min_periods=1).mean()
        rolling_std = time_series.rolling(window=window, min_periods=1).std()
        rolling_min = time_series.rolling(window=window, min_periods=1).min()
        rolling_max = time_series.rolling(window=window, min_periods=1).max()

        # 计算置信区间（均值 ± 2倍标准差）
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std

        # 转换为列表格式
        dates = [str(date) for date in time_series.index]

        return {
            "dates": dates,
            "values": [float(v) for v in time_series.values],
            "rolling_mean": [float(v) if pd.notna(v) else None for v in rolling_mean.values],
            "rolling_std": [float(v) if pd.notna(v) else None for v in rolling_std.values],
            "rolling_min": [float(v) if pd.notna(v) else None for v in rolling_min.values],
            "rolling_max": [float(v) if pd.notna(v) else None for v in rolling_max.values],
            "upper_band": [float(v) if pd.notna(v) else None for v in upper_band.values],
            "lower_band": [float(v) if pd.notna(v) else None for v in lower_band.values]
        }


# 全局服务实例
factor_exposure_service = FactorExposureService()
