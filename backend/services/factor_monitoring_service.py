"""
时间序列动态监测服务
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from backend.utils.factor_data_utils import find_longest_stock as _find_longest_stock_util

logger = logging.getLogger(__name__)


class FactorMonitoringService:
    """时间序列动态监测服务类"""

    def __init__(self):
        pass

    def _find_longest_stock(self, factor_data: Dict[str, pd.DataFrame], factor_name: str) -> tuple:
        """找到数据最长的股票（代码和数据），委托公共工具"""
        code, df = _find_longest_stock_util(factor_data, factor_name)
        return code, len(df[factor_name].dropna()) if factor_name in df.columns else len(df)

    def monitor_dynamics(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Any]:
        """
        时间序列动态监测

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_name: 因子名称

        Returns:
            {
                "rolling_chart": {...},        # 滚动窗口图
                "transition_matrix": {...},    # 暴露度转移矩阵
                "structural_break": {...},     # 结构断点检测
                "seasonality": {...}           # 周期性分析
            }
        """
        results = {}

        # 1. 滚动窗口图（折线图 + 滚动均值/标准差带）
        results["rolling_chart"] = self._calculate_rolling_bands(
            factor_data, factor_name
        )

        # 2. 暴露度转移矩阵（分位数迁移概率）
        results["transition_matrix"] = self._calculate_transition_matrix(
            factor_data, factor_name
        )

        # 3. 结构断点检测
        results["structural_break"] = self._detect_structural_breaks(
            factor_data, factor_name
        )

        # 4. 周期性分析
        results["seasonality"] = self._analyze_seasonality(
            factor_data, factor_name
        )

        return results

    def _calculate_rolling_bands(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        window: int = 20,
        std_multiplier: float = 2.0
    ) -> Dict[str, Any]:
        """
        计算滚动窗口带状图

        Returns:
            {
                "dates": list,
                "values": list,
                "rolling_mean": list,
                "upper_band": list,  # 均值 + N倍标准差
                "lower_band": list   # 均值 - N倍标准差
            }
        """
        # 找一个数据完整且时间最长的股票
        longest_stock, _ = self._find_longest_stock(factor_data, factor_name)

        if not longest_stock:
            return {"error": "没有可用的因子数据"}

        time_series = factor_data[longest_stock][factor_name].dropna()

        # 计算滚动统计
        rolling_mean = time_series.rolling(window=window, min_periods=1).mean()
        rolling_std = time_series.rolling(window=window, min_periods=1).std()

        # 计算置信区间
        upper_band = rolling_mean + std_multiplier * rolling_std
        lower_band = rolling_mean - std_multiplier * rolling_std

        return {
            "dates": [str(date) for date in time_series.index],
            "values": [float(v) for v in time_series.values],
            "rolling_mean": [float(v) if pd.notna(v) else None for v in rolling_mean.values],
            "rolling_std": [float(v) if pd.notna(v) else None for v in rolling_std.values],
            "upper_band": [float(v) if pd.notna(v) else None for v in upper_band.values],
            "lower_band": [float(v) if pd.notna(v) else None for v in lower_band.values],
            "window": window,
            "std_multiplier": std_multiplier
        }

    def _calculate_transition_matrix(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        n_bins: int = 5
    ) -> Dict[str, Any]:
        """
        计算暴露度转移矩阵（马尔可夫转移概率）

        Returns:
            {
                "matrix": [[...]],  # n_bins x n_bins 转移概率矩阵
                "bin_labels": list,
                "interpretation": str
            }
        """
        # 找一个数据完整且时间最长的股票
        longest_stock, _ = self._find_longest_stock(factor_data, factor_name)

        if not longest_stock:
            return {"error": "没有可用的因子数据"}

        time_series = factor_data[longest_stock][factor_name].dropna()

        # 计算分位数 bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = [time_series.quantile(q) for q in quantiles]
        bin_labels = [f"Q{i+1}" for i in range(n_bins)]

        # 将时间序列离散化为 bin 索引
        # 使用 duplicates='drop' 处理重复的边界值（当因子有大量重复值时）
        binned = pd.cut(time_series, bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')

        # 获取实际的 bin 数量（可能少于 n_bins，因为有重复值被丢弃）
        actual_bins = binned.nunique() if binned is not None else 1
        actual_bins = int(actual_bins) if not pd.isna(actual_bins) else n_bins

        # 如果实际 bin 数量少于预期，使用实际数量
        effective_bins = min(n_bins, actual_bins)

        # 计算转移矩阵（使用实际的 bin 数量）
        transition_matrix = np.zeros((effective_bins, effective_bins))
        transition_counts = np.zeros((effective_bins, effective_bins))

        # 过滤掉 NaN 值
        binned_clean = binned.dropna()

        for i in range(len(binned_clean) - 1):
            from_state = int(binned_clean.iloc[i])
            to_state = int(binned_clean.iloc[i + 1])
            if 0 <= from_state < effective_bins and 0 <= to_state < effective_bins:
                transition_counts[from_state, to_state] += 1

        # 归一化为概率
        for i in range(effective_bins):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum

        # 调整标签数量以匹配实际 bin 数量
        actual_labels = bin_labels[:effective_bins] if effective_bins <= len(bin_labels) else bin_labels

        return {
            "matrix": transition_matrix.tolist(),
            "bin_labels": actual_labels,
            "bin_edges": [float(e) for e in bin_edges],
            "actual_bins": effective_bins,
            "interpretation": f"基于{effective_bins}个分位数的暴露度转移概率矩阵（原计划{n_bins}个，因有重复值调整为{effective_bins}个）"
        }

    def _detect_structural_breaks(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Any]:
        """
        结构断点检测 - 使用简单的滚动均值变化检测

        Returns:
            {
                "breakpoints": [dates],
                "num_breaks": int,
                "method": str,
                "interpretation": str
            }
        """
        # 找一个数据完整且时间最长的股票
        longest_stock, _ = self._find_longest_stock(factor_data, factor_name)

        if not longest_stock:
            return {"error": "没有可用的因子数据"}

        time_series = factor_data[longest_stock][factor_name].dropna()

        # 简单的断点检测：使用滚动均值变化
        window = 60
        rolling_mean = time_series.rolling(window=window, min_periods=1).mean()

        # 计算均值变化率
        mean_change = rolling_mean.diff().abs()

        # 找出变化率超过阈值的点（使用3倍标准差作为阈值）
        threshold = mean_change.mean() + 3 * mean_change.std()

        # 找峰值
        peaks, _ = find_peaks(mean_change.values, height=threshold, distance=window//2)

        breakpoint_dates = [str(time_series.index[i]) for i in peaks if i < len(time_series)]

        return {
            "breakpoints": breakpoint_dates,
            "num_breaks": len(breakpoint_dates),
            "method": "Rolling Mean Change Detection",
            "threshold": float(threshold),
            "window": window,
            "interpretation": f"检测到 {len(breakpoint_dates)} 个结构性断点（基于滚动均值变化）"
        }

    def _analyze_seasonality(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Any]:
        """
        周期性分析 - 傅里叶变换检测周期

        Returns:
            {
                "frequencies": [...],
                "powers": [...],
                "dominant_periods": [...],
                "interpretation": str
            }
        """
        # 找一个数据完整且时间最长的股票
        longest_stock, _ = self._find_longest_stock(factor_data, factor_name)

        if not longest_stock:
            return {"error": "没有可用的因子数据"}

        time_series = factor_data[longest_stock][factor_name].dropna()

        # 确保数据足够长
        if len(time_series) < 50:
            return {"error": "数据不足以进行周期性分析（需要至少50个数据点）"}

        # 去趋势
        detrended = time_series - time_series.mean()

        # 傅里叶变换
        fft_values = fft(detrended.values)
        n = len(time_series)

        # 计算频率
        freqs = fftfreq(n, d=1)  # 假设日频数据

        # 计算功率谱
        power = np.abs(fft_values)

        # 只保留正频率部分
        positive_freq_idx = freqs > 0
        positive_freqs = freqs[positive_freq_idx]
        positive_power = power[positive_freq_idx]

        # 找出主要周期（功率谱峰值的对应频率）
        if len(positive_power) == 0:
            return {
                "frequencies": [],
                "powers": [],
                "dominant_periods": [],
                "data_length": len(time_series),
                "interpretation": "无正频率成分，无法进行周期性分析"
            }
        peaks, _ = find_peaks(positive_power, height=np.mean(positive_power))

        # 转换频率为周期（天数）
        dominant_periods = []
        for peak in peaks[:5]:  # 只取前5个主要周期
            freq = positive_freqs[peak]
            if freq > 0:
                period_days = 1.0 / freq
                dominant_periods.append({
                    "period_days": float(period_days),
                    "frequency": float(freq),
                    "power": float(positive_power[peak])
                })

        return {
            "frequencies": [float(f) for f in positive_freqs[:n//2]],
            "powers": [float(p) for p in positive_power[:n//2]],
            "dominant_periods": dominant_periods,
            "data_length": len(time_series),
            "interpretation": f"检测到 {len(dominant_periods)} 个显著周期成分"
        }


# 全局服务实例
factor_monitoring_service = FactorMonitoringService()
