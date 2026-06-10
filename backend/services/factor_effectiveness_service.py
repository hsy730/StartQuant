"""
因子有效性分析服务
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy.stats import spearmanr

from backend.utils.returns import calculate_future_returns
from backend.utils.safe_math import safe_ir

logger = logging.getLogger(__name__)

from backend.services.alphalens_analysis_service import alphalens_analysis_service


class FactorEffectivenessService:
    """因子有效性分析服务类"""

    def __init__(self):
        pass

    def analyze_effectiveness(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        future_periods: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, Any]:
        """
        分析因子有效性

        Args:
            factor_data: 股票代码到因子数据的映射
            factor_name: 因子名称
            future_periods: 未来收益周期列表

        Returns:
            {
                "scatter_plot": {...},
                "ic_time_series": {...},
                "event_response": {...},
                "decay_analysis": {...}
            }
        """
        # 规则5：禁止就地修改传入的DataFrame，必须先copy
        factor_data = {k: v.copy() for k, v in factor_data.items()}

        results = {}

        results["scatter_plot"] = self._create_scatter_data(
            factor_data, factor_name
        )

        results["ic_time_series"] = self._calculate_ic_series(
            factor_data, factor_name
        )

        results["event_response"] = self._analyze_event_response(
            factor_data, factor_name
        )

        results["decay_analysis"] = self._analyze_decay(
            factor_data, factor_name, future_periods
        )

        return results

    def _create_scatter_data(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> Dict[str, Any]:
        """创建因子-收益散点图数据"""
        factor_values = []
        returns = []

        for stock_code, df in factor_data.items():
            if factor_name in df.columns and "close" in df.columns:
                df_copy = calculate_future_returns(df[[factor_name, "close"]], periods=[1])
                valid_data = df_copy[[factor_name, "future_return_1"]].dropna()
                valid_data = valid_data[~np.isinf(valid_data["future_return_1"])]
                factor_values.extend(valid_data[factor_name].tolist())
                returns.extend(valid_data["future_return_1"].tolist())

        if len(factor_values) < 2:
            return {"error": "数据不足以计算相关性"}

        # 多股票时使用横截面IC（规则7.1），单股票时池化是唯一选择
        if len(factor_data) >= 2:
            cross_sectional_ics = []
            # 合并所有股票数据，按日期分组计算横截面IC
            panel_frames = []
            for stock_code, df in factor_data.items():
                if factor_name in df.columns and "close" in df.columns:
                    df_copy = calculate_future_returns(df[[factor_name, "close"]], periods=[1])
                    df_copy = df_copy[[factor_name, "future_return_1"]].dropna()
                    if len(df_copy) > 0:
                        df_copy = df_copy.copy()
                        df_copy["_stock_code"] = stock_code
                        panel_frames.append(df_copy)

            if panel_frames:
                panel_df = pd.concat(panel_frames)
                for date, group in panel_df.groupby(panel_df.index):
                    if len(group) < 3:
                        continue
                    ic_val, p_val = spearmanr(group[factor_name], group["future_return_1"])
                    if not np.isnan(ic_val):
                        cross_sectional_ics.append(ic_val)

                if cross_sectional_ics:
                    correlation = float(np.mean(cross_sectional_ics))
                    p_value = None  # 横截面IC均值无单一p值
                else:
                    correlation, p_value = spearmanr(factor_values, returns)
            else:
                correlation, p_value = spearmanr(factor_values, returns)
        else:
            correlation, p_value = spearmanr(factor_values, returns)

        return {
            "x": [float(v) for v in factor_values],
            "y": [float(v) for v in returns],
            "correlation": float(correlation),
            "correlation_pvalue": float(p_value) if p_value is not None else None,
            "count": len(factor_values)
        }

    def _calculate_ic_series(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        window: int = 20
    ) -> Dict[str, Any]:
        """计算IC时序分析（使用Alphalens）"""
        num_stocks = len(factor_data)

        if num_stocks >= 2:
            al_result = self._calc_ic_via_alphalens(factor_data, factor_name)
            if al_result and "error" not in al_result:
                return al_result

        all_data = []
        for stock_code, df in factor_data.items():
            if factor_name in df.columns and "close" in df.columns:
                df_copy = calculate_future_returns(df[[factor_name, "close"]], periods=[1])
                df_copy = df_copy.rename(columns={"future_return_1": "future_return"})
                df_copy["stock_code"] = stock_code
                all_data.append(df_copy[[factor_name, "future_return", "stock_code"]])

        if not all_data:
            return {"error": "没有可用的数据"}

        merged_df = pd.concat(all_data, ignore_index=False)

        if num_stocks == 1:
            return self._calculate_timeseries_ic(merged_df, factor_name, window)
        else:
            return self._calculate_cross_sectional_ic(merged_df, factor_name)

    def _calc_ic_via_alphalens(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
    ) -> Optional[Dict[str, Any]]:
        """使用Alphalens计算IC（业界金标准）"""
        try:
            factor_values_dict = {}
            for stock_code, df in factor_data.items():
                if factor_name in df.columns and "close" in df.columns:
                    series = df[factor_name].dropna()
                    if len(series) > 0:
                        factor_values_dict[stock_code] = series

            if len(factor_values_dict) < 2:
                return None

            all_dates = set()
            for s in factor_values_dict.values():
                all_dates.update(s.index)
            all_dates = sorted(all_dates)

            prices = pd.DataFrame(index=all_dates)
            for stock_code, df in factor_data.items():
                if "close" in df.columns:
                    prices[stock_code] = df["close"]
            prices = prices.dropna(how="all")

            al_result = alphalens_analysis_service.full_analysis(
                factor_values_dict=factor_values_dict,
                pricing_df=prices,
            )

            if "error" in al_result:
                return None

            ic_analysis = al_result.get("ic_analysis", {})
            spearman_data = ic_analysis.get("spearman_ic", {})
            first_period = next(iter(spearman_data.values()), {})
            if not isinstance(first_period, dict) or "error" in first_period:
                return None

            ic_series_data = first_period.get("ic_series", {})
            dates = ic_series_data.get("dates", [])
            values = ic_series_data.get("values", [])
            valid_values = [v for v in values if v is not None]

            if not valid_values:
                return None

            ic_s = pd.Series(valid_values)

            return {
                "dates": dates,
                "ic_values": valid_values,
                "ic_mean": float(ic_s.mean()),
                "ic_std": float(ic_s.std()) if len(ic_s) > 1 else None,
                "ir": safe_ir(float(ic_s.mean()), float(ic_s.std()), default=None) if len(ic_s) > 1 else None,
                "ic_positive_ratio": float((ic_s > 0).mean()),
                "source": "Alphalens"
            }
        except Exception as e:
            logger.debug(f"Alphalens IC计算失败，使用fallback: {e}")
            return None

    def _calculate_timeseries_ic(
        self,
        df: pd.DataFrame,
        factor_name: str,
        window: int = 20
    ) -> Dict[str, Any]:
        """计算时间序列滚动IC（适用于单只股票）"""
        factor_vals = df[factor_name].dropna()
        return_vals = df["future_return"].dropna()

        common_index = factor_vals.index.intersection(return_vals.index)
        if len(common_index) < window + 1:
            return {"error": f"数据不足，需要至少{window+1}个数据点，当前只有{len(common_index)}个"}

        factor_aligned = factor_vals.loc[common_index]
        return_aligned = return_vals.loc[common_index]

        def _rolling_spearman(x):
            y_aligned = return_aligned.loc[x.index]
            valid = x.notna() & y_aligned.notna()
            if valid.sum() < 10:
                return np.nan
            return spearmanr(x[valid], y_aligned[valid])[0]

        rolling_ic = factor_aligned.rolling(window=window, min_periods=10).apply(_rolling_spearman, raw=False)
        valid_ic = rolling_ic.dropna()
        valid_ic = valid_ic[~np.isinf(valid_ic)]

        if len(valid_ic) == 0:
            return {"error": "无法计算有效的IC序列"}

        ic_values = valid_ic.tolist()
        dates = [str(d) for d in valid_ic.index]
        ic_series = pd.Series(ic_values)
        ic_std = float(ic_series.std())

        return {
            "dates": dates,
            "ic_values": [float(v) for v in ic_values],
            "ic_mean": float(ic_series.mean()),
            "ic_std": ic_std,
            "ir": safe_ir(float(ic_series.mean()), ic_std, default=None),
            "ic_positive_ratio": float((ic_series > 0).mean())
        }

    def _calculate_cross_sectional_ic(
        self,
        df: pd.DataFrame,
        factor_name: str
    ) -> Dict[str, Any]:
        """计算横截面IC（适用于多只股票）"""
        ic_values = []
        dates = []

        grouped = df.groupby(level=0)
        for date, group in grouped:
            if len(group) < 2:
                continue
            # 直接在group内dropna，避免重复索引下loc返回错误行数
            valid = group[[factor_name, "future_return"]].dropna()
            if len(valid) < 2:
                continue
            try:
                # 使用Spearman秩相关（业界标准），与Alphalens一致
                ic, _ = spearmanr(valid[factor_name], valid["future_return"])
                if not np.isnan(ic) and not np.isinf(ic):
                    ic_values.append(float(ic))
                    dates.append(str(date))
            except Exception as e:
                logger.debug(f"IC计算失败: {e}")
                continue

        if not ic_values:
            return {"error": "无法计算IC序列"}

        ic_series = pd.Series(ic_values)
        return {
            "dates": dates,
            "ic_values": [float(v) for v in ic_values],
            "ic_mean": float(ic_series.mean()),
            "ic_std": float(ic_series.std()),
            "ir": safe_ir(float(ic_series.mean()), float(ic_series.std()), default=None),
            "ic_positive_ratio": float((ic_series > 0).mean())
        }

    def _analyze_event_response(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        threshold_percentile: float = 0.8,
        holding_periods: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        事件响应分析 - 因子突破阈值后N日收益
        """
        all_data = []
        for stock_code, df in factor_data.items():
            if factor_name in df.columns and "close" in df.columns:
                all_data.append(df[[factor_name, "close"]].copy())

        if not all_data:
            return {"error": "没有可用的数据"}

        merged_df = pd.concat(all_data, ignore_index=False)
        factor_values = merged_df[factor_name].dropna()

        # 多股票时使用横截面分位数阈值（规则7.17），单股票时全局阈值是唯一选择
        if len(factor_data) >= 2:
            # 按日期计算横截面分位数阈值
            high_returns = {p: [] for p in holding_periods}
            low_returns = {p: [] for p in holding_periods}

            for stock_code, df in factor_data.items():
                df = calculate_future_returns(df[[factor_name, "close"]], periods=holding_periods)
                if factor_name not in df.columns or "close" not in df.columns:
                    continue

                # 按日期计算该股票的横截面分位数阈值
                for date in df.index:
                    # 获取该日期所有股票的因子值
                    date_factor_vals = merged_df.loc[
                        merged_df.index == date, factor_name
                    ].dropna() if date in merged_df.index else None

                    if date_factor_vals is None or len(date_factor_vals) < 3:
                        continue

                    high_threshold = float(date_factor_vals.quantile(threshold_percentile))
                    low_threshold = float(date_factor_vals.quantile(1 - threshold_percentile))

                    row = df.loc[[date]] if date in df.index else None
                    if row is None or len(row) == 0:
                        continue

                    factor_val = row[factor_name].iloc[0]
                    if pd.isna(factor_val):
                        continue

                    for period in holding_periods:
                        future_ret_col = f"future_return_{period}"
                        if future_ret_col not in df.columns:
                            continue
                        future_ret = row[future_ret_col].iloc[0]
                        if pd.notna(future_ret) and not np.isinf(future_ret):
                            if factor_val > high_threshold:
                                high_returns[period].append(future_ret)
                            if factor_val < low_threshold:
                                low_returns[period].append(future_ret)

            threshold_value = float(factor_values.quantile(threshold_percentile))
        else:
            # 单股票：全局阈值是唯一选择
            threshold_value = float(factor_values.quantile(threshold_percentile))
            low_threshold = float(factor_values.quantile(1 - threshold_percentile))

            high_returns = {p: [] for p in holding_periods}
            low_returns = {p: [] for p in holding_periods}

            for stock_code, df in factor_data.items():
                df = calculate_future_returns(df[[factor_name, "close"]], periods=holding_periods)
                if factor_name not in df.columns or "close" not in df.columns:
                    continue

                high_events = df[df[factor_name] > threshold_value].index
                for event_date in high_events:
                    for period in holding_periods:
                        if event_date in df.index:
                            future_ret = df.loc[event_date, f"future_return_{period}"]
                            if pd.notna(future_ret) and not np.isinf(future_ret):
                                high_returns[period].append(future_ret)

                low_events = df[df[factor_name] < low_threshold].index
                for event_date in low_events:
                    for period in holding_periods:
                        if event_date in df.index:
                            future_ret = df.loc[event_date, f"future_return_{period}"]
                            if pd.notna(future_ret) and not np.isinf(future_ret):
                                low_returns[period].append(future_ret)

        high_avg = {}
        low_avg = {}
        excess = {}
        for period in holding_periods:
            high_avg[period] = float(np.mean(high_returns[period])) if high_returns[period] else None
            low_avg[period] = float(np.mean(low_returns[period])) if low_returns[period] else None
            excess[period] = (high_avg[period] - low_avg[period]) if (high_avg[period] is not None and low_avg[period] is not None) else None

        return {
            "threshold_value": threshold_value,
            "threshold_percentile": threshold_percentile,
            "high_exposure_returns": {f"{p}日": high_avg[p] for p in holding_periods},
            "low_exposure_returns": {f"{p}日": low_avg[p] for p in holding_periods},
            "excess_returns": {f"{p}日": excess[p] for p in holding_periods},
            "holding_periods": holding_periods
        }

    def _analyze_decay(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        periods: List[int]
    ) -> Dict[str, Any]:
        """
        因子衰减分析 - 计算不同持有期的IC（优先使用Alphalens多周期IC）
        """
        if len(factor_data) >= 2:
            al_decay = self._calc_decay_via_alphalens(factor_data, factor_name, periods)
            if al_decay and "error" not in al_decay:
                return al_decay

        decay_data = []
        for period in periods:
            # 合并所有股票数据，按日期分组计算横截面IC
            daily_ics = []
            combined_frames = []
            for stock_code, df in factor_data.items():
                if factor_name not in df.columns or "close" not in df.columns:
                    continue
                df_with_returns = calculate_future_returns(df[[factor_name, "close"]], periods=[period])
                future_returns = df_with_returns[f"future_return_{period}"]
                temp_df = pd.DataFrame({
                    "factor": df[factor_name],
                    "return": future_returns,
                })
                # 保留日期索引
                if isinstance(df.index, pd.DatetimeIndex):
                    temp_df["date"] = df.index
                elif "date" in df.columns:
                    temp_df["date"] = df["date"]
                else:
                    temp_df["date"] = df.index
                combined_frames.append(temp_df)

            if combined_frames:
                combined = pd.concat(combined_frames, ignore_index=True)
                combined = combined.dropna(subset=["factor", "return"])

                for date, group in combined.groupby("date"):
                    if len(group) >= 5:  # 最少5只股票
                        ic = group["factor"].corr(group["return"], method="spearman")
                        if not np.isnan(ic) and not np.isinf(ic):
                            daily_ics.append(ic)

            mean_ic = float(np.mean(daily_ics)) if daily_ics else None

            if len(daily_ics) >= 1:
                try:
                    decay_data.append({
                        "period": f"{period}日",
                        "period_days": period,
                        "ic": float(mean_ic),
                        "abs_ic": abs(float(mean_ic))
                    })
                except Exception as e:
                    logger.debug(f"衰减数据追加失败: {e}")

        if not decay_data:
            return {"error": "无法计算衰减曲线"}

        return {"decay_curve": decay_data}

    def _calc_decay_via_alphalens(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        periods: List[int],
    ) -> Optional[Dict[str, Any]]:
        """使用Alphalens计算衰减曲线"""
        try:
            factor_values_dict = {}
            for stock_code, df in factor_data.items():
                if factor_name in df.columns and "close" in df.columns:
                    series = df[factor_name].dropna()
                    if len(series) > 0:
                        factor_values_dict[stock_code] = series

            if len(factor_values_dict) < 2:
                return None

            all_dates = set()
            for s in factor_values_dict.values():
                all_dates.update(s.index)
            all_dates = sorted(all_dates)

            prices = pd.DataFrame(index=all_dates)
            for stock_code, df in factor_data.items():
                if "close" in df.columns:
                    prices[stock_code] = df["close"]
            prices = prices.dropna(how="all")

            al_result = alphalens_analysis_service.full_analysis(
                factor_values_dict=factor_values_dict,
                pricing_df=prices,
                periods=tuple(periods),
            )

            if "error" in al_result:
                return None

            ic_analysis = al_result.get("ic_analysis", {})
            spearman_data = ic_analysis.get("spearman_ic", {})

            decay_curve = []
            for period in periods:
                period_label = f"{period}D"
                period_stats = spearman_data.get(period_label, {})
                if isinstance(period_stats, dict) and "error" not in period_stats:
                    mean_ic = period_stats.get("mean_ic", 0)
                    if mean_ic is not None:
                        decay_curve.append({
                            "period": f"{period}日",
                            "period_days": period,
                            "ic": float(mean_ic),
                            "abs_ic": abs(float(mean_ic))
                        })

            if not decay_curve:
                return None

            return {"decay_curve": decay_curve, "source": "Alphalens"}
        except Exception as e:
            logger.debug(f"Alphalens衰减分析失败，使用fallback: {e}")
            return None


# 全局服务实例
factor_effectiveness_service = FactorEffectivenessService()
