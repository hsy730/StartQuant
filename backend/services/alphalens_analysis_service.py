"""
Alphalens因子分析服务 - 基于alphalens-reloaded库提供专业因子分析
"""
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

import alphalens

from backend.utils.safe_math import safe_divide, safe_ir
from backend.utils.serialization import safe_numeric_value


def _to_python_float(value, default=None):
    """转换numpy类型为Python原生类型（委托safe_numeric_value）"""
    return safe_numeric_value(value, default=default)


def _series_to_list(s: pd.Series) -> List[Optional[float]]:
    return [_to_python_float(v) for v in s.values]


def _index_to_str_list(idx: pd.Index) -> List[str]:
    return [str(i) for i in idx]


class AlphalensAnalysisService:
    """Alphalens因子分析服务类"""

    def __init__(self):
        pass

    def prepare_factor_data(
        self,
        factor_values_dict: Dict[str, pd.Series],
        pricing_df: pd.DataFrame,
        groupby_dict: Optional[Dict[str, str]] = None,
        periods: Tuple[int, ...] = (1, 5, 10),
        quantiles: int = 5,
        max_loss: float = 0.50,
    ) -> Optional[pd.DataFrame]:
        """
        准备alphalens格式的因子数据

        Args:
            factor_values_dict: {股票代码: pd.Series(因子值, index=日期)}
            pricing_df: DataFrame, index=日期, columns=股票代码, 值为收盘价
            groupby_dict: {股票代码: 行业名称}，可选
            periods: 远期收益计算周期
            quantiles: 分位数数量
            max_loss: 最大允许数据丢失比例 (默认0.50，即50%)

        Returns:
            alphalens格式的factor_data (MultiIndex: date, asset)，失败返回None
        """
        if len(factor_values_dict) < 1:
            logger.error("factor_values_dict为空，无法准备因子数据")
            return None

        if len(factor_values_dict) == 1:
            logger.warning("仅包含单只股票，横截面IC分析需要多只股票数据")

        try:
            records = []
            for stock_code, series in factor_values_dict.items():
                if not isinstance(series, pd.Series):
                    logger.warning(f"股票{stock_code}的因子值不是pd.Series，跳过")
                    continue
                for date, value in series.items():
                    if pd.notna(value) and not np.isinf(value):
                        records.append((pd.Timestamp(date), stock_code, value))

            if not records:
                logger.error("所有因子值均为NaN或Inf，无法构建因子数据")
                return None

            factor_series = pd.Series(
                [r[2] for r in records],
                index=pd.MultiIndex.from_tuples(
                    [(r[0], r[1]) for r in records],
                    names=["date", "asset"],
                ),
            )

            pricing_aligned = pricing_df.copy()
            pricing_aligned.index = pd.to_datetime(pricing_aligned.index)

            groupby_labels = None
            if groupby_dict is not None:
                groupby_labels = pd.Series(groupby_dict, name="group")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
                    factor=factor_series,
                    prices=pricing_aligned,
                    groupby=groupby_labels,
                    periods=list(periods),
                    quantiles=quantiles,
                    binning_by_group=groupby_labels is not None,
                    max_loss=max_loss,
                )

            logger.info(
                f"因子数据准备完成: {len(factor_data)}条记录, "
                f"股票数={factor_data.index.get_level_values('asset').nunique()}, "
                f"日期范围={factor_data.index.get_level_values('date').min()} ~ {factor_data.index.get_level_values('date').max()}"
            )
            return factor_data

        except Exception as e:
            logger.error(f"准备因子数据失败: {e}", exc_info=True)
            return None

    def analyze_ic(
        self,
        factor_data: pd.DataFrame,
        by_group: bool = False,
    ) -> Dict[str, Any]:
        """
        分析因子IC（信息系数）

        Args:
            factor_data: alphalens格式的因子数据
            by_group: 是否按行业分组

        Returns:
            IC分析结果字典，包含Pearson IC和Spearman Rank IC
        """
        if factor_data is None or factor_data.empty:
            logger.error("factor_data为空，无法执行IC分析")
            return {"error": "因子数据为空"}

        num_assets = factor_data.index.get_level_values("asset").nunique()
        if num_assets < 2:
            logger.warning("横截面IC分析需要至少2只股票，当前仅1只")

        results: Dict[str, Any] = {}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic_spearman = alphalens.performance.factor_information_coefficient(
                    factor_data, by_group=by_group
                )
                _mean_ic_spearman = alphalens.performance.mean_information_coefficient(
                    factor_data, by_group=by_group
                )

            results["spearman_ic"] = self._compute_ic_stats(ic_spearman, "Spearman Rank IC")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic_pearson = self._compute_pearson_ic(factor_data)

            results["pearson_ic"] = self._compute_ic_stats(ic_pearson, "Pearson IC")

            if by_group and "group" in factor_data.columns:
                results["by_group_spearman"] = self._compute_ic_stats_by_group(ic_spearman)
                results["by_group_pearson"] = self._compute_ic_stats_by_group(ic_pearson)

            logger.info("IC分析完成")
            return results

        except Exception as e:
            logger.error(f"IC分析失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _compute_pearson_ic(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """手动计算Pearson IC（alphalens默认只提供Spearman Rank IC）"""
        ic_results = {}
        return_cols = [c for c in factor_data.columns if c not in ['factor', 'factor_quantile', 'group']]
        dates = factor_data.index.get_level_values(0)
        for period_col in return_cols:
            merged = pd.DataFrame({
                'factor': factor_data['factor'].values,
                'return': factor_data[period_col].values,
            }, index=dates)
            merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
            if len(merged) < 2:
                continue
            daily_ic = merged.groupby(merged.index).apply(
                lambda g: g['factor'].corr(g['return']) if len(g) >= 2 else np.nan
            ).dropna()
            if len(daily_ic) > 0:
                ic_results[period_col] = daily_ic
        if ic_results:
            return pd.DataFrame(ic_results)
        return pd.DataFrame()

    def _compute_ic_stats(
        self,
        ic_df: pd.DataFrame,
        ic_type: str,
    ) -> Dict[str, Any]:
        """
        从IC DataFrame计算统计指标

        Args:
            ic_df: alphalens返回的IC DataFrame
            ic_type: IC类型名称

        Returns:
            各period的IC统计结果
        """
        stats_result: Dict[str, Any] = {}

        for period_col in ic_df.columns:
            period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
            ic_series = ic_df[period_col].dropna()

            if len(ic_series) == 0:
                stats_result[period_label] = {"error": f"{period_label}无有效IC数据"}
                continue

            n = len(ic_series)
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std()) if n > 1 else 0.0
            ir = safe_ir(float(ic_mean), float(ic_std), default=None)

            if n > 1 and ic_std > 1e-10:
                t_stat = safe_divide(float(ic_mean), float(ic_std / np.sqrt(n)), default=0.0)
                p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1)))
            elif n > 1 and ic_std <= 1e-10 and abs(ic_mean) > 1e-10:
                # 规则7.10：ic_std≈0但ic_mean显著非零时，因子极其稳定，t_stat→∞
                t_stat = float('inf')
                p_value = 0.0
            else:
                t_stat = 0.0
                p_value = 1.0

            ic_positive_ratio = float((ic_series > 0).mean())

            stats_result[period_label] = {
                "ic_type": ic_type,
                "period": period_label,
                "mean_ic": _to_python_float(ic_mean),
                "std_ic": _to_python_float(ic_std),
                "ir": _to_python_float(ir),
                "t_statistic": _to_python_float(t_stat),
                "p_value": _to_python_float(p_value),
                "ic_positive_ratio": _to_python_float(ic_positive_ratio),
                "n_observations": n,
                "ic_series": {
                    "dates": _index_to_str_list(ic_series.index),
                    "values": _series_to_list(ic_series),
                },
            }

        return stats_result

    def _compute_ic_stats_by_group(
        self,
        ic_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        按行业分组计算IC统计

        Args:
            ic_df: alphalens返回的按组IC DataFrame

        Returns:
            各行业各period的IC统计
        """
        group_stats: Dict[str, Any] = {}

        if isinstance(ic_df.index, pd.MultiIndex):
            for group_name in ic_df.index.get_level_values(0).unique():
                group_ic = ic_df.loc[group_name]
                group_stats[str(group_name)] = self._compute_ic_stats(
                    group_ic if isinstance(group_ic, pd.DataFrame) else group_ic.to_frame().T,
                    "",
                )
        else:
            group_stats = self._compute_ic_stats(ic_df, "")

        return group_stats

    def analyze_returns(
        self,
        factor_data: pd.DataFrame,
        by_group: bool = False,
    ) -> Dict[str, Any]:
        """
        分析因子收益

        Args:
            factor_data: alphalens格式的因子数据
            by_group: 是否按行业分组

        Returns:
            收益分析结果字典
        """
        if factor_data is None or factor_data.empty:
            logger.error("factor_data为空，无法执行收益分析")
            return {"error": "因子数据为空"}

        results: Dict[str, Any] = {}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mean_ret_by_q = alphalens.performance.mean_return_by_quantile(
                    factor_data, by_group=by_group, demeaned=True
                )

                factor_ret = alphalens.performance.factor_returns(factor_data, demeaned=True)

                alpha_beta = alphalens.performance.factor_alpha_beta(factor_data)

            if isinstance(mean_ret_by_q, tuple):
                mean_ret_q_df = mean_ret_by_q[0]
            else:
                mean_ret_q_df = mean_ret_by_q

            results["quantile_returns"] = self._extract_quantile_returns(mean_ret_q_df)

            results["factor_returns"] = self._extract_factor_returns(factor_ret)

            results["alpha_beta"] = self._extract_alpha_beta(alpha_beta)

            results["spread"] = self._compute_spread(mean_ret_q_df)

            if by_group and "group" in factor_data.columns:
                if isinstance(mean_ret_by_q, tuple) and len(mean_ret_by_q) > 1:
                    results["quantile_returns_by_group"] = self._extract_quantile_returns(
                        mean_ret_by_q[1]
                    )

            logger.info("收益分析完成")
            return results

        except Exception as e:
            logger.error(f"收益分析失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _extract_quantile_returns(
        self,
        mean_ret_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        提取分位数收益数据

        Args:
            mean_ret_df: alphalens返回的分位数收益DataFrame

        Returns:
            各分位数各period的收益数据
        """
        quantile_returns: Dict[str, Any] = {}

        for period_col in mean_ret_df.columns:
            period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
            period_data: Dict[str, Any] = {}

            for quantile in mean_ret_df.index:
                q_label = str(quantile)
                value = mean_ret_df.loc[quantile, period_col]
                period_data[q_label] = _to_python_float(value)

            quantile_returns[period_label] = period_data

        return quantile_returns

    def _extract_factor_returns(
        self,
        factor_ret: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        提取因子收益数据

        Args:
            factor_ret: alphalens返回的因子收益DataFrame

        Returns:
            各period的因子收益时序和统计
        """
        factor_returns: Dict[str, Any] = {}

        for period_col in factor_ret.columns:
            period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
            ret_series = factor_ret[period_col].dropna()

            factor_returns[period_label] = {
                "cumulative_return": _to_python_float((1 + ret_series).prod() - 1),
                "mean_return": _to_python_float(ret_series.mean()),
                "std_return": _to_python_float(ret_series.std()),
                "series": {
                    "dates": _index_to_str_list(ret_series.index),
                    "values": _series_to_list(ret_series),
                },
            }

        return factor_returns

    def _extract_alpha_beta(
        self,
        alpha_beta: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        提取alpha和beta

        Args:
            alpha_beta: alphalens返回的alpha/beta DataFrame

        Returns:
            各period的alpha和beta
        """
        result: Dict[str, Any] = {}

        for period_col in alpha_beta.columns:
            period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
            period_data: Dict[str, Any] = {}

            for stat_name in alpha_beta.index:
                period_data[str(stat_name)] = _to_python_float(alpha_beta.loc[stat_name, period_col])

            result[period_label] = period_data

        return result

    def _compute_spread(
        self,
        mean_ret_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        计算最高分位与最低分位的收益差

        Args:
            mean_ret_df: alphalens返回的分位数收益DataFrame

        Returns:
            各period的top-bottom spread
        """
        spread: Dict[str, Any] = {}

        quantiles = sorted(mean_ret_df.index.tolist())
        if len(quantiles) < 2:
            logger.warning("分位数不足2个，无法计算spread")
            return {"error": "分位数不足，无法计算spread"}

        top_q = quantiles[-1]
        bottom_q = quantiles[0]

        for period_col in mean_ret_df.columns:
            period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
            top_ret = mean_ret_df.loc[top_q, period_col]
            bottom_ret = mean_ret_df.loc[bottom_q, period_col]
            spread_val = top_ret - bottom_ret

            spread[period_label] = {
                "top_quantile": str(top_q),
                "bottom_quantile": str(bottom_q),
                "top_return": _to_python_float(top_ret),
                "bottom_return": _to_python_float(bottom_ret),
                "spread": _to_python_float(spread_val),
            }

        return spread

    def analyze_turnover(
        self,
        factor_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        分析因子换手率

        Args:
            factor_data: alphalens格式的因子数据

        Returns:
            换手率分析结果字典
        """
        if factor_data is None or factor_data.empty:
            logger.error("factor_data为空，无法执行换手率分析")
            return {"error": "因子数据为空"}

        results: Dict[str, Any] = {}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                turnover = alphalens.performance.quantile_turnover(
                    factor_data['factor_quantile'], quantile=factor_data['factor_quantile'].max()
                )

                autocorr = alphalens.performance.factor_rank_autocorrelation(factor_data)

            results["quantile_turnover"] = self._extract_turnover(turnover) if turnover is not None else {"error": "换手率数据为空"}

            results["factor_autocorrelation"] = self._extract_autocorrelation(autocorr) if autocorr is not None else {"error": "自相关数据为空"}

            logger.info("换手率分析完成")
            return results

        except Exception as e:
            logger.error(f"换手率分析失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _extract_turnover(
        self,
        turnover_data,
    ) -> Dict[str, Any]:
        """
        提取分位数换手率数据

        Args:
            turnover_data: alphalens返回的换手率数据 (DataFrame 或 Series)

        Returns:
            各分位数各period的换手率
        """
        turnover_result: Dict[str, Any] = {}

        # 处理 Series 类型（单列情况）
        if isinstance(turnover_data, pd.Series):
            return {
                "mean_turnover": _to_python_float(turnover_data.mean()),
                "std_turnover": _to_python_float(turnover_data.std()) if len(turnover_data) > 1 else 0.0,
                "series": {
                    "dates": _index_to_str_list(turnover_data.index),
                    "values": _series_to_list(turnover_data),
                },
            }

        if not hasattr(turnover_data, 'columns') or turnover_data.empty:
            return {"error": "换手率数据为空或格式不正确"}

        if isinstance(turnover_data.index, pd.MultiIndex):
            for period_col in turnover_data.columns:
                period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
                period_data: Dict[str, Any] = {}

                for quantile in turnover_data.index.get_level_values(0).unique():
                    q_label = str(quantile)
                    q_series = turnover_data.loc[quantile, period_col]
                    if isinstance(q_series, pd.Series):
                        period_data[q_label] = {
                            "mean_turnover": _to_python_float(q_series.mean()),
                            "std_turnover": _to_python_float(q_series.std()),
                            "series": {
                                "dates": _index_to_str_list(q_series.index),
                                "values": _series_to_list(q_series),
                            },
                        }
                    else:
                        period_data[q_label] = {
                            "mean_turnover": _to_python_float(q_series),
                        }

                turnover_result[period_label] = period_data
        else:
            for period_col in turnover_data.columns:
                period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
                period_data: Dict[str, Any] = {}

                for quantile in turnover_data.index:
                    q_label = str(quantile)
                    value = turnover_data.loc[quantile, period_col]
                    if isinstance(value, pd.Series):
                        period_data[q_label] = {
                            "mean_turnover": _to_python_float(value.mean()),
                            "series": {
                                "dates": _index_to_str_list(value.index),
                                "values": _series_to_list(value),
                            },
                        }
                    else:
                        period_data[q_label] = {
                            "mean_turnover": _to_python_float(value),
                        }

                turnover_result[period_label] = period_data

        return turnover_result

    def _extract_autocorrelation(
        self,
        autocorr_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        提取因子自相关数据

        Args:
            autocorr_df: alphalens返回的因子自相关DataFrame

        Returns:
            各period的因子自相关
        """
        autocorr_result: Dict[str, Any] = {}

        # 处理 Series 类型（单列情况）
        if isinstance(autocorr_df, pd.Series):
            return {
                "mean_autocorrelation": _to_python_float(autocorr_df.mean()),
                "std_autocorrelation": _to_python_float(autocorr_df.std()) if len(autocorr_df) > 1 else 0.0,
                "series": {
                    "dates": _index_to_str_list(autocorr_df.index),
                    "values": _series_to_list(autocorr_df),
                },
            }

        if not hasattr(autocorr_df, 'columns') or autocorr_df.empty:
            return {"error": "自相关数据为空或格式不正确"}

        for period_col in autocorr_df.columns:
            period_label = f"{period_col}D" if isinstance(period_col, int) else str(period_col)
            ac_series = autocorr_df[period_col].dropna()

            autocorr_result[period_label] = {
                "mean_autocorrelation": _to_python_float(ac_series.mean()),
                "std_autocorrelation": _to_python_float(ac_series.std()) if len(ac_series) > 1 else 0.0,
                "series": {
                    "dates": _index_to_str_list(ac_series.index),
                    "values": _series_to_list(ac_series),
                },
            }

        return autocorr_result

    def full_analysis(
        self,
        factor_values_dict: Dict[str, pd.Series],
        pricing_df: pd.DataFrame,
        groupby_dict: Optional[Dict[str, str]] = None,
        periods: Tuple[int, ...] = (1, 5, 10),
        quantiles: int = 5,
        max_loss: float = 0.50,
    ) -> Dict[str, Any]:
        """
        执行完整的alphalens因子分析

        Args:
            factor_values_dict: {股票代码: pd.Series(因子值, index=日期)}
            pricing_df: DataFrame, index=日期, columns=股票代码, 值为收盘价
            groupby_dict: {股票代码: 行业名称}，可选
            periods: 远期收益计算周期
            quantiles: 分位数数量
            max_loss: 最大允许数据丢失比例 (默认0.50)

        Returns:
            包含所有分析结果的字典
        """
        results: Dict[str, Any] = {
            "metadata": {
                "num_stocks": len(factor_values_dict),
                "periods": list(periods),
                "quantiles": quantiles,
                "has_groupby": groupby_dict is not None,
            },
        }

        factor_data = self.prepare_factor_data(
            factor_values_dict, pricing_df, groupby_dict, periods, quantiles, max_loss
        )

        if factor_data is None or factor_data.empty:
            logger.error("因子数据准备失败，无法继续分析")
            results["error"] = "因子数据准备失败"
            return results

        results["metadata"]["data_records"] = len(factor_data)
        results["metadata"]["date_range"] = {
            "start": str(factor_data.index.get_level_values("date").min()),
            "end": str(factor_data.index.get_level_values("date").max()),
        }
        results["metadata"]["num_assets"] = factor_data.index.get_level_values("asset").nunique()

        if len(factor_values_dict) == 1:
            results["metadata"]["warning"] = "仅包含单只股票，横截面IC分析结果可能不可靠"

        by_group = groupby_dict is not None

        ic_results = self.analyze_ic(factor_data, by_group=by_group)
        results["ic_analysis"] = ic_results

        returns_results = self.analyze_returns(factor_data, by_group=by_group)
        results["returns_analysis"] = returns_results

        turnover_results = self.analyze_turnover(factor_data)
        results["turnover_analysis"] = turnover_results

        logger.info("完整因子分析完成")
        return results


alphalens_analysis_service = AlphalensAnalysisService()
