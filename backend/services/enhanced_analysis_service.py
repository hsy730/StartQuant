"""
阶段3增强分析服务 - 扩展AnalysisService功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from scipy import stats

from backend.services.factor_neutralization_service import factor_neutralization_service
from backend.services.factor_stability_service import factor_stability_service
from backend.services.factor_summary_service import factor_summary_service
from backend.utils.safe_math import safe_ir, safe_divide
from backend.constants import STATISTICAL_SIGNIFICANCE_ALPHA, HIGHLY_SIGNIFICANT_ALPHA
from backend.utils.ic_calculator import calculate_cross_sectional_ic_from_panel

logger = logging.getLogger(__name__)


class EnhancedAnalysisService:
    """增强分析服务 - 集成阶段3新功能"""

    def __init__(self):
        pass

    def calculate_ic_significance(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        confidence_level: float = 0.95,
    ) -> Dict:
        """
        计算IC的显著性t检验

        按横截面（日期）计算IC，再对IC序列做t检验，避免池化相关违反独立性假设。

        Args:
            factor_values: 因子值序列（MultiIndex或带日期列的面板数据）
            return_values: 收益率序列
            confidence_level: 置信水平（默认95%）

        Returns:
            IC显著性检验结果
        """
        # 移除缺失值
        valid_data = pd.DataFrame(
            {"factor": factor_values, "return": return_values}
        ).dropna()

        if len(valid_data) < 10:
            return {
                "error": "有效数据不足（至少需要10个数据点）",
                "n_samples": len(valid_data),
            }

        # 尝试按横截面（日期）计算IC，再对IC序列做t检验
        # 如果数据有日期索引（MultiIndex或DatetimeIndex），按日期分组
        date_col = None
        if isinstance(valid_data.index, pd.MultiIndex):
            # MultiIndex: 第一层通常是日期
            date_col = valid_data.index.get_level_values(0)
        elif isinstance(valid_data.index, pd.DatetimeIndex):
            date_col = valid_data.index
        elif "date" in valid_data.columns:
            date_col = valid_data["date"]

        if date_col is not None:
            # 使用统一入口计算横截面IC（规则5：代码复用，规则7.1：横截面Spearman）
            panel_for_ic = valid_data.copy()
            if isinstance(date_col, pd.DatetimeIndex):
                panel_for_ic["_date"] = date_col
            elif isinstance(valid_data.index, pd.MultiIndex):
                panel_for_ic["_date"] = valid_data.index.get_level_values(0)
            else:
                panel_for_ic["_date"] = date_col

            ic_result = calculate_cross_sectional_ic_from_panel(
                panel_for_ic,
                factor_column="factor",
                return_column="return",
                date_column="_date",
                min_stocks=5,
                min_dates=2,
                method="spearman",
            )

            if ic_result is None:
                return {
                    "ic": None,
                    "p_value": None,
                    "is_significant": False,
                    "method": "spearman",
                    "n_samples": len(valid_data),
                    "n_cross_sections": 0,
                    "error": "有效截面不足（至少需要2个截面）",
                }

            ic_series = pd.Series(ic_result["ic_list"])
            mean_ic = ic_result["mean_ic"]
            n_cross = ic_result["n_dates"]

            # 使用 scipy.stats.ttest_1samp 替代手动 t 检验（规则0：开源库优先）
            t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0)

            # 计算置信区间
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, df=n_cross - 1)
            se = safe_divide(
                ic_series.std(ddof=1), np.sqrt(n_cross), default=None
            )
            if se is not None:
                ci_lower = mean_ic - t_critical * se
                ci_upper = mean_ic + t_critical * se
            else:
                ci_lower = ci_upper = None

            return {
                "ic": mean_ic,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "is_significant": p_value < STATISTICAL_SIGNIFICANCE_ALPHA,
                "significance_level": (
                    f"极高显著性 (p<{HIGHLY_SIGNIFICANT_ALPHA})"
                    if p_value < HIGHLY_SIGNIFICANT_ALPHA
                    else f"显著性 (p<{STATISTICAL_SIGNIFICANCE_ALPHA})"
                    if p_value < STATISTICAL_SIGNIFICANCE_ALPHA
                    else f"不显著 (p>={STATISTICAL_SIGNIFICANCE_ALPHA})"
                ),
                "confidence_interval": {
                    "lower": float(ci_lower) if ci_lower is not None else None,
                    "upper": float(ci_upper) if ci_upper is not None else None,
                    "level": confidence_level,
                },
                "n_samples": len(valid_data),
                "n_cross_sections": n_cross,
                "method": "spearman_cross_sectional",
                "interpretation": (
                    f"横截面IC均值在{confidence_level * 100:.0f}%置信区间为[{ci_lower:.4f}, {ci_upper:.4f}]"
                    if ci_lower is not None and ci_upper is not None
                    else "置信区间不可计算（标准误为零）"
                ),
            }

        # 无日期信息时无法计算横截面IC，池化Spearman违反独立性假设（规则7.1）
        logger.warning("无日期信息，无法计算横截面IC，池化Spearman违反独立性假设")
        return {
            "ic": None,
            "t_statistic": None,
            "p_value": None,
            "is_significant": None,
            "significance_level": "不可计算（无日期信息）",
            "confidence_interval": None,
            "n_samples": len(valid_data),
            "method": "spearman_pooled_unavailable",
            "warning": "无日期信息，无法计算横截面IC，池化Spearman违反独立性假设",
            "interpretation": "数据缺少日期维度，无法进行横截面IC检验",
        }

    def _calculate_cross_sectional_ic(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        date_series=None,
    ):
        """
        计算横截面Spearman IC（规则7.1：禁止池化Spearman）

        委托 ic_calculator.calculate_cross_sectional_ic_from_panel 统一入口（规则5：代码复用）

        Args:
            factor_values: 因子值序列
            return_values: 收益率序列
            date_series: 日期序列，用于横截面分组

        Returns:
            横截面IC均值（float），或None（无日期信息或截面不足）
        """
        if date_series is None or (
            isinstance(date_series, pd.Series) and date_series.isna().all()
        ):
            logger.warning("无日期信息，无法计算横截面IC")
            return None

        panel = pd.DataFrame({
            "factor": factor_values.reset_index(drop=True),
            "return": return_values.reset_index(drop=True),
        })
        if isinstance(date_series, pd.Series):
            panel["_date"] = date_series.reset_index(drop=True)
        else:
            panel["_date"] = pd.Series(date_series).reset_index(drop=True)

        result = calculate_cross_sectional_ic_from_panel(
            panel,
            factor_column="factor",
            return_column="return",
            date_column="_date",
            min_stocks=5,
            min_dates=1,
            method="spearman",
        )

        if result is None:
            return None

        return result["mean_ic"]

    def analyze_enhanced(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str],
        enable_neutralization: bool = False,
        enable_stability: bool = False,
        enable_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        增强版因子分析 - 集成所有阶段3新功能

        Args:
            factor_data: 因子数据字典
            factor_names: 因子名称列表
            enable_neutralization: 是否启用中性化
            enable_stability: 是否启用稳定性分析
            enable_summary: 是否生成摘要统计

        Returns:
            增强的分析结果
        """
        factor_data = {k: v.copy() for k, v in factor_data.items()}
        results = {
            "factors": {},
            "neutralization": {},
            "stability": {},
            "summary": {},
        }

        # factor_data 结构为 {stock_code: DataFrame}，每只股票的DataFrame包含所有因子列
        for factor_name in factor_names:
            # 收集所有股票的因子值和收益率
            all_factor_vals = []
            all_return_vals = []

            for stock_code, df in factor_data.items():
                df = df.copy()
                if factor_name not in df.columns:
                    continue
                if "close" not in df.columns:
                    continue

                # 计算未来收益率（如果不存在）
                if "future_return" not in df.columns:
                    df["future_return"] = df["close"].pct_change().shift(-1)

                valid = df[[factor_name, "future_return"]].dropna()
                if len(valid) > 0:
                    all_factor_vals.append(valid[factor_name])
                    all_return_vals.append(valid["future_return"])

            if not all_factor_vals:
                continue

            # 横截面IC分析：按日期截面计算Spearman相关，再取时间均值
            # （池化相关会产生伪相关，必须按截面计算）
            panel_data = []
            for stock_code, df in factor_data.items():
                df = df.copy()
                if factor_name not in df.columns:
                    continue
                if "close" not in df.columns:
                    continue
                if "future_return" not in df.columns:
                    df["future_return"] = df["close"].pct_change().shift(-1)
                valid = df[[factor_name, "future_return"]].dropna()
                if len(valid) > 0:
                    valid = valid.copy()
                    valid["_stock_code"] = stock_code
                    panel_data.append(valid)

            if panel_data:
                panel_df = pd.concat(panel_data)
                # 使用统一入口计算横截面IC（规则5：代码复用，规则7.1：横截面Spearman）
                ic_result = calculate_cross_sectional_ic_from_panel(
                    panel_df,
                    factor_column=factor_name,
                    return_column="future_return",
                    date_column=None,  # 使用DataFrame索引作为日期
                    min_stocks=3,
                    min_dates=1,
                    method="spearman",
                )

                if ic_result is not None:
                    mean_ic = ic_result["mean_ic"]
                    ic_std = ic_result["ic_std"]
                    n_days = ic_result["n_dates"]
                    daily_ics = ic_result["ic_list"]

                    # 使用 scipy.stats.ttest_1samp 替代手动 t 检验（规则0：开源库优先）
                    if ic_std > 1e-10 and len(daily_ics) > 1:
                        ic_series_for_test = pd.Series(daily_ics)
                        t_stat, p_value = stats.ttest_1samp(ic_series_for_test, 0)
                        t_stat = float(t_stat)
                        p_value = float(p_value)
                        is_significant = p_value < 0.05
                    else:
                        # IC标准差接近0时（规则7.10/7.15）
                        if abs(mean_ic) > 1e-10:
                            t_stat = float("inf")
                            p_value = 0.0
                            is_significant = True
                        else:
                            t_stat = 0.0
                            p_value = 1.0
                            is_significant = False
                    ic_significance = {
                        "ic": mean_ic,
                        "ic_std": ic_std,
                        "ir": safe_ir(float(mean_ic), float(ic_std), default=None),
                        "t_statistic": float(t_stat)
                        if t_stat is not None and np.isfinite(t_stat)
                        else None,
                        "p_value": float(p_value) if p_value is not None else None,
                        "is_significant": is_significant,
                        "n_samples": n_days,
                    }
                else:
                    ic_significance = {"error": "有效截面数据不足", "n_samples": 0}
            else:
                ic_significance = {"error": "有效数据不足", "n_samples": 0}

            results["factors"][factor_name] = {
                "ic_significance": ic_significance,
            }

            # 中性化处理（横截面操作，需合并所有股票数据）
            if enable_neutralization:
                try:
                    # 收集所有有效股票数据，合并后进行横截面中性化
                    all_stock_dfs = []
                    for stock_code, df in factor_data.items():
                        df = df.copy()
                        if factor_name not in df.columns:
                            continue
                        if "future_return" not in df.columns and "close" in df.columns:
                            df["future_return"] = df["close"].pct_change().shift(-1)
                        if "future_return" in df.columns:
                            df["_stock_code"] = stock_code
                            all_stock_dfs.append(df)

                    if all_stock_dfs:
                        combined_df = pd.concat(all_stock_dfs)
                        # 保留日期信息用于横截面IC计算
                        if isinstance(combined_df.index, pd.DatetimeIndex):
                            combined_df["_date"] = combined_df.index
                        elif isinstance(combined_df.index, pd.MultiIndex):
                            combined_df["_date"] = combined_df.index.get_level_values(0)
                        else:
                            combined_df["_date"] = None

                        # 市值中性化
                        if "market_cap" in combined_df.columns:
                            mc_neutralized = (
                                factor_neutralization_service.neutralize_market_cap(
                                    combined_df, factor_name, "market_cap"
                                )
                            )

                            if "future_return" in combined_df.columns:
                                ic_after_mc = self._calculate_cross_sectional_ic(
                                    mc_neutralized,
                                    combined_df["future_return"],
                                    combined_df.get("_date"),
                                )
                                ic_before = results["factors"][factor_name][
                                    "ic_significance"
                                ]["ic"]
                                results["neutralization"][f"{factor_name}_mc"] = {
                                    "method": "市值中性化",
                                    "ic_before": ic_before,
                                    "ic_after": ic_after_mc,
                                    "improvement": (
                                        (ic_after_mc - ic_before)
                                        if ic_after_mc is not None
                                        and ic_before is not None
                                        else None
                                    ),
                                }

                        # 行业中性化
                        if "industry" in combined_df.columns:
                            industry_neutralized = (
                                factor_neutralization_service.neutralize_industry(
                                    combined_df, factor_name, "industry"
                                )
                            )

                            if "future_return" in combined_df.columns:
                                ic_after_ind = self._calculate_cross_sectional_ic(
                                    industry_neutralized,
                                    combined_df["future_return"],
                                    combined_df.get("_date"),
                                )
                                ic_before = results["factors"][factor_name][
                                    "ic_significance"
                                ]["ic"]
                                results["neutralization"][f"{factor_name}_ind"] = {
                                    "method": "行业中性化",
                                    "ic_before": ic_before,
                                    "ic_after": ic_after_ind,
                                    "improvement": (
                                        (ic_after_ind - ic_before)
                                        if ic_after_ind is not None
                                        and ic_before is not None
                                        else None
                                    ),
                                }

                except Exception as e:
                    logger.warning(f"中性化处理失败({factor_name}): {e}")
                    results["neutralization"][factor_name] = {"error": str(e)}

            # 稳定性分析
            if enable_stability:
                try:
                    # 构建横截面因子值面板（date × stock_code），取横截面均值作为因子序列
                    cross_section_frames = []
                    for stock_code, df in factor_data.items():
                        df_copy = df.copy()
                        if factor_name in df_copy.columns:
                            stock_series = df_copy[factor_name].dropna()
                            stock_series.name = stock_code
                            cross_section_frames.append(stock_series)

                    if cross_section_frames:
                        cross_section_panel = pd.concat(cross_section_frames, axis=1)
                        combined_factor = cross_section_panel.mean(axis=1).dropna()
                    else:
                        combined_factor = pd.Series(dtype=float)

                    if len(combined_factor) >= 504:
                        dist_stability = (
                            factor_stability_service.calculate_distribution_stability(
                                combined_factor
                            )
                        )
                    else:
                        dist_stability = {
                            "warning": f"数据不足504个点(当前{len(combined_factor)})，跳过分布稳定性检验",
                            "data_points": len(combined_factor),
                            "required": 504,
                        }

                    # 时间序列稳定性（如果有IC序列）
                    ts_stability = None

                    # 滚动窗口稳定性
                    first_stock_df = None
                    for stock_code, df in factor_data.items():
                        if factor_name in df.columns:
                            first_stock_df = df.copy()
                            break

                    if first_stock_df is not None:
                        rolling_stability = (
                            factor_stability_service.calculate_rolling_stability(
                                first_stock_df, factor_name
                            )
                        )
                    else:
                        rolling_stability = None

                    results["stability"][factor_name] = {
                        "distribution_stability": dist_stability,
                        "time_series_stability": ts_stability,
                        "rolling_stability": rolling_stability,
                    }

                except Exception as e:
                    logger.warning(f"稳定性分析失败({factor_name}): {e}")
                    results["stability"][factor_name] = {"error": str(e)}

            # 生成摘要
            if enable_summary:
                try:
                    ic_analysis = results.get("factors", {})
                    stability_analysis = results.get("stability", {})

                    first_stock_df = None
                    for stock_code, df in factor_data.items():
                        if factor_name in df.columns:
                            first_stock_df = df.copy()
                            break

                    if first_stock_df is not None:
                        summary = factor_summary_service.generate_factor_summary(
                            factor_name, first_stock_df, ic_analysis, stability_analysis
                        )
                        results["summary"][factor_name] = summary

                except Exception as e:
                    logger.warning(f"摘要生成失败({factor_name}): {e}")
                    results["summary"][factor_name] = {"error": str(e)}

        return results


# 全局增强分析服务实例
enhanced_analysis_service = EnhancedAnalysisService()
