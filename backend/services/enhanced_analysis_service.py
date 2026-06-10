"""
阶段3增强分析服务 - 扩展AnalysisService功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats

from backend.services.factor_neutralization_service import factor_neutralization_service
from backend.services.factor_stability_service import factor_stability_service
from backend.services.factor_summary_service import factor_summary_service
from backend.utils.safe_math import safe_ir


class EnhancedAnalysisService:
    """增强分析服务 - 集成阶段3新功能"""

    def __init__(self):
        pass

    def calculate_ic_significance(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        confidence_level: float = 0.95
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
        valid_data = pd.DataFrame({
            "factor": factor_values,
            "return": return_values
        }).dropna()

        if len(valid_data) < 10:
            return {
                "error": "有效数据不足（至少需要10个数据点）",
                "n_samples": len(valid_data)
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
            ic_list = []
            for date, group in valid_data.groupby(date_col):
                if len(group) < 5:
                    continue
                from scipy.stats import spearmanr
                ic_val, _ = spearmanr(group["factor"], group["return"])
                if not np.isnan(ic_val):
                    ic_list.append(ic_val)

            if len(ic_list) < 2:
                return {
                    "ic": None,
                    "p_value": None,
                    "is_significant": False,
                    "method": "spearman",
                    "n_samples": len(valid_data),
                    "n_cross_sections": len(ic_list),
                    "error": "有效截面不足（至少需要2个截面）"
                }

            ic_series = pd.Series(ic_list)
            mean_ic = float(ic_series.mean())
            t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0)

            # 计算置信区间
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, df=len(ic_series) - 1)
            se = ic_series.std(ddof=1) / np.sqrt(len(ic_series))
            ci_lower = mean_ic - t_critical * se
            ci_upper = mean_ic + t_critical * se

            return {
                "ic": mean_ic,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "is_significant": p_value < 0.05,
                "significance_level": (
                    "极高显著性 (p<0.01)" if p_value < 0.01 else
                    "显著性 (p<0.05)" if p_value < 0.05 else
                    "不显著 (p>=0.05)"
                ),
                "confidence_interval": {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper),
                    "level": confidence_level,
                },
                "n_samples": len(valid_data),
                "n_cross_sections": len(ic_list),
                "method": "spearman_cross_sectional",
                "interpretation": (
                    f"横截面IC均值在{confidence_level*100:.0f}%置信区间为[{ci_lower:.4f}, {ci_upper:.4f}]"
                ),
            }

        # 无日期信息时回退到单次Spearman相关（并标注局限性）
        ic = valid_data["factor"].corr(valid_data["return"], method="spearman")

        n = len(valid_data)
        ic_clipped = np.clip(ic, -0.9999, 0.9999)
        t_statistic = ic_clipped * np.sqrt(n - 2) / np.sqrt(1 - ic_clipped**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n - 2))

        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 2)
        se = np.sqrt((1 - ic_clipped**2) / (n - 2))
        ci_lower = ic - t_critical * se
        ci_upper = ic + t_critical * se

        return {
            "ic": float(ic),
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
            "significance_level": (
                "极高显著性 (p<0.01)" if p_value < 0.01 else
                "显著性 (p<0.05)" if p_value < 0.05 else
                "不显著 (p>=0.05)"
            ),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": confidence_level,
            },
            "n_samples": n,
            "method": "spearman_pooled",
            "warning": "无日期信息，使用池化Spearman相关（可能违反独立性假设）",
            "interpretation": (
                f"IC在{confidence_level*100:.0f}%置信区间为[{ci_lower:.4f}, {ci_upper:.4f}]"
            ),
        }

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
                # 按日期分组计算横截面IC
                daily_ics = []
                for date, group in panel_df.groupby(panel_df.index):
                    if len(group) < 3:  # 截面至少3只股票才有意义
                        continue
                    ic_val = group[factor_name].corr(group["future_return"], method="spearman")
                    if not np.isnan(ic_val):
                        daily_ics.append(ic_val)

                if daily_ics:
                    mean_ic = float(np.mean(daily_ics))
                    ic_std = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else 0.0
                    n_days = len(daily_ics)
                    # t检验：IC均值是否显著不为0
                    if ic_std > 1e-10:
                        t_stat = mean_ic / (ic_std / np.sqrt(n_days))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_days - 1))
                    else:
                        t_stat = 0.0
                        p_value = 1.0
                    ic_significance = {
                        "ic": mean_ic,
                        "ic_std": ic_std,
                        "ir": safe_ir(float(mean_ic), float(ic_std), default=None),
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "is_significant": p_value < 0.05,
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
                        combined_df = pd.concat(all_stock_dfs, ignore_index=True)

                        # 市值中性化
                        if "market_cap" in combined_df.columns:
                            mc_neutralized = factor_neutralization_service.neutralize_market_cap(
                                combined_df, factor_name, "market_cap"
                            )

                            if "future_return" in combined_df.columns:
                                ic_after_mc = mc_neutralized.corr(combined_df["future_return"], method="spearman")
                                results["neutralization"][f"{factor_name}_mc"] = {
                                    "method": "市值中性化",
                                    "ic_before": results["factors"][factor_name]["ic_significance"]["ic"],
                                    "ic_after": float(ic_after_mc),
                                    "improvement": float(ic_after_mc - results["factors"][factor_name]["ic_significance"]["ic"]),
                                }

                        # 行业中性化
                        if "industry" in combined_df.columns:
                            industry_neutralized = factor_neutralization_service.neutralize_industry(
                                combined_df, factor_name, "industry"
                            )

                            if "future_return" in combined_df.columns:
                                ic_after_ind = industry_neutralized.corr(combined_df["future_return"], method="spearman")
                                results["neutralization"][f"{factor_name}_ind"] = {
                                    "method": "行业中性化",
                                    "ic_before": results["factors"][factor_name]["ic_significance"]["ic"],
                                    "ic_after": float(ic_after_ind),
                                    "improvement": float(ic_after_ind - results["factors"][factor_name]["ic_significance"]["ic"]),
                                }

                except Exception as e:
                    results["neutralization"][factor_name] = {
                        "error": str(e)
                    }

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
                        dist_stability = factor_stability_service.calculate_distribution_stability(
                            combined_factor
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
                        rolling_stability = factor_stability_service.calculate_rolling_stability(
                            first_stock_df, factor_name
                        )
                    else:
                        rolling_stability = None

                    results["stability"][factor_name] = {
                        "distribution_stability": dist_stability,
                        "time_series_stability": ts_stability,
                        "rolling_stability": rolling_stability,
                    }

                except Exception as e:
                    results["stability"][factor_name] = {
                        "error": str(e)
                    }

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
                            factor_name,
                            first_stock_df,
                            ic_analysis,
                            stability_analysis
                        )
                        results["summary"][factor_name] = summary

                except Exception as e:
                    results["summary"][factor_name] = {
                        "error": str(e)
                    }

        return results


# 全局增强分析服务实例
enhanced_analysis_service = EnhancedAnalysisService()
