"""
Tear Sheet 全貌报告生成器 - 专业因子分析报告系统

功能：
1. 整合所有因子分析结果为完整报告
2. 提供多维度评估（IC/IR/分组收益/稳定性/换手率等）
3. 生成综合评分和投资建议
4. 支持结构化数据输出（适合前端可视化）

设计原则：
- 类似 Alphalens 的 create_full_tear_sheet，但更现代化
- 返回结构化 JSON 数据，便于前端渲染图表
- 包含统计检验结果和专业解读
- 符合量化研究的报告标准
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TearSheetConfig:
    """
    Tear Sheet 配置
    
    Attributes:
        include_quantile_analysis: 是否包含分组收益分析
        include_cumulative_returns: 是否包含累计收益曲线
        include_turnover: 是否包含换手率分析
        include_bootstrap: 是否包含Bootstrap置信区间
        include_interpretation: 是否包含专业解读
        scoring_weights: 各维度评分权重
    """
    include_quantile_analysis: bool = True
    include_cumulative_returns: bool = True
    include_turnover: bool = True
    include_bootstrap: bool = True
    include_interpretation: bool = True
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "ic_strength": 25,
        "ir_quality": 20,
        "quantile_monotonicity": 20,
        "stability": 15,
        "turnover_efficiency": 10,
        "significance": 10,
    })


class TearSheetService:
    """
    Tear Sheet 服务类
    
    生成完整的因子分析报告，包含：
    - 因子概览
    - IC/IR 分析
    - 分组收益分析
    - 累计收益曲线
    - 换手率和稳定性
    - 综合评分和建议
    """

    def __init__(
        self,
        config: Optional[TearSheetConfig] = None,
    ):
        """
        初始化服务
        
        Args:
            config: 报告配置
        """
        self.config = config or TearSheetConfig()

    def create_full_tear_sheet(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
        price_column: str = "close",
        return_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        创建完整的Tear Sheet报告
        
        这是主要入口方法，整合所有分析模块的结果。
        
        Args:
            factor_data: {stock_code: DataFrame} 格式的因子数据
            factor_name: 因子名称
            price_column: 价格列名
            return_series: 收益率序列（可选）
            
        Returns:
            完整的Tear Sheet报告字典
        """
        try:
            report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = {
                "metadata": {
                    "report_type": "full_tear_sheet",
                    "factor_name": factor_name,
                    "generated_at": report_time,
                    "version": "2.0",
                },
                "summary": None,
                "sections": {},
            }
            
            sections_completed = []
            errors = []
            
            if self.config.include_quantile_analysis:
                try:
                    from backend.services.factor_return_analysis_service import (
                        factor_return_analysis_service
                    )
                    
                    quantile_result = (
                        factor_return_analysis_service.calculate_quantile_returns(
                            factor_data=factor_data,
                            factor_name=factor_name,
                            price_column=price_column,
                        )
                    )
                    
                    if "error" not in quantile_result:
                        report["sections"]["quantile_returns"] = quantile_result
                        sections_completed.append("quantile_returns")
                    else:
                        errors.append(f"分组收益分析: {quantile_result['error']}")
                        
                except Exception as e:
                    logger.warning(f"分组收益分析失败: {e}")
                    errors.append(f"分组收益分析: {str(e)}")
            
            if self.config.include_cumulative_returns:
                try:
                    from backend.services.factor_return_analysis_service import (
                        factor_return_analysis_service
                    )
                    
                    cumulative_result = (
                        factor_return_analysis_service.calculate_cumulative_returns(
                            factor_data=factor_data,
                            factor_name=factor_name,
                            price_column=price_column,
                            long_short=True,
                        )
                    )
                    
                    if "error" not in cumulative_result:
                        report["sections"]["cumulative_returns"] = cumulative_result
                        sections_completed.append("cumulative_returns")
                    else:
                        errors.append(f"累计收益分析: {cumulative_result['error']}")
                        
                except Exception as e:
                    logger.warning(f"累计收益分析失败: {e}")
                    errors.append(f"累计收益分析: {str(e)}")
            
            if self.config.include_turnover:
                try:
                    from backend.services.factor_return_analysis_service import (
                        factor_return_analysis_service
                    )
                    
                    turnover_result = (
                        factor_return_analysis_service.calculate_turnover_analysis(
                            factor_data=factor_data,
                            factor_name=factor_name,
                        )
                    )
                    
                    if "error" not in turnover_result:
                        report["sections"]["turnover_analysis"] = turnover_result
                        sections_completed.append("turnover_analysis")
                    else:
                        errors.append(f"换手率分析: {turnover_result['error']}")
                        
                except Exception as e:
                    logger.warning(f"换手率分析失败: {e}")
                    errors.append(f"换手率分析: {str(e)}")
            
            try:
                ic_ir_section = self._extract_ic_ir_summary(factor_data, factor_name)
                if ic_ir_section:
                    report["sections"]["ic_ir_analysis"] = ic_ir_section
                    sections_completed.append("ic_ir_analysis")
            except Exception as e:
                logger.warning(f"IC/IR摘要提取失败: {e}")
                errors.append(f"IC/IR分析: {str(e)}")
            
            overall_score, score_breakdown = self._calculate_overall_score(report["sections"])
            
            report["summary"] = {
                "overall_score": float(overall_score),
                "grade": self._score_to_grade(overall_score),
                "score_breakdown": score_breakdown,
                "sections_completed": sections_completed,
                "n_sections_total": 4,
                "completion_rate": len(sections_completed) / 4,
                "warnings": errors if errors else None,
            }
            
            if self.config.include_interpretation:
                report["interpretation"] = self._generate_interpretation(
                    report["sections"], 
                    overall_score
                )
            
            report["recommendations"] = self._generate_recommendations(
                report["sections"],
                overall_score
            )
            
            logger.info(
                f"Tear Sheet生成完成: 因子={factor_name}, "
                f"得分={overall_score:.1f}, "
                f"完成{len(sections_completed)}/4个板块"
            )
            
            return {"success": True, "tear_sheet": report}
            
        except Exception as e:
            logger.error(f"Tear Sheet生成失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _extract_ic_ir_summary(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_name: str,
    ) -> Optional[Dict[str, Any]]:
        """提取IC/IR分析的摘要信息"""
        try:
            all_ics = []
            
            # 横截面IC：每个日期计算因子值与未来收益的Spearman秩相关
            from scipy.stats import spearmanr
            
            # 构建日期-股票面板
            date_factors = {}
            date_returns = {}
            for stock_code, df in factor_data.items():
                if factor_name in df.columns and "close" in df.columns:
                    df_copy = df.copy()
                    df_copy["future_return"] = df_copy["close"].pct_change(1).shift(-1)
                    valid_mask = df_copy[factor_name].notna() & df_copy["future_return"].notna()
                    for date, row in df_copy[valid_mask].iterrows():
                        if date not in date_factors:
                            date_factors[date] = []
                            date_returns[date] = []
                        date_factors[date].append(row[factor_name])
                        date_returns[date].append(row["future_return"])
            
            # 每日横截面Rank IC
            for date in sorted(date_factors.keys()):
                fv = date_factors[date]
                rv = date_returns[date]
                if len(fv) >= 5 and np.std(fv) > 1e-12:
                    ic, _ = spearmanr(fv, rv)
                    if not np.isnan(ic):
                        all_ics.append(ic)
            
            if not all_ics:
                return None
            
            ic_series = pd.Series(all_ics)
            
            return {
                "mean_ic": float(ic_series.mean()),
                "std_ic": float(ic_series.std()) if ic_series.std() > 0 else 0.0,
                "ir": float(
                    ic_series.mean() / ic_series.std()
                ) if ic_series.std() > 0 else 0.0,
                "ic_positive_ratio": float((ic_series > 0).mean()),
                "n_observations": len(ic_series),
                "median_ic": float(ic_series.median()),
                "skewness": float(ic_series.skew()),
                "kurtosis": float(ic_series.kurtosis()),
            }
            
        except Exception as e:
            logger.debug(f"IC/IR摘要提取异常: {e}")
            return None

    def _calculate_overall_score(
        self,
        sections: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算综合得分
        
        基于各分析维度的加权评分。
        """
        weights = self.config.scoring_weights
        scores = {}
        
        if "ic_ir_analysis" in sections:
            ic_ir = sections["ic_ir_analysis"]
            ir = abs(ic_ir.get("ir", 0))
            positive_ratio = ic_ir.get("ic_positive_ratio", 0.5)
            
            ic_score = min(ir * 15 + (positive_ratio - 0.5) * 40, 100)
            scores["ic_strength"] = max(0, ic_score)
            
            ir_score = min(ir * 25, 100)
            scores["ir_quality"] = max(0, ir_score)
        else:
            scores["ic_strength"] = 0
            scores["ir_quality"] = 0
        
        if "quantile_returns" in sections:
            qr = sections["quantile_returns"]
            mono = qr.get("monotonicity_test", {})
            spread = qr.get("spread", {})
            
            mono_score = (
                mono.get("monotonicity_ratio", 0) * 100 if 
                isinstance(mono.get("monotonicity_ratio"), (int, float)) 
                else 0
            )
            scores["quantile_monotonicity"] = max(0, min(mono_score, 100))
            
            is_sig = spread.get("is_significant", False)
            spread_val = abs(spread.get("long_short_spread", 0))
            sig_score = (50 if is_sig else 20) + min(spread_val * 1000, 50)
            scores["significance"] = max(0, min(sig_score, 100))
        else:
            scores["quantile_monotonicity"] = 0
            scores["significance"] = 0
        
        if "turnover_analysis" in sections:
            ta = sections["turnover_analysis"]
            stability = ta.get("stability_analysis", {})
            stab_score = stability.get("stability_score", 0) * 100
            
            turnover_stats = ta.get("turnover_stats", {})
            mean_turnover = turnover_stats.get("mean_turnover", 0.5)
            eff_score = (1 - mean_turnover) * 100
            
            scores["stability"] = max(0, min(stab_score, 100))
            scores["turnover_efficiency"] = max(0, min(eff_score, 100))
        else:
            scores["stability"] = 0
            scores["turnover_efficiency"] = 0
        
        total_score = sum(
            scores.get(key, 0) * weight / 100 
            for key, weight in weights.items()
        )
        
        return total_score, scores

    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级"""
        if score >= 80:
            return "A（优秀）"
        elif score >= 65:
            return "B（良好）"
        elif score >= 50:
            return "C（一般）"
        elif score >= 35:
            return "D（较弱）"
        else:
            return "F（不合格）"

    def _generate_interpretation(
        self,
        sections: Dict[str, Any],
        overall_score: float,
    ) -> Dict[str, str]:
        """生成专业解读"""
        interpretations = {}
        
        if "ic_ir_analysis" in sections:
            ic_ir = sections["ic_ir_analysis"]
            ir = ic_ir.get("ir", 0)
            mean_ic = ic_ir.get("mean_ic", 0)
            pos_ratio = ic_ir.get("ic_positive_ratio", 0.5)
            
            interpretations["ic_ir"] = (
                f"该因子的IC均值为{mean_ic:.4f}，IR为{ir:.3f}，"
                f"IC正方向占比{pos_ratio:.1%}。"
            )
            
            if ir > 1.0:
                interpretations["ic_ir"] += "IC表现优秀，因子具有很强的预测能力和稳定性。"
            elif ir > 0.5:
                interpretations["ic_ir"] += "IC表现良好，因子具有一定的预测能力。"
            elif ir > 0:
                interpretations["ic_ir"] += "IC表现一般，预测能力有限。"
            else:
                interpretations["ic_ir"] += "IC表现较差，可能需要优化或更换因子。"
        
        if "quantile_returns" in sections:
            qr = sections["quantile_returns"]
            spread = qr.get("spread", {})
            mono = qr.get("monotonicity_test", {})
            
            spread_val = spread.get("long_short_spread", 0)
            is_sig = spread.get("is_significant", False)
            is_mono = mono.get("is_monotonic", False)
            
            interpretations["quantile"] = (
                f"多空利差为{spread_val:.4%}"
                f"{'（显著）' if is_sig else '（不显著）'}"
                f"，分组收益{'呈现' if is_mono else '未呈现'}单调性。"
            )
            
            if is_sig and is_mono and spread_val > 0.01:
                interpretations["quantile"] += "因子选股能力强，适合构建多头组合。"
            elif is_sig and spread_val > 0:
                interpretations["quantile"] += "因子具有一定的选股价值。"
            else:
                interpretations["quantile"] += "因子选股效果不明显，建议谨慎使用。"
        
        if "turnover_analysis" in sections:
            ta = sections["turnover_analysis"]
            stability = ta.get("stability_analysis", {})
            turnover_stats = ta.get("turnover_stats", {})
            
            mean_to = turnover_stats.get("mean_turnover", 0.5)
            is_stable = stability.get("is_stable", False)
            
            interpretations["turnover"] = (
                f"平均换手率为{mean_to:.2%}"
                f"，因子{'稳定' if is_stable else '不稳定'}。"
            )
            
            if is_stable and mean_to < 0.3:
                interpretations["turnover"] += "低换手率+高稳定性，交易成本低。"
            elif is_stable:
                interpretations["turnover"] += "因子较稳定，但需关注交易成本。"
            else:
                interpretations["turnover"] += "因子变动频繁，可能产生较高交易成本。"
        
        interpretations["overall"] = (
            f"综合评分为{overall_score:.1f}/100（{self._score_to_grade(overall_score)}）。"
        )
        
        if overall_score >= 70:
            interpretations["overall"] += "该因子质量优秀，推荐用于实盘策略。"
        elif overall_score >= 50:
            interpretations["overall"] += "该因子质量良好，可用于研究但需关注风险。"
        elif overall_score >= 35:
            interpretations["overall"] += "该因子质量一般，建议优化后使用。"
        else:
            interpretations["overall"] += "该因子质量较差，不建议直接使用。"
        
        return interpretations

    def _generate_recommendations(
        self,
        sections: Dict[str, Any],
        overall_score: float,
    ) -> List[Dict[str, Any]]:
        """生成改进建议"""
        recommendations = []
        
        if "ic_ir_analysis" in sections:
            ic_ir = sections["ic_ir_analysis"]
            ir = ic_ir.get("ir", 0)
            
            if ir < 0.5:
                recommendations.append({
                    "priority": "high",
                    "category": "IC/IR提升",
                    "suggestion": (
                        "因子IR较低(<0.5)，建议："
                        "1) 检查因子逻辑是否合理 "
                        "2) 尝试不同的时间周期 "
                        "3) 考虑与其他因子组合"
                    ),
                })
            
            positive_ratio = ic_ir.get("ic_positive_ratio", 0.5)
            if positive_ratio < 0.55 and positive_ratio > 0.45:
                recommendations.append({
                    "priority": "medium",
                    "category": "方向一致性",
                    "suggestion": (
                        f"IC正方向占比仅{positive_ratio:.1%}(接近随机)，"
                        "建议检查因子在不同市场环境下的表现"
                    ),
                })
        
        if "quantile_returns" in sections:
            qr = sections["quantile_returns"]
            mono = qr.get("monotonicity_test", {})
            
            if not mono.get("is_monotonic", False):
                recommendations.append({
                    "priority": "high",
                    "category": "单调性改善",
                    "suggestion": (
                        "分组收益不满足单调性，建议："
                        "1) 优化因子计算方式 "
                        "2) 尝试非线性变换 "
                        "3) 检查是否存在极端值干扰"
                    ),
                })
        
        if "turnover_analysis" in sections:
            ta = sections["turnover_analysis"]
            turnover_stats = ta.get("turnover_stats", {})
            mean_to = turnover_stats.get("mean_turnover", 0)
            
            if mean_to > 0.5:
                recommendations.append({
                    "priority": "medium",
                    "category": "换手率控制",
                    "suggestion": (
                        f"换手率过高({mean_to:.2%})，建议："
                        "1) 降低调仓频率 "
                        "2) 对因子值进行平滑处理 "
                        "3) 设置换手率约束"
                    ),
                })
        
        if overall_score < 50:
            recommendations.append({
                "priority": "high",
                "category": "整体优化",
                "suggestion": (
                    "综合评分偏低，建议进行全面审查："
                    "1) 回顾因子构造逻辑 "
                    "2) 检查数据处理流程 "
                    "3) 对比同类基准因子 "
                    "4) 考虑使用因子合成方法"
                ),
            })
        
        if not recommendations:
            recommendations.append({
                "priority": "low",
                "category": "持续监控",
                "suggestion": (
                    "因子表现良好，建议定期监控："
                    "1) 跟踪IC衰减情况 "
                    "2) 关注市场环境变化 "
                    "3) 定期更新参数"
                ),
            })
        
        return recommendations


# 全局默认实例
tear_sheet_service = TearSheetService()