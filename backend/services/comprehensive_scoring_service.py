"""
综合评分服务 - 多维度综合评分系统
"""
from typing import Dict, List, Optional
import logging

from backend.services.smart_slippage_detector import smart_slippage_detector
from backend.utils.safe_math import safe_divide

logger = logging.getLogger(__name__)


class ComprehensiveScoringService:
    """综合评分服务"""

    def __init__(self):
        # 默认评分配置
        self.default_weights = {
            "return": 0.3,          # 收益率权重
            "risk": 0.25,            # 风险权重
            "efficiency": 0.2,       # 效率权重（夏普、IR等）
            "stability": 0.15,       # 稳定性权重
            "cost": 0.1,             # 成本权重（换手率）
        }

    def score_factor(
        self,
        factor_metrics: Dict,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        对因子进行综合评分

        Args:
            factor_metrics: 因子指标字典
                必须包含:
                - ic_mean: IC均值
                - ir: IR值
                - ic_std: IC标准差（可选）
                - stability_score: 稳定性得分（可选）
                - turnover: 换手率（可选）
            weights: 自定义权重

        Returns:
            评分结果
        """
        if weights is None:
            weights = {
                "ic": 0.35,
                "ir": 0.30,
                "stability": 0.20,
                "turnover": 0.15,
            }

        # 防御性复制：避免修改传入数据
        factor_metrics = factor_metrics.copy()

        total_score = 0.0
        details = {}

        # 1. IC得分 (0-100) — 规则7.36: dict.get(key, default) 在值为None时不生效
        ic_mean_val = factor_metrics.get("ic_mean")
        ic_mean = abs(ic_mean_val) if ic_mean_val is not None else 0
        ic_score = min(ic_mean * 400, 100)  # IC=0.25时满分
        total_score += weights["ic"] * ic_score
        details["ic_score"] = float(ic_score)

        # 2. IR得分 (0-100)
        ir_val = factor_metrics.get("ir")
        ir_score = min(abs(ir_val) * 40, 100) if ir_val is not None else 0  # IR=2.5时满分
        total_score += weights["ir"] * ir_score
        details["ir_score"] = float(ir_score)

        # 3. 稳定性得分 (0-100)
        stability = factor_metrics.get("stability_score")
        stability_score = (stability if stability is not None else 0.8) * 100
        total_score += weights["stability"] * stability_score
        details["stability_score"] = float(stability_score)

        # 4. 换手率得分 (0-100)
        turnover = factor_metrics.get("turnover")
        turnover_val = turnover if turnover is not None else 0.3
        # 换手率越低越好
        turnover_score = max(100 - turnover_val * 200, 0)
        total_score += weights["turnover"] * turnover_score
        details["turnover_score"] = float(turnover_score)

        # 评级
        grade = self._get_grade(total_score)

        return {
            "total_score": max(0.0, min(round(total_score, 2), 100.0)),
            "grade": grade,
            "details": details,
            "weights": weights,
        }

    def analyze_slippage_sensitivity(
        self,
        strategy_metrics: Dict,
        stock_codes: Optional[List[str]] = None,
        base_slippage: float = 0.002,
        test_slippages: Optional[List[float]] = None,
    ) -> Dict:
        """
        分析策略对交易滑点的敏感性

        评估不同滑点水平对策略收益的影响，帮助用户理解交易成本风险

        Args:
            strategy_metrics: 策略指标字典，必须包含：
                            - annual_return: 年化收益率
                            - turnover: 换手率（可选，默认12倍/年）
            stock_codes: 股票代码列表（可选，用于智能检测）
            base_slippage: 基准滑点率（默认0.2%）
            test_slippages: 测试的滑点列表（可选）

        Returns:
            滑点敏感性分析结果字典
        """
        # 获取基础指标
        annual_return = strategy_metrics.get("annual_return")
        if annual_return is None:
            annual_return = 0.15
        turnover = strategy_metrics.get("turnover")
        if turnover is None:
            turnover = 12.0

        # 如果提供了股票代码，使用智能检测器获取更准确的推荐
        smart_recommendation = None
        if stock_codes and len(stock_codes) > 0:
            try:
                smart_recommendation = smart_slippage_detector.recommend_slippage(
                    stock_codes=stock_codes,
                    strategy_turnover=turnover,
                )
                # 使用智能推荐的基准滑点
                base_slippage = smart_recommendation.recommended_slippage
            except Exception as e:
                logger.warning(f"智能滑点检测失败，回退到默认值: {e}")

        # 定义测试场景
        if test_slippages is None:
            test_slippages = [0.0, 0.001, 0.002, 0.003, 0.005, 0.01, base_slippage]

        # 计算不同滑点下的净收益
        scenarios = []
        for slip in sorted(test_slippages):
            # 年化滑点成本 = 滑点率 * 换手率 * 2（买入+卖出）
            annual_cost = slip * turnover * 2
            net_return = annual_return - annual_cost
            if annual_return is None or abs(annual_return) < 1e-10:
                return_decay = float('inf')
            else:
                return_decay = safe_divide(float(annual_cost), float(annual_return), default=float('inf')) * 100

            scenario = {
                "slippage_rate": slip,
                "slippage_pct": f"{slip * 100:.2f}%",
                "annual_cost_pct": round(annual_cost * 100, 2),
                "net_annual_return": round(net_return * 100, 2),
                "return_decay_pct": round(return_decay, 2),
                "is_recommended": abs(slip - base_slippage) < 0.0001,
            }
            scenarios.append(scenario)

        # 敏感性等级评估
        base_cost = base_slippage * turnover * 2
        sensitivity_ratio = safe_divide(float(abs(base_cost)), float(annual_return), default=float('inf'))

        if sensitivity_ratio < 0.1:
            sensitivity_level = "low"
            sensitivity_desc = "低敏感：滑点对收益影响较小（<10%）"
        elif sensitivity_ratio < 0.25:
            sensitivity_level = "medium"
            sensitivity_desc = "中敏感：滑点对收益有适度影响（10-25%）"
        elif sensitivity_ratio < 0.5:
            sensitivity_level = "high"
            sensitivity_desc = "高敏感：滑点显著侵蚀收益（25-50%）"
        else:
            sensitivity_level = "very_high"
            sensitivity_desc = "极高敏感：滑点严重损害收益（>50%），需优化执行"

        # 生成建议
        recommendations = self._generate_slippage_recommendations(
            sensitivity_level, turnover, base_slippage, annual_return
        )

        # 构建结果
        result = {
            "base_slippage": base_slippage,
            "base_slippage_pct": f"{base_slippage * 100:.3f}%",
            "strategy_turnover": turnover,
            "original_annual_return": round(annual_return * 100, 2),
            "sensitivity_level": sensitivity_level,
            "sensitivity_description": sensitivity_desc,
            "cost_impact_ratio": round(sensitivity_ratio * 100, 2),
            "scenarios": scenarios,
            "recommendations": recommendations,
        }

        # 如果有智能推荐，添加到结果中
        if smart_recommendation:
            result["smart_recommendation"] = {
                "recommended_slippage": smart_recommendation.recommended_slippage,
                "conservative_slippage": smart_recommendation.conservative_slippage,
                "aggressive_slippage": smart_recommendation.aggressive_slippage,
                "confidence": smart_recommendation.confidence,
                "reasoning": smart_recommendation.reasoning,
                "sensitivity_analysis": smart_recommendation.sensitivity_analysis,
                "warnings": smart_recommendation.warnings,
                "tips": smart_recommendation.tips,
            }

        return result

    def _generate_slippage_recommendations(
        self,
        sensitivity_level: str,
        turnover: float,
        base_slippage: float,
        annual_return: float,
    ) -> List[Dict]:
        """根据敏感性分析生成优化建议"""
        recommendations = []

        if sensitivity_level in ["high", "very_high"]:
            recommendations.append({
                "priority": "critical",
                "category": "execution",
                "title": "优化交易执行",
                "description": f"当前策略对滑点{sensitivity_level.replace('_', ' ')}敏感，年化成本可能达到{base_slippage * turnover * 2 * 100:.1f}%",
                "actions": [
                    "使用算法交易（VWAP/TWAP）降低冲击成本",
                    "分批建仓/平仓，避免大额单笔交易",
                    "选择流动性较好的时段交易（开盘后30分钟或收盘前30分钟）",
                    "考虑使用限价单而非市价单",
                ]
            })

            if turnover > 24:
                recommendations.append({
                    "priority": "critical",
                    "category": "strategy",
                    "title": "降低换手频率",
                    "description": f"年化换手率{turnover:.0f}倍过高，导致交易成本累积严重",
                    "actions": [
                        "增加持仓周期，减少不必要的调仓",
                        "设置调仓阈值（如因子排名变化>20%才调仓）",
                        "考虑使用因子平滑或信号过滤降低噪声交易",
                    ]
                })
        elif sensitivity_level == "medium":
            recommendations.append({
                "priority": "moderate",
                "category": "optimization",
                "title": "适度优化执行质量",
                "description": "滑点成本适中，可通过简单优化改善",
                "actions": [
                    "监控实际成交价与预期价格的偏差",
                    "在波动剧烈时减少交易频率",
                    "优先选择流动性好的标的",
                ]
            })
        else:
            recommendations.append({
                "priority": "low",
                "category": "monitoring",
                "title": "保持当前策略",
                "description": "滑点影响较小，当前参数设置合理",
                "actions": [
                    "定期回顾实际滑点与假设的差异",
                    "关注市场微观结构变化",
                ]
            })

        # 针对不同换手率的通用建议
        if turnover > 12 and sensitivity_level != "low":
            recommendations.append({
                "priority": "moderate",
                "category": "cost_control",
                "title": "建立交易成本预算",
                "description": f"建议将年化交易成本控制在{annual_return * 0.1 * 100:.1f}%以内（目标收益的10%）",
                "actions": [
                    "设定单笔交易成本上限",
                    "定期统计实际执行滑点",
                    "将交易成本纳入策略绩效归因分析",
                ]
            })

        return recommendations

    def score_strategy(
        self,
        strategy_metrics: Dict,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        对策略进行综合评分

        Args:
            strategy_metrics: 策略指标字典
                必须包含:
                - annual_return: 年化收益率
                - volatility: 波动率
                - sharpe_ratio: 夏普比率
                - max_drawdown: 最大回撤
                - win_rate: 胜率（可选）
                - turnover: 换手率（可选）
            weights: 自定义权重

        Returns:
            评分结果
        """
        if weights is None:
            weights = self.default_weights

        total_score = 0.0
        details = {}

        # 1. 收益率得分 (0-100)
        annual_return = strategy_metrics.get("annual_return")
        annual_return = annual_return if annual_return is not None else 0
        # 假设年化收益率目标为20%
        return_score = min(max(annual_return / 0.2 * 100, 0), 100)
        total_score += weights["return"] * return_score
        details["return_score"] = float(return_score)

        # 2. 风险得分 (0-100)
        max_drawdown = strategy_metrics.get("max_drawdown")
        max_drawdown = max_drawdown if max_drawdown is not None else 0.2
        # 最大回撤越小越好，假设目标为10%
        drawdown_score = max(100 - abs(max_drawdown) / 0.1 * 100, 0)
        total_score += weights["risk"] * drawdown_score
        details["risk_score"] = float(drawdown_score)

        # 3. 效率得分 (0-100)
        sharpe_ratio = strategy_metrics.get("sharpe_ratio")
        sharpe_ratio = sharpe_ratio if sharpe_ratio is not None else 0
        # 夏普比率目标为2.0
        sharpe_score = max(min(sharpe_ratio / 2.0 * 100, 100), 0)
        total_score += weights["efficiency"] * sharpe_score
        details["efficiency_score"] = float(sharpe_score)

        # 4. 稳定性得分 (0-100)
        win_rate = strategy_metrics.get("win_rate")
        win_rate = win_rate if win_rate is not None else 0.5
        win_rate_score = win_rate * 100
        total_score += weights["stability"] * win_rate_score
        details["stability_score"] = float(win_rate_score)

        # 5. 成本得分 (0-100)
        turnover = strategy_metrics.get("turnover")
        turnover = turnover if turnover is not None else 0.5
        # 换手率越低越好
        cost_score = max(100 - turnover * 100, 0)
        total_score += weights["cost"] * cost_score
        details["cost_score"] = float(cost_score)

        # 评级
        grade = self._get_grade(total_score)

        return {
            "total_score": max(0.0, min(round(total_score, 2), 100.0)),
            "grade": grade,
            "details": details,
            "weights": weights,
        }

    def score_portfolio(
        self,
        portfolio_metrics: Dict,
        benchmark_metrics: Optional[Dict] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        对投资组合进行综合评分

        Args:
            portfolio_metrics: 组合指标字典
            benchmark_metrics: 基准指标字典（可选）
            weights: 自定义权重

        Returns:
            评分结果
        """
        if weights is None:
            weights = {
                "return": 0.35,
                "risk": 0.30,
                "diversification": 0.2,
                "efficiency": 0.15,
            }

        total_score = 0.0
        details = {}

        # 1. 收益率得分
        annual_return = portfolio_metrics.get("annual_return")
        annual_return = annual_return if annual_return is not None else 0
        return_score = min(max(annual_return / 0.15 * 100, 0), 100)

        # 如果有基准，计算超额收益
        if benchmark_metrics:
            benchmark_return = benchmark_metrics.get("annual_return")
            benchmark_return = benchmark_return if benchmark_return is not None else 0
            excess_return = annual_return - benchmark_return
            return_score = min(max(excess_return / 0.05 * 100, 0), 100)

        total_score += weights["return"] * return_score
        details["return_score"] = float(return_score)

        # 2. 风险得分
        volatility = portfolio_metrics.get("volatility")
        volatility = volatility if volatility is not None else 0.15
        max_drawdown = portfolio_metrics.get("max_drawdown")
        max_drawdown = max_drawdown if max_drawdown is not None else 0.1
        # empyrical 返回负值（如 -0.15 表示15%回撤），需取绝对值
        risk_score = max(100 - (volatility / 0.2 * 50 + abs(max_drawdown) / 0.15 * 50), 0)
        total_score += weights["risk"] * risk_score
        details["risk_score"] = float(risk_score)

        # 3. 分散化得分
        concentration = portfolio_metrics.get("herfindahl_index")
        concentration = concentration if concentration is not None else 0.1
        # Herfindahl指数越低越好
        diversification_score = max(100 - concentration * 100, 0)
        total_score += weights["diversification"] * diversification_score
        details["diversification_score"] = float(diversification_score)

        # 4. 效率得分
        sharpe_ratio = portfolio_metrics.get("sharpe_ratio")
        sharpe_ratio = sharpe_ratio if sharpe_ratio is not None else 1.0
        sharpe_score = max(min(sharpe_ratio / 2.0 * 100, 100), 0)
        total_score += weights["efficiency"] * sharpe_score
        details["efficiency_score"] = float(sharpe_score)

        # 评级
        grade = self._get_grade(total_score)

        return {
            "total_score": max(0.0, min(round(total_score, 2), 100.0)),
            "grade": grade,
            "details": details,
            "weights": weights,
        }

    def compare_and_rank(
        self,
        items: List[Dict],
        scoring_type: str = "strategy",
    ) -> List[Dict]:
        """
        对多个项目进行评分和排名

        Args:
            items: 项目列表，每个项目包含metrics和name
            scoring_type: 评分类型 ("factor", "strategy", "portfolio")

        Returns:
            排序后的项目列表
        """
        scored_items = []

        for item in items:
            metrics = item.get("metrics", {})
            name = item.get("name", "Unknown")

            # 根据类型选择评分方法
            if scoring_type == "factor":
                score_result = self.score_factor(metrics)
            elif scoring_type == "strategy":
                score_result = self.score_strategy(metrics)
            elif scoring_type == "portfolio":
                score_result = self.score_portfolio(metrics)
            else:
                raise ValueError(f"未知的评分类型: {scoring_type}")

            scored_items.append({
                "name": name,
                "score": score_result["total_score"],
                "grade": score_result["grade"],
                "details": score_result["details"],
            })

        # 按得分排序
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        # 添加排名
        for i, item in enumerate(scored_items, 1):
            item["rank"] = i

        return scored_items

    def _get_grade(self, score: float) -> str:
        """
        根据得分返回评级

        Args:
            score: 得分 (0-100)

        Returns:
            评级
        """
        if score >= 95:
            return "S+"
        elif score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        else:
            return "D"

    def generate_scoring_report(self, score_result: Dict, name: str) -> str:
        """
        生成评分报告

        Args:
            score_result: 评分结果
            name: 项目名称

        Returns:
            报告文本
        """
        report = f"# {name} 评分报告\n\n"
        report += "## 综合得分\n\n"
        report += f"**得分**: {score_result['total_score']:.2f}/100\n"
        report += f"**评级**: {score_result['grade']}\n\n"

        report += "## 分项得分\n\n"

        details = score_result.get("details", {})
        weights = score_result.get("weights", {})

        for key, weight in weights.items():
            score = details.get(f"{key}_score", 0)
            report += f"- **{key.upper()}**: {score:.2f}/100 (权重 {weight:.0%})\n"

        return report


# 全局综合评分服务实例
comprehensive_scoring_service = ComprehensiveScoringService()
