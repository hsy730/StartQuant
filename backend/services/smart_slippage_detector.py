"""
智能交易滑点检测器

根据市场微观结构特征自动推荐最优的滑点参数
支持：
- 市场板块自适应（主板/创业板/科创板/北交所）
- 流动性感知（基于成交量和市值）
- 波动率调节（高波动环境增加滑点）
- 换手率影响（高频策略需要更高滑点假设）
- 置信度评估和推荐理由生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from backend.services.risk_metrics import calculate_volatility
from enum import Enum
import logging

from backend.core.market_board import MarketBoard, detect_market_board
from backend.utils.safe_math import safe_divide

logger = logging.getLogger(__name__)


class LiquidityLevel(str, Enum):
    """流动性等级"""

    HIGH = "high"  # 高流动性
    MEDIUM = "medium"  # 中等流动性
    LOW = "low"  # 低流动性
    VERY_LOW = "very_low"  # 极低流动性


@dataclass
class MarketCharacteristics:
    """市场特征描述"""

    stock_codes: List[str]
    n_stocks: int

    market_board: MarketBoard
    board_distribution: Dict[MarketBoard, float]  # 各板块占比

    avg_market_cap: float  # 平均市值（元）
    median_market_cap: float  # 中位数市值
    market_cap_cv: float  # 市值变异系数

    avg_daily_volume: float  # 日均成交量（股）
    avg_daily_amount: float  # 日均成交额（元）
    avg_turnover_rate: float  # 平均换手率

    price_volatility: float  # 价格波动率（日收益率标准差）
    spread_estimate: float  # 估计买卖价差（基于波动率）

    is_large_cap_dominant: bool  # 是否大盘股主导
    has_illiquid_stocks: bool  # 是否包含低流动性股票


@dataclass
class SlippageRecommendation:
    """滑点推荐结果"""

    recommended_slippage: float  # 推荐滑点率（如0.002表示0.2%）
    conservative_slippage: float  # 保守估计
    aggressive_slippage: float  # 激进估计

    confidence: float  # 推荐置信度 (0-1)
    reasoning: str  # 推荐理由（人类可读）

    market_characteristics: MarketCharacteristics
    sensitivity_analysis: Dict  # 敏感性分析结果

    warnings: List[str]  # 警告信息
    tips: List[str]  # 优化建议


class SmartSlippageDetector:
    """
    智能交易滑点检测器

    算法流程：
    1. 识别市场板块 → 确定基础滑点范围
    2. 评估流动性水平 → 调整滑点基准值
    3. 分析波动率 → 增加波动性溢价
    4. 考虑策略换手率 → 应用频率惩罚因子
    5. 输出推荐配置 + 置信度 + 敏感性分析
    """

    # 市场板块基础配置
    _BOARD_BASE_CONFIG = {
        MarketBoard.MAIN: {
            "base_slippage": 0.001,  # 基础滑点 0.1%
            "volatility_factor": 1.0,
            "liquidity_premium": 0.0,
            "description": "主板市场",
            "typical_spread_bps": 10,  # 典型价差（基点）
        },
        MarketBoard.CHINEXT: {
            "base_slippage": 0.002,  # 基础滑点 0.2%
            "volatility_factor": 1.3,
            "liquidity_premium": 0.0005,  # 流动性溢价
            "description": "创业板市场",
            "typical_spread_bps": 20,
        },
        MarketBoard.STAR: {
            "base_slippage": 0.0025,  # 基础滑点 0.25%
            "volatility_factor": 1.4,
            "liquidity_premium": 0.0008,
            "description": "科创板市场",
            "typical_spread_bps": 25,
        },
        MarketBoard.BSE: {
            "base_slippage": 0.003,  # 基础滑点 0.3%
            "volatility_factor": 1.6,
            "liquidity_premium": 0.001,
            "description": "北交所市场",
            "typical_spread_bps": 30,
        },
        MarketBoard.UNKNOWN: {
            "base_slippage": 0.0018,  # 混合市场取中间值
            "volatility_factor": 1.2,
            "liquidity_premium": 0.0003,
            "description": "混合板块市场",
            "typical_spread_bps": 18,
        },
    }

    # 流动性等级阈值（日均成交额，单位：亿元）
    _LIQUIDITY_THRESHOLDS = {
        LiquidityLevel.HIGH: 5.0,  # >5亿/日
        LiquidityLevel.MEDIUM: 1.0,  # 1-5亿/日
        LiquidityLevel.LOW: 0.2,  # 0.2-1亿/日
        LiquidityLevel.VERY_LOW: 0.0,  # <0.2亿/日
    }

    def __init__(self):
        # 板块识别已委托给 backend.core.market_board.detect_market_board
        pass

    def analyze_market(
        self,
        stock_codes: List[str],
        market_data: Optional[pd.DataFrame] = None,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> MarketCharacteristics:
        """
        分析市场特征

        Args:
            stock_codes: 股票代码列表
            market_data: 市场数据DataFrame（可选），应包含：
                        - stock_code: 股票代码
                        - market_cap: 市值
                        - volume: 成交量
                        - amount: 成交额
                        - turnover_rate: 换手率
            price_data: 价格数据字典 {stock_code: DataFrame}（可选）
                      应包含 close, volume 列

        Returns:
            市场特征描述对象
        """
        if not stock_codes:
            return self._empty_characteristics([])

        n_stocks = len(stock_codes)

        # 2️⃣ 识别市场板块分布
        board_distribution = self._detect_board_distribution(stock_codes)

        # 判断主导板块：单一板块占比>60%才认为是该板块，否则为混合
        dominant_board = MarketBoard.UNKNOWN
        if board_distribution:
            max_board, max_ratio = max(board_distribution.items(), key=lambda x: x[1])
            if max_ratio > 0.6:  # 单一板块占比超过60%
                dominant_board = max_board

        # 2️⃣ 提取市场数据特征
        avg_mc = 0
        median_mc = 0
        mc_cv = 0
        avg_volume = 0
        avg_amount = 0
        avg_turnover = 0

        if market_data is not None and len(market_data) > 0:
            valid_mc = market_data["market_cap"].dropna()
            if len(valid_mc) > 0:
                avg_mc = valid_mc.mean()
                median_mc = valid_mc.median()
                mc_cv = safe_divide(float(valid_mc.std()), abs(float(avg_mc)), default=None)

            if "volume" in market_data.columns:
                avg_volume = market_data["volume"].dropna().mean()

            if "amount" in market_data.columns:
                avg_amount = market_data["amount"].dropna().mean()

            if "turnover_rate" in market_data.columns:
                avg_turnover = market_data["turnover_rate"].dropna().mean()

        # 3️⃣ 从价格数据计算波动率
        price_vol = 0
        spread_est = 0

        if price_data:
            vols = []
            for code, df in price_data.items():
                if "close" in df.columns and len(df) > 1:
                    returns = df["close"].pct_change().dropna()
                    vol = calculate_volatility(returns)
                    if vol is not None:
                        vols.append(vol)

            if vols:
                price_vol = float(np.median(vols))

                # 估算买卖价差（基于波动率的简化模型）
                # 价差 ≈ 波动率 * sqrt(1/250) * 2 （粗略估计）
                spread_est = price_vol * np.sqrt(1 / 250) * 2

        # 4️⃣ 判断是否大盘股主导
        is_large_cap = avg_mc > 50e8  # >50亿市值算大盘

        # 5️⃣ 判断是否有低流动性股票
        has_illiquid = False
        if avg_amount > 0 and avg_amount < 0.5e8:  # <5000万/日成交额
            has_illiquid = True

        return MarketCharacteristics(
            stock_codes=stock_codes,
            n_stocks=n_stocks,
            market_board=dominant_board,
            board_distribution=board_distribution,
            avg_market_cap=avg_mc,
            median_market_cap=median_mc,
            market_cap_cv=mc_cv,
            avg_daily_volume=avg_volume,
            avg_daily_amount=avg_amount,
            avg_turnover_rate=avg_turnover,
            price_volatility=price_vol,
            spread_estimate=spread_est,
            is_large_cap_dominant=is_large_cap,
            has_illiquid_stocks=has_illiquid,
        )

    def _detect_board_distribution(self, stock_codes: List[str]) -> Dict[MarketBoard, float]:
        """检测各市场板块占比"""
        board_counts: Dict[MarketBoard, int] = {}

        for code in stock_codes:
            board = detect_market_board(code)
            board_counts[board] = board_counts.get(board, 0) + 1

        # 转换为占比
        total = sum(board_counts.values())
        if total == 0:
            return {MarketBoard.UNKNOWN: 1.0}

        distribution = {board: count / total for board, count in board_counts.items()}
        return distribution

    def recommend_slippage(
        self,
        stock_codes: List[str],
        strategy_turnover: float = 12.0,  # 年化换手率（默认12倍/年）
        market_data: Optional[pd.DataFrame] = None,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        user_preference: Optional[str] = None,  # "conservative"/"aggressive"/None
    ) -> SlippageRecommendation:
        """
        推荐最优滑点设置

        Args:
            stock_codes: 股票代码列表
            strategy_turnover: 策略年化换手率（倍数/年）
                            - 低频策略：2-6倍/年
                            - 中频策略：6-12倍/年
                            - 高频策略：12-50倍/年
            market_data: 市场数据（可选）
            price_data: 价格数据（可选）
            user_preference: 用户偏好 ("conservative"/"aggressive"/None自动)

        Returns:
            滑点推荐对象
        """
        # 分析市场特征
        chars = self.analyze_market(stock_codes, market_data, price_data)

        # 生成推荐配置
        recommended, conservative, aggressive, confidence, reasoning, warnings, tips = self._generate_recommendation(
            chars, strategy_turnover, user_preference
        )

        # 进行敏感性分析
        sensitivity = self._perform_sensitivity_analysis(recommended, strategy_turnover)

        return SlippageRecommendation(
            recommended_slippage=recommended,
            conservative_slippage=conservative,
            aggressive_slippage=aggressive,
            confidence=confidence,
            reasoning=reasoning,
            market_characteristics=chars,
            sensitivity_analysis=sensitivity,
            warnings=warnings,
            tips=tips,
        )

    def _generate_recommendation(
        self,
        chars: MarketCharacteristics,
        turnover: float,
        preference: Optional[str],
    ) -> Tuple[float, float, float, float, str, List[str], List[str]]:
        """生成推荐的核心逻辑"""

        warnings = []
        tips = []
        confidence_scores = []
        reasoning_parts = []

        # ==================== 1️⃣ 基础滑点（基于市场板块）====================
        base_config = self._BOARD_BASE_CONFIG.get(chars.market_board, self._BOARD_BASE_CONFIG[MarketBoard.UNKNOWN])
        base_slippage = base_config["base_slippage"]
        vol_factor = base_config["volatility_factor"]

        confidence_scores.append(0.9)
        reasoning_parts.append(f"{base_config['description']}基础滑点{base_slippage*100:.2f}%")

        # ==================== 2️⃣ 流动性调整 ====================
        liquidity_level = self._assess_liquidity(chars.avg_daily_amount)
        liquidity_adjustment = 0.0

        if liquidity_level == LiquidityLevel.VERY_LOW:
            liquidity_adjustment = 0.002  # +0.2%
            confidence_scores.append(0.85)
            reasoning_parts.append("极低流动性，增加滑点溢价")
            warnings.append("⚠️ 包含极低流动性股票，实际滑点可能显著高于估计")
            tips.append("💡 建议分批建仓以降低冲击成本")
        elif liquidity_level == LiquidityLevel.LOW:
            liquidity_adjustment = 0.001  # +0.1%
            confidence_scores.append(0.88)
            reasoning_parts.append("低流动性，适度增加滑点")
            tips.append("💡 注意控制单笔交易规模")
        elif liquidity_level == LiquidityLevel.HIGH:
            liquidity_adjustment = -0.0002  # -0.02%
            confidence_scores.append(0.92)
            reasoning_parts.append("高流动性，可适当降低滑点假设")

        # ==================== 3️⃣ 波动率调整 ====================
        vol_adjustment = 0.0
        if chars.price_volatility > 0:
            if chars.price_volatility > 0.4:  # 年化波动率>40%（高波动）
                vol_adjustment = 0.001 * vol_factor
                confidence_scores.append(0.87)
                reasoning_parts.append(f"高波动环境({chars.price_volatility*100:.1f}%/年)，增加波动性溢价")
                tips.append("💡 高波动期建议使用限价单而非市价单")
            elif chars.price_volatility > 0.3:  # 年化波动率>30%（中高波动）
                vol_adjustment = 0.0005 * vol_factor
                confidence_scores.append(0.89)
                reasoning_parts.append(f"中等偏高波动({chars.price_volatility*100:.1f}%/年)")
            else:
                confidence_scores.append(0.91)
                reasoning_parts.append(f"正常波动环境({chars.price_volatility*100:.1f}%/年)")

        # ==================== 4️⃣ 换手率惩罚因子 ====================
        turnover_penalty = 0.0
        if turnover > 0:
            # 换手率越高，有效滑点越大（因为交易更频繁）
            # 使用对数函数建模：penalty = log(1 + turnover/12) * base
            turnover_penalty = np.log1p(turnover / 12) * base_slippage * 0.5

            if turnover > 30:
                warnings.append(f"⚠️ 超高换手率策略(>{turnover:.0f}倍/年)，滑点影响将被显著放大")
                tips.append("💡 考虑降低换手率或使用算法交易")
            elif turnover > 12:
                tips.append("💡 中高频策略，建议关注交易执行质量")

            confidence_scores.append(0.88)
            reasoning_parts.append(f"换手率{turnover:.1f}倍/年，应用频率调整因子")

        # ==================== 5️⃣ 综合计算 ====================
        raw_slippage = (
            base_slippage
            + liquidity_adjustment
            + vol_adjustment
            + turnover_penalty
            + base_config.get("liquidity_premium", 0)
        )

        # 根据用户偏好调整
        if preference == "conservative":
            recommended = raw_slippage * 1.3  # 保守：+30%
            confidence_scores.append(0.95)
            reasoning_parts.append("用户偏好保守估计，上浮30%")
        elif preference == "aggressive":
            recommended = raw_slippage * 0.7  # 激进：-30%
            confidence_scores.append(0.82)
            reasoning_parts.append("用户偏好激进估计，下浮30%")
        else:
            recommended = raw_slippage
            confidence_scores.append(0.90)

        # 边界约束
        recommended = max(0.0005, min(0.01, recommended))  # 限制在[0.05%, 1%]区间

        # 保守/激进估计
        conservative = min(recommended * 1.5, 0.015)  # 上限1.5%
        aggressive = max(recommended * 0.6, 0.0003)  # 下限0.03%

        # 综合置信度
        overall_confidence = np.mean(confidence_scores)
        full_reasoning = "；".join(reasoning_parts)

        return recommended, conservative, aggressive, overall_confidence, full_reasoning, warnings, tips

    def _assess_liquidity(self, avg_daily_amount: float) -> LiquidityLevel:
        """评估流动性等级"""
        amount_yi = avg_daily_amount / 1e8  # 转换为亿元

        if amount_yi >= self._LIQUIDITY_THRESHOLDS[LiquidityLevel.HIGH]:
            return LiquidityLevel.HIGH
        elif amount_yi >= self._LIQUIDITY_THRESHOLDS[LiquidityLevel.MEDIUM]:
            return LiquidityLevel.MEDIUM
        elif amount_yi >= self._LIQUIDITY_THRESHOLDS[LiquidityLevel.LOW]:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW

    def _perform_sensitivity_analysis(
        self,
        base_slippage: float,
        turnover: float,
    ) -> Dict:
        """
        执行滑点敏感性分析

        分析不同滑点设置对策略收益的影响

        Args:
            base_slippage: 基准滑点
            turnover: 年化换手率

        Returns:
            敏感性分析结果字典
        """
        # 测试不同滑点水平
        test_slippages = [0.0, 0.001, 0.002, 0.003, 0.005, base_slippage]

        # 假设年化收益（示例值，实际应由回测引擎提供）
        assumed_annual_return = 0.15  # 15%

        results = {}
        for slip in test_slippages:
            # 简化的滑点成本模型：cost = slippage * turnover * 2（买入+卖出）
            annual_slippage_cost = slip * turnover * 2
            net_return = assumed_annual_return - annual_slippage_cost
            return_decay = safe_divide(annual_slippage_cost, assumed_annual_return, default=float("inf")) * 100

            results[f"{slip*100:.2f}%"] = {
                "slippage_rate": slip,
                "estimated_annual_cost": round(annual_slippage_cost, 4),
                "net_return_estimate": round(net_return, 4),
                "return_decay_pct": round(return_decay, 2),
            }

        # 计算敏感性指标
        base_cost = base_slippage * turnover * 2
        sensitivity = {
            "test_scenarios": results,
            "base_scenario": {
                "slippage": base_slippage,
                "estimated_annual_impact": round(base_cost, 4),
                "impact_per_1bps_change": round(turnover * 2 * 0.0001, 4),  # 每1个基点变化的影响
            },
            "sensitivity_level": self._classify_sensitivity(base_cost, assumed_annual_return),
            "recommendation": self._get_sensitivity_recommendation(turnover, base_slippage),
        }

        return sensitivity

    def _classify_sensitivity(self, annual_cost: float, annual_return: float) -> str:
        """分类敏感性水平"""
        if abs(annual_return) < 1e-10:
            return "unknown"

        cost_ratio = safe_divide(abs(annual_cost), abs(annual_return), default=None)
        if cost_ratio is None:
            return "unknown"

        if cost_ratio < 0.1:
            return "low"  # 低敏感：<10%收益被侵蚀
        elif cost_ratio < 0.25:
            return "medium"  # 中敏感：10-25%
        elif cost_ratio < 0.5:
            return "high"  # 高敏感：25-50%
        else:
            return "very_high"  # 极高敏感：>50%

    def _get_sensitivity_recommendation(self, turnover: float, slippage: float) -> str:
        """根据敏感性给出建议"""
        annual_cost = slippage * turnover * 2

        if turnover > 24 and annual_cost > 0.05:
            return "超高换手率+高滑点成本，强烈建议优化执行算法或降低换手"
        elif turnover > 12 and annual_cost > 0.03:
            return "中高频策略需关注交易成本，建议使用VWAP/TWAP等算法交易"
        elif annual_cost > 0.02:
            return "滑点成本适中，可通过优化下单时机降低成本"
        else:
            return "滑点成本较低，当前设置合理"

    def get_recommendation_summary(self, rec: SlippageRecommendation) -> str:
        """生成人类可读的推荐报告"""
        chars = rec.market_characteristics

        lines = [
            "# 🎯 智能交易滑点推荐报告",
            "",
            "## 📊 市场概况",
            f"- **股票数量**: {chars.n_stocks} 只",
            f"- **市场板块**: {self._get_board_name(chars.market_board)} "
            f"({self._get_board_distribution_str(chars.board_distribution)})",
            (
                f"- **平均市值**: {chars.avg_market_cap/1e8:.2f} 亿"
                if chars.avg_market_cap > 0
                else "- **平均市值**: 数据不足"
            ),
            (
                f"- **日均成交额**: {chars.avg_daily_amount/1e8:.2f} 亿"
                if chars.avg_daily_amount > 0
                else "- **日均成交额**: 数据不足"
            ),
            (
                f"- **价格波动率**: {chars.price_volatility*100:.1f}%/年"
                if chars.price_volatility > 0
                else "- **价格波动率**: 数据不足"
            ),
            "",
            f"## 🎯 推荐滑点设置 (置信度: {rec.confidence*100:.0f}%)",
            "",
            "| 场景 | 滑点率 | 说明 |",
            "|------|--------|------|",
            f"| **推荐值** | **{rec.recommended_slippage*100:.3f}%** | 基于市场特征的最优估计 |",
            f"| 保守估计 | {rec.conservative_slippage*100:.3f}% | 不利情况下的上限 |",
            f"| 激进估计 | {rec.aggressive_slippage*100:.3f}% | 理想情况下的下限 |",
            "",
            "## 💡 推荐理由",
            f"{rec.reasoning}",
            "",
            "## 📈 敏感性分析",
            "",
            "### 不同滑点下的预估影响（假设年化收益15%，回测换手率需实际计算）",
            "",
            "| 滑点设置 | 年化成本 | 净收益估计 | 收益衰减 |",
            "|---------|---------|-----------|---------|",
        ]

        for scenario, data in rec.sensitivity_analysis["test_scenarios"].items():
            lines.append(
                f"| {scenario} | {data['estimated_annual_cost']*100:.2f}% "
                f"| {data['net_return_estimate']*100:.2f}% "
                f"| {data['return_decay_pct']:.1f}% |"
            )

        lines.extend(
            [
                "",
                f"**敏感性等级**: {rec.sensitivity_analysis['sensitivity_level'].upper()}",
                f"**优化建议**: {rec.sensitivity_analysis['recommendation']}",
            ]
        )

        if rec.warnings:
            lines.extend(["", "## ⚠️ 风险警告"])
            for w in rec.warnings:
                lines.append(f"- {w}")

        if rec.tips:
            lines.extend(["", "## ✨ 优化建议"])
            for t in rec.tips:
                lines.append(f"- {t}")

        return "\n".join(lines)

    def _get_board_name(self, board: MarketBoard) -> str:
        names = {
            MarketBoard.MAIN: "主板",
            MarketBoard.CHINEXT: "创业板",
            MarketBoard.STAR: "科创板",
            MarketBoard.BSE: "北交所",
            MarketBoard.UNKNOWN: "混合板块",
        }
        return names.get(board, "未知")

    def _get_board_distribution_str(self, distribution: Dict[MarketBoard, float]) -> str:
        parts = [
            f"{self._get_board_name(k)}{v*100:.0f}%"
            for k, v in sorted(distribution.items(), key=lambda x: -x[1])
            if v > 0.05
        ]
        return ", ".join(parts) if parts else "未知"

    def _empty_characteristics(self, stock_codes: List[str]) -> MarketCharacteristics:
        return MarketCharacteristics(
            stock_codes=stock_codes or [],
            n_stocks=len(stock_codes) if stock_codes else 0,
            market_board=MarketBoard.UNKNOWN,
            board_distribution={},
            avg_market_cap=0,
            median_market_cap=0,
            market_cap_cv=0,
            avg_daily_volume=0,
            avg_daily_amount=0,
            avg_turnover_rate=0,
            price_volatility=0,
            spread_estimate=0,
            is_large_cap_dominant=False,
            has_illiquid_stocks=False,
        )


# 全局实例
smart_slippage_detector = SmartSlippageDetector()
