"""
未来函数（Look-ahead Bias）检测器

基于统计特征的自动化检测引擎，用于识别因子计算中可能存在的未来信息泄漏。

核心检测维度：
1. IC 异常检测 — IC/IR 远超正常范围
2. 完美排序检测 — 因子排名与收益排名的相关性异常高
3. 自相关异常 — 未来函数通常导致极高的序列自相关
4. 分层收益异常 — 分层收益单调性过强或 spread 过大
5. 时序一致性检验 — 前后段 IC 差异过大（过拟合信号）
6. 回测真实性校验 — 年化收益/夏普/胜率等指标异常

设计原则：
- 零误杀：宁可漏报，不误报正常因子
- 可解释：每个检测结果都附带明确的判定理由
- 可配置：阈值支持自定义，适应不同市场环境
- 高性能：纯 pandas/numpy 向量化操作

参考：
- Grinold & Kahn 《Active Portfolio Management》
- BigQuant 量化平台《因子清洗与预处理》
- JoinQuant 因子研究最佳实践
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.stats import spearmanr

from backend.utils.safe_math import safe_divide, safe_ir
from backend.utils.ic_calculator import calculate_rolling_ic

logger = logging.getLogger(__name__)


class BiasRiskLevel(str, Enum):
    """风险等级枚举"""
    SAFE = "safe"                # 安全，未检测到明显问题
    LOW = "low"                  # 低风险，存在轻微异常信号
    MEDIUM = "medium"            # 中等风险，多个检测项触发警告
    HIGH = "high"                # 高风险，强烈怀疑存在未来函数
    CRITICAL = "critical"        # 严重风险，几乎可以确认存在未来函数


@dataclass
class BiasCheckResult:
    """单个检测项的结果"""
    check_name: str               # 检测项名称
    passed: bool                 # 是否通过（True=安全，False=可疑）
    value: float                 # 检测到的实际值
    threshold: float             # 判定阈值
    severity: str                # 严重程度: "info" / "warning" / "error" / "critical"
    message: str                 # 人类可读的说明
    detail: Dict[str, Any] = field(default_factory=dict)  # 详细数据


@dataclass
class LookaheadBiasDetectionResult:
    """未来函数检测结果"""
    has_bias: bool               # 是否检测到未来函数
    risk_level: BiasRiskLevel    # 综合风险等级
    risk_score: float            # 风险评分 (0-100)，越高越危险
    checks: List[BiasCheckResult]  # 所有检测项的详细结果
    summary: str                 # 人类可读的摘要
    recommendations: List[str]   # 改进建议
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元信息


class LookaheadBiasDetector:
    """
    未来函数（Look-ahead Bias）统计检测器

    通过多维度统计特征识别因子中的未来信息泄漏。
    适用于所有类型的因子（公式因子、技术因子、基本面因子等）。

    使用方式：
        detector = LookaheadBiasDetector()
        result = detector.detect(factor_values=factor_series, return_values=return_series)
        if result.has_bias:
            logger.info(f"风险等级: {result.risk_level}")
            logger.info(result.summary)
    """

    # ========== 默认阈值配置（基于A股市场经验值）==========
    DEFAULT_THRESHOLDS = {
        # IC 相关
        "ic_abs_max": 0.15,              # IC 绝对值上限（正常因子通常 < 0.08）
        "ir_max": 3.0,                   # IR 上限（正常因子通常 < 2.0）
        "ic_positive_ratio_max": 0.85,   # IC>0 占比上限

        # 排名相关
        "rank_corr_max": 0.50,           # 排名相关系数上限
        "rank_ic_abs_max": 0.15,         # Rank IC 绝对值上限

        # 自相关
        "autocorr_lag1_max": 0.99,       # 一阶自相关系数上限
        "autocorr_lag5_max": 0.95,       # 五阶自相关系数上限

        # 分层收益
        "quantile_spread_daily_max": 0.05,  # 日均分层收益差上限 (5%)
        "quantile_monotonicity_min": 0.95,   # 分层单调性下限（过高也可疑）

        # 时序一致性
        "ic_split_ratio_max": 5.0,       # 前后半段 IC 比值上限

        # 回测指标
        "annual_return_max": 5.0,        # 年化收益率上限 (500%)
        "sharpe_max": 10.0,             # 夏普比率上限
        "win_rate_max": 0.95,           # 胜率上限
        "max_drawdown_min": 0.001,      # 最大回撤下限 (0.1%，过低可疑)
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        strict_mode: bool = False,
    ):
        """
        初始化检测器

        Args:
            thresholds: 自定义阈值字典，覆盖默认值
            strict_mode: 严格模式（降低阈值，更敏感但可能增加误报）
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

        if strict_mode:
            # 严格模式下收紧阈值约 30-50%
            self.thresholds["ic_abs_max"] *= 0.6
            self.thresholds["ir_max"] *= 0.6
            self.thresholds["rank_corr_max"] *= 0.6
            self.thresholds["autocorr_lag1_max"] *= 0.995
            self.thresholds["annual_return_max"] *= 0.5
            self.thresholds["sharpe_max"] *= 0.5

    def detect(
        self,
        factor_values: pd.Series,
        return_values: Optional[pd.Series] = None,
        factor_name: str = "factor",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> LookaheadBiasDetectionResult:
        """
        执行完整的未来函数检测流程

        Args:
            factor_values: 因子值序列（索引为日期）
            return_values: 收益率序列（可选，有则启用更多检测）
            factor_name: 因子名称（用于日志和报告）
            extra_context: 额外上下文信息（如回测指标、分层收益等）

        Returns:
            LookaheadBiasDetectionResult: 完整的检测结果
        """
        checks: List[BiasCheckResult] = []
        context = extra_context or {}

        logger.info(f"[未来函数检测] 开始检测因子: {factor_name}, 样本数: {len(factor_values)}")

        # ---- 基础校验 ----
        factor_clean = factor_values.dropna()
        if len(factor_clean) < 20:
            logger.warning(f"[未来函数检测] 因子 {factor_name} 有效数据不足 (n={len(factor_clean)} < 20)，跳过检测")
            return self._insufficient_data_result(factor_name, len(factor_clean))

        # ---- 维度 1: IC 异常检测 ----
        if return_values is not None and len(return_values) > 0:
            checks.append(self._check_ic_magnitude(factor_clean, return_values))
            checks.append(self._check_ir_magnitude(factor_clean, return_values))
            checks.append(self._check_ic_positive_ratio(factor_clean, return_values))

        # ---- 维度 2: 完美排序检测 ----
        if return_values is not None and len(return_values) > 0:
            checks.append(self._check_rank_correlation(factor_clean, return_values))
            checks.append(self._check_rank_ic_magnitude(factor_clean, return_values))

        # ---- 维度 3: 自相关异常检测 ----
        checks.append(self._check_autocorrelation_lag1(factor_clean))
        checks.append(self._check_autocorrelation_lag5(factor_clean))

        # ---- 维度 4: 分布异常检测 ----
        checks.append(self._check_distribution_anomaly(factor_clean))
        checks.append(self._check_value_constancy(factor_clean))

        # ---- 维度 5: 分层收益异常（需要额外上下文）----
        quantile_returns = context.get("quantile_returns")
        if quantile_returns is not None:
            checks.append(self._check_quantile_spread(quantile_returns))
            checks.append(self._check_quantile_monotonicity(quantile_returns))

        # ---- 维度 6: 回测指标异常（需要额外上下文）----
        backtest_metrics = context.get("backtest_metrics")
        if backtest_metrics is not None:
            checks.append(self._check_backtest_metrics(backtest_metrics))

        # ---- 维度 7: 时序一致性检验 ----
        if return_values is not None and len(return_values) >= 40:
            checks.append(self._check_temporal_consistency(factor_clean, return_values))

        # ---- 汇总结果 ----
        failed_checks = [c for c in checks if not c.passed]
        critical_count = sum(1 for c in failed_checks if c.severity == "critical")
        error_count = sum(1 for c in failed_checks if c.severity == "error")

        risk_score = self._calculate_risk_score(checks)
        risk_level = self._determine_risk_level(risk_score, critical_count, error_count)
        has_bias = risk_level in (BiasRiskLevel.HIGH, BiasRiskLevel.CRITICAL)
        summary = self._generate_summary(factor_name, checks, risk_level)
        recommendations = self._generate_recommendations(failed_checks)

        logger.info(
            f"[未来函数检测] 检测完成: {factor_name}, "
            f"风险等级={risk_level.value}, 评分={risk_score:.1f}, "
            f"通过={len(checks)-len(failed_checks)}/{len(checks)}"
        )

        return LookaheadBiasDetectionResult(
            has_bias=has_bias,
            risk_level=risk_level,
            risk_score=risk_score,
            checks=checks,
            summary=summary,
            recommendations=recommendations,
            metadata={
                "factor_name": factor_name,
                "sample_size": len(factor_clean),
                "n_checks": len(checks),
                "n_failed": len(failed_checks),
                "thresholds_used": self.thresholds,
            },
        )

    def detect_cross_sectional(
        self,
        factor_df: pd.DataFrame,
        return_df: pd.DataFrame,
        date_column: str = "date",
        stock_column: str = "stock_code",
        factor_name: str = "factor",
    ) -> LookaheadBiasDetectionResult:
        """
        横截面模式下的未来函数检测

        适用于多股票因子的检测，逐日计算横截面 IC 然后分析其统计特征。

        Args:
            factor_df: 因子数据 DataFrame（必须包含 date, stock_code, 因子值列）
            return_df: 收益率 DataFrame（必须包含 date, stock_code, return 列）
            date_column: 日期列名
            stock_column: 股票代码列名
            factor_name: 因子名称

        Returns:
            LookaheadBiasDetectionResult: 检测结果
        """
        checks: List[BiasCheckResult] = []

        # 合并数据
        merged = factor_df.merge(return_df, on=[date_column, stock_column], how="inner")
        if len(merged) < 50:
            return self._insufficient_data_result(factor_name, len(merged))

        # 逐日计算横截面 IC
        daily_ics = []
        daily_rank_ics = []
        daily_spreads = []  # Q1 vs Q5 的收益差

        for date, group in merged.groupby(date_column):
            if len(group) < 5:
                continue

            fv = group[factor_name] if factor_name in group.columns else group.iloc[:, 2]
            ret = group["return"] if "return" in group.columns else group.iloc[:, 3]

            valid = fv.notna() & ret.notna()
            if valid.sum() < 5:
                continue

            fv_valid = fv[valid]
            ret_valid = ret[valid]

            # 防止常数值导致 corr 返回 NaN
            if fv_valid.nunique() < 2 or ret_valid.nunique() < 2:
                continue

            try:
                ic_result = spearmanr(fv_valid, ret_valid)
                ic = ic_result[0] if not np.isnan(ic_result[0]) else np.nan
                rank_ic = ic  # Spearman IS rank correlation（规则7.25）
            except Exception as e:
                logger.debug(f"IC计算异常: {e}")
                continue

            if pd.notna(ic) and not np.isinf(ic):
                daily_ics.append(ic)
            if pd.notna(rank_ic) and not np.isinf(rank_ic):
                daily_rank_ics.append(rank_ic)

            # 计算当日分层收益差
            try:
                q5_ret = ret_valid[fv_valid.rank(pct=True) >= 0.8].mean()
                q1_ret = ret_valid[fv_valid.rank(pct=True) <= 0.2].mean()
                if pd.notna(q5_ret) and pd.notna(q1_ret):
                    daily_spreads.append(q5_ret - q1_ret)
            except Exception as e:
                logger.debug(f"计算分层收益差失败: {e}")

        ic_series = pd.Series(daily_rank_ics) if daily_rank_ics else pd.Series(daily_ics) if daily_ics else pd.Series()

        # 检测 1: 横截面 IC 均值过高
        if len(ic_series) > 0:
            mean_ic = ic_series.mean()
            abs_mean_ic = abs(mean_ic)
            threshold = self.thresholds["ic_abs_max"]
            checks.append(BiasCheckResult(
                check_name="cross_sectional_ic_mean",
                passed=abs_mean_ic < threshold,
                value=float(abs_mean_ic),
                threshold=threshold,
                severity="critical" if abs_mean_ic > threshold * 2 else "error",
                message=(
                    f"横截面IC均值={mean_ic:.4f}，"
                    f"{'远超' if abs_mean_ic > threshold * 2 else '超过'}正常范围(±{threshold:.2f})。"
                    f"正常因子IC通常在±0.02~±0.08之间。"
                ),
                detail={"mean_ic": float(mean_ic), "n_days": len(ic_series)},
            ))

            # 检测 2: IR 过高
            ic_std = ic_series.std()
            ir = safe_ir(float(abs(mean_ic)), float(ic_std), default=None)
            if ir is not None:
                ir_threshold = self.thresholds["ir_max"]
                checks.append(BiasCheckResult(
                    check_name="cross_sectional_ir",
                    passed=ir < ir_threshold,
                    value=float(ir),
                    threshold=ir_threshold,
                    severity="critical" if ir > ir_threshold * 2 else "error",
                    message=(
                        f"横截面IR={ir:.2f}，"
                        f"{'极高' if ir > ir_threshold * 2 else '偏高'}。"
                        f"正常因子IR通常<2.0，IR>3需重点审查。"
                    ),
                    detail={"ir": float(ir), "ic_std": float(ic_std)},
                ))

            # 检测 3: IC>0 占比过高
            pos_ratio = (ic_series > 0).mean()
            pos_threshold = self.thresholds["ic_positive_ratio_max"]
            checks.append(BiasCheckResult(
                check_name="ic_positive_ratio",
                passed=(1 - pos_threshold) <= pos_ratio <= pos_threshold,
                value=float(pos_ratio),
                threshold=pos_threshold,
                severity="warning",
                message=(
                    f"IC>0占比={pos_ratio:.1%}，接近100%或0%意味着因子方向过于确定。"
                    f"正常因子通常在50%~80%之间波动。"
                ),
                detail={"positive_ratio": float(pos_ratio), "n_days": len(ic_series)},
            ))

        # 检测 4: 日均分层收益差过大
        if daily_spreads:
            avg_spread = np.mean(daily_spreads)
            spread_threshold = self.thresholds["quantile_spread_daily_max"]
            checks.append(BiasCheckResult(
                check_name="daily_quantile_spread",
                passed=abs(avg_spread) < spread_threshold,
                value=float(abs(avg_spread)),
                threshold=spread_threshold,
                severity="error" if abs(avg_spread) > spread_threshold * 3 else "warning",
                message=(
                    f"日均Top-Bottom收益差={avg_spread:.4f}({avg_spread*100:.2f}%)，"
                    f"{'极大' if abs(avg_spread) > spread_threshold * 3 else '偏大'}。"
                    f"日均>5%需警惕未来函数。"
                ),
                detail={"avg_spread": float(avg_spread), "n_days": len(daily_spreads)},
            ))

        # 汇总
        failed_checks = [c for c in checks if not c.passed]
        critical_count = sum(1 for c in failed_checks if c.severity == "critical")
        error_count = sum(1 for c in failed_checks if c.severity == "error")

        risk_score = self._calculate_risk_score(checks)
        risk_level = self._determine_risk_level(risk_score, critical_count, error_count)
        has_bias = risk_level in (BiasRiskLevel.HIGH, BiasRiskLevel.CRITICAL)
        summary = self._generate_summary(factor_name, checks, risk_level)
        recommendations = self._generate_recommendations(failed_checks)

        return LookaheadBiasDetectionResult(
            has_bias=has_bias,
            risk_level=risk_level,
            risk_score=risk_score,
            checks=checks,
            summary=summary,
            recommendations=recommendations,
            metadata={
                "factor_name": factor_name,
                "mode": "cross_sectional",
                "n_days": len(ic_series),
                "thresholds_used": self.thresholds,
            },
        )

    # ==================== 单项检测方法 ====================

    def _check_ic_magnitude(
        self, factor: pd.Series, returns: pd.Series
    ) -> BiasCheckResult:
        """检测 1a: IC 绝对值是否异常偏高（使用Spearman秩相关）"""
        aligned = pd.DataFrame({"f": factor, "r": returns}).dropna()
        if len(aligned) < 10:
            return self._skip_result("ic_magnitude", "数据量不足")

        ic_result = spearmanr(aligned["f"], aligned["r"])
        ic = ic_result[0] if not np.isnan(ic_result[0]) else 0.0
        abs_ic = abs(ic) if pd.notna(ic) else 0.0
        threshold = self.thresholds["ic_abs_max"]

        return BiasCheckResult(
            check_name="ic_magnitude",
            passed=abs_ic < threshold,
            value=float(abs_ic),
            threshold=threshold,
            severity="critical" if abs_ic > threshold * 2 else "error",
            message=(
                f"IC={ic:.4f}，绝对值{abs_ic:.4f}"
                f"{'远超' if abs_ic > threshold * 2 else '超过'}正常阈值(±{threshold:.2f})。"
                f"A股正常因子IC通常在±0.02~±0.08之间，IC>0.15极罕见。"
            ),
            detail={"ic": float(ic), "n_samples": len(aligned)},
        )

    def _check_ir_magnitude(
        self, factor: pd.Series, returns: pd.Series
    ) -> BiasCheckResult:
        """检测 1b: IR 是否异常偏高（基于滚动Spearman IC）"""
        aligned = pd.DataFrame({"f": factor, "r": returns}).dropna()
        if len(aligned) < 40:
            return self._skip_result("ir_magnitude", "数据量不足(<40)")

        window = min(20, len(aligned) // 2)

        # 使用统一入口计算滚动Spearman IC（符合规则0和规则5）
        rolling_ic = calculate_rolling_ic(
            aligned["f"], aligned["r"],
            window=window, method="spearman",
        )
        rolling_ic = rolling_ic.replace([np.inf, -np.inf], np.nan).dropna()

        if len(rolling_ic) < 10:
            return self._skip_result("ir_magnitude", "有效滚动IC不足")

        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()
        ir = safe_ir(float(abs(ic_mean)), float(ic_std), default=None)
        if ir is None:
            return self._skip_result("ir_magnitude", "IR无法计算（IC标准差为0）")
        threshold = self.thresholds["ir_max"]

        return BiasCheckResult(
            check_name="ir_magnitude",
            passed=ir < threshold,
            value=float(ir),
            threshold=threshold,
            severity="critical" if ir > threshold * 2 else "error",
            message=(
                f"IR={ir:.2f}(IC均值={ic_mean:.4f}, IC标准差={ic_std:.4f})，"
                f"{'极高' if ir > threshold * 2 else '偏高'}。"
                f"正常因子IR通常<2.0，IR>3需审查是否存在前视偏差。"
            ),
            detail={"ir": float(ir), "ic_mean": float(ic_mean), "ic_std": float(ic_std), "window": window},
        )

    def _check_ic_positive_ratio(
        self, factor: pd.Series, returns: pd.Series
    ) -> BiasCheckResult:
        """检测 1c: IC 正向比例是否异常（接近 100% 或 0%，使用Spearman IC）"""
        aligned = pd.DataFrame({"f": factor, "r": returns}).dropna()
        if len(aligned) < 20:
            return self._skip_result("ic_positive_ratio", "数据量不足")

        window = min(20, len(aligned) // 2)

        # 使用统一入口计算滚动Spearman IC（符合规则0和规则5）
        rolling_ic = calculate_rolling_ic(
            aligned["f"], aligned["r"],
            window=window, method="spearman",
        )
        rolling_ic = rolling_ic.replace([np.inf, -np.inf], np.nan).dropna()

        if len(rolling_ic) < 5:
            return self._skip_result("ic_positive_ratio", "有效滚动IC不足")

        pos_ratio = (rolling_ic > 0).mean()
        neg_ratio = (rolling_ic < 0).mean()
        extreme_ratio = max(pos_ratio, neg_ratio)
        threshold = self.thresholds["ic_positive_ratio_max"]

        return BiasCheckResult(
            check_name="ic_positive_ratio",
            passed=extreme_ratio < threshold,
            value=float(extreme_ratio),
            threshold=threshold,
            severity="warning",
            message=(
                f"IC同向占比={extreme_ratio:.1%}({'正向' if pos_ratio > neg_ratio else '负向'})，"
                f"接近100%意味着因子预测方向过于确定，需检查是否有泄漏。"
            ),
            detail={"positive_ratio": float(pos_ratio), "negative_ratio": float(neg_ratio)},
        )

    def _check_rank_correlation(
        self, factor: pd.Series, returns: pd.Series
    ) -> BiasCheckResult:
        """检测 2a: Spearman 排名相关系数（完美排名检测）"""
        aligned = pd.DataFrame({"f": factor, "r": returns}).dropna()
        if len(aligned) < 10:
            return self._skip_result("rank_correlation", "数据量不足")

        rank_corr = aligned["f"].rank().corr(aligned["r"].rank())
        abs_rc = abs(rank_corr) if pd.notna(rank_corr) else 0.0
        threshold = self.thresholds["rank_corr_max"]

        return BiasCheckResult(
            check_name="rank_correlation",
            passed=abs_rc < threshold,
            value=float(abs_rc),
            threshold=threshold,
            severity="critical" if abs_rc > threshold * 1.5 else "error",
            message=(
                f"Spearman排名相关系数={rank_corr:.4f}，"
                f"{'极高' if abs_rc > threshold * 1.5 else '偏高'}。"
                f"正常因子排名相关性通常<0.1，>0.3需重点排查。"
            ),
            detail={"rank_corr": float(rank_corr)},
        )

    def _check_rank_ic_magnitude(
        self, factor: pd.Series, returns: pd.Series
    ) -> BiasCheckResult:
        """检测 2b: Rank IC 均值"""
        aligned = pd.DataFrame({"f": factor, "r": returns}).dropna()
        if len(aligned) < 20:
            return self._skip_result("rank_ic_magnitude", "数据量不足")

        window = min(20, len(aligned) // 2)

        # 使用统一入口计算滚动Spearman IC（符合规则0和规则5）
        rolling_rank_ic = calculate_rolling_ic(
            aligned["f"], aligned["r"],
            window=window, method="spearman",
        )
        rolling_rank_ic = rolling_rank_ic.replace([np.inf, -np.inf], np.nan).dropna()

        if len(rolling_rank_ic) < 5:
            return self._skip_result("rank_ic_magnitude", "有效Rank IC不足")

        mean_rank_ic = abs(rolling_rank_ic.mean())
        threshold = self.thresholds["rank_ic_abs_max"]

        return BiasCheckResult(
            check_name="rank_ic_magnitude",
            passed=mean_rank_ic < threshold,
            value=float(mean_rank_ic),
            threshold=threshold,
            severity="error" if mean_rank_ic > threshold * 1.5 else "warning",
            message=(
                f"Rank IC均值={mean_rank_ic:.4f}，{'偏高' if mean_rank_ic > threshold else '正常'}。"
            ),
            detail={"mean_rank_ic": float(mean_rank_ic)},
        )

    def _check_autocorrelation_lag1(self, factor: pd.Series) -> BiasCheckResult:
        """检测 3a: 一阶自相关系数（未来函数常导致接近1的自相关）"""
        if len(factor) < 20:
            return self._skip_result("autocorrelation_lag1", "数据量不足")

        ac1 = factor.autocorr(lag=1)
        abs_ac1 = abs(ac1) if pd.notna(ac1) else 0.0
        threshold = self.thresholds["autocorr_lag1_max"]

        return BiasCheckResult(
            check_name="autocorrelation_lag1",
            passed=abs_ac1 < threshold,
            value=float(abs_ac1),
            threshold=threshold,
            severity="error" if abs_ac1 > 0.999 else "warning",
            message=(
                f"一阶自相关系数={ac1:.6f}，"
                f"{'极高(接近1)' if abs_ac1 > 0.999 else '偏高'}。"
                f"未来函数常导致因子值几乎不变或完美复制其他序列。"
            ),
            detail={"autocorr_lag1": float(ac1)},
        )

    def _check_autocorrelation_lag5(self, factor: pd.Series) -> BiasCheckResult:
        """检测 3b: 五阶自相关系数"""
        if len(factor) < 30:
            return self._skip_result("autocorrelation_lag5", "数据量不足")

        ac5 = factor.autocorr(lag=5)
        abs_ac5 = abs(ac5) if pd.notna(ac5) else 0.0
        threshold = self.thresholds["autocorr_lag5_max"]

        return BiasCheckResult(
            check_name="autocorrelation_lag5",
            passed=abs_ac5 < threshold,
            value=float(abs_ac5),
            threshold=threshold,
            severity="warning",
            message=(
                f"五阶自相关系数={ac5:.6f}，{'偏高' if abs_ac5 > threshold else '正常'}。"
            ),
            detail={"autocorr_lag5": float(ac5)},
        )

    def _check_distribution_anomaly(self, factor: pd.Series) -> BiasCheckResult:
        """检测 4a: 分布异常（极端峰度、异常值比例）"""
        if len(factor) < 30:
            return self._skip_result("distribution_anomaly", "数据量不足")

        kurtosis = factor.kurtosis()
        skewness = factor.skew()

        # 异常值比例（超出3σ的范围）
        std = factor.std()
        mean = factor.mean()
        if std > 0:
            outlier_ratio = ((factor < mean - 3 * std) | (factor > mean + 3 * std)).mean()
        else:
            outlier_ratio = 0.0

        # 峰度过高（>50 极端）或过低（<-3 几乎均匀分布）
        kurtosis_abnormal = abs(kurtosis) > 50 or kurtosis < -3
        outlier_abnormal = outlier_ratio > 0.3

        is_abnormal = kurtosis_abnormal or outlier_abnormal

        return BiasCheckResult(
            check_name="distribution_anomaly",
            passed=not is_abnormal,
            value=float(abs(kurtosis)),
            threshold=50.0,
            severity="warning",
            message=(
                f"峰度={kurtosis:.2f}, 偏度={skewness:.2f}, 异常值比例={outlier_ratio:.1%}。"
                f"{'分布异常' if is_abnormal else '分布正常'}。"
            ),
            detail={
                "kurtosis": float(kurtosis),
                "skewness": float(skewness),
                "outlier_ratio": float(outlier_ratio),
            },
        )

    def _check_value_constancy(self, factor: pd.Series) -> BiasCheckResult:
        """检测 4b: 因子值恒定性（完全不变或每日全变都是异常信号）"""
        if len(factor) < 10:
            return self._skip_result("value_constancy", "数据量不足")

        unique_ratio = factor.nunique() / len(factor)
        day_over_day_change = factor.diff().abs()
        avg_change = day_over_day_change.mean()
        std_factor = factor.std()

        # 完全不变或几乎不变
        is_constant = unique_ratio < 0.01 or (std_factor < 1e-10 and len(factor) > 10)
        # 每日变化率异常一致（可能是复制了某个带噪声的未来序列）
        is_too_stable = (
            std_factor > 1e-10
            and avg_change > 0
            and safe_divide(float(day_over_day_change.std()), float(avg_change), default=0.0) < 0.01
        )

        is_abnormal = is_constant or is_too_stable

        return BiasCheckResult(
            check_name="value_constancy",
            passed=not is_abnormal,
            value=float(unique_ratio),
            threshold=0.01,
            severity="error" if is_constant else "warning",
            message=(
                f"唯一值比例={unique_ratio:.4f}, "
                f"{'因子值几乎不变，可能存在计算错误' if is_constant else ''}"
                f"{'变化过于规律，需检查数据来源' if is_too_stable else ''}"
                f"{'正常' if not is_abnormal else ''}".strip()
            ),
            detail={
                "unique_ratio": float(unique_ratio),
                "std": float(std_factor),
                "avg_daily_change": float(avg_change),
            },
        )

    def _check_quantile_spread(self, quantile_returns: Dict[str, Any]) -> BiasCheckResult:
        """检测 5a: 分层收益差异（Top-Bottom Spread）"""
        # quantile_returns 格式: {"Q1": series, "Q2": series, ...} 或类似
        try:
            if isinstance(quantile_returns, dict):
                keys = sorted(quantile_returns.keys())
                if len(keys) >= 2:
                    q_low = quantile_returns[keys[0]]
                    q_high = quantile_returns[keys[-1]]

                    # 转为 Series 并计算均值收益
                    if isinstance(q_low, (pd.Series, list, np.ndarray)):
                        low_mean = np.nanmean(q_low)
                        high_mean = np.nanmean(q_high)
                        spread = high_mean - low_mean
                        abs_spread = abs(spread)
                        threshold = self.thresholds["quantile_spread_daily_max"]

                        return BiasCheckResult(
                            check_name="quantile_spread",
                            passed=abs_spread < threshold,
                            value=float(abs_spread),
                            threshold=threshold,
                            severity="critical" if abs_spread > threshold * 5 else "error",
                            message=(
                                f"分层收益差(Q{keys[-1]}-Q{keys[0]})={spread:.4f}({spread*100:.2f}%/天)，"
                                f"{'极其异常' if abs_spread > threshold * 5 else '偏大'}。"
                                f"正常分层收益差通常<1%/天。"
                            ),
                            detail={"spread": float(spread), "q_low_mean": float(low_mean), "q_high_mean": float(high_mean)},
                        )
        except Exception as e:
            logger.debug(f"分层收益检测失败: {e}")

        return self._skip_result("quantile_spread", "数据格式不支持")

    def _check_quantile_monotonicity(self, quantile_returns: Dict[str, Any]) -> BiasCheckResult:
        """检测 5b: 分层收益单调性（过强也可疑——完美单调可能意味着泄漏）"""
        try:
            if isinstance(quantile_returns, dict):
                keys = sorted(quantile_returns.keys())
                if len(keys) >= 3:
                    means = [np.nanmean(quantile_returns[k]) for k in keys]
                    # 计算单调性：相邻层之间的方向一致性（双向检测）
                    directions = [means[i + 1] - means[i] for i in range(len(means) - 1)]
                    increase_ratio = sum(1 for d in directions if d > 0) / len(directions)
                    decrease_ratio = sum(1 for d in directions if d < 0) / len(directions)
                    same_direction = max(increase_ratio, decrease_ratio)
                    # 同向比例过高（>0.95 且各层差距很大）是可疑信号
                    max_gap = max(means) - min(means) if means else 0

                    is_too_perfect = same_direction > self.thresholds["quantile_monotonicity_min"] and max_gap > 0.02

                    return BiasCheckResult(
                        check_name="quantile_monotonicity",
                        passed=not is_too_perfect,
                        value=float(same_direction),
                        threshold=self.thresholds["quantile_monotonicity_min"],
                        severity="warning",
                        message=(
                            f"分层收益单调性={same_direction:.1%}(各层均值={means})，"
                            f"{'近乎完美单调且差距较大，需注意' if is_too_perfect else '正常'}。"
                        ),
                        detail={"monotonicity": float(same_direction), "layer_means": means, "max_gap": float(max_gap)},
                    )
        except Exception as e:
            logger.debug(f"分层单调性检测失败: {e}")

        return self._skip_result("quantile_monotonicity", "数据格式不支持")

    def _check_backtest_metrics(self, metrics: Dict[str, Any]) -> BiasCheckResult:
        """检测 6: 回测指标真实性校验"""
        warnings_list = []

        annual_return = metrics.get("annual_return")
        if annual_return is None:
            annual_return = metrics.get("total_return", 0)
        sharpe = metrics.get("sharpe_ratio")
        if sharpe is None:
            sharpe = 0
        win_rate = metrics.get("win_rate")
        if win_rate is None:
            win_rate = 0.5
        max_dd = metrics.get("max_drawdown")
        if max_dd is None:
            max_dd = 1.0

        if annual_return > self.thresholds["annual_return_max"]:
            warnings_list.append(f"年化收益={annual_return:.1%}>={self.thresholds['annual_return_max']*100:.0f}%")
        if sharpe > self.thresholds["sharpe_max"]:
            warnings_list.append(f"夏普比率={sharpe:.1f}>={self.thresholds['sharpe_max']:.0f}")
        if win_rate > self.thresholds["win_rate_max"]:
            warnings_list.append(f"胜率={win_rate:.1%}>={self.thresholds['win_rate_max']*100:.0f}%")
        if 0 < max_dd < self.thresholds["max_drawdown_min"]:
            warnings_list.append(f"最大回撤={max_dd:.4f}极低，几乎无回撤")

        n_warnings = len(warnings_list)
        is_abnormal = n_warnings >= 2

        return BiasCheckResult(
            check_name="backtest_reality_check",
            passed=not is_abnormal,
            value=float(n_warnings),
            threshold=2.0,
            severity="critical" if n_warnings >= 3 else ("error" if n_warnings >= 2 else "info"),
            message=(
                f"回测指标{'严重异常' if n_warnings >= 3 else '存疑' if n_warnings >= 2 else '正常'}。"
                f"{'; '.join(warnings_list) if warnings_list else '各项指标在合理范围内'}"
            ),
            detail={
                "warnings": warnings_list,
                "n_warnings": n_warnings,
                "metrics_snapshot": {
                    k: v for k, v in metrics.items()
                    if k in ("annual_return", "sharpe_ratio", "win_rate", "max_drawdown")
                },
            },
        )

    def _check_temporal_consistency(
        self, factor: pd.Series, returns: pd.Series
    ) -> BiasCheckResult:
        """检测 7: 时段一致性检验（前后半段 IC 对比，使用Spearman）"""
        aligned = pd.DataFrame({"f": factor, "r": returns}).dropna()
        if len(aligned) < 40:
            return self._skip_result("temporal_consistency", "数据量不足")

        mid = len(aligned) // 2
        first_half = aligned.iloc[:mid]
        second_half = aligned.iloc[mid:]

        ic_first_result = spearmanr(first_half["f"], first_half["r"])
        ic_second_result = spearmanr(second_half["f"], second_half["r"])
        ic_first = ic_first_result[0]
        ic_second = ic_second_result[0]

        if pd.isna(ic_first) or pd.isna(ic_second):
            return self._skip_result("temporal_consistency", "IC计算无效")

        # 保护除零
        abs_ic_second = abs(ic_second)
        abs_ic_first = abs(ic_first)
        ratio = max(
            safe_divide(float(abs_ic_second), float(abs_ic_first), default=1.0),
            safe_divide(float(abs_ic_first), float(abs_ic_second), default=1.0),
        )
        threshold = self.thresholds["ic_split_ratio_max"]

        return BiasCheckResult(
            check_name="temporal_consistency",
            passed=ratio < threshold,
            value=float(ratio),
            threshold=threshold,
            severity="warning",
            message=(
                f"前后半段IC比值={ratio:.2f}("
                f"前半段IC={ic_first:.4f}, 后半段IC={ic_second:.4f})，"
                f"{'差异较大，可能存在过拟合或数据特性变化' if ratio > threshold else '一致性好'}。"
            ),
            detail={
                "ic_first_half": float(ic_first),
                "ic_second_half": float(ic_second),
                "ratio": float(ratio),
            },
        )

    # ==================== 辅助方法 ====================

    def _skip_result(self, check_name: str, reason: str) -> BiasCheckResult:
        """生成跳过的检测结果"""
        return BiasCheckResult(
            check_name=check_name,
            passed=True,
            value=0.0,
            threshold=0.0,
            severity="info",
            message=f"跳过: {reason}",
            detail={"skipped": True, "reason": reason},
        )

    def _insufficient_data_result(
        self, factor_name: str, n_samples: int
    ) -> LookaheadBiasDetectionResult:
        """数据不足时的默认结果"""
        return LookaheadBiasDetectionResult(
            has_bias=False,
            risk_level=BiasRiskLevel.SAFE,
            risk_score=0.0,
            checks=[
                BiasCheckResult(
                    check_name="data_sufficiency",
                    passed=False,
                    value=float(n_samples),
                    threshold=20.0,
                    severity="info",
                    message=f"样本数{n_samples}<最低要求20，无法执行检测",
                )
            ],
            summary=f"因子 [{factor_name}] 数据不足（{n_samples}个样本），跳过未来函数检测。",
            recommendations=["增加数据量后重新检测"],
            metadata={"factor_name": factor_name, "sample_size": n_samples, "reason": "insufficient_data"},
        )

    def _calculate_risk_score(self, checks: List[BiasCheckResult]) -> float:
        """
        计算综合风险评分 (0-100)

        权重设计：
        - critical: 每个 +30 分
        - error: 每个 +15 分
        - warning: 每个 +5 分
        - info: 0 分
        """
        score = 0.0
        for c in checks:
            if c.passed:
                continue
            if c.severity == "critical":
                score += 30
            elif c.severity == "error":
                score += 15
            elif c.severity == "warning":
                score += 5
        return min(score, 100.0)

    def _determine_risk_level(
        self, risk_score: float, critical_count: int, error_count: int
    ) -> BiasRiskLevel:
        """根据评分和严重程度确定风险等级"""
        if critical_count >= 2 or risk_score >= 70:
            return BiasRiskLevel.CRITICAL
        elif critical_count >= 1 or error_count >= 3 or risk_score >= 50:
            return BiasRiskLevel.HIGH
        elif error_count >= 2 or risk_score >= 25:
            return BiasRiskLevel.MEDIUM
        elif error_count >= 1 or risk_score >= 10:
            return BiasRiskLevel.LOW
        return BiasRiskLevel.SAFE

    def _generate_summary(
        self, factor_name: str, checks: List[BiasCheckResult], level: BiasRiskLevel
    ) -> str:
        """生成人类可读的摘要"""
        failed = [c for c in checks if not c.passed]
        level_emoji = {
            BiasRiskLevel.SAFE: "✅",
            BiasRiskLevel.LOW: "⚠️",
            BiasRiskLevel.MEDIUM: "🔶",
            BiasRiskLevel.HIGH: "🔴",
            BiasRiskLevel.CRITICAL: "💀",
        }

        lines = [
            f"{level_emoji.get(level, '')} 因子 [{factor_name}] 未来函数检测: {level.value.upper()}",
            f"   检测项: {len([c for c in checks if c.passed])}/{len(checks)} 通过",
        ]

        if failed:
            lines.append("   未通过项目:")
            for c in failed[:5]:  # 最多显示5条
                lines.append(f"     - [{c.severity.upper()}] {c.check_name}: {c.message[:80]}")
            if len(failed) > 5:
                lines.append(f"     ... 还有 {len(failed) - 5} 项未通过")

        return "\n".join(lines)

    def _generate_recommendations(self, failed_checks: List[BiasCheckResult]) -> List[str]:
        """根据失败的检测项生成改进建议"""
        recommendations = set()
        severity_names = {c.check_name for c in failed_checks if not c.passed}

        if "ic_magnitude" in severity_names or "rank_correlation" in severity_names:
            recommendations.add("检查因子公式是否使用了 shift(-N) 或引用了未来字段（如 next_close）")
            recommendations.add("确认因子值的计算时间点：应使用 T-1 及之前的数据计算 T 时刻的因子值")

        if "ir_magnitude" in severity_names:
            recommendations.add("IR 过高提示因子预测能力过强，请检查是否使用了未来信息")

        if "autocorrelation_lag1" in severity_names or "value_constancy" in severity_names:
            recommendations.add("因子自相关接近 1 或值几乎不变，可能存在计算错误或数据源问题")

        if any(k in severity_names for k in ("quantile_spread", "quantile_monotonicity")):
            recommendations.add("分层收益异常完美，建议用样本外数据验证因子有效性")

        if "backtest_reality_check" in severity_names:
            recommendations.add("回测指标不真实（年化收益/夏普/胜率过高），强烈建议检查是否存在未来函数")

        if "temporal_consistency" in severity_names:
            recommendations.add("前后时段 IC 差异大，因子可能过拟合到特定时间段")

        if not recommendations:
            recommendations.add("未发现明显的未来函数迹象，但仍建议人工复核因子逻辑")

        return sorted(recommendations)


# ==================== 全局实例 ====================

# 标准模式检测器（默认阈值）
lookahead_bias_detector = LookaheadBiasDetector()

# 严格模式检测器（更敏感，适用于正式发布前的审查）
strict_lookahead_bias_detector = LookaheadBiasDetector(strict_mode=True)
