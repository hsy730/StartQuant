"""
因子验证服务 - 验证因子质量（含未来函数检测）
"""
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances
from backend.services.alphalens_analysis_service import alphalens_analysis_service
from backend.services.lookahead_bias_detector import (
    LookaheadBiasDetector,
    lookahead_bias_detector,
    BiasRiskLevel,
)
from backend.utils.safe_math import safe_ir


class FactorValidationService:
    """因子验证服务"""

    def __init__(
        self,
        ic_threshold: float = 0.03,
        ir_threshold: float = 0.5,
        turnover_threshold: float = 0.5,
        max_correlation: float = 0.8,
        ic_type: str = "time_series",
    ):
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold
        self.turnover_threshold = turnover_threshold
        self.max_correlation = max_correlation
        self.ic_type = ic_type

    def validate_factor(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        existing_factors: Optional[Dict[str, pd.Series]] = None,
    ) -> Dict:
        """
        全面验证因子质量

        Args:
            factor_values: 因子值序列
            return_values: 收益率序列
            existing_factors: 已有因子字典（用于相关性检测）

        Returns:
            验证结果
        """
        results = {
            "ic_validation": None,
            "rank_ic_validation": None,
            "ir_validation": None,
            "turnover_validation": None,
            "stability_validation": None,
            "correlation_validation": None,
            "lookahead_bias": None,
            "overall_passed": False,
            "score": 0.0,
        }

        results["ic_validation"] = self._validate_ic(factor_values, return_values)

        results["rank_ic_validation"] = self._validate_rank_ic(factor_values, return_values)

        # 2. IR验证
        results["ir_validation"] = self._validate_ir(factor_values, return_values)

        # 3. 换手率验证
        results["turnover_validation"] = self._validate_turnover(factor_values)

        # 4. 稳定性验证
        results["stability_validation"] = self._validate_stability(factor_values)

        # 5. 相关性验证
        if existing_factors:
            results["correlation_validation"] = self._validate_correlation(
                factor_values, existing_factors
            )
        else:
            results["correlation_validation"] = {"passed": True, "max_correlation": 0.0}

        # 6. 未来函数检测（Look-ahead Bias Detection）
        results["lookahead_bias"] = self.detect_lookahead_bias(
            factor_values, return_values
        )

        # overall_passed: 原有验证项全部通过 且 无高风险/严重未来函数
        base_checks_passed = all([
            results["ic_validation"]["passed"],
            results["rank_ic_validation"]["passed"],
            results["ir_validation"]["passed"],
            results["turnover_validation"]["passed"],
            results["stability_validation"]["passed"],
            results["correlation_validation"]["passed"],
        ])
        bias_result = results["lookahead_bias"]
        bias_safe = (
            bias_result is None or
            bias_result.get("risk_level") in (None, BiasRiskLevel.SAFE.value, BiasRiskLevel.LOW.value)
        )
        results["overall_passed"] = base_checks_passed and bias_safe

        results["score"] = self._calculate_score(results)

        if self.ic_type == "time_series":
            results["warnings"] = ["时序IC仅评估择时能力，建议使用横截面IC"]

        return results

    def _validate_ic(
        self,
        factor_values: pd.Series,
        return_values: pd.Series
    ) -> Dict:
        """
        验证IC

        Args:
            factor_values: 因子值
            return_values: 收益率

        Returns:
            IC验证结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            "factor": factor_values,
            "return": return_values
        }).dropna()

        if len(aligned_data) < 10:
            return {
                "passed": False,
                "ic": 0.0,
                "message": "数据量不足",
            }

        # 计算IC
        ic = aligned_data["factor"].corr(aligned_data["return"])

        n = len(aligned_data)
        if abs(ic) >= 1.0:
            t_stat = float('inf') if ic > 0 else float('-inf')
            p_value = 0.0
        else:
            t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

        is_significant = p_value < 0.05

        passed = abs(ic) >= self.ic_threshold and is_significant

        return {
            "passed": passed,
            "ic": float(ic),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "threshold": self.ic_threshold,
            "message": f"IC={ic:.4f} t={t_stat:.4f} p={p_value:.4f} {'通过' if passed else '未通过'} (阈值±{self.ic_threshold}, 显著性p<0.05)",
        }

    def _validate_rank_ic(
        self,
        factor_values: pd.Series,
        return_values: pd.Series
    ) -> Dict:
        aligned_data = pd.DataFrame({
            "factor": factor_values,
            "return": return_values
        }).dropna()

        if len(aligned_data) < 10:
            return {
                "passed": False,
                "rank_ic": 0.0,
                "message": "数据量不足",
            }

        rank_ic = aligned_data["factor"].rank().corr(aligned_data["return"].rank())

        n = len(aligned_data)
        if abs(rank_ic) >= 1.0:
            t_stat = float('inf') if rank_ic > 0 else float('-inf')
            p_value = 0.0
        else:
            t_stat = rank_ic * np.sqrt(n - 2) / np.sqrt(1 - rank_ic ** 2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

        is_significant = p_value < 0.05

        passed = abs(rank_ic) >= self.ic_threshold and is_significant

        return {
            "passed": passed,
            "rank_ic": float(rank_ic),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "threshold": self.ic_threshold,
            "message": f"Rank IC={rank_ic:.4f} t={t_stat:.4f} p={p_value:.4f} {'通过' if passed else '未通过'} (阈值±{self.ic_threshold}, 显著性p<0.05)",
        }

    def _validate_ir(
        self,
        factor_values: pd.Series,
        return_values: pd.Series
    ) -> Dict:
        """
        验证IR

        Phase 5: 使用 pandas rolling().corr() 向量化计算替代 Python 循环，
        性能提升 ~10-50x。

        Args:
            factor_values: 因子值
            return_values: 收益率

        Returns:
            IR验证结果
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            "factor": factor_values,
            "return": return_values
        }).dropna()

        if len(aligned_data) < 20:
            return {
                "passed": False,
                "ir": 0.0,
                "message": "数据量不足",
            }

        # Phase 5: 向量化滚动IC计算（替代 O(N*W) Python 循环）
        window = 20
        min_periods = 10

        rolling_ic = aligned_data["factor"].rolling(
            window=window, min_periods=min_periods
        ).corr(aligned_data["return"])

        # 计算IR（IC均值 / IC标准差）
        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()

        if ic_std > 0:
            ir = safe_ir(float(ic_mean), float(ic_std), default=0.0)
            ir = min(ir, 5.0)
        else:
            ir = 0.0

        # 判断是否通过
        passed = ir >= self.ir_threshold

        return {
            "passed": passed,
            "ir": float(ir),
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "threshold": self.ir_threshold,
            "message": f"IR={ir:.4f} {'通过' if passed else '未通过'} (阈值{self.ir_threshold})",
        }

    def _validate_turnover(
        self,
        factor_values: pd.Series
    ) -> Dict:
        """
        验证换手率

        Args:
            factor_values: 因子值

        Returns:
            换手率验证结果
        """
        # 计算因子排名变化
        factor_rank = factor_values.rolling(
            window=252, min_periods=1
        ).rank(pct=True)

        # 计算换手率（排名变化的比例）
        rank_change = factor_rank.diff().abs()
        turnover = rank_change.mean()

        # 判断是否通过
        passed = turnover <= self.turnover_threshold

        return {
            "passed": passed,
            "turnover": float(turnover),
            "threshold": self.turnover_threshold,
            "message": f"换手率={turnover:.4f} {'通过' if passed else '未通过'} (阈值{self.turnover_threshold})",
        }

    def _validate_stability(
        self,
        factor_values: pd.Series
    ) -> Dict:
        """
        验证因子稳定性（分布稳定性）

        Args:
            factor_values: 因子值

        Returns:
            稳定性验证结果
        """
        if len(factor_values) < 252:
            return {
                "passed": True,
                "stability_score": 1.0,
                "message": "数据量不足，跳过稳定性检验",
            }

        # 分段检验（每252天一段）
        n_segments = len(factor_values) // 252
        if n_segments < 2:
            return {
                "passed": True,
                "stability_score": 1.0,
                "message": "数据长度不足2段，跳过稳定性检验",
            }

        segments = []
        for i in range(n_segments):
            start_idx = i * 252
            end_idx = start_idx + 252
            segment = factor_values.iloc[start_idx:end_idx].dropna()
            if len(segment) > 0:
                segments.append(segment)

        # 两两KS检验
        p_values = []
        for i in range(len(segments) - 1):
            for j in range(i + 1, len(segments)):
                statistic, p_value = stats.ks_2samp(segments[i], segments[j])
                p_values.append(p_value)

        # 稳定性得分（p值 > 0.05的比例）
        if p_values:
            stable_ratio = sum(1 for p in p_values if p > 0.05) / len(p_values)
            passed = stable_ratio >= 0.6  # 60%的比较显示稳定
        else:
            stable_ratio = 1.0
            passed = True

        return {
            "passed": passed,
            "stability_score": float(stable_ratio),
            "n_comparisons": len(p_values),
            "message": f"稳定性得分={stable_ratio:.2f} {'通过' if passed else '未通过'}",
        }

    def _validate_correlation(
        self,
        factor_values: pd.Series,
        existing_factors: Dict[str, pd.Series]
    ) -> Dict:
        """
        验证因子相关性

        Args:
            factor_values: 新因子值
            existing_factors: 已有因子字典

        Returns:
            相关性验证结果
        """
        correlations = []

        for factor_name, factor_data in existing_factors.items():
            # 对齐索引
            aligned_data = pd.DataFrame({
                "new_factor": factor_values,
                "existing_factor": factor_data
            }).dropna()

            if len(aligned_data) >= 10:
                corr = aligned_data["new_factor"].corr(
                    aligned_data["existing_factor"]
                )
                correlations.append(corr)

        if not correlations:
            return {
                "passed": True,
                "max_correlation": 0.0,
                "message": "无现有因子可对比",
            }

        max_corr = max(abs(c) for c in correlations)
        passed = max_corr <= self.max_correlation

        return {
            "passed": passed,
            "max_correlation": float(max_corr),
            "all_correlations": [float(c) for c in correlations],
            "message": f"最大相关性={max_corr:.4f} {'通过' if passed else '未通过'} (阈值{self.max_correlation})",
        }

    def _calculate_score(self, validation_results: Dict) -> float:
        """
        计算综合得分（0-100）

        Args:
            validation_results: 验证结果

        Returns:
            综合得分
        """
        score = 0.0

        ic_result = validation_results["ic_validation"]
        if ic_result["passed"]:
            ic_abs = abs(ic_result["ic"])
            score += min(ic_abs * 300, 25)

        rank_ic_result = validation_results["rank_ic_validation"]
        if rank_ic_result["passed"]:
            rank_ic_abs = abs(rank_ic_result["rank_ic"])
            score += min(rank_ic_abs * 300, 25)

        ir_result = validation_results["ir_validation"]
        if ir_result["passed"]:
            ir = ir_result["ir"]
            score += min(ir * 20, 20)

        stab_result = validation_results["stability_validation"]
        if stab_result["passed"]:
            stability_score = stab_result["stability_score"]
            score += stability_score * 15

        turnover_result = validation_results["turnover_validation"]
        if turnover_result["passed"]:
            turnover = turnover_result["turnover"]
            score += max(15 - turnover * 30, 0)

        # 未来函数风险扣分
        bias_result = validation_results.get("lookahead_bias")
        if bias_result and isinstance(bias_result, dict):
            risk_score = bias_result.get("risk_score", 0)
            risk_level = bias_result.get("risk_level")
            if risk_level in (BiasRiskLevel.HIGH.value, BiasRiskLevel.CRITICAL.value):
                # 高风险/严重: 扣除所有分数
                score = max(0, score * (1 - risk_score / 100))
            elif risk_level == BiasRiskLevel.MEDIUM.value:
                # 中等风险: 扣一半
                score = max(0, score * 0.5)
            elif risk_level == BiasRiskLevel.LOW.value:
                # 低风险: 轻微扣分
                score = max(0, score * 0.9)

        return round(score, 2)

    def detect_lookahead_bias(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        factor_name: str = "factor",
        extra_context: Optional[Dict] = None,
    ) -> Dict:
        """
        检测因子是否存在未来函数（Look-ahead Bias）

        委托给 LookaheadBiasDetector 执行多维度统计检测，
        并将结果转换为兼容的字典格式。

        Args:
            factor_values: 因子值序列
            return_values: 收益率序列
            factor_name: 因子名称
            extra_context: 额外上下文（如回测指标、分层收益等）

        Returns:
            包含检测结果的字典，关键字段：
            - has_bias: bool
            - risk_level: str ("safe"/"low"/"medium"/"high"/"critical")
            - risk_score: float (0-100)
            - summary: str
            - recommendations: list[str]
            - checks: list[dict]  各检测项详情
        """
        detection_result = lookahead_bias_detector.detect(
            factor_values=factor_values,
            return_values=return_values,
            factor_name=factor_name,
            extra_context=extra_context,
        )

        return {
            "has_bias": detection_result.has_bias,
            "risk_level": detection_result.risk_level.value,
            "risk_score": detection_result.risk_score,
            "summary": detection_result.summary,
            "recommendations": detection_result.recommendations,
            "checks": [
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in detection_result.checks
            ],
            "metadata": detection_result.metadata,
        }

    def batch_validate(
        self,
        factors: Dict[str, pd.Series],
        return_values: pd.Series,
    ) -> Dict[str, Dict]:
        """
        批量验证多个因子

        Args:
            factors: 因子字典 {factor_name: factor_values}
            return_values: 收益率序列

        Returns:
            验证结果字典
        """
        results = {}

        # 按顺序验证，每次将已通过的因子加入existing_factors
        existing_factors = {}

        for factor_name, factor_values in factors.items():
            results[factor_name] = self.validate_factor(
                factor_values=factor_values,
                return_values=return_values,
                existing_factors=existing_factors,
            )

            # 如果通过验证，加入已有因子列表
            if results[factor_name]["overall_passed"]:
                existing_factors[factor_name] = factor_values

        return results


# 全局因子验证服务实例
factor_validation_service = FactorValidationService()
