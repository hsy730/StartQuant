"""
因子验证服务 - 验证因子质量（含未来函数检测）

多股票场景下，IC和换手率计算委托alphalens-reloaded（符合开源库优先原则和规则7.1/7.2/7.12），
单股票场景保留自实现作为回退。
"""

import logging
import time
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from backend.services.lookahead_bias_detector import (
    lookahead_bias_detector,
    BiasRiskLevel,
)
from backend.utils.safe_math import safe_ir
from backend.constants import IC_PASS_THRESHOLD, IR_PASS_THRESHOLD
from backend.services.factor_stability_service import FactorStabilityService

logger = logging.getLogger(__name__)


def _to_python_float(value, default=None):
    """转换numpy类型为Python原生类型"""
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class FactorValidationService:
    """因子验证服务"""

    def __init__(
        self,
        ic_threshold: float = IC_PASS_THRESHOLD,
        ir_threshold: float = IR_PASS_THRESHOLD,
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
        cross_sectional_panel: Optional[pd.DataFrame] = None,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        全面验证因子质量

        Args:
            factor_values: 因子值序列
            return_values: 收益率序列
            existing_factors: 已有因子字典（用于相关性检测）
            cross_sectional_panel: 横截面面板数据（MultiIndex: date×asset），
                多股票场景下提供以计算正确的横截面换手率
            factor_data: alphalens格式的因子数据（MultiIndex: date×asset），
                传入时IC和换手率计算委托alphalens（推荐）

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

        # IC验证：优先使用alphalens（多股票场景），回退到自实现（单股票场景）
        _t = time.time()
        results["ic_validation"] = self._validate_ic(
            factor_values, return_values, factor_data=factor_data
        )
        logger.debug(f"[验证计时] IC: {time.time()-_t:.3f}s")

        _t = time.time()
        results["rank_ic_validation"] = self._validate_rank_ic(
            factor_values, return_values
        )
        logger.debug(f"[验证计时] RankIC: {time.time()-_t:.3f}s")

        # IR验证：优先使用alphalens IC序列
        _t = time.time()
        results["ir_validation"] = self._validate_ir(
            factor_values, return_values, factor_data=factor_data
        )
        logger.debug(f"[验证计时] IR: {time.time()-_t:.3f}s")

        # 换手率验证：优先使用alphalens，回退到自实现
        _t = time.time()
        results["turnover_validation"] = self._validate_turnover(
            factor_values,
            cross_sectional_panel=cross_sectional_panel,
            factor_data=factor_data,
        )
        logger.debug(f"[验证计时] Turnover: {time.time()-_t:.3f}s")

        # 4. 稳定性验证
        _t = time.time()
        results["stability_validation"] = self._validate_stability(factor_values)
        logger.debug(f"[验证计时] Stability: {time.time()-_t:.3f}s")

        # 5. 相关性验证
        if existing_factors:
            results["correlation_validation"] = self._validate_correlation(
                factor_values, existing_factors
            )
        else:
            results["correlation_validation"] = {"passed": True, "max_correlation": 0.0}

        # 6. 未来函数检测（Look-ahead Bias Detection）
        _t = time.time()
        try:
            results["lookahead_bias"] = self.detect_lookahead_bias(
                factor_values, return_values
            )
        except Exception as e:
            logger.warning(f"未来函数检测异常，跳过: {e}")
            results["lookahead_bias"] = None
        logger.debug(f"[验证计时] LookaheadBias: {time.time()-_t:.3f}s")

        # overall_passed: 原有验证项全部通过 且 无高风险/严重未来函数
        base_checks_passed = all(
            [
                results["ic_validation"]["passed"],
                results["rank_ic_validation"]["passed"],
                results["ir_validation"]["passed"],
                results["turnover_validation"]["passed"],
                results["stability_validation"]["passed"],
                results["correlation_validation"]["passed"],
            ]
        )
        bias_result = results["lookahead_bias"]
        bias_safe = bias_result is None or bias_result.get("risk_level") in (
            None,
            BiasRiskLevel.SAFE.value,
            BiasRiskLevel.LOW.value,
        )
        results["overall_passed"] = base_checks_passed and bias_safe

        results["score"] = self._calculate_score(results)

        if self.ic_type == "time_series":
            results["warnings"] = ["时序IC仅评估择时能力，建议使用横截面IC"]

        return results

    def _validate_ic(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        验证IC

        多股票场景下委托alphalens计算横截面Spearman IC（规则7.1/7.12），
        单股票场景回退到scipy.stats.spearmanr。

        Args:
            factor_values: 因子值
            return_values: 收益率
            factor_data: alphalens格式的因子数据（可选）

        Returns:
            IC验证结果
        """
        # 优先使用alphalens横截面IC（多股票场景）
        if factor_data is not None and isinstance(factor_data.index, pd.MultiIndex):
            num_assets = factor_data.index.get_level_values("asset").nunique()
            if num_assets >= 2:
                try:
                    import alphalens
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ic_df = alphalens.performance.factor_information_coefficient(
                            factor_data
                        )
                    # 取1D周期的IC序列
                    ic_col = [c for c in ic_df.columns if "1" in str(c)]
                    if ic_col:
                        ic_series = ic_df[ic_col[0]].dropna()
                    else:
                        ic_series = ic_df.iloc[:, 0].dropna()

                    if len(ic_series) == 0:
                        return {
                            "passed": False,
                            "ic": None,
                            "message": "alphalens IC序列为空",
                        }

                    ic = float(ic_series.mean())
                    n = len(ic_series)
                    ic_std = float(ic_series.std()) if n > 1 else 0.0

                    # 基于IC序列的t检验（比单期Fisher z更可靠）
                    if n > 1 and ic_std > 1e-10:
                        # 直接除法：se = ic_std/sqrt(n)，ic_std >= 1e-10（上方检查）且 n >= 2，
                        # 因此 se > 0，无需 safe_divide（规则7.34：safe_divide 的 min_threshold
                        # 可能吞掉 se < 1e-10 时的有效极大 t 统计量）
                        se = float(ic_std) / np.sqrt(n)
                        t_stat = float(ic) / se
                        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
                    else:
                        # 规则7.15：ic_std≈0 且 ic_mean≠0 → t_stat=inf
                        if abs(ic) > 1e-10:
                            t_stat = float("inf")
                            p_value = 0.0
                        else:
                            t_stat = 0.0
                            p_value = 1.0

                    is_significant = p_value < 0.05
                    passed = abs(ic) >= self.ic_threshold and is_significant

                    return {
                        "passed": passed,
                        "ic": ic,
                        "ic_std": _to_python_float(ic_std),
                        "t_statistic": _to_python_float(t_stat),
                        "p_value": _to_python_float(p_value),
                        "is_significant": is_significant,
                        "threshold": self.ic_threshold,
                        "method": "alphalens横截面Spearman IC",
                        "n_dates": n,
                        "message": (
                            f"IC={ic:.4f}(std={ic_std:.4f}) t={t_stat:.4f} p={p_value:.4f} "
                            f"{'通过' if passed else '未通过'} (阈值±{self.ic_threshold}, 显著性p<0.05)"
                        ),
                    }
                except Exception as e:
                    logger.warning(f"alphalens IC计算失败，回退到自实现: {e}")

        # 单股票回退：使用scipy.stats.spearmanr（符合规则7.12）
        aligned_data = pd.DataFrame(
            {"factor": factor_values, "return": return_values}
        ).dropna()

        if len(aligned_data) < 10:
            return {
                "passed": False,
                "ic": 0.0,
                "message": "数据量不足",
            }

        ic, _ = spearmanr(aligned_data["factor"], aligned_data["return"])
        if np.isnan(ic):
            return {
                "passed": False,
                "ic": None,
                "message": "IC计算结果为NaN",
            }

        n = len(aligned_data)
        ic_clipped = np.clip(ic, -0.9999, 0.9999)
        denom = np.sqrt(max(1 - ic_clipped**2, 1e-20))  # 截断避免除零
        t_stat = ic_clipped * np.sqrt(n - 2) / denom
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
            "method": "scipy单期Spearman IC（单股票回退）",
            "message": (
                f"IC={ic:.4f} t={t_stat:.4f} p={p_value:.4f} "
                f"{'通过' if passed else '未通过'} (阈值±{self.ic_threshold}, 显著性p<0.05)"
            ),
        }

    def _validate_rank_ic(
        self, factor_values: pd.Series, return_values: pd.Series
    ) -> Dict:
        """
        验证Rank IC — 委托给 _validate_ic（规范5：代码复用）

        Rank IC 即 Spearman IC，与 _validate_ic 的单股票回退路径
        使用相同的 spearmanr 计算，无需重复实现。
        """
        result = self._validate_ic(factor_values, return_values)
        # 将结果键名映射为 rank_ic 格式
        # 注意：result.get("ic") 在键存在但值为 None 时返回 None（规则7.41）
        # 下游 abs(rank_ic) 需要安全处理 None
        rank_ic_val = result.get("ic")
        return {
            "passed": result["passed"],
            "rank_ic": rank_ic_val,
            "t_statistic": result.get("t_statistic"),
            "p_value": result.get("p_value"),
            "is_significant": result.get("is_significant", False),
            "threshold": self.ic_threshold,
            "message": result.get("message", ""),
        }

    def _validate_ir(
        self,
        factor_values: pd.Series,
        return_values: pd.Series,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        验证IR

        多股票场景下使用alphalens横截面IC序列计算IR（规则7.1/7.12），
        单股票场景回退到滚动Spearman IC。

        Args:
            factor_values: 因子值
            return_values: 收益率
            factor_data: alphalens格式的因子数据（可选）

        Returns:
            IR验证结果
        """
        # 优先使用alphalens IC序列计算IR（多股票场景）
        if factor_data is not None and isinstance(factor_data.index, pd.MultiIndex):
            num_assets = factor_data.index.get_level_values("asset").nunique()
            if num_assets >= 2:
                try:
                    import alphalens
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ic_df = alphalens.performance.factor_information_coefficient(
                            factor_data
                        )
                    ic_col = [c for c in ic_df.columns if "1" in str(c)]
                    if ic_col:
                        ic_series = ic_df[ic_col[0]].dropna()
                    else:
                        ic_series = ic_df.iloc[:, 0].dropna()

                    if len(ic_series) >= 2:
                        ic_mean = float(ic_series.mean())
                        ic_std = float(ic_series.std())

                        if ic_std < 1e-10:
                            ir = None  # 不可计算（规则7.10）
                        else:
                            ir = safe_ir(ic_mean, ic_std, default=None)
                            if ir is not None:
                                # IR双向截断到[-5, 5]（业界合理范围）
                                ir = max(min(ir, 5.0), -5.0)

                        # t检验
                        n = len(ic_series)
                        if n > 1 and ic_std > 1e-10:
                            # 直接除法：se = ic_std/sqrt(n)，ic_std >= 1e-10 且 n >= 2，
                            # 因此 se > 0（规则7.34）
                            se = float(ic_std) / np.sqrt(n)
                            t_stat = float(ic_mean) / se
                            p_value = float(
                                2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
                            )
                        else:
                            # 规则7.15：ic_std≈0 且 ic_mean≠0 → t_stat=inf
                            if abs(ic_mean) > 1e-10:
                                t_stat = float("inf")
                                p_value = 0.0
                            else:
                                t_stat = 0.0
                                p_value = 1.0

                        passed = ir is not None and ir >= self.ir_threshold

                        return {
                            "passed": passed,
                            "ir": float(ir) if ir is not None else None,
                            "ic_mean": ic_mean,
                            "ic_std": ic_std,
                            "t_statistic": _to_python_float(t_stat),
                            "p_value": _to_python_float(p_value),
                            "threshold": self.ir_threshold,
                            "method": "alphalens横截面IC序列IR",
                            "n_dates": n,
                            "message": (
                                f"IR={'不可计算' if ir is None else f'{ir:.4f}'} "
                                f"{'通过' if passed else '未通过'} (阈值{self.ir_threshold})"
                            ),
                        }
                except Exception as e:
                    logger.warning(f"alphalens IR计算失败，回退到自实现: {e}")

        # 单股票回退：滚动Spearman IC
        aligned_data = pd.DataFrame(
            {"factor": factor_values, "return": return_values}
        ).dropna()

        if len(aligned_data) < 20:
            return {
                "passed": False,
                "ir": None,
                "message": "数据量不足",
            }

        window = 20

        from backend.utils.ic_calculator import calculate_rolling_ic

        rolling_ic = calculate_rolling_ic(
            aligned_data["factor"],
            aligned_data["return"],
            window=window,
            method="spearman",
        )

        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()

        if ic_std < 1e-10:
            ir = None
        else:
            ir = safe_ir(float(ic_mean), float(ic_std), default=None)
            if ir is not None:
                # IR双向截断到[-5, 5]（业界合理范围）
                ir = max(min(ir, 5.0), -5.0)

        passed = ir is not None and ir >= self.ir_threshold

        return {
            "passed": passed,
            "ir": float(ir) if ir is not None else None,
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "threshold": self.ir_threshold,
            "method": "滚动Spearman IC（单股票回退）",
            "message": (
                f"IR={'不可计算' if ir is None else f'{ir:.4f}'} "
                f"{'通过' if passed else '未通过'} (阈值{self.ir_threshold})"
            ),
        }

    def _validate_turnover(
        self,
        factor_values: pd.Series,
        cross_sectional_panel: Optional[pd.DataFrame] = None,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        验证换手率

        优先使用alphalens quantile_turnover（横截面分位数换手率，规则7.2），
        回退到自实现横截面分位数或时序分位数。

        Args:
            factor_values: 因子值
            cross_sectional_panel: 横截面面板数据（回退方案）
            factor_data: alphalens格式的因子数据（推荐）

        Returns:
            换手率验证结果
        """
        # 优先使用alphalens换手率（多股票场景，规则7.2）
        if factor_data is not None and isinstance(factor_data.index, pd.MultiIndex):
            num_assets = factor_data.index.get_level_values("asset").nunique()
            if num_assets >= 2:
                try:
                    import alphalens
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # alphalens分位数换手率：基于横截面分位数桶变化
                        turnover_df = alphalens.performance.quantile_turnover(
                            factor_data["factor_quantile"],
                            quantile=factor_data["factor_quantile"].max(),
                        )
                        # 因子排名自相关（换手率的互补指标）
                        autocorr_df = alphalens.performance.factor_rank_autocorrelation(
                            factor_data
                        )

                    # 提取1D周期的换手率均值
                    if isinstance(turnover_df, pd.DataFrame):
                        turnover_col = [c for c in turnover_df.columns if "1" in str(c)]
                        if turnover_col:
                            turnover_series = turnover_df[turnover_col[0]].dropna()
                        else:
                            turnover_series = turnover_df.iloc[:, 0].dropna()
                    elif isinstance(turnover_df, pd.Series):
                        turnover_series = turnover_df.dropna()
                    else:
                        turnover_series = pd.Series(dtype=float)

                    turnover = (
                        float(turnover_series.mean())
                        if len(turnover_series) > 0
                        else 0.0
                    )

                    # 自相关
                    if isinstance(autocorr_df, pd.DataFrame):
                        ac_col = [c for c in autocorr_df.columns if "1" in str(c)]
                        if ac_col:
                            autocorr_series = autocorr_df[ac_col[0]].dropna()
                        else:
                            autocorr_series = autocorr_df.iloc[:, 0].dropna()
                    elif isinstance(autocorr_df, pd.Series):
                        autocorr_series = autocorr_df.dropna()
                    else:
                        autocorr_series = pd.Series(dtype=float)

                    autocorr = (
                        float(autocorr_series.mean())
                        if len(autocorr_series) > 0
                        else None
                    )

                    passed = turnover <= self.turnover_threshold

                    return {
                        "passed": passed,
                        "turnover": turnover,
                        "autocorrelation": _to_python_float(autocorr),
                        "threshold": self.turnover_threshold,
                        "method": "alphalens横截面分位数换手率",
                        "message": (
                            f"换手率={turnover:.4f} 自相关={_to_python_float(autocorr)} "
                            f"{'通过' if passed else '未通过'} (阈值{self.turnover_threshold})"
                        ),
                    }
                except Exception as e:
                    logger.warning(f"alphalens换手率计算失败，回退到自实现: {e}")

        # 回退方案1：自实现横截面分位数换手率
        n_bins = 5
        if cross_sectional_panel is not None and isinstance(
            cross_sectional_panel.index, pd.MultiIndex
        ):
            factor_name = factor_values.name or "factor"
            if factor_name in cross_sectional_panel.columns:
                panel = cross_sectional_panel[[factor_name]].dropna()
                factor_bins = panel.groupby(level=0)[factor_name].transform(
                    lambda x: pd.cut(x.rank(pct=True), bins=n_bins, labels=False)
                )
                rank_change = (
                    factor_bins != factor_bins.groupby(level=1).shift(1)
                ).astype(float)
                turnover = rank_change.dropna().mean()
            else:
                turnover = self._time_series_turnover(factor_values, n_bins)
        else:
            # 回退方案2：单股票时序分位数换手率
            turnover = self._time_series_turnover(factor_values, n_bins)

        passed = turnover <= self.turnover_threshold

        return {
            "passed": passed,
            "turnover": float(turnover),
            "threshold": self.turnover_threshold,
            "method": "自实现分位数换手率（回退）",
            "message": f"换手率={turnover:.4f} {'通过' if passed else '未通过'} (阈值{self.turnover_threshold})",
        }

    @staticmethod
    def _time_series_turnover(factor_values: pd.Series, n_bins: int = 5) -> float:
        """单股票时序分位数换手率计算"""
        factor_ranks = factor_values.rank(pct=True)
        factor_bins = pd.cut(factor_ranks, bins=n_bins, labels=False)
        rank_change = (factor_bins != factor_bins.shift(1)).astype(float)
        return rank_change.mean()

    def _validate_stability(self, factor_values: pd.Series) -> Dict:
        """
        验证因子稳定性（分布稳定性）— 委托给 FactorStabilityService（规范5：代码复用）

        KS分段检验逻辑与 factor_stability_service.calculate_distribution_stability 重复，
        统一委托以避免逻辑漂移。
        """
        try:
            stability_service = FactorStabilityService()
            result = stability_service.calculate_distribution_stability(factor_values)
            stability_score = result.get("stability_score", 1.0)
            passed = stability_score >= 0.6
            return {
                "passed": passed,
                "stability_score": float(stability_score),
                "n_comparisons": result.get("n_comparisons", 0),
                "message": f"稳定性得分={stability_score:.2f} {'通过' if passed else '未通过'}",
            }
        except Exception as e:
            logger.warning(f"稳定性检验失败: {e}")
            return {
                "passed": True,
                "stability_score": 1.0,
                "message": f"稳定性检验异常，跳过: {e}",
            }

    def _validate_correlation(
        self, factor_values: pd.Series, existing_factors: Dict[str, pd.Series]
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
            aligned_data = pd.DataFrame(
                {"new_factor": factor_values, "existing_factor": factor_data}
            ).dropna()

            if len(aligned_data) >= 10:
                corr = aligned_data["new_factor"].corr(
                    aligned_data["existing_factor"], method="spearman"
                )
                correlations.append(corr)

        if not correlations:
            return {
                "passed": True,
                "max_correlation": 0.0,
                "message": "无现有因子可对比",
            }

        valid_correlations = [c for c in correlations if pd.notna(c)]
        if not valid_correlations:
            return {
                "passed": True,
                "max_correlation": 0.0,
                "message": "无有效相关性可计算",
            }
        max_corr = max(abs(c) for c in valid_correlations)
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
        if ic_result["passed"] and ic_result.get("ic") is not None:
            ic_abs = abs(ic_result["ic"])
            score += min(ic_abs * 300, 25)

        rank_ic_result = validation_results["rank_ic_validation"]
        if rank_ic_result["passed"] and rank_ic_result.get("rank_ic") is not None:
            rank_ic_abs = abs(rank_ic_result["rank_ic"])
            score += min(rank_ic_abs * 300, 25)

        ir_result = validation_results["ir_validation"]
        if ir_result["passed"] and ir_result["ir"] is not None:
            ir = ir_result["ir"]
            score += max(min(ir * 20, 20), 0)

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

        return max(0.0, min(round(score, 2), 100.0))

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
