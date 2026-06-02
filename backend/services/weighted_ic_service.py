"""
因子加权IC服务 - 多因子组合分析的核心工具

功能列表：
1. 加权IC计算（多种加权方案）
2. 因子重要性排序（基于加权IC贡献度）
3. 因子冗余检测（基于相关性调整后的IC）
4. 最优权重搜索（基于历史IC序列）

设计原则：
- 符合业界标准（Barra/Bloomberg多因子模型）
- 支持多种加权方法，适应不同场景
- 提供详细的归因分析
- 高性能向量化实现
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class WeightingMethod(str, Enum):
    """加权方法枚举"""
    EQUAL_WEIGHT = "equal_weight"           # 等权
    IR_WEIGHT = "ir_weight"                 # IR加权 (IC_mean / IC_std)
    ABS_IC_WEIGHT = "abs_ic_weight"         # IC绝对值加权
    DECAY_WEIGHT = "decay_weight"           # 衰减加权（近期权重高）
    OPTIMAL_WEIGHT = "optimal_weight"       # 最优权重（最大化风险调整收益）


@dataclass
class WeightedICConfig:
    """
    加权IC配置
    
    Attributes:
        weighting_method: 加权方法
        decay_half_life: 衰减半衰期（仅用于decay_weight方法）
        min_observations: 最小观察期数
        lookback_window: 回看窗口期
        correlation_adjustment: 是否进行相关性调整（避免重复计算）
        risk_aversion: 风险厌恶系数（用于最优权重计算）
    """
    weighting_method: WeightingMethod = WeightingMethod.IR_WEIGHT
    decay_half_life: int = 60  # 60个交易日半衰期
    min_observations: int = 20
    lookback_window: int = 252
    correlation_adjustment: bool = True
    risk_aversion: float = 1.0


class WeightedICService:
    """
    因子加权IC服务类
    
    提供专业的多因子加权IC分析功能，包括：
    - 多种加权方案（等权、IR、IC绝对值、衰减、最优）
    - 相关性调整（消除多重共线性影响）
    - 因子重要性归因
    - 最优权重搜索
    
    所有方法均使用pandas/numpy向量化操作。
    """

    def __init__(self, config: Optional[WeightedICConfig] = None):
        """
        初始化服务
        
        Args:
            config: 配置对象，默认使用标准配置
        """
        self.config = config or WeightedICConfig()

    def calculate_weighted_ic(
        self,
        factor_ic_dict: Dict[str, pd.Series],
        return_series: Optional[pd.Series] = None,
        factor_correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        计算加权IC
        
        根据配置的加权方法，将多个因子的IC序列合成为加权IC。
        
        Args:
            factor_ic_dict: {factor_name: IC序列} 字典
            return_series: 收益率序列（可选，用于计算最优权重）
            factor_correlation_matrix: 因子相关性矩阵（可选，用于相关性调整）
            
        Returns:
            {
                "weighted_ic": {...},
                "factor_weights": {...},
                "contribution_analysis": {...},
                ...
            }
        """
        try:
            if not factor_ic_dict or len(factor_ic_dict) == 0:
                return {"error": "没有提供因子IC数据"}
            
            factor_names = list(factor_ic_dict.keys())
            n_factors = len(factor_names)
            
            ic_stats = {}
            for name, ic_series in factor_ic_dict.items():
                valid_ic = ic_series.dropna()
                if len(valid_ic) < self.config.min_observations:
                    continue
                
                ic_stats[name] = {
                    "mean_ic": float(valid_ic.mean()),
                    "std_ic": float(valid_ic.std()) if valid_ic.std() > 0 else 1e-6,
                    "ir": float(valid_ic.mean() / valid_ic.std()) if valid_ic.std() > 0 else 0.0,
                    "ic_positive_ratio": float((valid_ic > 0).mean()),
                    "n_observations": len(valid_ic),
                    "ic_series": valid_ic,
                }
            
            if not ic_stats:
                return {"error": "所有因子的IC数据都不足"}
            
            weights = self._calculate_weights(ic_stats, factor_names)
            
            if self.config.correlation_adjustment and factor_correlation_matrix is not None:
                weights, adjustment_info = self._adjust_for_correlation(
                    weights, factor_correlation_matrix, factor_names
                )
            else:
                adjustment_info = None
            
            aligned_ics = self._align_ic_series(factor_ic_dict, factor_names)
            
            weighted_ic_series = pd.Series(0.0, index=aligned_ics.index)
            contribution_dict = {}
            
            for i, name in enumerate(factor_names):
                if name in weights and name in aligned_ics.columns:
                    weight = weights[name]
                    ic_contribution = aligned_ics[name].fillna(0) * weight
                    weighted_ic_series += ic_contribution
                    
                    contribution_dict[name] = {
                        "weight": float(weight),
                        "mean_contribution": float(ic_contribution.mean()),
                        "std_contribution": float(ic_contribution.std()),
                        "contribution_ratio": float(
                            abs(ic_contribution.mean()) / 
                            (abs(weighted_ic_series.mean()) if weighted_ic_series.mean() != 0 else 1.0)
                        ),
                    }
            
            valid_weighted_ic = weighted_ic_series.dropna()
            
            result = {
                "success": True,
                "n_factors": n_factors,
                "factors_analyzed": factor_names,
                "weighting_method": self.config.weighting_method.value,
                "weighted_ic": {
                    "mean": float(valid_weighted_ic.mean()) if len(valid_weighted_ic) > 0 else 0.0,
                    "std": float(valid_weighted_ic.std()) if len(valid_weighted_ic) > 1 else 0.0,
                    "ir": float(
                        valid_weighted_ic.mean() / valid_weighted_ic.std()
                    ) if len(valid_weighted_ic) > 1 and valid_weighted_ic.std() > 0 else 0.0,
                    "positive_ratio": float((valid_weighted_ic > 0).mean()) if len(valid_weighted_ic) > 0 else 0.0,
                    "n_observations": len(valid_weighted_ic),
                    "series_dates": [str(d) for d in valid_weighted_ic.index],
                    "series_values": [float(v) for v in valid_weighted_ic.values],
                },
                "factor_weights": {
                    name: {
                        "weight": float(weights.get(name, 0.0)),
                        **{k: v for k, v in ic_stats.get(name, {}).items() if k != "ic_series"}
                    }
                    for name in factor_names
                    if name in weights
                },
                "contribution_analysis": contribution_dict,
                "correlation_adjustment": adjustment_info,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"计算加权IC失败: {e}", exc_info=True)
            return {"error": str(e)}

    def calculate_factor_importance(
        self,
        factor_ic_dict: Dict[str, pd.Series],
        factor_correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        计算因子重要性排名
        
        综合考虑IC、IR、稳定性、独特性等多个维度，
        给出因子的综合重要性评分和排名。
        
        Args:
            factor_ic_dict: {factor_name: IC序列} 字典
            factor_correlation_matrix: 因子相关性矩阵（可选）
            
        Returns:
            {
                "ranking": [...],
                "scores": {...},
                "interpretation": {...}
            }
        """
        try:
            importance_scores = {}
            
            for name, ic_series in factor_ic_dict.items():
                valid_ic = ic_series.dropna()
                
                if len(valid_ic) < self.config.min_observations:
                    continue
                
                mean_ic = abs(valid_ic.mean())
                std_ic = valid_ic.std()
                ir = mean_ic / std_ic if std_ic > 0 else 0
                positive_ratio = (valid_ic > 0).mean()
                
                recent_ic = valid_ic.tail(self.config.lookback_window // 4)
                momentum_score = (
                    (recent_ic.mean() - valid_ic.mean()) / std_ic 
                    if std_ic > 0 and len(recent_ic) > 10 else 0
                )
                
                stability_score = self._calculate_stability_score(valid_ic)
                
                raw_score = (
                    ir * 30 +
                    mean_ic * 100 +
                    (positive_ratio - 0.5) * 20 +
                    stability_score * 20 +
                    max(min(momentum_score * 10, 10), -10)
                )
                
                importance_scores[name] = {
                    "raw_score": float(raw_score),
                    "components": {
                        "ir": float(ir),
                        "mean_abs_ic": float(mean_ic),
                        "positive_ratio": float(positive_ratio),
                        "stability": float(stability_score),
                        "momentum": float(momentum_score),
                    },
                }
            
            if factor_correlation_matrix is not None:
                for name in importance_scores.keys():
                    if name in factor_correlation_matrix.columns:
                        other_factors = [
                            n for n in importance_scores.keys() 
                            if n != name and n in factor_correlation_matrix.columns
                        ]
                        
                        if other_factors:
                            max_corr_with_others = max([
                                abs(factor_correlation_matrix.loc[name, n])
                                for n in other_factors
                            ])
                            
                            uniqueness_penalty = max_corr_with_others ** 2 * 20
                            importance_scores[name]["raw_score"] -= uniqueness_penalty
                            importance_scores[name]["components"]["uniqueness_penalty"] = float(uniqueness_penalty)
                            importance_scores[name]["components"]["max_correlation"] = float(max_corr_with_others)
            
            sorted_factors = sorted(
                importance_scores.items(),
                key=lambda x: x[1]["raw_score"],
                reverse=True
            )
            
            ranking = []
            for rank, (name, scores) in enumerate(sorted_factors, 1):
                ranking.append({
                    "rank": rank,
                    "factor_name": name,
                    "total_score": float(scores["raw_score"]),
                    **scores["components"],
                })
            
            return {
                "success": True,
                "n_factors_evaluated": len(ranking),
                "ranking": ranking,
                "top_factor": ranking[0] if ranking else None,
                "interpretation": self._generate_importance_interpretation(ranking),
            }
            
        except Exception as e:
            logger.error(f"计算因子重要性失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_weights(
        self,
        ic_stats: Dict[str, Dict],
        factor_names: List[str],
    ) -> Dict[str, float]:
        """根据配置的方法计算权重"""
        
        method = self.config.weighting_method
        
        if method == WeightingMethod.EQUAL_WEIGHT:
            n_valid = len([n for n in factor_names if n in ic_stats])
            return {name: 1.0 / n_valid for name in factor_names if name in ic_stats}
        
        elif method == WeightingMethod.IR_WEIGHT:
            raw_weights = {}
            for name in factor_names:
                if name in ic_stats:
                    stats = ic_stats[name]
                    ir = stats["ir"]
                    raw_weights[name] = max(ir, 0)
            
            total = sum(raw_weights.values())
            if total > 0:
                return {k: v / total for k, v in raw_weights.items()}
            else:
                n = len(raw_weights)
                return {k: 1.0 / n for k in raw_weights}
        
        elif method == WeightingMethod.ABS_IC_WEIGHT:
            raw_weights = {}
            for name in factor_names:
                if name in ic_stats:
                    stats = ic_stats[name]
                    raw_weights[name] = abs(stats["mean_ic"])
            
            total = sum(raw_weights.values())
            if total > 0:
                return {k: v / total for k, v in raw_weights.items()}
            else:
                n = len(raw_weights)
                return {k: 1.0 / n for k in raw_weights}
        
        elif method == WeightingMethod.DECAY_WEIGHT:
            raw_weights = {}
            decay_factor = 0.5 ** (1.0 / self.config.decay_half_life)
            
            for name in factor_names:
                if name in ic_stats and "ic_series" in ic_stats[name]:
                    ic_series = ic_stats[name]["ic_series"]
                    
                    weights_array = np.array([
                        decay_factor ** (len(ic_series) - 1 - i)
                        for i in range(len(ic_series))
                    ])
                    
                    weighted_mean = np.average(ic_series.values, weights=weights_array)
                    raw_weights[name] = abs(weighted_mean)
                elif name in ic_stats:
                    raw_weights[name] = abs(ic_stats[name]["mean_ic"])
            
            total = sum(raw_weights.values())
            if total > 0:
                return {k: v / total for k, v in raw_weights.items()}
            else:
                n = len(raw_weights)
                return {k: 1.0 / n for k in raw_weights}
        
        elif method == WeightingMethod.OPTIMAL_WEIGHT:
            return self._calculate_optimal_weights(ic_stats, factor_names)
        
        else:
            n_valid = len([n for n in factor_names if n in ic_stats])
            return {name: 1.0 / n_valid for name in factor_names if name in ic_stats}

    def _adjust_for_correlation(
        self,
        weights: Dict[str, float],
        corr_matrix: pd.DataFrame,
        factor_names: List[str],
    ) -> Tuple[Dict[str, float], Dict]:
        """
        进行相关性调整
        
        当因子间高度相关时，降低冗余因子的权重，
        以避免重复计算相似的信号。
        """
        adjusted_weights = weights.copy()
        adjustment_info = {"original_weights": dict(weights), "adjustments": {}}
        
        for i, name_i in enumerate(factor_names):
            if name_i not in adjusted_weights:
                continue
            
            for j, name_j in enumerate(factor_names):
                if i >= j or name_j not in adjusted_weights:
                    continue
                
                try:
                    corr_value = abs(corr_matrix.loc[name_i, name_j])
                except KeyError:
                    continue
                
                if corr_value > 0.7:
                    reduction_factor = 1.0 - (corr_value - 0.7) * 0.5
                    
                    if adjusted_weights[name_i] >= adjusted_weights[name_j]:
                        adjusted_weights[name_j] *= reduction_factor
                        adjustment_info["adjustments"][f"{name_j}_reduced_by_{name_i}"] = {
                            "correlation": float(corr_value),
                            "reduction_factor": float(reduction_factor),
                            "reason": f"与{name_i}高度相关({corr_value:.2f})",
                        }
                    else:
                        adjusted_weights[name_i] *= reduction_factor
                        adjustment_info["adjustments"][f"{name_i}_reduced_by_{name_j}"] = {
                            "correlation": float(corr_value),
                            "reduction_factor": float(reduction_factor),
                            "reason": f"与{name_j}高度相关({corr_value:.2f})",
                        }
        
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        adjustment_info["adjusted_weights"] = dict(adjusted_weights)
        adjustment_info["total_reduction"] = float(
            1.0 - sum(adjusted_weights.values()) / sum(weights.values())
            if sum(weights.values()) > 0 else 0.0
        )
        
        return adjusted_weights, adjustment_info

    def _align_ic_series(
        self,
        factor_ic_dict: Dict[str, pd.Series],
        factor_names: List[str],
    ) -> pd.DataFrame:
        """对齐多个因子的IC序列到共同的时间索引"""
        all_indices = []
        
        for name in factor_names:
            if name in factor_ic_dict:
                all_indices.append(factor_ic_dict[name].index)
        
        if not all_indices:
            return pd.DataFrame()
        
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        
        aligned_data = {}
        for name in factor_names:
            if name in factor_ic_dict:
                aligned_data[name] = factor_ic_dict[name].loc[common_index]
        
        return pd.DataFrame(aligned_data)

    def _calculate_optimal_weights(
        self,
        ic_stats: Dict[str, Dict],
        factor_names: List[str],
    ) -> Dict[str, float]:
        """
        计算最优权重（简化版）
        
        目标：最大化风险调整后收益 (IR)
        约束：权重之和为1，权重非负
        """
        n_valid = len([n for n in factor_names if n in ic_stats])
        
        if n_valid < 2:
            return {name: 1.0 / n_valid for name in factor_names if name in ic_stats}
        
        ics = []
        stds = []
        valid_names = []
        
        for name in factor_names:
            if name in ic_stats:
                ics.append(ic_stats[name]["mean_ic"])
                stds.append(max(ic_stats[name]["std_ic"], 1e-6))
                valid_names.append(name)
        
        ics = np.array(ics)
        stds = np.array(stds)
        
        inv_variances = 1.0 / (stds ** 2)
        
        raw_weights = np.abs(ics) * inv_variances
        total = raw_weights.sum()
        
        if total > 0:
            optimal_weights = raw_weights / total
        else:
            optimal_weights = np.ones(len(valid_names)) / len(valid_names)
        
        return dict(zip(valid_names, optimal_weights.tolist()))

    def _calculate_stability_score(self, ic_series: pd.Series) -> float:
        """计算IC稳定性得分"""
        if len(ic_series) < 40:
            return 0.5
        
        first_half = ic_series.iloc[:len(ic_series)//2]
        second_half = ic_series.iloc[len(ic_series)//2:]
        
        if len(first_half) < 10 or len(second_half) < 10:
            return 0.5
        
        first_mean = first_half.mean()
        second_mean = second_half.mean()
        overall_std = ic_series.std()
        
        if overall_std == 0:
            return 1.0
        
        change_score = 1.0 - min(abs(second_mean - first_mean) / overall_std, 1.0)
        
        rolling_std = ic_series.rolling(window=20).std()
        cv_of_std = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else 1.0
        consistency_score = max(0, 1.0 - cv_of_std)
        
        return change_score * 0.6 + consistency_score * 0.4

    def _generate_importance_interpretation(self, ranking: List[Dict]) -> str:
        """生成重要性排名解读"""
        if not ranking:
            return "无法生成解读"
        
        top_factor = ranking[0]
        n_factors = len(ranking)
        
        interpretation_parts = [
            f"共评估{n_factors}个因子，排名如下：\n"
        ]
        
        for r in ranking[:5]:
            interpretation_parts.append(
                f"  {r['rank']}. {r['factor_name']}: "
                f"综合得分={r['total_score']:.2f}, "
                f"IR={r.get('ir', 0):.3f}\n"
            )
        
        if n_factors > 5:
            interpretation_parts.append(f"  ... 共{n_factors}个因子\n")
        
        top = top_factor["factor_name"]
        score = top_factor["total_score"]
        
        if score > 50:
            quality = "优秀"
        elif score > 30:
            quality = "良好"
        elif score > 15:
            quality = "一般"
        else:
            quality = "较弱"
        
        interpretation_parts.append(f"\n最佳因子: {top}（{quality}，得分{score:.1f}）")
        
        return "".join(interpretation_parts)


# 全局默认实例
weighted_ic_service = WeightedICService()