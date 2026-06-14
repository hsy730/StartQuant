"""
因子挖掘统一数据模型

将挖掘算法与因子分析解耦的核心契约：
- MiningResult: 挖掘算法的返回值（只包含因子表达式和算法元数据）
- FactorCandidate: 单个因子候选（表达式 + 算法元数据，不含验证结果）

设计原则：
1. 挖掘算法只负责"发现因子表达式"，不负责验证
2. 因子分析（验证、评分、存储）由 FactorAnalyzer 统一执行
3. 新增挖掘算法只需实现 mine_factors() -> MiningResult
4. 算法特有元数据通过 algorithm_metadata 字典透传，不影响通用流程
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FactorCandidate:
    """单个因子候选 — 挖掘算法的产出

    挖掘算法只负责填充此数据类，不执行验证。
    验证由 FactorAnalyzer 在后续步骤中统一执行。

    Attributes:
        expression: 因子代码表达式（如 "ts_delta(close, 5) / close"）
        placeholder_expression: 算法原始表达式格式（如 DEAP树字符串、sympy表达式）
        fitness: 算法原始适应度（不同算法含义不同，仅作参考）
        complexity: 表达式复杂度（节点数/参数量等）
        source: 算法来源标识（如 "genetic", "pysr", "gflownet"）
        rank: 算法内部排名（可选，由算法自行决定）
        precomputed_values: 预计算的因子值（可选，避免重复计算）
            遗传算法等可提供此值，FactorAnalyzer 会复用而非重新计算
        algorithm_metadata: 算法特有元数据（可选，透传到前端）
            如 PySR 的 loss/score、Deep 的 model_id 等
    """

    expression: str
    placeholder_expression: str = ""
    fitness: float = 0.0
    complexity: float = 0.0
    source: str = ""
    rank: int = 0
    precomputed_values: Optional[pd.Series] = None
    algorithm_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（precomputed_values 不序列化，仅内存传递）"""
        d = {
            "expression": self.expression,
            "placeholder_expression": self.placeholder_expression,
            "fitness": self.fitness,
            "complexity": self.complexity,
            "source": self.source,
            "rank": self.rank,
        }
        if self.algorithm_metadata:
            d["algorithm_metadata"] = self.algorithm_metadata
        return d


@dataclass
class MiningResult:
    """挖掘算法的统一返回值 — 挖掘与分析的契约边界

    挖掘算法返回此数据类，API 层将其传递给 FactorAnalyzer 进行验证。
    MiningResult 不包含任何验证结果，只包含"发现了什么"。

    Attributes:
        success: 挖掘是否成功
        candidates: 发现的因子候选列表
        cancelled: 是否被用户取消
        fitness_history: 适应度历史 {"best": [...], "average": [...]}
        algorithm_metadata: 算法级别的元数据（透传到前端）
            如 genetic 的 logbook、deep 的 model_info、tree_prescreen 的 feature_importance
        error: 错误信息（success=False 时）
    """

    success: bool
    candidates: List[FactorCandidate] = field(default_factory=list)
    cancelled: bool = False
    fitness_history: Dict[str, List[float]] = field(
        default_factory=lambda: {"best": [], "average": []}
    )
    algorithm_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于传递给 _finalize_task 等下游逻辑）"""
        return {
            "success": self.success,
            "cancelled": self.cancelled,
            "best_factors": [c.to_dict() for c in self.candidates],
            "fitness_history": self.fitness_history,
            "algorithm_metadata": self.algorithm_metadata,
            "error": self.error,
        }

    @classmethod
    def from_legacy_dict(cls, data: Dict) -> "MiningResult":
        """从旧格式字典构建 MiningResult（兼容过渡期）

        旧格式：{"success": True, "best_factors": [{"expression": ..., ...}], ...}
        """
        candidates = []
        for f in data.get("best_factors", []):
            candidates.append(
                FactorCandidate(
                    expression=f.get("expression", ""),
                    placeholder_expression=f.get("placeholder_expression", ""),
                    fitness=f.get("fitness", 0.0),
                    complexity=f.get("complexity", 0.0),
                    source=f.get("source", ""),
                    rank=f.get("rank", 0),
                    precomputed_values=f.get("_precomputed_factor_values"),
                    algorithm_metadata={
                        k: v
                        for k, v in f.items()
                        if k
                        not in {
                            "expression",
                            "placeholder_expression",
                            "fitness",
                            "complexity",
                            "source",
                            "rank",
                            "validation",
                            "_precomputed_factor_values",
                        }
                    },
                )
            )
        return cls(
            success=data.get("success", False),
            candidates=candidates,
            cancelled=data.get("cancelled", False),
            fitness_history=data.get(
                "fitness_history", {"best": [], "average": []}
            ),
            algorithm_metadata={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "success",
                    "best_factors",
                    "cancelled",
                    "fitness_history",
                    "error",
                }
            },
            error=data.get("error"),
        )
