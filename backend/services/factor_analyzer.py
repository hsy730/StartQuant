"""
因子分析器 — 挖掘与验证的解耦层

职责：
1. 接收 MiningResult（挖掘算法的产出）
2. 对每个 FactorCandidate 执行统一验证（IC/IR/换手率/稳定性/前视偏差）
3. 将验证结果存储到 generated_factors 表
4. 返回前端格式的结果

设计原则：
- 挖掘算法不需要知道验证逻辑
- 验证标准对所有算法一致
- 新增挖掘算法无需修改此模块
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.core.database import get_db
from backend.models.generated_factor import GeneratedFactorModel
from backend.repositories.generated_factor_repository import GeneratedFactorRepository
from backend.services.factor_validation_service import factor_validation_service
from backend.services.mining_models import FactorCandidate, MiningResult
from backend.utils.safe_math import safe_divide, safe_float as _safe_float

logger = logging.getLogger(__name__)


class FactorAnalyzer:
    """因子分析器 — 对挖掘结果执行统一验证和存储

    使用方式：
        analyzer = FactorAnalyzer(factor_calculator, data, return_values)
        result_data = analyzer.analyze(mining_result, source="genetic")
    """

    def __init__(
        self,
        factor_calculator,
        data: pd.DataFrame,
        return_values: pd.Series,
    ):
        """
        Args:
            factor_calculator: 因子计算器（FactorCalculator 实例）
            data: 原始数据 DataFrame
            return_values: 收益率 Series
        """
        self.factor_calculator = factor_calculator
        self.data = data
        self.return_values = return_values

    def analyze(
        self,
        mining_result: MiningResult,
        source: str = "unknown",
    ) -> Dict[str, Any]:
        """对挖掘结果执行统一验证和存储

        Args:
            mining_result: 挖掘算法的产出
            source: 算法来源标识

        Returns:
            前端格式的结果字典，包含 discovered_factors 和算法元数据
        """
        discovered_factors = []
        n_candidates = len(mining_result.candidates)
        _validation_timeout = 120.0  # 单因子验证超时 120s
        _total_validation_start = time.time()

        logger.info(f"开始统一验证 {n_candidates} 个候选因子...")

        with get_db() as db:
            repo = GeneratedFactorRepository(db)

            try:
                for idx, candidate in enumerate(mining_result.candidates, 1):
                    _expr_short = candidate.expression[:60] + ("..." if len(candidate.expression) > 60 else "")
                    _cand_start = time.time()
                    logger.info(f"  验证候选因子 [{idx}/{n_candidates}] {_expr_short}")

                    try:
                        # 单因子超时保护：在线程中执行，超时返回空结果
                        # 注意：ThreadPoolExecutor 的 with 语句 __exit__ 会调用
                        # shutdown(wait=True)，即使 future.result(timeout=X) 抛出
                        # TimeoutError，仍会阻塞等待线程完成。
                        # 修复：不使用 with 语句，超时后直接 shutdown(wait=False)
                        import concurrent.futures

                        def _do_analyze():
                            return self._analyze_candidate(
                                candidate, repo, source, idx
                            )

                        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        _future = _executor.submit(_do_analyze)
                        try:
                            factor_result = _future.result(timeout=_validation_timeout)
                        except concurrent.futures.TimeoutError:
                            elapsed = time.time() - _cand_start
                            logger.warning(
                                f"  候选因子 [{idx}/{n_candidates}] 验证超时 ({elapsed:.1f}s > {_validation_timeout:.0f}s)，跳过: {_expr_short}"
                            )
                            factor_result = None
                        finally:
                            _executor.shutdown(wait=False, cancel_futures=True)
                    except Exception as e:
                        elapsed = time.time() - _cand_start
                        logger.warning(
                            f"  候选因子 [{idx}/{n_candidates}] 验证失败 ({elapsed:.1f}s): {e}"
                        )
                        factor_result = None

                    if factor_result:
                        discovered_factors.append(factor_result)

                    _elapsed = time.time() - _cand_start
                    _total_elapsed = time.time() - _total_validation_start
                    logger.info(
                        f"  候选因子 [{idx}/{n_candidates}] 完成 ({_elapsed:.1f}s), "
                        f"累计 {_total_elapsed:.1f}s, 已发现 {len(discovered_factors)} 个有效因子"
                    )
            except Exception as e:
                db.rollback()
                logger.warning(f"保存挖掘结果到 generated_factors 表失败: {e}")
                # 降级：仍然返回结果给前端
                discovered_factors = self._fallback_analyze(mining_result, source)

        # 构建前端结果
        result_data = {
            "discovered_factors": discovered_factors,
            "total_discovered": len(discovered_factors),
            "valid_factors": sum(
                1 for f in discovered_factors if f.get("overall_passed")
            ),
            "best_fitness": (
                max((f.get("raw_fitness", 0) for f in discovered_factors), default=0.0)
            ),
            "avg_fitness": (
                safe_divide(
                    sum(f.get("raw_fitness", 0) for f in discovered_factors),
                    len(discovered_factors),
                    default=0.0,
                )
            ),
        }

        # 透传算法特有元数据
        if mining_result.algorithm_metadata:
            result_data.update(mining_result.algorithm_metadata)

        return result_data

    def _analyze_candidate(
        self,
        candidate: FactorCandidate,
        repo: GeneratedFactorRepository,
        source: str,
        idx: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """分析单个因子候选：验证 + 存储"""
        expression = candidate.expression

        # 统一验证
        unified = self._unified_validate(expression, candidate.precomputed_values)

        # 提取验证指标
        ic = unified["ic"]
        ir = unified["ir"]
        fitness = unified["fitness"]
        validation_score = unified["validation_score"]
        overall_passed = unified["overall_passed"]
        turnover_val = unified["turnover"]
        stability_val = unified["stability"]

        # 如果统一验证完全无结果，回退到算法原始 fitness
        raw_fitness = candidate.fitness
        if (
            (validation_score is None or abs(validation_score) < 1e-10)
            and (ic is None or abs(ic) < 1e-10)
            and (ir is None or abs(ir) < 1e-10)
        ):
            ic = None
            ir = None
            fitness = _safe_float(raw_fitness)
            validation_score = None
            overall_passed = False

        # 存储到 generated_factors 表
        generated_id = self._store_factor(
            repo,
            expression=expression,
            source=source,
            ic=ic,
            ir=ir,
            fitness=fitness,
            validation_score=validation_score,
            overall_passed=overall_passed,
            turnover=turnover_val,
            stability=stability_val,
            complexity=candidate.complexity,
        )

        return {
            "name": f"Mined_Factor_{candidate.rank or idx}",
            "expression": expression,
            "ic": _safe_float(ic),
            "ir": _safe_float(ir),
            "fitness": _safe_float(fitness),
            "raw_fitness": _safe_float(raw_fitness),
            "complexity": _safe_float(candidate.complexity),
            "source": source,
            "overall_passed": overall_passed,
            "validation_score": _safe_float(validation_score),
            "generated_factor_id": generated_id,
        }

    def _unified_validate(
        self,
        expression: str,
        precomputed_values: Optional[pd.Series] = None,
    ) -> Dict:
        """统一验证 — 所有算法的因子使用同一套标准

        Args:
            expression: 因子表达式
            precomputed_values: 预计算的因子值（可选，避免重复计算）

        Returns:
            验证结果字典
        """
        # 优先使用预计算的因子值
        if precomputed_values is not None:
            fv = precomputed_values
        else:
            try:
                fv = self.factor_calculator.calculate(self.data, expression)
            except Exception as e:
                logger.debug(f"统一验证: 无法计算表达式 '{expression[:60]}': {e}")
                return self._empty_result()

        if fv is None or len(fv.dropna()) < 10:
            return self._empty_result()

        fv = fv.replace([np.inf, -np.inf], np.nan).dropna()
        if len(fv) < 10 or fv.isna().all():
            return self._empty_result()

        if self.return_values is not None and len(self.return_values) > 0:
            try:
                validation = factor_validation_service.validate_factor(
                    factor_values=fv,
                    return_values=self.return_values,
                    existing_factors=None,
                )
                ic_val = validation.get("ic_validation", {})
                ir_val = validation.get("ir_validation", {})
                # IC和IR保持符号一致：若取abs则两者都取abs
                ic_raw = float(ic_val.get("ic")) if ic_val.get("ic") is not None else 0.0
                ir_raw = float(ir_val.get("ir")) if ir_val.get("ir") is not None else 0.0
                ic = abs(ic_raw)
                ir_capped = abs(ir_raw)
                # IR双向截断到[-5, 5]（防御性保护，正常由_validate_ir截断）
                ir_capped = max(min(ir_capped, 5.0), -5.0)
                score = (
                    float(validation.get("score"))
                    if validation.get("score") is not None
                    else 0.0
                )
                overall_passed = validation.get("overall_passed", False)
                turnover_val = validation.get("turnover_validation", {})
                stability_val = validation.get("stability_validation", {})

                return {
                    "ic": ic,
                    "ir": ir_capped,
                    "fitness": score / 100.0,
                    "validation_score": score,
                    "overall_passed": overall_passed,
                    "turnover": turnover_val,
                    "stability": stability_val,
                    "validation": validation,
                }
            except Exception as e:
                logger.debug(f"统一验证: 验证失败 '{expression[:60]}': {e}")
                return self._empty_result()

        return self._empty_result()

    @staticmethod
    def _empty_result() -> dict:
        """返回空的验证结果"""
        return {
            "ic": None,
            "ir": None,
            "fitness": None,
            "validation_score": None,
            "overall_passed": False,
            "turnover": None,
            "stability": None,
            "validation": {},
        }

    @staticmethod
    def _store_factor(
        repo: GeneratedFactorRepository,
        expression: str,
        source: str,
        ic: Optional[float],
        ir: Optional[float],
        fitness: Optional[float],
        validation_score: Optional[float],
        overall_passed: bool,
        turnover: Any,
        stability: Any,
        complexity: float,
    ) -> Optional[int]:
        """存储因子到 generated_factors 表

        注意：使用 repo 的 session 统一 commit，不用 Session.object_session，
        因为 object_session 可能返回不同的 session 实例，导致 commit 不一致。
        """
        existing = repo.get_by_expression(expression)
        if existing:
            existing.ic_value = _safe_float(ic)
            existing.ir_value = _safe_float(ir)
            existing.turnover_value = (
                _safe_float(turnover.get("turnover"))
                if isinstance(turnover, dict)
                else None
            )
            existing.stability_score = (
                _safe_float(stability.get("stability_score"))
                if isinstance(stability, dict)
                else None
            )
            existing.validation_score = _safe_float(validation_score)
            existing.is_valid = overall_passed
            existing.generation_method = source
            existing.complexity = str(complexity)
            # 使用 repo 的 db session 统一 commit（由外层 with get_db() 管理）
            repo.db.commit()
            repo.db.refresh(existing)
            return existing.id
        else:
            gen_factor = GeneratedFactorModel(
                expression=expression,
                generation_method=source,
                ic_value=_safe_float(ic),
                ir_value=_safe_float(ir),
                turnover_value=(
                    _safe_float(turnover.get("turnover"))
                    if isinstance(turnover, dict)
                    else None
                ),
                stability_score=(
                    _safe_float(stability.get("stability_score"))
                    if isinstance(stability, dict)
                    else None
                ),
                validation_score=_safe_float(validation_score),
                is_valid=overall_passed,
                is_saved=False,
                complexity=str(complexity),
            )
            created = repo.create(gen_factor)
            return created.id

    def _fallback_analyze(
        self,
        mining_result: MiningResult,
        source: str,
    ) -> List[Dict[str, Any]]:
        """降级分析：数据库存储失败时，仍返回基本结果给前端"""
        results = []
        for candidate in mining_result.candidates:
            results.append(
                {
                    "name": f"Mined_Factor_{candidate.rank or len(results) + 1}",
                    "expression": candidate.expression,
                    "ic": None,
                    "ir": None,
                    "fitness": _safe_float(candidate.fitness),
                    "raw_fitness": _safe_float(candidate.fitness),
                    "complexity": _safe_float(candidate.complexity),
                    "source": source,
                    "overall_passed": False,
                    "validation_score": None,
                    "generated_factor_id": None,
                }
            )
        return results
