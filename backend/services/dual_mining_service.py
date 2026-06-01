"""
双算法并行因子挖掘服务

同时运行DEAP遗传规划和PySR符号回归，取两者中更优秀的结果。
支持三种模式:
  - "genetic": 仅DEAP遗传规划（向后兼容）
  - "pysr": 仅PySR符号回归
  - "dual": 两者并行执行，合并最优结果
"""
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from backend.services.genetic_factor_mining_service import (
    GeneticFactorMiningService,
    create_genetic_mining_service,
    DEAP_AVAILABLE,
)
from backend.services.pysr_factor_mining_service import (
    PySRFactorMiningService,
    create_pysr_mining_service,
    PYSR_AVAILABLE,
)
from backend.services.factor_validation_service import factor_validation_service


class DualMiningService:
    """双算法并行因子挖掘服务

    Runs DEAP GP and PySR symbolic regression in parallel, then merges
    and ranks the results from both algorithms.  The best factors from
    each algorithm are kept, deduplicated, and sorted by validation
    score.
    """

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        factor_calculator=None,
        max_eval_stocks: int = 50,
        algorithm: str = "dual",
        # ---- DEAP GP parameters ----
        population_size: int = 50,
        n_generations: int = 20,
        cx_prob: float = 0.7,
        mut_prob: float = 0.3,
        elite_size: int = 5,
        fitness_objective: str = "ic_mean",
        parsimony_coeff: float = 0.001,
        diversity_penalty_coeff: float = 0.1,
        max_cache_size: int = 512,
        cv_folds: int = 0,
        use_extended_primitives: bool = True,
        max_tree_depth: int = 17,
        use_nsga2: bool = True,
        # ---- PySR parameters ----
        pysr_niterations: int = 40,
        pysr_populations: int = 30,
        pysr_binary_operators: Optional[List[str]] = None,
        pysr_unary_operators: Optional[List[str]] = None,
        pysr_maxsize: int = 30,
        pysr_maxdepth: int = 5,
        pysr_constraints: Optional[Dict] = None,
        pysr_nested_constraints: Optional[Dict] = None,
        pysr_parsimony: float = 0.0032,
        pysr_procs: int = 8,
        pysr_population_size: int = 33,
    ):
        self.base_factor_codes = base_factors
        self.data = data
        self.return_column = return_column
        self.factor_calculator = factor_calculator
        self.max_eval_stocks = max_eval_stocks
        self.algorithm = algorithm

        self._gp_params = dict(
            population_size=population_size,
            n_generations=n_generations,
            cx_prob=cx_prob,
            mut_prob=mut_prob,
            elite_size=elite_size,
            fitness_objective=fitness_objective,
            parsimony_coeff=parsimony_coeff,
            diversity_penalty_coeff=diversity_penalty_coeff,
            max_cache_size=max_cache_size,
            cv_folds=cv_folds,
            use_extended_primitives=use_extended_primitives,
            max_tree_depth=max_tree_depth,
            use_nsga2=use_nsga2,
        )

        self._pysr_params = dict(
            niterations=pysr_niterations,
            populations=pysr_populations,
            binary_operators=pysr_binary_operators,
            unary_operators=pysr_unary_operators,
            maxsize=pysr_maxsize,
            maxdepth=pysr_maxdepth,
            constraints=pysr_constraints,
            nested_constraints=pysr_nested_constraints,
            parsimony=pysr_parsimony,
            procs=pysr_procs,
            population_size=pysr_population_size,
            fitness_objective=fitness_objective,
            cv_folds=cv_folds,
        )

        self.progress_callback = None
        self._gp_service: Optional[GeneticFactorMiningService] = None
        self._pysr_service: Optional[PySRFactorMiningService] = None

    def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
        if self._gp_service is not None:
            self._gp_service.set_stock_pool(stock_codes, start_date, end_date)
        if self._pysr_service is not None:
            self._pysr_service.set_stock_pool(stock_codes, start_date, end_date)

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _run_genetic(self) -> Dict:
        """Run DEAP genetic programming mining."""
        if not DEAP_AVAILABLE:
            logger.warning("DEAP not available, skipping genetic mining")
            return {"success": False, "message": "DEAP库未安装", "best_factors": []}

        logger.info("启动DEAP遗传规划挖掘...")
        self._gp_service = create_genetic_mining_service(
            base_factors=self.base_factor_codes,
            data=self.data,
            return_column=self.return_column,
            factor_calculator=self.factor_calculator,
            max_eval_stocks=self.max_eval_stocks,
            **self._gp_params,
        )

        if self.progress_callback:
            def gp_progress(gen, total_gen, best_fitness, avg_fitness):
                self.progress_callback(
                    gen, total_gen, best_fitness, avg_fitness,
                    algorithm="genetic"
                )
            self._gp_service.set_progress_callback(gp_progress)

        result = self._gp_service.mine_factors()
        result["source"] = "genetic"
        return result

    def _run_pysr(self) -> Dict:
        """Run PySR symbolic regression mining."""
        if not PYSR_AVAILABLE:
            logger.warning("PySR not available, skipping PySR mining")
            return {"success": False, "message": "PySR库未安装", "best_factors": []}

        logger.info("启动PySR符号回归挖掘...")
        self._pysr_service = create_pysr_mining_service(
            base_factors=self.base_factor_codes,
            data=self.data,
            return_column=self.return_column,
            factor_calculator=self.factor_calculator,
            max_eval_stocks=self.max_eval_stocks,
            **self._pysr_params,
        )

        if self.progress_callback:
            def pysr_progress(iteration, total_iter, best_fitness, avg_fitness):
                self.progress_callback(
                    iteration, total_iter, best_fitness, avg_fitness,
                    algorithm="pysr"
                )
            self._pysr_service.set_progress_callback(pysr_progress)

        result = self._pysr_service.mine_factors()
        result["source"] = "pysr"
        return result

    def _merge_results(self, gp_result: Dict, pysr_result: Dict) -> Dict:
        """Merge and rank results from both algorithms.

        Deduplication is based on expression string similarity (Jaccard
        > 0.8).  Factors are sorted by validation score (or fitness as
        fallback), and the top factors from both algorithms are kept.
        """
        all_factors = []

        if gp_result.get("success"):
            for f in gp_result.get("best_factors", []):
                f_copy = dict(f)
                f_copy["source"] = "genetic"
                all_factors.append(f_copy)

        if pysr_result.get("success"):
            for f in pysr_result.get("best_factors", []):
                f_copy = dict(f)
                f_copy["source"] = "pysr"
                all_factors.append(f_copy)

        if not all_factors:
            return {
                "success": True,
                "best_factors": [],
                "gp_result": gp_result,
                "pysr_result": pysr_result,
                "source": "dual",
            }

        all_factors = self._deduplicate_factors(all_factors)

        def _sort_key(f):
            v = f.get("validation", {})
            if v and isinstance(v, dict):
                return v.get("score", f.get("fitness", 0))
            return f.get("fitness", 0)

        all_factors.sort(key=_sort_key, reverse=True)
        for i, fi in enumerate(all_factors):
            fi["rank"] = i + 1

        gp_count = sum(1 for f in all_factors if f.get("source") == "genetic")
        pysr_count = sum(1 for f in all_factors if f.get("source") == "pysr")

        logger.info(
            f"Dual mining merged {len(all_factors)} factors: "
            f"{gp_count} from GP, {pysr_count} from PySR"
        )

        fitness_history = self._merge_fitness_history(gp_result, pysr_result)

        return {
            "success": True,
            "best_factors": all_factors,
            "gp_result": gp_result,
            "pysr_result": pysr_result,
            "fitness_history": fitness_history,
            "source": "dual",
        }

    def _deduplicate_factors(self, factors: List[Dict]) -> List[Dict]:
        """Remove near-duplicate factors based on expression similarity."""
        if len(factors) <= 1:
            return factors

        def _token_set(expr: str) -> set:
            return set(expr.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split())

        def _jaccard(a: str, b: str) -> float:
            sa = _token_set(a)
            sb = _token_set(b)
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        kept = []
        for f in factors:
            expr = f.get("expression", "")
            is_dup = False
            for k in kept:
                k_expr = k.get("expression", "")
                if _jaccard(expr, k_expr) > 0.8:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(f)

        removed = len(factors) - len(kept)
        if removed > 0:
            logger.info(f"Deduplicated: removed {removed} near-duplicate factors")

        return kept

    def _merge_fitness_history(self, gp_result: Dict, pysr_result: Dict) -> Dict:
        """Merge fitness histories from both algorithms for frontend charting."""
        gp_history = gp_result.get("fitness_history", {"best": [], "average": []})
        pysr_history = pysr_result.get("fitness_history", {"best": [], "average": []})

        if isinstance(gp_history, dict) and "best" in gp_history:
            gp_best = gp_history.get("best", [])
            gp_avg = gp_history.get("average", [])
        else:
            gp_best = []
            gp_avg = []

        if isinstance(pysr_history, dict) and "best" in pysr_history:
            pysr_best = pysr_history.get("best", [])
            pysr_avg = pysr_history.get("average", [])
        else:
            pysr_best = []
            pysr_avg = []

        max_len = max(len(gp_best), len(pysr_best))

        merged_best = []
        merged_avg = []
        for i in range(max_len):
            vals = []
            avg_vals = []
            if i < len(gp_best):
                vals.append(gp_best[i])
                avg_vals.append(gp_avg[i] if i < len(gp_avg) else 0)
            if i < len(pysr_best):
                vals.append(pysr_best[i])
                avg_vals.append(pysr_avg[i] if i < len(pysr_avg) else 0)
            merged_best.append(max(vals) if vals else 0)
            merged_avg.append(sum(avg_vals) / len(avg_vals) if avg_vals else 0)

        return {
            "best": merged_best,
            "average": merged_avg,
            "gp_best": gp_best,
            "gp_average": gp_avg,
            "pysr_best": pysr_best,
            "pysr_average": pysr_avg,
        }

    def mine_factors(self) -> Dict:
        """Execute factor mining using the configured algorithm mode.

        Modes:
        - ``"genetic"``: DEAP GP only
        - ``"pysr"``: PySR only
        - ``"dual"``: both in parallel, merge best results
        """
        if self.algorithm == "genetic":
            return self._run_genetic()
        elif self.algorithm == "pysr":
            return self._run_pysr()
        elif self.algorithm == "dual":
            return self._run_dual()
        else:
            logger.warning(f"Unknown algorithm mode '{self.algorithm}', falling back to genetic")
            return self._run_genetic()

    def _run_dual(self) -> Dict:
        """Run both algorithms in parallel and merge results."""
        logger.info("启动双算法并行挖掘 (DEAP GP + PySR)...")

        gp_result = {"success": False, "best_factors": []}
        pysr_result = {"success": False, "best_factors": []}

        gp_available = DEAP_AVAILABLE
        pysr_available = PYSR_AVAILABLE

        if not gp_available and not pysr_available:
            return {
                "success": False,
                "message": "DEAP和PySR均不可用，请至少安装一个",
                "best_factors": [],
            }

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if gp_available:
                futures[executor.submit(self._run_genetic)] = "genetic"
            if pysr_available:
                futures[executor.submit(self._run_pysr)] = "pysr"

            for future in as_completed(futures):
                algo = futures[future]
                try:
                    result = future.result()
                    if algo == "genetic":
                        gp_result = result
                        logger.info(f"GP mining completed: {len(gp_result.get('best_factors', []))} factors")
                    else:
                        pysr_result = result
                        logger.info(f"PySR mining completed: {len(pysr_result.get('best_factors', []))} factors")
                except Exception as e:
                    logger.error(f"{algo} mining failed with exception: {e}", exc_info=True)
                    if algo == "genetic":
                        gp_result = {"success": False, "message": str(e), "best_factors": []}
                    else:
                        pysr_result = {"success": False, "message": str(e), "best_factors": []}

        return self._merge_results(gp_result, pysr_result)


def create_dual_mining_service(
    base_factors: List[str],
    data: pd.DataFrame,
    factor_calculator=None,
    **kwargs
) -> DualMiningService:
    """Create a configured :class:`DualMiningService` instance.

    Accepted keyword arguments include all DEAP GP and PySR parameters,
    plus:

    * ``algorithm`` – "genetic" / "pysr" / "dual" (default "dual")
    """
    return DualMiningService(
        base_factors=base_factors,
        data=data,
        factor_calculator=factor_calculator,
        **kwargs
    )
