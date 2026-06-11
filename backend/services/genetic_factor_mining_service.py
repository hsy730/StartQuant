"""
遗传算法因子挖掘服务 - 使用 DEAP gp 模块自动发现最优因子

8-Phase 优化:
  Phase 1: 精英策略 + 适应度目标路由（修复 elite_size/fitness_objective 未使用的 bug）
  Phase 2: 简约性压力（防膨胀）
  Phase 3: 多样性保护（去重 + 相似度惩罚）
  Phase 4: 因子值缓存（避免重复计算）
  Phase 5: 向量化滚动 IC（在 factor_validation_service 中实现）
  Phase 6: 交叉验证过拟合控制
  Phase 7: 扩展基元集（9→~25，含时序窗口操作）
  Phase 8: 前端更新
"""

import logging
import operator
import threading
from typing import List, Dict, Optional
from collections import OrderedDict
import pandas as pd
import numpy as np
import random

logger = logging.getLogger(__name__)

try:
    from deap import base, creator, tools, algorithms, gp  # noqa: F401

    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP库未安装，遗传算法功能将不可用")

from backend.services.base_mining_service import BaseMiningService  # noqa: E402
from backend.services.factor_validation_service import factor_validation_service  # noqa: E402
from backend.services.alphalens_analysis_service import alphalens_analysis_service  # noqa: E402
from backend.services.factor_primitives import (  # noqa: E402
    create_pset,
    tree_to_expression,
    tree_to_placeholder_expr,
    compile_tree,
    expression_similarity,
)

MAX_EVAL_STOCKS = 50
DEFAULT_MAX_CACHE_SIZE = 512


def _ensure_creator_types():
    """Idempotently register DEAP creator types (safe across multiple instances).

    Registers both single-objective (FitnessMax) and multi-objective
    (FitnessMulti) fitness classes so that NSGA-II can be used when
    the user requests multi-objective optimisation.
    """
    # Single-objective fitness (backward compatible)
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except RuntimeError:
        logger.debug("FitnessMax已注册，跳过")
    # Multi-objective fitness: (maximise IC, minimise complexity)
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    except RuntimeError:
        logger.debug("FitnessMulti已注册，跳过")
    # Individual for single-objective
    try:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    except RuntimeError:
        logger.debug("Individual已注册，跳过")
    # Individual for multi-objective (NSGA-II)
    try:
        creator.create("IndividualMulti", gp.PrimitiveTree, fitness=creator.FitnessMulti)
    except RuntimeError:
        logger.debug("IndividualMulti已注册，跳过")


class GeneticFactorMiningService(BaseMiningService):
    """遗传算法因子挖掘服务（基于 DEAP gp.PrimitiveTree）

    Quality-boosting mechanisms:

    1. **Elitism** – top elite_size individuals carried over unchanged. (Phase 1)
    2. **Fitness Objective Routing** – ic_mean / ir_ratio / sharpe / combined. (Phase 1)
    3. **Parsimony Pressure** – penalises overly complex trees. (Phase 2)
    4. **Diversity Protection** – duplicate removal + similarity penalty. (Phase 3)
    5. **Factor Value Cache** – avoids redundant computation. (Phase 4)
    6. **Cross-Validation** – train/test split penalty for over-fitting control. (Phase 6)
    7. **NSGA-II Multi-objective** – maximise IC, minimise complexity. (Phase 7)
    8. **Extended Primitives** – 9→~25 operators incl. time-series windows. (Phase 7)
    """

    _service_name = "遗传规划"

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        population_size: int = 50,
        n_generations: int = 20,
        cx_prob: float = 0.7,
        mut_prob: float = 0.3,
        factor_calculator=None,
        max_eval_stocks: int = MAX_EVAL_STOCKS,
        # ---- Phase 1: Elitism + fitness objective ----
        elite_size: int = 5,
        fitness_objective: str = "ic_mean",
        # ---- Phase 2: Parsimony pressure ----
        parsimony_coeff: float = 0.001,
        # ---- Phase 3 & 4: Diversity + cache ----
        diversity_penalty_coeff: float = 0.1,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        # ---- Phase 6: Cross-validation ----
        cv_folds: int = 0,
        # ---- Phase 7: Extended primitives ----
        use_extended_primitives: bool = True,
        max_tree_depth: int = 17,
        # ---- NSGA-II ----
        use_nsga2: bool = True,
    ):
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP库未安装，请运行: pip install DEAP")

        super().__init__(
            base_factors=base_factors,
            data=data,
            return_column=return_column,
            factor_calculator=factor_calculator,
            max_eval_stocks=max_eval_stocks,
            fitness_objective=fitness_objective,
            cv_folds=cv_folds,
        )

        self.population_size = population_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob

        # Phase 1
        self.elite_size = min(elite_size, population_size)

        # Phase 2
        self.parsimony_coeff = parsimony_coeff

        # Phase 3 & 4
        self.diversity_penalty_coeff = diversity_penalty_coeff
        self.max_cache_size = max_cache_size

        # Phase 7
        self.use_extended_primitives = use_extended_primitives
        self.max_tree_depth = max_tree_depth

        # NSGA-II
        self.use_nsga2 = use_nsga2

        self._current_generation: int = 0

        # Phase 4: factor value cache (keyed by tree string)
        self._factor_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()

        # Build tradable_mask from OHLC data (detect limit-up/down days)
        self.tradable_mask: Optional[pd.Series] = self._build_tradable_mask()

        # GP primitives & toolbox
        self.pset: Optional[gp.PrimitiveSet] = None
        self._setup_genetic_algorithm()

    # ------------------------------------------------------------------
    # Tradable mask construction (limit-up/down detection)
    # ------------------------------------------------------------------

    def _build_tradable_mask(self) -> Optional[pd.Series]:
        """Build a tradable mask from OHLC data to filter limit-up/down days.

        A day is marked as NOT tradable (False) when:
        - Close == High AND Close >= Open * 1.095 (limit-up, ~10% for main board)
        - Close == Low  AND Close <= Open * 0.905 (limit-down, ~10%)

        Returns None if the data lacks the required columns.
        """
        required = {"open", "close", "high", "low"}
        if not required.issubset(self.data.columns):
            logger.info("无法构建tradable_mask: 数据缺少OHLC列")
            return None

        try:
            close = self.data["close"]
            open_ = self.data["open"]
            high = self.data["high"]
            low = self.data["low"]

            # 涨停: 收盘=最高 且 涨幅>=9.5%
            limit_up = (close == high) & (close >= open_ * 1.095)
            # 跌停: 收盘=最低 且 跌幅>=9.5%
            limit_down = (close == low) & (close <= open_ * 0.905)

            mask = ~(limit_up | limit_down)
            n_excluded = (~mask).sum()
            logger.info(f"tradable_mask构建完成: {mask.sum()}个可交易日, {n_excluded}个涨跌停日已排除")
            return mask
        except Exception as e:
            logger.warning(f"构建tradable_mask失败: {e}")
            return None

    # ------------------------------------------------------------------
    # GP setup
    # ------------------------------------------------------------------

    def _setup_genetic_algorithm(self):
        _ensure_creator_types()

        n_factors = max(len(self.base_factor_values), 1)
        self.pset = create_pset(n_factors, extended=self.use_extended_primitives, tradable_mask=self.tradable_mask)

        self.toolbox = base.Toolbox()
        # Phase 7: deeper initial trees when extended primitives + parsimony control bloat
        init_max_depth = 5 if self.use_extended_primitives else 3
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=init_max_depth)

        # Choose individual class based on multi-objective flag
        if self.use_nsga2:
            individual_class = creator.IndividualMulti
        else:
            individual_class = creator.Individual

        self.toolbox.register("individual", tools.initIterate, individual_class, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # GP operators
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)

        # Depth limiter for crossover & mutation (prevents bloat)
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_depth),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_tree_depth),
        )

        # Selection operator: NSGA-II for multi-objective, tournament otherwise
        if self.use_nsga2:
            self.toolbox.register("select", tools.selNSGA2)
        else:
            self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Evaluation
        if self.use_nsga2:
            self.toolbox.register("evaluate", self._evaluate_factor_multi)
        else:
            self.toolbox.register("evaluate", self._evaluate_factor)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        # Statistics — only track the primary objective (IC fitness) to avoid
        # complexity values contaminating max/avg in NSGA-II mode.
        self.stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 4: Factor value cache
    # ------------------------------------------------------------------

    def _cache_get(self, tree_str: str) -> Optional[Dict[str, pd.Series]]:
        """Look up pre-computed factor values for a tree expression."""
        with self._cache_lock:
            return self._factor_cache.get(tree_str)

    def _cache_set(self, tree_str: str, values: Dict[str, pd.Series]):
        """Store factor values in the LRU cache."""
        with self._cache_lock:
            if len(self._factor_cache) >= self.max_cache_size:
                # evict oldest entry
                self._factor_cache.popitem(last=False)
            self._factor_cache[tree_str] = values

    def _cache_clear(self):
        """Clear the factor value cache (call once per generation)."""
        with self._cache_lock:
            self._factor_cache.clear()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_factor(self, individual) -> tuple:
        """Evaluate a PrimitiveTree individual by cross-sectional or time-series IC.

        Applies: Parsimony Pressure (Phase 2), Diversity Penalty (Phase 3),
        Cross-Validation penalty (Phase 6), Fitness Objective Routing (Phase 1).
        """
        if len(individual) == 0:
            return (0.0,)

        try:
            if len(self.stock_pool_data) >= 2:
                raw_fitness = self._evaluate_cross_sectional_ic(individual)[0]
            elif len(self.stock_pool_data) == 1:
                logger.warning("Only 1 stock in pool, falling back to time-series IC evaluation")
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
            else:
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
        except Exception as e:
            logger.debug(f"适应度评估失败: {e}")
            return (0.0,)

        # --- Parsimony Pressure (Phase 2) ---
        parsimony_penalty = self.parsimony_coeff * len(individual)

        # --- Diversity Penalty (Phase 3) ---
        diversity_penalty = 0.0
        if self.diversity_penalty_coeff > 0 and hasattr(self, "_halloffame") and self._halloffame is not None:
            ind_expr = tree_to_placeholder_expr(individual)
            for hof_ind in self._halloffame:
                hof_expr = tree_to_placeholder_expr(hof_ind)
                sim = expression_similarity(ind_expr, hof_expr)
                if sim > 0.7:
                    diversity_penalty += self.diversity_penalty_coeff * sim

        adjusted_fitness = raw_fitness - parsimony_penalty - diversity_penalty
        return (adjusted_fitness,)

    def _evaluate_factor_multi(self, individual) -> tuple:
        """Multi-objective evaluation for NSGA-II.

        Returns a 2-tuple ``(ic_fitness, complexity)`` where:
        - ``ic_fitness`` is the raw IC-based fitness (to be maximised).
        - ``complexity`` is the tree node count (to be minimised, hence the
          negative weight in ``FitnessMulti``).

        Parsimony pressure is *not* applied here because complexity is
        already an explicit second objective.  Diversity penalty and CV
        penalty are still applied to the IC objective.
        """
        if len(individual) == 0:
            return (0.0, 1.0)

        try:
            if len(self.stock_pool_data) >= 2:
                raw_fitness = self._evaluate_cross_sectional_ic(individual)[0]
            elif len(self.stock_pool_data) == 1:
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
            else:
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
        except Exception as e:
            logger.debug(f"NSGA2适应度评估失败: {e}")
            return (0.0, 1.0)

        # --- Diversity Penalty (Phase 3, applied to IC objective only) ---
        diversity_penalty = 0.0
        if self.diversity_penalty_coeff > 0 and hasattr(self, "_halloffame") and self._halloffame is not None:
            ind_expr = tree_to_placeholder_expr(individual)
            for hof_ind in self._halloffame:
                hof_expr = tree_to_placeholder_expr(hof_ind)
                sim = expression_similarity(ind_expr, hof_expr)
                if sim > 0.7:
                    diversity_penalty += self.diversity_penalty_coeff * sim

        ic_fitness = max(raw_fitness - diversity_penalty, 0.0)
        complexity = float(len(individual))
        return (ic_fitness, complexity)

    def _eval_tree_on_stock(self, tree, stock_code: str, stock_base_factors: dict) -> Optional[pd.Series]:
        """Compile a tree and evaluate it using one stock's base factor values.

        Phase 4: Results are cached per (tree_str, stock_code) within a
        generation so that the same expression is never computed twice.
        """
        tree_str = str(tree)
        # Phase 4: check cache first
        cached = self._cache_get(tree_str)
        if cached is not None and stock_code in cached:
            return cached[stock_code]

        try:
            func = compile_tree(tree, self.pset)
        except Exception as e:
            logger.debug(f"编译表达式失败: {e}")
            return None

        # Build ordered positional args matching factor_0 … factor_N
        ordered = []
        for i in range(len(self.base_factor_values)):
            info = stock_base_factors.get(f"factor_{i}")
            if info is None:
                return None
            ordered.append(info["values"])

        try:
            result = func(*ordered)
        except Exception as e:
            logger.debug(f"执行表达式失败: {e}")
            return None

        if isinstance(result, (int, float, np.number)):
            # scalar → broadcast to a Series using the first factor's index
            idx = ordered[0].index if ordered else None
            if idx is None:
                return None
            result = pd.Series(float(result), index=idx)

        if not isinstance(result, pd.Series):
            return None

        result = result.replace([np.inf, -np.inf], np.nan)
        valid_count = result.notna().sum()
        if valid_count == 0 or valid_count < len(result) * 0.1:
            return None

        # Phase 4: store in cache
        with self._cache_lock:
            cached = self._factor_cache.get(tree_str)
            if cached is None:
                cached = {}
                if len(self._factor_cache) >= self.max_cache_size:
                    self._factor_cache.popitem(last=False)
                self._factor_cache[tree_str] = cached
            cached[stock_code] = result

        return result

    def _evaluate_cross_sectional_ic(self, tree) -> tuple:
        """Cross-sectional IC evaluation (multi-stock).

        Phase 1: Uses ``_route_fitness`` to select the objective metric.
        Phase 4: Uses factor value cache to avoid redundant computation.
        Phase 6: Applies cross-validation penalty when cv_folds > 0.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        eval_codes = self._sampled_stock_codes
        factor_values_dict: Dict[str, pd.Series] = {}

        # Phase 4: check if full result is cached
        tree_str = str(tree)
        cached_all = self._cache_get(tree_str)
        if cached_all is not None and "_complete" in cached_all:
            factor_values_dict = cached_all["_complete"]
        else:

            def _calc_one_stock(code):
                try:
                    base_factors = self.stock_pool_base_factor_values[code]
                    fv = self._eval_tree_on_stock(tree, code, base_factors)
                    if fv is not None and len(fv.dropna()) >= 10:
                        return code, fv.dropna()
                except Exception as e:
                    logger.debug(f"评估股票 {code} 因子失败: {e}")
                return code, None

            with ThreadPoolExecutor(max_workers=min(len(eval_codes), 10)) as executor:
                futures = {executor.submit(_calc_one_stock, code): code for code in eval_codes}
                for future in as_completed(futures):
                    code, fv = future.result()
                    if fv is not None:
                        factor_values_dict[code] = fv

            # Phase 4: cache the complete result
            self._cache_set(tree_str, {"_complete": factor_values_dict})

        if len(factor_values_dict) < 2:
            return (0.0,)

        try:
            all_dates = set()
            for s in factor_values_dict.values():
                all_dates.update(s.index)
            all_dates = sorted(all_dates)

            pricing_df = pd.DataFrame(index=all_dates)
            for stock_code in factor_values_dict:
                df = self.stock_pool_data.get(stock_code)
                if df is not None and "close" in df.columns:
                    pricing_df[stock_code] = df["close"]
            pricing_df = pricing_df.dropna(how="all")

            factor_data = alphalens_analysis_service.prepare_factor_data(
                factor_values_dict=factor_values_dict,
                pricing_df=pricing_df,
            )

            if factor_data is None or factor_data.empty:
                return (0.0,)

            ic_results = alphalens_analysis_service.analyze_ic(factor_data)

            if "error" in ic_results:
                return (0.0,)

            # Phase 1: route fitness based on objective
            raw_fitness = self._route_fitness(ic_results, factor_values_dict)

            # Phase 6: cross-validation penalty
            cv_penalty = self._cv_penalty(factor_values_dict)
            raw_fitness = raw_fitness * (1.0 - cv_penalty)

            return (raw_fitness,)

        except Exception as e:
            logger.warning(f"Cross-sectional IC evaluation failed: {e}")
            return (0.0,)

    def _evaluate_single_stock_ic(self, tree) -> tuple:
        """Time-series IC evaluation (single stock / fallback).

        Uses factor validation service when possible (returns fitness in [0, 1]).
        Falls back to rank-IC against forward returns only when validation is
        unavailable.  The old ``std / mean`` fallback is removed because it
        produces unbounded fitness values that are incompatible with the
        progress callback and fitness_history display.
        """
        fv = self._compute_factor_expression(tree)
        if fv is None or len(fv.dropna()) < 10:
            return (0.0,)

        if self.return_values is not None:
            validation = factor_validation_service.validate_factor(
                factor_values=fv,
                return_values=self.return_values,
                existing_factors=None,
            )
            score_val = validation.get("score")
            fitness = (score_val / 100.0) if score_val is not None else 0.0
        else:
            # Compute forward returns from close prices for time-series IC
            try:
                close = self.data["close"]
                from backend.utils.safe_math import safe_series_divide

                fwd_ret = safe_series_divide(close.shift(-1), close, default=np.nan) - 1
                fwd_ret = fwd_ret.dropna()
                fv_aligned = fv.reindex(fwd_ret.index).dropna()

                if len(fv_aligned) < 10:
                    return (0.0,)

                # Spearman rank IC as fitness (bounded in [-1, 1], take abs)
                from scipy.stats import spearmanr

                corr, _ = spearmanr(fv_aligned, fwd_ret.reindex(fv_aligned.index))
                fitness = abs(corr) if not np.isnan(corr) else 0.0
            except Exception as e:
                logger.debug(f"Single stock IC fallback failed: {e}")
                fitness = 0.0

        return (fitness,)

    # ------------------------------------------------------------------
    # Expression computation (compiled tree → Series)
    # ------------------------------------------------------------------

    def _compute_factor_expression(self, tree) -> Optional[pd.Series]:
        """Evaluate a PrimitiveTree using the global base factor cache."""
        if len(tree) == 0:
            return None
        try:
            func = compile_tree(tree, self.pset)
        except Exception as e:
            logger.debug(f"编译表达式失败: {e}")
            return None

        ordered = []
        for i in range(len(self.base_factor_values)):
            info = self.base_factor_values.get(f"factor_{i}")
            if info is None:
                return None
            ordered.append(info["values"])

        try:
            result = func(*ordered)
        except Exception as e:
            logger.debug(f"执行表达式失败: {e}")
            return None

        if isinstance(result, (int, float, np.number)):
            result = pd.Series(float(result), index=self.data.index)

        if not isinstance(result, pd.Series):
            return None

        result = result.replace([np.inf, -np.inf], np.nan)
        valid = result.notna().sum()
        if valid == 0 or valid < len(result) * 0.1:
            return None
        return result

    # ------------------------------------------------------------------
    # Expression conversion
    # ------------------------------------------------------------------

    def _convert_expression_to_code(self, tree) -> str:
        """Convert a PrimitiveTree to an expression string with real factor codes."""
        mapping = {}
        for var_name, info in self.base_factor_values.items():
            mapping[var_name] = info["code"]
        return tree_to_expression(tree, mapping)

    # ------------------------------------------------------------------
    # Mining entry point
    # ------------------------------------------------------------------

    def mine_factors(self) -> Dict:
        """Execute genetic-programming-based factor mining.

        The evolutionary loop now includes:

        * **Elitism** – the top ``elite_size`` individuals are carried over
          to the next generation unchanged.
        * **Diversity Penalty** – the Hall-of-Fame is kept as ``self._halloffame``
          so that the evaluation functions can penalise individuals that are
          structurally similar to elite ones.
        * **NSGA-II** – when ``use_nsga2=True``, selection uses the
          non-dominated sorting algorithm, balancing IC fitness against
          expression complexity.

        Returns
        -------
        dict with keys: ``success``, ``best_factors``, ``logbook``, ``final_population``
        """
        if not DEAP_AVAILABLE:
            return {"success": False, "message": "DEAP库未安装", "best_factors": []}

        logger.info("开始遗传规划因子挖掘...")
        logger.info(f"种群大小: {self.population_size}, 迭代代数: {self.n_generations}")
        logger.info(
            f"增强参数: parsimony={self.parsimony_coeff}, nsga2={self.use_nsga2}, "
            f"elite={self.elite_size}, fitness_objective={self.fitness_objective}, "
            f"diversity_penalty={self.diversity_penalty_coeff}, "
            f"cv_folds={self.cv_folds}, extended_primitives={self.use_extended_primitives}, "
            f"max_depth={self.max_tree_depth}"
        )

        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Update Z-Score normalization stats from initial population
        self._update_zscore_stats()

        # Hall-of-Fame (stored as instance attr so evaluation can access it)
        hof_size = max(self.elite_size * 2, 10)
        halloffame = tools.HallOfFame(hof_size)
        halloffame.update(population)
        self._halloffame = halloffame  # expose for diversity penalty

        logbook = tools.Logbook()
        record = self.stats.compile(population)
        logbook.record(gen=0, **record)

        # 委托通用进化循环
        self._evolutionary_loop(
            population,
            self.n_generations,
            halloffame,
            logbook=logbook,
            use_nsga2=self.use_nsga2,
            diversity_protection=True,
            progress=True,
        )

        # Build result
        best_factors = []
        for i, tree in enumerate(halloffame):
            actual_expr = self._convert_expression_to_code(tree)
            placeholder_expr = tree_to_placeholder_expr(tree)

            # Extract primary fitness (IC-based)
            fitness_values = tree.fitness.values
            if self.use_nsga2:
                primary_fitness = float(fitness_values[0])
                complexity = float(fitness_values[1]) if len(fitness_values) > 1 else float(len(tree))
            else:
                primary_fitness = float(fitness_values[0])
                complexity = float(len(tree))

            factor_info = {
                "rank": i + 1,
                "expression": actual_expr,
                "placeholder_expression": placeholder_expr,
                "fitness": primary_fitness,
                "complexity": complexity,
            }

            try:
                fv = self._compute_factor_expression(tree)
                if fv is not None and self.return_values is not None:
                    validation = factor_validation_service.validate_factor(
                        factor_values=fv,
                        return_values=self.return_values,
                    )
                    factor_info["validation"] = validation
            except Exception as e:
                logger.debug(f"因子验证失败: {e}")

            best_factors.append(factor_info)

        # Sort by validation score, fallback to fitness
        def _sort_key(f):
            v = f.get("validation", {})
            if v and isinstance(v, dict):
                score = v.get("score")
                if score is not None:
                    return score
            fitness = f.get("fitness")
            return fitness if fitness is not None else 0

        best_factors.sort(key=_sort_key, reverse=True)
        for idx, fi in enumerate(best_factors):
            fi["rank"] = idx + 1

        result = {
            "success": True,
            "best_factors": best_factors,
            "logbook": logbook,
            "final_population_size": len(population),
        }

        # Release memory held by DEAP population and Hall-of-Fame objects
        self._halloffame = None

        return result

    # ------------------------------------------------------------------
    # Evolve from seed
    # ------------------------------------------------------------------

    def _evolutionary_loop(
        self,
        population,
        n_generations: int,
        halloffame,
        logbook=None,
        use_nsga2: bool = False,
        diversity_protection: bool = True,
        progress: bool = True,
    ):
        """通用进化循环 — mine_factors 和 evolve_factor 共享的核心逻辑。

        Parameters
        ----------
        population : list
            初始种群（已评估）
        n_generations : int
            迭代代数
        halloffame : tools.HallOfFame
            精英保留容器
        logbook : tools.Logbook or None
            日志记录器（evolve_factor 不需要）
        use_nsga2 : bool
            是否使用 NSGA-II 选择
        diversity_protection : bool
            是否启用 Phase 3 去重替换
        progress : bool
            是否记录进度日志和回调
        """
        for gen in range(1, n_generations + 1):
            # 取消检查
            if self._cancel_flag:
                logger.info(f"挖掘任务在第 {gen} 代被用户取消")
                break

            self._current_generation = gen
            self._refresh_stock_sample()

            # Phase 4: clear factor value cache at generation boundary
            self._cache_clear()

            # ---- Elitism: copy top elite_size individuals unchanged ----
            if use_nsga2:
                elites = tools.selNSGA2(population, self.elite_size)
            else:
                elites = tools.selBest(population, self.elite_size)
            elites = list(map(self.toolbox.clone, elites))

            # ---- Selection ----
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # ---- Crossover ----
            for i in range(1, len(offspring), 2):
                if random.random() < self.cx_prob:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            # ---- Mutation ----
            for i in range(len(offspring)):
                if random.random() < self.mut_prob:
                    (offspring[i],) = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # ---- Phase 3: Diversity protection – replace duplicates ----
            if diversity_protection:
                seen_exprs: Dict[str, int] = {}
                n_duplicates = 0
                for i, ind in enumerate(offspring):
                    expr_key = str(ind)
                    if expr_key in seen_exprs:
                        new_ind = self.toolbox.individual()
                        offspring[i] = new_ind
                        n_duplicates += 1
                    else:
                        seen_exprs[expr_key] = i

            # ---- Re-evaluate invalid individuals ----
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # ---- Replace population: offspring + elites ----
            population[:] = offspring + elites

            # Update Z-Score normalization stats from this generation's evaluations
            self._update_zscore_stats()

            halloffame.update(population)

            if logbook is not None:
                record = self.stats.compile(population)
                logbook.record(gen=gen, **record)

            if progress:
                record = self.stats.compile(population) if logbook is None else logbook[-1]
                best_fit = float(record["max"]) if record.get("max") is not None else 0.0
                avg_fit = float(record["avg"]) if record.get("avg") is not None else 0.0

                if self.progress_callback:
                    self.progress_callback(gen, n_generations, best_fit, avg_fit)

                logger.info(
                    f"Generation {gen}/{n_generations} - Best: {best_fit:.4f}, "
                    f"Avg: {avg_fit:.4f}, Elite: {self.elite_size}"
                )

        return population

    def evolve_factor(self, initial_expression: str, n_generations: int = 10) -> Dict:
        """Evolve a population seeded with a user-provided expression.

        The *initial_expression* is treated as a label only – the actual initial
        population is generated via ``genHalfAndHalf`` because the string-based
        infix format from the old representation cannot be reliably parsed into a
        ``PrimitiveTree``.  This keeps the API compatible while using GP
        initialisation.

        The method now also benefits from elitism and diversity penalty.
        """
        if not DEAP_AVAILABLE:
            return {"success": False, "message": "DEAP库未安装"}

        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Update Z-Score normalization stats from initial population
        self._update_zscore_stats()

        halloffame = tools.HallOfFame(5)
        halloffame.update(population)
        self._halloffame = halloffame

        # 委托通用进化循环（简化版：无日志、无NSGA2、无去重保护）
        self._evolutionary_loop(
            population,
            n_generations,
            halloffame,
            logbook=None,
            use_nsga2=False,
            diversity_protection=False,
            progress=False,
        )

        best = halloffame[0]
        original_fitness = float(best.fitness.values[0])

        return {
            "success": True,
            "original_expression": initial_expression,
            "evolved_expression": str(best),
            "original_fitness": original_fitness,
            "evolved_fitness": original_fitness,
            "improvement": 0.0,
        }


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_genetic_mining_service(
    base_factors: List[str], data: pd.DataFrame, factor_calculator=None, **kwargs
) -> GeneticFactorMiningService:
    """Create a configured :class:`GeneticFactorMiningService` instance.

    Accepted keyword arguments (forwarded to the constructor):

    * ``population_size``, ``n_generations``, ``cx_prob``, ``mut_prob``
    * ``elite_size`` – number of elite individuals preserved (default 5)
    * ``fitness_objective`` – ic_mean / ir_ratio / sharpe / combined (default ic_mean)
    * ``parsimony_coeff`` – weight of the complexity penalty (default 0.001)
    * ``diversity_penalty_coeff`` – penalty for similarity to HoF (default 0.1)
    * ``max_cache_size`` – max entries in factor value cache (default 512)
    * ``cv_folds`` – cross-validation folds for over-fitting control (0=off, default 0)
    * ``use_extended_primitives`` – enable ~25 operators incl. time-series (default True)
    * ``use_nsga2`` – enable NSGA-II multi-objective (default True)
    * ``max_tree_depth`` – hard depth limit for GP trees (default 17)
    """
    return GeneticFactorMiningService(
        base_factors=base_factors, data=data, factor_calculator=factor_calculator, **kwargs
    )
