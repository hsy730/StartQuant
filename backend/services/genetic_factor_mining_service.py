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
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import pandas as pd
import numpy as np
import random

logger = logging.getLogger(__name__)

try:
    from deap import base, creator, tools, algorithms, gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP库未安装，遗传算法功能将不可用")

from backend.services.factor_generator_service import factor_generator_service
from backend.services.factor_validation_service import factor_validation_service
from backend.services.alphalens_analysis_service import alphalens_analysis_service, ALPHALENS_AVAILABLE
from backend.services.data_service import data_service
from backend.services.factor_primitives import (
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
        pass
    # Multi-objective fitness: (maximise IC, minimise complexity)
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    except RuntimeError:
        pass
    # Individual for single-objective
    try:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    except RuntimeError:
        pass
    # Individual for multi-objective (NSGA-II)
    try:
        creator.create("IndividualMulti", gp.PrimitiveTree, fitness=creator.FitnessMulti)
    except RuntimeError:
        pass


class GeneticFactorMiningService:
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

        self.base_factor_codes = base_factors
        self.data = data
        self.return_column = return_column
        self.population_size = population_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.factor_calculator = factor_calculator
        self.max_eval_stocks = max_eval_stocks

        # Phase 1
        self.elite_size = min(elite_size, population_size)
        self.fitness_objective = fitness_objective

        # Phase 2
        self.parsimony_coeff = parsimony_coeff

        # Phase 3 & 4
        self.diversity_penalty_coeff = diversity_penalty_coeff
        self.max_cache_size = max_cache_size

        # Phase 6
        self.cv_folds = cv_folds

        # Phase 7
        self.use_extended_primitives = use_extended_primitives
        self.max_tree_depth = max_tree_depth

        # NSGA-II
        self.use_nsga2 = use_nsga2

        self.return_values = data[return_column] if return_column in data.columns else None

        self.stock_codes: List[str] = []
        self.stock_pool_data: Dict[str, pd.DataFrame] = {}
        self.stock_pool_return_values: Dict[str, pd.Series] = {}
        self.stock_pool_base_factor_values: Dict[str, dict] = {}

        self._sampled_stock_codes: List[str] = []
        self._current_generation: int = 0

        # Phase 4: factor value cache (keyed by tree string)
        self._factor_cache: OrderedDict = OrderedDict()

        # Phase 1: Generational Z-Score normalization for combined objective
        # Collect raw IC/IR per generation → compute Z-Score stats → apply next gen
        self._gen_ic_values: List[float] = []  # raw IC values collected in current generation
        self._gen_ir_values: List[float] = []  # raw IR values collected in current generation
        # Prior cold-start values based on domain knowledge of quantitative factors
        _PRIOR_IC_MEAN = 0.03
        _PRIOR_IC_STD = 0.02
        _PRIOR_IR_MEAN = 0.5
        _PRIOR_IR_STD = 0.3
        self._zscore_ic_mean: float = _PRIOR_IC_MEAN  # μ of IC from previous generation
        self._zscore_ic_std: float = _PRIOR_IC_STD    # σ of IC from previous generation
        self._zscore_ir_mean: float = _PRIOR_IR_MEAN  # μ of IR from previous generation
        self._zscore_ir_std: float = _PRIOR_IR_STD    # σ of IR from previous generation
        self._has_zscore_stats: bool = True  # Prior values are valid from the start

        # Pre-computed factor cache
        self.base_factor_values: Dict[str, dict] = {}
        self._precompute_base_factors()

        # Build tradable_mask from OHLC data (detect limit-up/down days)
        self.tradable_mask: Optional[pd.Series] = self._build_tradable_mask()

        # GP primitives & toolbox
        self.pset: Optional[gp.PrimitiveSet] = None
        self._setup_genetic_algorithm()

    # ------------------------------------------------------------------
    # Stock pool (cross-sectional IC evaluation support)
    # ------------------------------------------------------------------

    def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
        self.stock_codes = stock_codes
        self.stock_pool_data = data_service.get_multiple_stocks_data(stock_codes, start_date, end_date)

        for code, df in self.stock_pool_data.items():
            if "close" in df.columns:
                df["return"] = df["close"].pct_change()
            self.stock_pool_return_values[code] = (
                df[self.return_column] if self.return_column in df.columns else None
            )

            if self.factor_calculator is None:
                from backend.services.factor_service import factor_service
                self.factor_calculator = factor_service.calculator

            stock_base_factors = {}
            for i, factor_code in enumerate(self.base_factor_codes):
                try:
                    fv = self.factor_calculator.calculate(df, factor_code)
                    if fv is not None and len(fv.dropna()) > 0:
                        var_name = f"factor_{i}"
                        stock_base_factors[var_name] = {
                            "code": factor_code,
                            "values": fv,
                        }
                except Exception as e:
                    logger.warning(f"Stock {code} factor {factor_code} compute error: {e}")
            self.stock_pool_base_factor_values[code] = stock_base_factors

        self._refresh_stock_sample()
        logger.info(
            f"Stock pool set with {len(self.stock_pool_data)} stocks, "
            f"eval sample={len(self._sampled_stock_codes)}: {stock_codes}"
        )

    def _refresh_stock_sample(self):
        available = list(self.stock_pool_base_factor_values.keys())
        if len(available) <= self.max_eval_stocks:
            self._sampled_stock_codes = available
        else:
            self._sampled_stock_codes = random.sample(available, self.max_eval_stocks)

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
    # Base factor precomputation
    # ------------------------------------------------------------------

    def _precompute_base_factors(self):
        if self.factor_calculator is None:
            from backend.services.factor_service import factor_service
            self.factor_calculator = factor_service.calculator

        logger.info(f"预计算 {len(self.base_factor_codes)} 个基础因子...")

        for i, factor_code in enumerate(self.base_factor_codes):
            try:
                fv = self.factor_calculator.calculate(self.data, factor_code)
                if fv is not None and len(fv.dropna()) > 0:
                    var_name = f"factor_{i}"
                    self.base_factor_values[var_name] = {
                        "code": factor_code,
                        "values": fv,
                    }
                    logger.info(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: {len(fv.dropna())} 个有效值")
                else:
                    logger.warning(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算失败或无有效值")
            except Exception as e:
                logger.warning(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算出错 - {e}")

        logger.info(f"成功预计算 {len(self.base_factor_values)} 个基础因子")

    # ------------------------------------------------------------------
    # GP setup
    # ------------------------------------------------------------------

    def _setup_genetic_algorithm(self):
        _ensure_creator_types()

        n_factors = max(len(self.base_factor_values), 1)
        self.pset = create_pset(n_factors, extended=self.use_extended_primitives,
                                tradable_mask=self.tradable_mask)

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

        self.progress_callback = None
        self._cancel_flag = False

    def set_progress_callback(self, callback):
        """设置进度回调函数

        Args:
            callback: 签名为 callback(generation, total_generations, best_fitness, avg_fitness)
        """
        self.progress_callback = callback

    def request_cancel(self):
        """请求取消挖掘任务"""
        self._cancel_flag = True
        logger.info("收到取消请求，将在当前代结束后停止")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 4: Factor value cache
    # ------------------------------------------------------------------

    def _cache_get(self, tree_str: str) -> Optional[Dict[str, pd.Series]]:
        """Look up pre-computed factor values for a tree expression."""
        return self._factor_cache.get(tree_str)

    def _cache_set(self, tree_str: str, values: Dict[str, pd.Series]):
        """Store factor values in the LRU cache."""
        if len(self._factor_cache) >= self.max_cache_size:
            # evict oldest entry
            self._factor_cache.popitem(last=False)
        self._factor_cache[tree_str] = values

    def _cache_clear(self):
        """Clear the factor value cache (call once per generation)."""
        self._factor_cache.clear()

    # ------------------------------------------------------------------
    # Phase 6: Cross-validation
    # ------------------------------------------------------------------

    def _cv_penalty(self, factor_values_dict: Dict[str, pd.Series]) -> float:
        """Compute a cross-validation penalty for over-fitting control.

        Splits the time-series into *cv_folds* segments, computes IC on
        each, and returns ``1.0 - (min_fold_ic / max_fold_ic)`` clamped
        to [0, 1].  A factor whose IC is consistent across folds gets
        penalty ≈ 0; one that collapses on some folds gets penalty → 1.
        """
        if self.cv_folds < 2:
            return 0.0

        fold_ics: List[float] = []
        for stock_code, fv in factor_values_dict.items():
            ret = self.stock_pool_return_values.get(stock_code)
            if ret is None:
                continue
            aligned = pd.DataFrame({"factor": fv, "return": ret}).dropna()
            if len(aligned) < self.cv_folds * 20:
                continue

            n = len(aligned)
            fold_size = n // self.cv_folds
            for k in range(self.cv_folds):
                start = k * fold_size
                end = start + fold_size if k < self.cv_folds - 1 else n
                segment = aligned.iloc[start:end]
                if len(segment) >= 10:
                    ic = segment["factor"].corr(segment["return"])
                    if not np.isnan(ic):
                        fold_ics.append(abs(ic))

        if len(fold_ics) < self.cv_folds:
            return 0.0

        min_ic = min(fold_ics)
        max_ic = max(fold_ics)
        if max_ic < 1e-10:
            return 1.0
        penalty = 1.0 - (min_ic / max_ic)
        return max(0.0, min(penalty, 1.0))

    # ------------------------------------------------------------------
    # Phase 1: Fitness objective routing
    # ------------------------------------------------------------------

    def _route_fitness(self, ic_results: dict, factor_values_dict: Optional[Dict[str, pd.Series]] = None) -> float:
        """Select the fitness value according to ``self.fitness_objective``.

        Supported objectives:
        - ``ic_mean``  : best absolute mean IC across Spearman/Pearson (default)
        - ``ir_ratio`` : IC mean / IC std (information ratio)
        - ``sharpe``   : Sharpe-like ratio of the long-short portfolio
        - ``combined`` : weighted blend of *Z-Score normalized* ic_mean and ir_ratio

        For ``combined``, IC and IR have very different scales (IC ≈ 0.01–0.10,
        IR ≈ 0.3–2.0), which makes naive 0.6*IC + 0.4*IR meaningless (IR dominates).
        Instead, we apply Z-Score normalization using statistics from the *previous*
        generation's batch of evaluations, then clip to [-3, 3] and shift to [0, 1]:

            z_ic = clip((IC - μ_ic) / (σ_ic + ε), -3, 3)
            z_ir = clip((IR - μ_ir) / (σ_ir + ε), -3, 3)
            Norm(IC) = (z_ic + 3) / 6      → maps [-3σ, +3σ] to [0, 1]
            Norm(IR) = (z_ir + 3) / 6
            combined  = 0.6 * Norm(IC) + 0.4 * Norm(IR)

        The 60/40 weighting is now meaningful because both components are on the
        same [0, 1] scale. Statistics are computed at generation boundaries in
        ``_update_zscore_stats()``.
        """
        best_ic = 0.0
        best_ir = 0.0

        for ic_type in ["spearman_ic", "pearson_ic"]:
            ic_type_data = ic_results.get(ic_type, {})
            for period_key, period_stats in ic_type_data.items():
                if not isinstance(period_stats, dict) or "error" in period_stats:
                    continue
                mean_ic = period_stats.get("mean_ic")
                std_ic = period_stats.get("std_ic")
                if mean_ic is None or std_ic is None:
                    continue
                mean_ic = abs(float(mean_ic))
                std_ic = float(std_ic)
                ir = abs(mean_ic / std_ic) if std_ic > 1e-10 else 0.0
                if mean_ic > best_ic:
                    best_ic = mean_ic
                if ir > best_ir:
                    best_ir = ir

        # Collect raw IC/IR for generational Z-Score computation
        self._gen_ic_values.append(best_ic)
        self._gen_ir_values.append(best_ir)

        if self.fitness_objective == "ir_ratio":
            return best_ir
        elif self.fitness_objective == "sharpe":
            # Approximate Sharpe: use IR as proxy
            return best_ir
        elif self.fitness_objective == "combined":
            # Z-Score normalize using previous generation's statistics (with prior cold-start)
            z_ic = max(-3.0, min((best_ic - self._zscore_ic_mean) / (self._zscore_ic_std + 1e-8), 3.0))
            z_ir = max(-3.0, min((best_ir - self._zscore_ir_mean) / (self._zscore_ir_std + 1e-8), 3.0))
            # Map from [-3, 3] to [0, 1]
            norm_ic = (z_ic + 3.0) / 6.0
            norm_ir = (z_ir + 3.0) / 6.0
            return 0.6 * norm_ic + 0.4 * norm_ir
        else:  # ic_mean (default)
            return best_ic

    def _update_zscore_stats(self):
        """Compute Z-Score normalization stats from the current generation's
        collected IC/IR values.  Called at each generation boundary.

        Requirements: at least 5 valid values to compute stable statistics.
        After computing, clears the collection lists for the next generation.
        Applies σ lower-bound protection: max(σ, max(0.01*μ, 0.005)) to
        prevent Z-Score explosion when the population converges.
        """
        valid_ic = [v for v in self._gen_ic_values if v > 1e-10]
        valid_ir = [v for v in self._gen_ir_values if v > 1e-10]

        if len(valid_ic) >= 5 and len(valid_ir) >= 5:
            ic_mean = float(np.mean(valid_ic))
            ic_std = float(np.std(valid_ic))
            ir_mean = float(np.mean(valid_ir))
            ir_std = float(np.std(valid_ir))

            # σ lower-bound protection: prevent Z-Score explosion on convergence
            ic_std = max(ic_std, max(0.01 * ic_mean, 0.005))
            ir_std = max(ir_std, max(0.01 * ir_mean, 0.005))

            if ic_std < self._zscore_ic_std * 0.1 or ir_std < self._zscore_ir_std * 0.1:
                logger.warning(
                    f"Z-Score σ very small (IC σ={ic_std:.6f}, IR σ={ir_std:.6f}), "
                    f"search may be stagnating"
                )

            self._zscore_ic_mean = ic_mean
            self._zscore_ic_std = ic_std
            self._zscore_ir_mean = ir_mean
            self._zscore_ir_std = ir_std
            self._has_zscore_stats = True
            logger.debug(
                f"Z-Score stats updated: IC μ={self._zscore_ic_mean:.4f} σ={self._zscore_ic_std:.4f}, "
                f"IR μ={self._zscore_ir_mean:.4f} σ={self._zscore_ir_std:.4f}"
            )

        # Clear for next generation
        self._gen_ic_values = []
        self._gen_ir_values = []

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
        except Exception:
            return (0.0,)

        # --- Parsimony Pressure (Phase 2) ---
        parsimony_penalty = self.parsimony_coeff * len(individual)

        # --- Diversity Penalty (Phase 3) ---
        diversity_penalty = 0.0
        if self.diversity_penalty_coeff > 0 and hasattr(self, '_halloffame') and self._halloffame is not None:
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
        except Exception:
            return (0.0, 1.0)

        # --- Diversity Penalty (Phase 3, applied to IC objective only) ---
        diversity_penalty = 0.0
        if self.diversity_penalty_coeff > 0 and hasattr(self, '_halloffame') and self._halloffame is not None:
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
        except Exception:
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
        except Exception:
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
        if cached is None:
            cached = {}
        cached[stock_code] = result
        self._cache_set(tree_str, cached)

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
                except Exception:
                    pass
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

        if not ALPHALENS_AVAILABLE:
            logger.warning("alphalens not available, falling back to time-series IC")
            return self._evaluate_single_stock_ic(tree)

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
            fitness = validation["score"] / 100.0
        else:
            # Compute forward returns from close prices for time-series IC
            try:
                close = self.data["close"]
                fwd_ret = close.shift(-1) / close - 1
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
        except Exception:
            return None

        ordered = []
        for i in range(len(self.base_factor_values)):
            info = self.base_factor_values.get(f"factor_{i}")
            if info is None:
                return None
            ordered.append(info["values"])

        try:
            result = func(*ordered)
        except Exception:
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

        for gen in range(1, self.n_generations + 1):
            # 取消检查
            if self._cancel_flag:
                logger.info(f"挖掘任务在第 {gen} 代被用户取消")
                break

            self._current_generation = gen
            self._refresh_stock_sample()

            # Phase 4: clear factor value cache at generation boundary
            self._cache_clear()

            # ---- Elitism: copy top elite_size individuals unchanged ----
            if self.use_nsga2:
                elites = tools.selNSGA2(population, self.elite_size)
            else:
                elites = tools.selBest(population, self.elite_size)
            elites = list(map(self.toolbox.clone, elites))

            # ---- Selection ----
            if self.use_nsga2:
                offspring = self.toolbox.select(population, len(population) - self.elite_size)
            else:
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
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # ---- Phase 3: Diversity protection – replace duplicates ----
            seen_exprs: Dict[str, int] = {}
            n_duplicates = 0
            for i, ind in enumerate(offspring):
                expr_key = str(ind)
                if expr_key in seen_exprs:
                    # Replace duplicate with a fresh random individual
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

            record = self.stats.compile(population)
            logbook.record(gen=gen, **record)

            # Stats now only track the primary objective (IC fitness),
            # so record["max"]/["avg"] are already scalar IC values.
            best_fit = float(record.get("max", 0.0))
            avg_fit = float(record.get("avg", 0.0))

            if self.progress_callback:
                self.progress_callback(gen, self.n_generations, best_fit, avg_fit)

            logger.info(
                f"Generation {gen}/{self.n_generations} - Best: {best_fit:.4f}, "
                f"Avg: {avg_fit:.4f}, Elite: {self.elite_size}"
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
            except Exception:
                pass

            best_factors.append(factor_info)

        # Sort by validation score, fallback to fitness
        def _sort_key(f):
            v = f.get("validation", {})
            if v and isinstance(v, dict):
                return v.get("score", f.get("fitness", 0))
            return f.get("fitness", 0)

        best_factors.sort(key=_sort_key, reverse=True)
        for idx, fi in enumerate(best_factors):
            fi["rank"] = idx + 1

        return {
            "success": True,
            "best_factors": best_factors,
            "logbook": logbook,
            "final_population": population,
        }

    # ------------------------------------------------------------------
    # Evolve from seed
    # ------------------------------------------------------------------

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

        for gen in range(1, n_generations + 1):
            # Elitism
            elites = list(map(self.toolbox.clone, tools.selBest(population, self.elite_size)))

            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))

            for i in range(1, len(offspring), 2):
                if random.random() < self.cx_prob:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.mut_prob:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring + elites

            # Update Z-Score normalization stats from this generation's evaluations
            self._update_zscore_stats()

            halloffame.update(population)

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
    base_factors: List[str],
    data: pd.DataFrame,
    factor_calculator=None,
    **kwargs
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
        base_factors=base_factors,
        data=data,
        factor_calculator=factor_calculator,
        **kwargs
    )
