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

性能优化:
  进化循环中用 _fast_cross_sectional_ic 替代 alphalens 全流程，
  避免每次评估都执行分位数分箱+远期收益计算，加速 10-20x。
  最终结果仍由 factor_validation_service 补充完整分析。
"""

import logging
import operator
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Dict, Optional
from collections import Counter, OrderedDict
import pandas as pd
import numpy as np
import random
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

try:
    from deap import base, creator, tools, algorithms, gp  # noqa: F401

    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP库未安装，遗传算法功能将不可用")

from backend.services.base_mining_service import BaseMiningService  # noqa: E402
from backend.services.factor_validation_service import factor_validation_service  # noqa: E402
from backend.services.factor_primitives import (  # noqa: E402
    create_pset_numpy,
    tree_to_expression,
    tree_to_placeholder_expr,
    compile_tree,
    expression_similarity,
    zobrist_hash,
    simplify_gp_expression,
    compute_weighted_complexity,
    count_duplicate_subtrees,
    replace_duplicate_subtrees,
    sympy_canonical_key,
)
from backend.utils.safe_math import safe_divide, safe_ir  # noqa: E402

MAX_EVAL_STOCKS = 20
# AlphaForge 论文建议 pool_size=10 即可产生高质量因子。
# 50 只股票评估耗时是 20 只的 2.5 倍，但 IC 均值差异 < 5%。
# 20 只在速度和稳定性之间取得平衡。
# 缓存容量从 512 降至 128：每个缓存条目存储一棵GP树在所有评估股票上的因子值，
# 128棵树 × 20只股票 × 250行 × 8字节 ≈ 51MB，远小于之前的 128MB。
# 缓存每代清空一次，代内命中率通常 < 10%，128 足够。
DEFAULT_MAX_CACHE_SIZE = 128


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
        creator.create(
            "IndividualMulti", gp.PrimitiveTree, fitness=creator.FitnessMulti
        )
    except RuntimeError:
        logger.debug("IndividualMulti已注册，跳过")


def _same_tree_structure(a, b) -> bool:
    """HallOfFame 去重谓词：Zobrist 哈希相同视为同一个体。

    实测结论（见 test_hof）：DEAP 默认的 ``operator.eq`` 已能去除
    结构完全相同的树（pset 内原语节点是共享对象，逐元素 == 成立），
    但对交换律同构无能为力（``add(a, b)`` 与 ``add(b, a)`` 判不等）。
    改用 Zobrist 哈希可额外合并 ``add``/``mul``/``max``/``min``
    的交换律重复，避免 HOF 名额被等价因子占满。
    """
    return zobrist_hash(a) == zobrist_hash(b)


class _FactorValueEquivalence:
    """HallOfFame 等价谓词：因子值同序（|Spearman| ≥ 阈值）视为同一因子。

    结构哈希只能识别语法同构，识别不了单调变换：``sqrt(sqrt(x))``、
    ``log(x)``、``ts_mean_20(x)``、``ts_delay_5(x)`` 结构各不相同，但因子值
    完全同序（单调变换不改变秩），在因子库 / 组合优化里提供的是同一份信息。
    实测稳健适应度收敛后，HOF 前 5 个里常出现 3~4 个是同一因子的变换
    （样本外 IC、t、有效天数完全相同），"挖出 10 个因子"实际只有 2~3 个。

    计算口径：在首只评估股票上用 ``_compute_factor_expression`` 重算因子值，
    逐元素取 Spearman。只用于 HOF 插入判定，不参与适应度。
    """

    def __init__(self, service, threshold: float = 0.98, max_cache: int = 512):
        self._service = service
        self._threshold = threshold
        self._max_cache = max_cache
        self._cache = {}

    def _values(self, individual):
        key = zobrist_hash(individual)
        if key in self._cache:
            return self._cache[key]

        arr = None
        # 优先复用评估阶段已缓存的逐股票因子值：HOF 更新时种群刚评估完，
        # 缓存命中可避免重复 compile + eval（实测可省 5~8s/轮）。
        try:
            cached = self._service._cache_get(key)
            if cached:
                preferred = None
                sampled = getattr(self._service, "_sampled_stock_codes", None)
                if sampled:
                    preferred = sampled[0]
                if preferred is not None and preferred in cached:
                    arr = np.asarray(cached[preferred], dtype=float)
                else:
                    for stock, values in cached.items():
                        if stock == "_index":
                            continue
                        arr = np.asarray(values, dtype=float)
                        break
        except Exception:
            arr = None

        if arr is None:
            try:
                series = self._service._compute_factor_expression(individual)
                arr = series.to_numpy(dtype=float) if series is not None else None
            except Exception:
                arr = None
        if len(self._cache) >= self._max_cache:
            self._cache.clear()
        self._cache[key] = arr
        return arr

    def __call__(self, a, b) -> bool:
        if zobrist_hash(a) == zobrist_hash(b):
            return True
        va = self._values(a)
        vb = self._values(b)
        if va is None or vb is None or va.shape != vb.shape:
            return False
        mask = np.isfinite(va) & np.isfinite(vb)
        if mask.sum() < 20:
            return False
        x = va[mask]
        y = vb[mask]
        # 常数序列不含信息，但也不能当作"等价锚点"吞掉整个 HOF
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return False
        rho = spearmanr(x, y)[0]
        if rho is None or np.isnan(rho):
            return False
        # 因子与其相反数信息等价（方向可翻转），取绝对值
        return bool(abs(float(rho)) >= self._threshold)


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
        # 系数需与 IC 量纲匹配：修复逐股 dropna 后真实 |IC| 约 0.08~0.12，
        # 原 0.1 意味着"与 1 个 HOF 成员相似即扣掉全部 IC"（HOF 最多 10 个，
        # 可累计到 0.8+），实测 5 个种子上 0.1 → 0.02 使最优适应度均值
        # 0.1035 → 0.1085、每代均值 0.0267 → 0.0293。
        diversity_penalty_coeff: float = 0.02,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        # ---- Phase 6: Cross-validation ----
        cv_folds: int = 0,
        # ---- 评估稳健性：IC 下置信界系数 ----
        # 进化循环用 |mean IC| 排序，但稀疏因子（大量 NaN，仅 14~30 个有效
        # 交易日）的 |IC| 由极少样本估计，噪声会把它抬到 0.2+ 并挤掉稳健因子。
        # >0 时改用 mean_ic - sigma * ic_std / sqrt(n_days)：
        # sigma=1.65 约等于单侧 95% 下置信界，样本天数少或 IC 波动大的因子
        # 会被自动降权，样本天数多的稳健因子相对上升。
        ic_lcb_sigma: float = 1.65,
        # 有效 IC 天数覆盖率下限：len(daily_ics) 低于该比例（且不少于 20 天）
        # 视为不可评估。实测稀疏因子只靠 9~14 天估出 |IC|=0.2+，样本外完全
        # 失效；>=0.2 时这类因子直接记为最差，不进入 HOF。
        min_ic_day_ratio: float = 0.2,
        # ---- HOF 因子值去重 ----
        # >0 时 HOF 插入判定改为"因子值 |Spearman| ≥ 阈值即视为同一因子"，
        # 用于阻止 sqrt/log/ts_mean/ts_delay 等单调变换占满名额。
        # <=0 时退回纯结构哈希（_same_tree_structure）。
        hof_dedup_corr: float = 0.98,
        # ---- Phase 7: Extended primitives ----
        use_extended_primitives: bool = True,
        max_tree_depth: int = 17,
        # ---- NSGA-II ----
        use_nsga2: bool = True,
        # ---- SymPy 代数等价去重（默认关闭：simplify 对大树上无超时保护，易卡死）----
        use_sympy_dedup: bool = False,
        # ---- Factor pool sampling ----
        max_base_factors: int = 30,
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
            max_base_factors=max_base_factors,
        )

        self.population_size = population_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.ic_lcb_sigma = ic_lcb_sigma
        self.min_ic_day_ratio = min_ic_day_ratio
        self.hof_dedup_corr = hof_dedup_corr

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
        # SymPy 代数等价去重开关（评估路径与 Phase 3 去重共用）
        self.use_sympy_dedup = use_sympy_dedup

        self._current_generation: int = 0
        # _task_id 用于 checkpoint 保存时关联挖掘任务。
        # 仅在 API 路由层通过 mining_service._task_id = task_id 设置，
        # evolve_factor 等非任务场景下为 None，此时不保存 checkpoint。
        self._task_id: Optional[str] = None

        # Phase 4: factor value cache (keyed by Zobrist hash)
        self._factor_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        # IC 面板缓存：(股票集合) → (all_dates, return_matrix, stock_codes)
        # 股票样本整轮固定，收益矩阵与日期交集与个体无关，可复用
        self._panel_cache: dict = {}

        # 终端使用统计：var_name → 累计出现次数（每代种群快照累加）
        # 用于诊断因子池冗余：从未进入 HOF 的终端即为"死终端"
        self._terminal_use_counts: Counter = Counter()

        # P1: SymPy canonical-form → Zobrist hash mapping
        # 用于检测代数等价表达式（如 add(a, sub(b, a)) ≡ b）
        # 仅在 zobrist 未命中时查询，避免每次评估都做 SymPy 简化
        self._sympy_key_map: dict = {}
        self._sympy_key_lock = threading.Lock()

        # Build tradable_mask from OHLC data (detect limit-up/down days)
        self.tradable_mask: Optional[pd.Series] = self._build_tradable_mask()

        # GP primitives & toolbox
        self.pset: Optional[gp.PrimitiveSet] = None
        self._setup_genetic_algorithm()

    # ------------------------------------------------------------------
    # 股票池管理（覆盖基类，删除不需要的原始数据以节省内存）
    # ------------------------------------------------------------------

    def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
        """设置股票池，计算派生值后立即释放原始数据

        内存优化说明：
        - 基类 set_stock_pool 会设置 stock_pool_data（原始OHLCV DataFrame）、
          stock_pool_base_factor_values（预计算因子值）、stock_pool_return_values（收益率）。
        - 遗传规划评估循环只使用后两者，不需要原始 OHLCV 数据。
        - 删除 stock_pool_data 可节省约 10MB/任务（50只股票 × 250行 × 10列 × 8字节）。
        - 注意：此覆写必须在 super().set_stock_pool() 之后清空，
          因为基类方法中遍历 stock_pool_data 计算派生值。
        """
        # 重建股票池时需要重新抽样（_refresh_stock_sample 默认只在样本为空时抽样）
        self._sampled_stock_codes = []
        # 收益矩阵与股票池绑定，换池后必须失效
        self._panel_cache = {}
        super().set_stock_pool(stock_codes, start_date, end_date)
        self.stock_pool_data = {}
        logger.info(f"[{self._service_name}] stock_pool_data 已释放")

    def _refresh_stock_sample(self):
        """固定评估股票样本：整轮进化只抽样一次。

        基类每代随机重抽评估股票，精英（沿用上一代 fitness）与子代
        （在新样本上计算）的适应度因此不可比，NSGA-II / 锦标赛退化为
        噪声比较。遗传规划需要所有个体在同一评估样本上比较，
        所以仅在首次调用时抽样，后续代直接复用。
        """
        if self._sampled_stock_codes:
            return
        super()._refresh_stock_sample()

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
            logger.info(
                f"tradable_mask构建完成: {mask.sum()}个可交易日, {n_excluded}个涨跌停日已排除"
            )
            return mask
        except Exception as e:
            logger.warning(f"构建tradable_mask失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 终端使用统计（诊断因子池冗余）
    # ------------------------------------------------------------------

    def _record_terminal_usage(self, population):
        """累计当代种群中每个终端（基础因子）的出现次数。

        每代种群快照累加，用于衡量"探索预算"被各终端分摊的情况。
        """
        for ind in population:
            for node in ind:
                if isinstance(node, gp.Terminal):
                    value = str(node.value)
                    if value.startswith("factor_"):
                        self._terminal_use_counts[value] += 1

    def _terminal_usage_report(self, halloffame) -> Dict:
        """统计每个终端进入 HOF 的次数，并输出诊断日志。

        判定规则：
        - ``hof_hits == 0`` → 死终端（整轮进化从未产出 top 个体），可从因子池删除
        - 死终端占比 ≥ 50% → 判定因子池冗余，输出裁剪建议
        """
        hof_counts: Counter = Counter()
        for ind in halloffame:
            for node in ind:
                if isinstance(node, gp.Terminal):
                    value = str(node.value)
                    if value.startswith("factor_"):
                        hof_counts[value] += 1

        total_use = sum(self._terminal_use_counts.values())
        hof_total = sum(hof_counts.values())

        rows = []
        for var_name, info in self.base_factor_values.items():
            use = self._terminal_use_counts.get(var_name, 0)
            hof = hof_counts.get(var_name, 0)
            rows.append(
                {
                    "var": var_name,
                    "code": info.get("code"),
                    "population_hits": use,
                    "population_share": round(use / total_use, 4) if total_use else 0.0,
                    "hof_hits": hof,
                    "hof_share": round(hof / hof_total, 4) if hof_total else 0.0,
                    "dead": hof == 0,
                }
            )

        rows.sort(key=lambda r: (r["hof_hits"], r["population_hits"]), reverse=True)
        n_total = len(rows)
        n_dead = sum(1 for r in rows if r["dead"])
        dead_ratio = n_dead / n_total if n_total else 0.0

        logger.info(
            f"[终端使用统计] 基础因子={n_total}, HOF终端出现次数={hof_total}, "
            f"死终端={n_dead} ({dead_ratio:.0%})"
        )
        for r in rows[: min(10, n_total)]:
            logger.info(
                f"  {r['var']} | HOF {r['hof_hits']}次 ({r['hof_share']:.0%}) | "
                f"种群 {r['population_hits']}次 ({r['population_share']:.0%}) | "
                f"{r['code']}"
            )
        if n_dead:
            dead_desc = [f"{r['var']}({r['code']})" for r in rows if r["dead"]]
            logger.info(
                f"  从未进入HOF的终端 {n_dead} 个: "
                + ", ".join(dead_desc[:15])
                + (" ..." if n_dead > 15 else "")
            )
        if n_total and dead_ratio >= 0.5:
            logger.warning(
                f"超过一半基础因子（{n_dead}/{n_total}）从未进入HOF，因子池存在冗余；"
                f"有效终端仅 {n_total - n_dead} 个，建议裁剪冗余因子并补充新的候选因子"
            )

        return {
            "n_base_factors": n_total,
            "n_dead_in_hof": n_dead,
            "dead_ratio": round(dead_ratio, 4),
            "terminals": rows,
        }

    # ------------------------------------------------------------------
    # GP setup
    # ------------------------------------------------------------------

    def _setup_genetic_algorithm(self):
        _ensure_creator_types()

        n_factors = max(len(self.base_factor_values), 1)
        # 使用 numpy 版原语集 — 所有原语接受/返回 ndarray，
        # 消除 pandas 索引对齐开销，加速约 3-5x。
        # GP 树字符串表示与 pd.Series 版完全一致，缓存 key 兼容。
        self.pset = create_pset_numpy(
            n_factors,
            extended=self.use_extended_primitives,
        )

        self.toolbox = base.Toolbox()
        # Phase 7: deeper initial trees when extended primitives + parsimony control bloat
        init_max_depth = 5 if self.use_extended_primitives else 3
        self.toolbox.register(
            "expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=init_max_depth
        )

        # Choose individual class based on multi-objective flag
        if self.use_nsga2:
            individual_class = creator.IndividualMulti
        else:
            individual_class = creator.Individual

        self.toolbox.register(
            "individual", tools.initIterate, individual_class, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # GP operators
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset
        )

        # Depth limiter for crossover & mutation (prevents bloat)
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=operator.attrgetter("height"), max_value=self.max_tree_depth
            ),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=operator.attrgetter("height"), max_value=self.max_tree_depth
            ),
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
    #
    # 缓存结构：{tree_key: {"_index": pd.Index, "600036.SH": np.ndarray, ...}}
    # - key: GP树的 Zobrist 哈希（检测结构相同 + 交换律同构）
    # - "_index": 共享的 DatetimeIndex（所有股票共享同一时间索引）
    # - 其他 key: 股票代码 → numpy ndarray（因子值）
    #
    # 内存优化：用 numpy ndarray 替代 pd.Series 存储因子值。
    # pd.Series 每个实例自带一份 Index 副本（约 2KB/250行），
    # 128棵树 × 50只股票 = 6400 份重复 Index ≈ 12.5MB。
    # 改为共享 _index + ndarray 后，每棵树只需 1 份 Index ≈ 0.25MB。
    # 读取时按需 pd.Series(arr, index=cached["_index"]) 包装回 Series。
    # ------------------------------------------------------------------

    def _cache_get(self, tree_key: int) -> Optional[Dict]:
        """Look up pre-computed factor values for a tree expression.

        Uses Zobrist hash as key for O(1) lookup and isomorphic tree detection.
        """
        _locked = self._cache_lock.acquire(timeout=5.0)
        if not _locked:
            logger.warning(f"[诊断] _cache_get 无法获取 _cache_lock（5s超时），tree_key={tree_key}")
            return None
        try:
            return self._factor_cache.get(tree_key)
        finally:
            self._cache_lock.release()

    def _cache_set(self, tree_key: int, values: Dict):
        """Store factor values in the LRU cache."""
        _locked = self._cache_lock.acquire(timeout=5.0)
        if not _locked:
            logger.warning(f"[诊断] _cache_set 无法获取 _cache_lock（5s超时），tree_key={tree_key}")
            return
        try:
            if len(self._factor_cache) >= self.max_cache_size:
                # evict oldest entry (OrderedDict FIFO)
                self._factor_cache.popitem(last=False)
            self._factor_cache[tree_key] = values
        finally:
            self._cache_lock.release()

    # ------------------------------------------------------------------
    # 内存管理
    # ------------------------------------------------------------------

    def _sympy_canonical_key_with_timeout(self, tree, timeout: float = 3.0) -> Optional[str]:
        """带超时保护的 SymPy 规范形计算。

        sp.simplify() 对复杂 GP 树（深度>10、含时序算子嵌套）可能进入
        极长计算甚至死循环。实测卡住 16+ 分钟，阻塞整个评估流程。

        使用线程池 + future.result(timeout) 实现超时，
        超时后返回 None（降级为纯 zobrist 去重，功能不受影响）。

        ⚠️ 已知陷阱：``with ThreadPoolExecutor`` 退出时 ``shutdown(wait=True)``
        会等待工作线程完成。即使 ``future.result(timeout)`` 已超时返回 None，
        如果底层 ``sympy_canonical_key`` 仍在运行，``shutdown`` 会一直阻塞。
        因此改用显式 ``shutdown`` + 耗时诊断日志。
        """
        def _compute():
            return sympy_canonical_key(tree)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_compute)
        try:
            result = future.result(timeout=timeout)
            _t_shutdown_start = time.time()
            executor.shutdown(wait=True)
            _t_shutdown = time.time() - _t_shutdown_start
            if _t_shutdown > 1.0:
                logger.warning(
                    f"SymPy shutdown耗时 {_t_shutdown:.1f}s"
                    f"（树节点数={len(tree)}），线程未及时退出"
                )
            return result
        except TimeoutError:
            logger.warning(
                f"SymPy规范形计算超时（>{timeout}s），"
                f"树节点数={len(tree)}，跳过该优化"
            )
            # shutdown(wait=False) 不等待卡死的线程，避免阻塞主流程。
            # 卡死的线程会在后台继续运行，最终被垃圾回收。
            # ⚠️ 不能用 wait=True，否则会阻塞到 sp.simplify() 返回（实测可达 490+ 秒）。
            executor.shutdown(wait=False)
            return None
        except Exception as e:
            # sympy 缺失或表达式无法解析：降级为纯 zobrist 去重
            logger.debug("SymPy规范形计算失败，降级为纯 zobrist: %s", e)
            executor.shutdown(wait=False)
            return None

    def release_memory(self):
        """释放大对象占用的内存，在挖掘任务完成后调用

        调用时机：mining.py _run_mining() 的 finally 块。
        此时 mine_factors() 已返回，不会再访问这些属性。
        必须在 finally 块中调用而非 cancel_mining_task() 中调用，
        因为取消只是设置 flag，mine_factors() 仍在子线程中运行。
        """
        super().release_memory()
        _locked = self._cache_lock.acquire(timeout=5.0)
        if not _locked:
            logger.warning("[诊断] release_memory 无法获取 _cache_lock（5s超时），跳过缓存清理")
        else:
            try:
                self._factor_cache = OrderedDict()
            finally:
                self._cache_lock.release()
        self._halloffame = None
        self.tradable_mask = None
        logger.info(f"[{self._service_name}] 遗传规划内存已释放（含因子缓存）")

    # ------------------------------------------------------------------
    # Checkpoint（断点续跑）
    # ------------------------------------------------------------------

    def _save_checkpoint(self, gen: int, n_generations: int, population, halloffame):
        """保存进化检查点到数据库

        设计说明：
        - 每5代保存一次（gen % 5 == 0），最后一代也保存。
        - 只保留最近3个检查点（cleanup_old_checkpoints），避免磁盘占用过多。
        - 序列化内容：种群树字符串+适应度、精英、Z-Score统计量。
        - 不序列化 DEAP 对象本身（不可 pickle），只保存字符串表示。
        - 恢复时需重新创建种群（见 resume API）。
        - 整个方法用 try/except 包裹，保存失败不影响进化流程。
        """
        if not self._task_id:
            return  # 无task_id时不保存（如evolve_factor场景）

        try:
            import json as _json
            from backend.core.database import get_db
            from backend.models.mining_checkpoint import MiningCheckpointModel
            from backend.repositories.mining_checkpoint_repository import (
                MiningCheckpointRepository,
            )

            # 序列化种群
            pop_data = []
            for ind in population:
                pop_data.append({
                    "tree_str": str(ind),
                    "fitness_values": (
                        list(ind.fitness.values) if ind.fitness.valid else None
                    ),
                })
            population_json = _json.dumps(pop_data, ensure_ascii=False)

            # 序列化精英
            hof_data = []
            for ind in halloffame:
                hof_data.append({
                    "tree_str": str(ind),
                    "fitness_values": (
                        list(ind.fitness.values) if ind.fitness.valid else None
                    ),
                })
            hof_json = _json.dumps(hof_data, ensure_ascii=False)

            # 序列化Z-Score统计量
            zscore_stats = _json.dumps({
                "ic_mean": self._zscore_ic_mean,
                "ic_std": self._zscore_ic_std,
                "ir_mean": self._zscore_ir_mean,
                "ir_std": self._zscore_ir_std,
            })

            # 序列化适应度历史（从logbook提取，如果有）
            fitness_history_json = None
            if hasattr(self, "_gen_fitness_history"):
                fitness_history_json = _json.dumps(self._gen_fitness_history)

            with get_db() as db:
                repo = MiningCheckpointRepository(db)
                checkpoint = MiningCheckpointModel(
                    task_id=self._task_id,
                    generation=gen,
                    total_generations=n_generations,
                    population_json=population_json,
                    hof_json=hof_json,
                    zscore_stats_json=zscore_stats,
                    fitness_history_json=fitness_history_json,
                )
                repo.create(checkpoint)
                # 清理旧检查点，只保留最近3个
                repo.cleanup_old_checkpoints(self._task_id, keep_last=3)

            logger.debug(f"Checkpoint saved: gen={gen}/{n_generations}")
        except Exception as e:
            logger.debug(f"保存checkpoint失败（不影响进化）: {e}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_factor(self, individual) -> tuple:
        """Evaluate a PrimitiveTree individual by cross-sectional or time-series IC.

        Applies: Parsimony Pressure (Phase 2), Diversity Penalty (Phase 3),
        Cross-Validation penalty (Phase 6), Fitness Objective Routing (Phase 1).
        """
        if len(individual) == 0:
            return (self._no_ic_fitness(),)

        try:
            # 使用 stock_pool_base_factor_values 而非 stock_pool_data 判断股票数量，
            # 因为 set_stock_pool 覆写中已释放 stock_pool_data 以节省内存
            if len(self.stock_pool_base_factor_values) >= 2:
                raw_fitness = self._evaluate_cross_sectional_ic(individual)[0]
            elif len(self.stock_pool_base_factor_values) == 1:
                logger.warning(
                    "Only 1 stock in pool, falling back to time-series IC evaluation"
                )
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
            else:
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
        except Exception as e:
            logger.debug(f"适应度评估失败: {e}")
            return (self._no_ic_fitness(),)

        # --- Parsimony Pressure (Phase 2, P1: operator-weighted) ---
        # 替换原 len(individual)：算子加权复杂度更精准地反映计算成本
        # ts_corr=4.0, log/sqrt/rank=2.0, ts_mean=3.0, add/sub=1.0, terminal=0.5
        parsimony_penalty = self.parsimony_coeff * compute_weighted_complexity(individual)

        # --- Subtree Duplicate Penalty (P1增强: 抑制表达式膨胀) ---
        # 检测单个表达式内部的重复子树，对重复施加额外惩罚
        # 重复子树 = 冗余计算，不提升表达能力但增加复杂度
        dup_info = count_duplicate_subtrees(individual)
        # 渐进惩罚：前2个重复0.15/个，之后0.4/个
        # 8个重复 → 0.3 + 6*0.4 = 2.7（远大于 IC 值 0.03~0.34），
        # 有效抑制终端大量重复（如 factor_0 出现8次）
        # 注意：当只有1个基础因子时，replace_duplicate_subtrees 会跳过终端重复
        # （避免复杂度爆炸），因此适应度惩罚是唯一的抑制手段，需要更强
        n_dups = dup_info["n_duplicates"]
        if n_dups <= 2:
            dup_penalty = n_dups * 0.15
        else:
            dup_penalty = 0.3 + (n_dups - 2) * 0.4

        # --- Diversity Penalty (Phase 3) ---
        diversity_penalty = 0.0
        if (
            self.diversity_penalty_coeff > 0
            and hasattr(self, "_halloffame")
            and self._halloffame is not None
        ):
            ind_expr = tree_to_placeholder_expr(individual)
            for hof_ind in self._halloffame:
                hof_expr = tree_to_placeholder_expr(hof_ind)
                sim = expression_similarity(ind_expr, hof_expr)
                if sim > 0.7:
                    diversity_penalty += self.diversity_penalty_coeff * sim

        adjusted_fitness = raw_fitness - parsimony_penalty - dup_penalty - diversity_penalty
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
            # 使用 stock_pool_base_factor_values 而非 stock_pool_data 判断股票数量，
            # 因为 set_stock_pool 覆写中已释放 stock_pool_data 以节省内存
            if len(self.stock_pool_base_factor_values) >= 2:
                raw_fitness = self._evaluate_cross_sectional_ic(individual)[0]
            elif len(self.stock_pool_base_factor_values) == 1:
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
            else:
                raw_fitness = self._evaluate_single_stock_ic(individual)[0]
        except Exception as e:
            logger.warning(f"NSGA2适应度评估失败: {e}")
            return (0.0, 1.0)

        # --- Diversity Penalty (Phase 3, applied to IC objective only) ---
        diversity_penalty = 0.0
        if (
            self.diversity_penalty_coeff > 0
            and hasattr(self, "_halloffame")
            and self._halloffame is not None
        ):
            ind_expr = tree_to_placeholder_expr(individual)
            for hof_ind in self._halloffame:
                hof_expr = tree_to_placeholder_expr(hof_ind)
                sim = expression_similarity(ind_expr, hof_expr)
                if sim > 0.7:
                    diversity_penalty += self.diversity_penalty_coeff * sim

        # 不截断到 0：负 IC 个体之间仍需保留相对排序，
        # 否则劣质个体全部并列，NSGA-II 的支配关系与拥挤度失去区分度。
        ic_fitness = raw_fitness - diversity_penalty
        # P1: 算子加权复杂度替代朴素节点计数
        # P1增强: 加入子表达式重复惩罚（重复子树增加复杂度但不提升表达能力）
        # NSGA-II 第二目标：复杂度越小越好
        dup_info = count_duplicate_subtrees(individual)
        # 渐进惩罚：前2个重复0.15/个，之后0.4/个
        # 注意：当只有1个基础因子时，replace_duplicate_subtrees 会跳过终端重复
        # （避免复杂度爆炸），因此适应度惩罚是唯一的抑制手段，需要更强
        n_dups = dup_info["n_duplicates"]
        if n_dups <= 2:
            dup_penalty = n_dups * 0.15
        else:
            dup_penalty = 0.3 + (n_dups - 2) * 0.4
        complexity = compute_weighted_complexity(individual) + dup_penalty
        return (ic_fitness, complexity)

    def _penalty_fitness(self) -> tuple:
        """返回惩罚适应度值（评估异常/超时时使用）。

        NSGA-II 双目标：(ic_fitness, complexity)
        - ic_fitness 越大越好 → 惩罚值设为 -1.0（比任何合法个体都差）
        - complexity 越小越好 → 惩罚值设为 999（最差）

        注意：允许 ic_lcb_sigma>0 后合法个体的 ic_fitness 可以为负，
        惩罚值不能再取 0.0，否则"评估失败"会排在合法负适应度个体之前。
        """
        return (-1.0, 999.0)

    def _eval_tree_on_stock(
        self, tree, stock_code: str, stock_base_factors: dict
    ) -> Optional[pd.Series]:
        """Compile a tree and evaluate it using one stock's base factor values.

        Phase 4: Results are cached per (tree_key, stock_code) within a
        generation so that the same expression is never computed twice.
        Uses Zobrist hash to detect isomorphic trees (e.g. add(a,b) = add(b,a)).

        内存优化：缓存中存储 numpy ndarray 而非 pd.Series，
        共享索引存储在 cached["_index"]，节省约50%缓存内存。
        """
        tree_key = zobrist_hash(tree)
        # Phase 4: check cache first
        # 缓存结构：{tree_key: {"_index": pd.Index, "600036.SH": np.ndarray, ...}}
        # 读取时将 ndarray + 共享 index 包装回 pd.Series
        cached = self._cache_get(tree_key)
        if cached is not None and stock_code in cached:
            val = cached[stock_code]
            if isinstance(val, np.ndarray):
                index = cached.get("_index")
                return pd.Series(val, index=index) if index is not None else None
            return val  # 兼容旧格式（pd.Series 直存，理论上不会再出现）

        _t_start = time.time()
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

        # Phase 4: store in cache (numpy array + shared index)
        _locked = self._cache_lock.acquire(timeout=5.0)
        if not _locked:
            logger.warning(f"[诊断] _eval_on_stock 无法获取 _cache_lock（5s超时），stock={stock_code}")
            return result
        try:
            cached = self._factor_cache.get(tree_key)
            if cached is None:
                cached = {"_index": result.index}
                if len(self._factor_cache) >= self.max_cache_size:
                    self._factor_cache.popitem(last=False)
                self._factor_cache[tree_key] = cached
            elif "_index" not in cached:
                cached["_index"] = result.index
            cached[stock_code] = result.values  # 存储 numpy ndarray
        finally:
            self._cache_lock.release()

        _t_elapsed = time.time() - _t_start
        if _t_elapsed > 1.0:
            logger.debug(f"  [诊断] _eval_tree_on_stock 慢执行: stock={stock_code}, 耗时={_t_elapsed:.2f}s")
        return result

    def _eval_compiled_on_stock(
        self, compiled_func, tree_key: int, stock_code: str, stock_base_factors: dict
    ) -> Optional[pd.Series]:
        """使用已编译的函数在一只股票上评估因子值。

        与 _eval_tree_on_stock 不同，此方法接受已编译的函数，
        避免同一棵树被重复编译（性能优化：50只股票从50次编译降为1次）。

        性能优化：numpy 版原语集接受/返回 ndarray，消除 pandas 索引对齐开销。
        入口处将 pd.Series 转为 ndarray，出口处包装回 pd.Series。

        内存优化：缓存中存储 numpy ndarray 而非 pd.Series，
        共享索引存储在 cached["_index"]，节省约50%缓存内存。
        """
        # Phase 4: check cache first
        # 缓存结构同 _eval_tree_on_stock，读取时 ndarray → pd.Series
        cached = self._cache_get(tree_key)
        if cached is not None and stock_code in cached:
            val = cached[stock_code]
            if isinstance(val, np.ndarray):
                index = cached.get("_index")
                return pd.Series(val, index=index) if index is not None else None
            return val  # 兼容旧格式

        _t_start = time.time()
        # Build ordered positional args matching factor_0 … factor_N
        # 提取 .values 转为 numpy 数组，避免原语中的 pandas 索引对齐开销
        ordered_np = []
        index = None
        _data_len = 0
        for i in range(len(self.base_factor_values)):
            info = stock_base_factors.get(f"factor_{i}")
            if info is None:
                return None
            series = info["values"]
            if index is None:
                index = series.index
                _data_len = len(series)
            ordered_np.append(series.values)

        try:
            # 超时保护：单只股票的 GP 树执行不应超过 5 秒
            # 深度嵌套的 ts_corr（pandas rolling 回退）可能导致极长耗时
            _t0 = time.time()
            result = compiled_func(*ordered_np)
            _elapsed = time.time() - _t0
            # 单只股票执行日志：热路径（20 股 × 50 个体 × 20 代 = 2 万次），
            # 降到 DEBUG 并改用惰性格式化，避免 f-string + 双 handler 写盘开销
            logger.debug(
                "GP树执行 股票=%s 耗时=%.3fs 数据长度=%s",
                stock_code,
                _elapsed,
                _data_len,
            )
            if _elapsed > 3.0:
                logger.warning(
                    f"GP树执行耗时 {_elapsed:.1f}s（股票 {stock_code}），超过阈值跳过"
                )
                return None
        except Exception as e:
            logger.debug(f"执行表达式失败（股票 {stock_code}）: {e}")
            return None

        # 将 numpy 结果包装回 pd.Series（用于返回值和验证）
        if isinstance(result, (int, float, np.number)):
            if index is None:
                return None
            result_series = pd.Series(float(result), index=index)
            result_np = result_series.values
        elif isinstance(result, np.ndarray):
            if index is None:
                return None
            result_series = pd.Series(result, index=index)
            result_np = result
        elif isinstance(result, pd.Series):
            result_series = result
            result_np = result.values
            if index is None:
                index = result.index
        else:
            return None

        # 统一处理 inf 和有效值检查
        result_series = result_series.replace([np.inf, -np.inf], np.nan)
        valid_count = result_series.notna().sum()
        if valid_count == 0 or valid_count < len(result_series) * 0.1:
            return None

        # 更新 numpy 数组（replace 可能产生新数组）
        result_np = result_series.values

        # Phase 4: store in cache (numpy array + shared index)
        _locked = self._cache_lock.acquire(timeout=5.0)
        if not _locked:
            logger.warning(f"[诊断] _eval_compiled_on_stock 无法获取 _cache_lock（5s超时），stock={stock_code}")
            return result_series  # 返回结果但不缓存
        try:
            cached = self._factor_cache.get(tree_key)
            if cached is None:
                cached = {"_index": index}
                if len(self._factor_cache) >= self.max_cache_size:
                    self._factor_cache.popitem(last=False)
                self._factor_cache[tree_key] = cached
            elif "_index" not in cached:
                cached["_index"] = index
            cached[stock_code] = result_np  # 存储 numpy ndarray
        finally:
            self._cache_lock.release()

        _t_elapsed = time.time() - _t_start
        if _t_elapsed > 1.0:
            logger.debug(f"  [诊断] _eval_compiled_on_stock 慢执行: stock={stock_code}, 耗时={_t_elapsed:.2f}s")
        return result_series

    def _evaluate_cross_sectional_ic(self, tree) -> tuple:
        """Cross-sectional IC evaluation (multi-stock).

        Phase 1: Uses ``_route_fitness`` to select the objective metric.
        Phase 4: Uses factor value cache to avoid redundant computation.
        Phase 6: Applies cross-validation penalty when cv_folds > 0.

        Performance optimization:
            进化循环中使用轻量级横截面IC计算（_fast_cross_sectional_ic），
            而非完整的 alphalens 流水线。原因见 _fast_cross_sectional_ic 注释。
            GP 树编译一次后复用于所有股票，避免同一棵树被编译 N 次。
        """
        _t0 = time.time()
        eval_codes = self._sampled_stock_codes
        factor_values_dict: Dict[str, pd.Series] = {}
        # 性能诊断：各环节耗时
        _t_compile = 0.0
        _t_parallel = 0.0
        _t_fast_ic = 0.0
        _t_cv = 0.0

        # 热路径：str(tree) 对大树开销明显，仅在 DEBUG 时构造预览
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "截面IC评估开始 树高度=%s 节点数=%s 表达式预览=%s",
                tree.height,
                len(tree),
                str(tree)[:80],
            )

        # Phase 4: check if full result is cached
        # 缓存中 "_complete" 键存储横截面IC的完整结果（所有股票的因子值），
        # "_complete_index" 存储各股票对应的 DatetimeIndex（用于 ndarray → Series 转换）
        tree_key = zobrist_hash(tree)
        cached_all = self._cache_get(tree_key)

        # P1: SymPy 规范形去重 — zobrist 未命中时，检查代数等价表达式
        # 例如 add(a, sub(b, a)) ≡ b，zobrist 不同但 SymPy 规范形相同
        # 默认关闭：sp.simplify() 对复杂 GP 树可能极慢甚至死循环（实测卡住 16+ 分钟），
        # 而代数等价命中率很低，代价远大于收益。仅在 use_sympy_dedup=True 时启用。
        sympy_hit = False
        canon_key = None
        if self.use_sympy_dedup and (
            cached_all is None or "_complete" not in cached_all
        ):
            try:
                canon_key = self._sympy_canonical_key_with_timeout(tree, timeout=3.0)
                if canon_key is not None:
                    with self._sympy_key_lock:
                        mapped_key = self._sympy_key_map.get(canon_key)
                    if mapped_key is not None:
                        cached_all = self._cache_get(mapped_key)
                        if cached_all is not None and "_complete" in cached_all:
                            tree_key = mapped_key  # 复用已有缓存的 key
                            sympy_hit = True
                            logger.debug("SymPy规范形命中，复用已有缓存")
            except Exception:
                pass  # SymPy 不可用或失败/超时，降级为纯 zobrist

        if cached_all is not None and "_complete" in cached_all:
            # 从缓存恢复：numpy ndarray → pd.Series（按需包装）
            complete_cache = cached_all["_complete"]
            complete_index = cached_all.get("_complete_index", {})
            factor_values_dict = {}
            for code, val in complete_cache.items():
                if isinstance(val, np.ndarray):
                    idx = complete_index.get(code)
                    if idx is not None:
                        factor_values_dict[code] = pd.Series(val, index=idx)
                    else:
                        factor_values_dict[code] = pd.Series(val)
                elif isinstance(val, pd.Series):
                    factor_values_dict[code] = val
        else:
            # 编译 GP 树一次，复用于所有股票（原来每只股票编译一次，50只股票=50次编译）
            try:
                _t_compile_start = time.time()
                compiled_func = compile_tree(tree, self.pset)
                _t_compile = time.time() - _t_compile_start
                logger.debug("GP树编译完成 节点数=%s 耗时=%.3fs", len(tree), _t_compile)
            except Exception as e:
                logger.warning(f"编译表达式失败: {e}")
                return (self._no_ic_fitness(),)

            # 串行逐股评估：实测 20 股 × 250 行时，4 线程因 GIL 竞争 + 每个体
            # 新建线程池，反而比串行慢约 1.9x（1.69ms → 3.21ms）。
            # 卡死保护由外层「逐个体 30s 超时」包装器承担。
            _t_parallel_start = time.time()
            n_skipped = 0
            for code in eval_codes:
                base_factors = self.stock_pool_base_factor_values.get(code)
                if base_factors is None:
                    n_skipped += 1
                    continue
                try:
                    fv = self._eval_compiled_on_stock(
                        compiled_func, tree_key, code, base_factors
                    )
                except Exception as e:
                    logger.debug("评估股票 %s 因子失败: %s", code, e)
                    n_skipped += 1
                    continue
                if fv is None:
                    n_skipped += 1
                    continue
                fv_clean = fv.dropna()
                if len(fv_clean) >= 10:
                    # 必须保留完整索引（含 NaN），不能只存 dropna() 后的序列。
                    # 逐股 dropna() 会让每只股票的索引只剩"自身有效日"，而
                    # _fast_cross_sectional_ic 对所有股票索引求交集，等价于要求
                    # "当天所有股票都有效"。含 sqrt/log 的因子 NaN 极多，实测
                    # 10 股池 484 天里只剩 7 天能算 IC，适应度退化为 7 天噪声
                    # （|IC| 虚高到 0.25），HOF 全是对这 7 天的过拟合。
                    # 逐日有效样本数应交由 _fast_cross_sectional_ic 的 valid
                    # 掩码 + min_stocks 处理。
                    factor_values_dict[code] = fv
                else:
                    n_skipped += 1

            _t_parallel = time.time() - _t_parallel_start
            logger.debug(
                "逐股评估完成: %s/%s 只有效, %s 只跳过, 总耗时=%.3fs",
                len(factor_values_dict),
                len(eval_codes),
                n_skipped,
                _t_parallel,
            )
            logger.debug(f"  [诊断] 开始构建缓存数据... factor_values_dict={len(factor_values_dict)}项")

            # Phase 4: cache the complete result (numpy arrays for memory efficiency)
            # 将 pd.Series 拆分为 ndarray + DatetimeIndex 分别存储，
            # 避免每只股票的 Series 都带一份重复的 Index 副本
            complete_np = {}
            complete_index = {}
            for code, fv in factor_values_dict.items():
                complete_np[code] = fv.values if isinstance(fv, pd.Series) else fv
                if isinstance(fv, pd.Series):
                    complete_index[code] = fv.index
            logger.debug(f"  [诊断] 缓存数据构建完成，准备调用_cache_set...")
            self._cache_set(tree_key, {
                "_complete": complete_np,
                "_complete_index": complete_index,
            })
            logger.debug(f"  [诊断] _cache_set 完成")

            # P1: 记录 SymPy 规范形 → zobrist key 映射
            # 后续代数等价表达式可通过此映射复用缓存（仅在启用 SymPy 去重时构建）
            if canon_key is not None and not sympy_hit:
                try:
                    with self._sympy_key_lock:
                        if canon_key not in self._sympy_key_map:
                            self._sympy_key_map[canon_key] = tree_key
                except Exception:
                    pass
                logger.debug(f"  [诊断] SymPy key map 更新完成")

        logger.debug(f"  [诊断] 退出缓存/计算分支，factor_values_dict={len(factor_values_dict)}项")

        if len(factor_values_dict) < 2:
            return (self._no_ic_fitness(),)

        logger.debug(f"  [诊断] 准备调用 _fast_cross_sectional_ic...")

        try:
            # ---- 轻量级横截面IC计算（替代 alphalens 全流程） ----
            #
            # 为什么不用 alphalens？
            # alphalens 的 get_clean_factor_and_forward_returns() + analyze_ic()
            # 包含：分位数分箱、远期收益计算、多period IC、Pearson+Spearman 双通道
            # 等重量级操作，单次调用耗时约 50-200ms。
            #
            # 进化循环中每个个体都需要评估，20代×45个体=900次调用，
            # alphalens 总耗时约 45-180秒，占总运行时间的 60-80%。
            #
            # 但进化循环只需要一个值——横截面 Spearman IC 均值——来排序个体。
            # 完整的 alphalens 分析（分位数收益、换手率等）在最终结果阶段
            # 由 mine_factors() 中的 factor_validation_service.validate_factor() 补充，
            # 此处省略不影响最终结果质量。
            #
            # 轻量级实现直接按日期截面计算 Spearman IC 再取均值，
            # 与 alphalens 的 Spearman IC 语义一致（规则7.1），耗时约 5-10ms，
            # 加速 10-20 倍。
            _t_fast_ic_start = time.time()
            best_ic, best_ir = self._fast_cross_sectional_ic(factor_values_dict)
            _t_fast_ic = time.time() - _t_fast_ic_start
            logger.debug(f"  [诊断] _fast_cross_sectional_ic 完成, best_ic={best_ic:.4f}, best_ir={best_ir:.4f}")

            # 收集原始IC/IR用于代际Z-Score计算（与 _route_fitness 行为一致）
            self._gen_ic_values.append(best_ic)
            self._gen_ir_values.append(best_ir)
            logger.debug(f"  [诊断] IC/IR 值已收集，进入适应度路由...")

            # 根据 fitness_objective 路由适应度（与 _route_fitness 逻辑一致）
            if self.fitness_objective == "ir_ratio":
                raw_fitness = best_ir
            elif self.fitness_objective == "sharpe":
                raw_fitness = best_ir
            elif self.fitness_objective == "combined":
                # Z-Score归一化（与 _route_fitness 完全一致）
                z_ic = max(
                    -3.0,
                    min(
                        safe_divide(
                            float(best_ic - self._zscore_ic_mean),
                            float(self._zscore_ic_std),
                            default=0.0,
                        ),
                        3.0,
                    ),
                )
                z_ir = max(
                    -3.0,
                    min(
                        safe_divide(
                            float(best_ir - self._zscore_ir_mean),
                            float(self._zscore_ir_std),
                            default=0.0,
                        ),
                        3.0,
                    ),
                )
                norm_ic = (z_ic + 3.0) / 6.0
                norm_ir = (z_ir + 3.0) / 6.0
                raw_fitness = 0.6 * norm_ic + 0.4 * norm_ir
            else:  # ic_mean (default)
                raw_fitness = best_ic
            logger.debug(f"  [诊断] 适应度路由完成, raw_fitness={raw_fitness:.4f}")

            # Phase 6: cross-validation penalty
            _t_cv_start = time.time()
            cv_penalty = self._cv_penalty(factor_values_dict)
            _t_cv = time.time() - _t_cv_start
            raw_fitness = raw_fitness * (1.0 - cv_penalty)
            logger.debug(f"  [诊断] CV惩罚完成, cv_penalty={cv_penalty:.4f}, adjusted_fitness={raw_fitness:.4f}")

            _elapsed = time.time() - _t0
            # 逐个体耗时明细只在 DEBUG 输出，INFO 仅保留超时告警
            logger.debug(
                "截面IC评估 总耗时=%.2fs 树节点=%s 高度=%s 股票数=%s "
                "有效股票=%s 编译=%.2fs 并行=%.2fs 快速IC=%.2fs CV=%.2fs "
                "raw_fitness=%.4f",
                _elapsed,
                len(tree),
                tree.height,
                len(eval_codes),
                len(factor_values_dict),
                _t_compile,
                _t_parallel,
                _t_fast_ic,
                _t_cv,
                raw_fitness,
            )
            if _elapsed > 10.0:
                logger.warning(
                    f"截面IC评估严重超时 {_elapsed:.1f}s（>10s），"
                    f"树节点数={len(tree)}, 股票数={len(eval_codes)}"
                )
            return (raw_fitness,)

        except Exception as e:
            logger.warning(f"Cross-sectional IC evaluation failed: {e}")
            return (self._no_ic_fitness(),)

    def _no_ic_fitness(self) -> float:
        """无法计算 IC 时的适应度（有效股票/日期不足）。

        启用 ic_lcb_sigma 后合法个体的适应度可为负，"无法评估"必须比
        任何合法个体都差，否则稀疏到算不出 IC 的树会排在中游。
        """
        return -1.0 if self.ic_lcb_sigma > 0 else 0.0

    def _fast_cross_sectional_ic(
        self, factor_values_dict: Dict[str, pd.Series]
    ) -> tuple:
        """轻量级横截面IC计算 — 进化循环专用，替代 alphalens 全流程。

        设计决策说明（为什么不用 alphalens）：
        ────────────────────────────────────────
        1. 性能：alphalens 的 get_clean_factor_and_forward_returns() 内部执行
           分位数分箱 + 远期收益计算 + 数据对齐，单次耗时 50-200ms。
           进化循环中 20代×45个体=900次调用，总耗时 45-180秒。
           本方法直接按日期截面计算 Spearman IC，单次耗时 1-3ms，加速 20-100x。

        2. 语义等价性：本方法计算的是"每日横截面 Spearman 秩相关的均值"，
           与 alphalens 的 mean_information_coefficient(spearman) 语义一致，
           符合项目规范规则7.1（IC必须使用横截面Spearman，禁止池化Pearson）。

        3. 结果完整性：进化循环只需要 IC/IR 值来排序个体，不需要分位数收益、
           换手率等详细分析。这些在 mine_factors() 最终阶段由
           factor_validation_service.validate_factor() 补充计算，
           因此省略不影响最终输出质量。

        4. 精度差异：alphalens 内部会做 max_loss 过滤和分位数分箱，
           可能丢弃部分数据点。本方法使用全部有效数据，IC值可能略有差异
           （通常 <5%），但排序一致性不受影响——进化算法只关心相对排序。

        性能优化（v2）：
        ────────────────
        使用 numpy 矩阵替代 pd.concat + groupby，避免：
        - 50次 DataFrame 创建
        - pd.concat 合并
        - groupby 索引构建
        - Python 循环中冗余的 notna 检查

        Returns:
            (best_ic, best_ir) — 与 _extract_best_ic_ir 返回格式一致
        """
        _t_ic_start = time.time()
        logger.debug(f"  [诊断] _fast_cross_sectional_ic 开始, 输入 {len(factor_values_dict)} 只股票")
        # ---- 日期交集 + 收益矩阵（按股票集合缓存收益侧）----
        # 收益矩阵与日期候选集与个体无关，可跨个体复用；
        # 因子侧日期仍参与交集，保持与旧实现完全一致的样本口径。
        panel_key = tuple(sorted(factor_values_dict.keys()))
        panel = self._panel_cache.get(panel_key)
        if panel is None:
            stock_codes = []
            ret_map = {}
            for stock_code in panel_key:
                ret = self.stock_pool_return_values.get(stock_code)
                if ret is None:
                    continue
                ret_map[stock_code] = ret
                stock_codes.append(stock_code)

            if len(stock_codes) < 5:
                self._panel_cache[panel_key] = (None, [])
                logger.debug(
                    "  [诊断] 面板不可用: 有效股票数=%s (<5)", len(stock_codes)
                )
                return (self._no_ic_fitness(), 0.0)

            ret_df = pd.DataFrame(ret_map, columns=stock_codes)
            panel = (ret_df, stock_codes)
            self._panel_cache[panel_key] = panel
            logger.debug(
                "面板缓存构建: %s 股 × %s 天", len(stock_codes), len(ret_df.index)
            )

        ret_df, stock_codes = panel
        if ret_df is None:
            return (self._no_ic_fitness(), 0.0)

        # 与旧实现一致：逐股 factor.index ∩ return.index 后再取交集
        all_dates = ret_df.index
        for stock_code in stock_codes:
            all_dates = all_dates.intersection(factor_values_dict[stock_code].index)

        if len(all_dates) < 5:
            logger.debug(
                "  [诊断] _fast_cross_sectional_ic 提前退出: all_dates=%s (<5)",
                len(all_dates),
            )
            return (self._no_ic_fitness(), 0.0)

        return_matrix = ret_df.reindex(all_dates).values

        # ---- 构建因子矩阵：shape (n_dates, n_stocks) ----
        factor_cols = []
        for stock_code in stock_codes:
            fv = factor_values_dict[stock_code]
            if fv.index.equals(all_dates):
                factor_cols.append(fv.values)
            else:
                factor_cols.append(fv.reindex(all_dates).values)

        factor_matrix = np.column_stack(factor_cols)  # (n_dates, n_stocks)
        logger.debug(
            "  [诊断] numpy矩阵构建完成: shape=%s, n_dates=%s",
            factor_matrix.shape,
            len(all_dates),
        )

        # ---- 逐行（逐日期）计算 Spearman IC ----
        # 对每行的因子值和收益率分别排名，然后计算 Pearson 相关
        daily_ics: List[float] = []
        n_stocks = factor_matrix.shape[1]
        # 截面样本下限：大池保持 10 只；小池按 60% 比例放宽。
        # 原实现 min(10, n_stocks) 在 10 只股票池上等于"要求 10 只全部有效"，
        # 含 sqrt/log 的因子在很多股票上为 NaN，实测 484 天里只剩 7 天能算 IC，
        # 适应度变成 7 天噪声（|IC| 虚高到 0.25），HOF 全是对这 7 天的过拟合。
        min_stocks = min(
            n_stocks, max(5, min(10, int(np.ceil(0.6 * n_stocks))))
        )

        for i in range(len(all_dates)):
            f_row = factor_matrix[i]
            r_row = return_matrix[i]
            valid = ~np.isnan(f_row) & ~np.isnan(r_row)
            n_valid = valid.sum()
            if n_valid < min_stocks:
                continue
            f_valid = f_row[valid]
            r_valid = r_row[valid]
            # 排名（argsort of argsort = rank）
            f_rank = np.argsort(np.argsort(f_valid)).astype(float)
            r_rank = np.argsort(np.argsort(r_valid)).astype(float)
            # Pearson 相关（对排名序列 = Spearman 相关）
            f_centered = f_rank - f_rank.mean()
            r_centered = r_rank - r_rank.mean()
            denom = np.sqrt((f_centered ** 2).sum() * (r_centered ** 2).sum())
            if denom < 1e-10:
                continue
            ic = (f_centered * r_centered).sum() / denom
            daily_ics.append(ic)

        logger.debug(f"  [诊断] Spearman IC循环完成: {len(daily_ics)}/{len(all_dates)} 有效日期")

        if len(daily_ics) < 2:
            _t_ic = time.time() - _t_ic_start
            logger.debug(f"  [诊断] _fast_cross_sectional_ic 提前退出: 有效IC<2, 耗时={_t_ic:.3f}s")
            return (self._no_ic_fitness(), 0.0)

        # 覆盖率闸门：有效 IC 天数过少时 |IC| 完全由噪声决定，直接判为最差
        if self.min_ic_day_ratio > 0:
            required = max(20, int(round(self.min_ic_day_ratio * len(all_dates))))
            if len(daily_ics) < required:
                logger.debug(
                    "  [诊断] IC 覆盖率不足: %s/%s 天 (<%s)，判为不可评估",
                    len(daily_ics), len(all_dates), required,
                )
                return (-1.0, 0.0)

        ic_arr = np.array(daily_ics)
        # 有符号均值才是真实 IC；|IC| 不是无偏估计量（纯噪声下 mean(|IC|) ≈ sqrt(2/(π(N-1)))）
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        # 适应度方向无关（GP 可用 neg 翻转因子方向），但不改变 IC 的定义
        mean_ic = abs(ic_mean)

        # 估计不确定性惩罚：稀疏因子（有效交易日极少）的 |IC| 是噪声，
        # 用下置信界排序可避免其挤掉样本充足、IC 稳定的因子。
        if self.ic_lcb_sigma > 0 and len(ic_arr) > 1:
            se = ic_std / np.sqrt(len(ic_arr))
            mean_ic = mean_ic - self.ic_lcb_sigma * float(se)

        # IR 分子分母必须来自同一序列（规则7.10：std≈0时返回None）
        best_ir = safe_ir(ic_mean, ic_std, default=None)
        best_ir = abs(best_ir) if best_ir is not None else 0.0

        _t_ic = time.time() - _t_ic_start
        logger.info(
            f"  [诊断] _fast_cross_sectional_ic 完成: ic={ic_mean:.4f}, "
            f"|ic|={mean_ic:.4f}, ir={best_ir:.4f}, 耗时={_t_ic:.3f}s"
        )
        return (mean_ic, best_ir)

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
            return (self._no_ic_fitness(),)

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

    def _make_hof_similar(self):
        """构造 HallOfFame 去重谓词。

        hof_dedup_corr > 0：按因子值 |Spearman| ≥ 阈值判定等价，
            可识别 sqrt/log/ts_mean/ts_delay 等单调变换重复。
        hof_dedup_corr <= 0：退回纯结构哈希（兼容旧行为）。
        """
        if self.hof_dedup_corr > 0:
            return _FactorValueEquivalence(self, self.hof_dedup_corr)
        return _same_tree_structure

    def _compute_factor_expression(self, tree) -> Optional[pd.Series]:
        """Evaluate a PrimitiveTree using the global base factor cache.

        注意：此方法在 mine_factors() 的结果构建阶段调用，
        此时 release_memory() 尚未执行（在 finally 块中才调用）。
        但仍需防御 self.data 为 None 的情况（如未来调用时序变化）。
        """
        if len(tree) == 0:
            return None
        # 防御性检查：release_memory() 会将 self.data 设为 None
        if self.data is None:
            return None
        try:
            func = compile_tree(tree, self.pset)
        except Exception as e:
            logger.debug(f"编译表达式失败: {e}")
            return None

        ordered = []
        index = None
        for i in range(len(self.base_factor_values)):
            info = self.base_factor_values.get(f"factor_{i}")
            if info is None:
                return None
            series = info["values"]
            if index is None:
                index = series.index
            # numpy 版 pset 需要 ndarray 输入（与 _eval_compiled_on_stock 一致）
            ordered.append(series.values)

        try:
            result = func(*ordered)
        except Exception as e:
            logger.debug(f"执行表达式失败: {e}")
            return None

        # numpy 版原语返回 ndarray，需要包装回 pd.Series
        if isinstance(result, (int, float, np.number)):
            if index is None:
                return None
            result = pd.Series(float(result), index=index)
        elif isinstance(result, np.ndarray):
            if index is None:
                return None
            result = pd.Series(result, index=index)
        elif isinstance(result, pd.Series):
            pass  # 已经是 Series
        else:
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

    def _placeholder_to_code(self, placeholder_str: str) -> str:
        """Expand factor_N placeholders in a prefix expression string to actual codes.

        Used after SymPy simplification: the simplified placeholder string
        (e.g. ``ts_corr_5(mul(2, factor_0), factor_0)``) is expanded by
        replacing each ``factor_N`` with its real expression code.

        不加额外括号：GP 前缀表达式中 factor_N 总是作为函数参数出现（在括号内），
        逗号已分隔参数，无需再用括号包裹。避免 ``sqrt((SCALE(...)))`` 双括号。
        """
        expr = placeholder_str
        for var_name in sorted(self.base_factor_values, key=len, reverse=True):
            code = self.base_factor_values[var_name]["code"]
            expr = expr.replace(var_name, code)
        return expr

    def _generate_display_expression(self, simplified_placeholder: str) -> tuple:
        """Generate a compact display expression using short variable names.

        When only 1-2 base factors are used, the expanded expression contains
        the full base factor code repeated many times (e.g. ``-1 * DELTA(...)``
        appears 7 times), making it unreadable.

        This method replaces each ``factor_N`` with a short name ``FN`` and
        returns a mapping of short names to full codes.

        Returns
        -------
        display_expr : str
            Expression with short variable names (e.g. ``ts_corr_5(F0, F0)``).
        definitions : dict
            Mapping from short names to full code strings.
        """
        display_expr = simplified_placeholder
        definitions = {}
        for var_name in sorted(self.base_factor_values, key=len, reverse=True):
            code = self.base_factor_values[var_name]["code"]
            # factor_0 → F0, factor_1 → F1, ...
            short = var_name.replace("factor_", "F")
            display_expr = display_expr.replace(var_name, short)
            definitions[short] = code
        return display_expr, definitions

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

        # ---- Phase 2.6 (initial): Deduplicate initial population ----
        # 随机生成的初始种群可能包含个体内部重复终端（如 factor_0 出现两次）。
        # 由于 Alpha101 基础因子是复杂表达式，终端重复会导致最终表达式包含
        # 大段重复内容。在评估前去重，确保起始种群干净。
        _n_init_deduped = 0
        for i, ind in enumerate(population):
            try:
                new_nodes, n_replaced = replace_duplicate_subtrees(
                    ind, self.pset
                )
                if new_nodes is not None and n_replaced > 0:
                    new_tree = gp.PrimitiveTree(new_nodes)
                    if len(new_tree) > 0:
                        _ = new_tree.height
                        gp.compile(new_tree, self.pset)
                        population[i] = type(ind)(new_tree)
                        _n_init_deduped += n_replaced
            except Exception:
                pass  # 去重失败，保留原始个体
        if _n_init_deduped > 0:
            logger.info(
                f"初始种群去重: 替换{_n_init_deduped}个重复子树"
            )

        # Evaluate initial population — 串行评估
        # _evaluate_cross_sectional_ic 内部已改为串行逐股（线程池实测负收益），
        # 不存在嵌套线程池问题，逐个体串行评估即可。
        # 慢个体由下方 ts_corr 计数预检过滤；初始评估不做单个体超时包装。
        best_init_fitness = 0.0
        _init_eval_start = time.time()
        for i, ind in enumerate(population):
            _ind_t0 = time.time()

            # 单个体评估直接调用：内部逐股评估已串行，外层再包线程池只增加调度开销。
            # 慢个体由树复杂度预检（ts_corr 计数）与代际总预算共同控制。
            _should_skip = False
            # 预检1: 嵌套时序算子过多 → 直接跳过（ts_corr/ts_std 嵌套 >2层 极慢）
            _ts_op_count = sum(1 for node in ind if hasattr(node, 'name') and
                              node.name.startswith('ts_') and 'corr' in node.name)
            if _ts_op_count >= 2:
                logger.debug(
                    "初始评估 [%s/%s] 跳过: %s个ts_corr嵌套 (阈值>=2), 树高度=%s, 节点数=%s",
                    i + 1,
                    len(population),
                    _ts_op_count,
                    ind.height,
                    len(ind),
                )
                fit = self._penalty_fitness() if self.use_nsga2 else (-1.0,)
                _should_skip = True

            if not _should_skip:
                try:
                    # 单个个体评估超时保护（30秒）
                    _eval_executor = ThreadPoolExecutor(max_workers=1)
                    _eval_future = _eval_executor.submit(self.toolbox.evaluate, ind)
                    try:
                        fit = _eval_future.result(timeout=30.0)
                    except TimeoutError:
                        logger.warning(
                            f"初始评估 [{i+1}/{len(population)}] 超时(30s)，"
                            f"表达式={str(ind)[:80]}"
                        )
                        _eval_future.cancel()
                        fit = self._penalty_fitness() if self.use_nsga2 else (-1.0,)
                    finally:
                        _eval_executor.shutdown(wait=False)
                except Exception as e:
                    logger.warning(f"初始评估 [{i+1}/{len(population)}] 异常: {e}")
                    fit = self._penalty_fitness() if self.use_nsga2 else (-1.0,)

            _ind_elapsed = time.time() - _ind_t0
            ind.fitness.values = fit
            primary_fit = float(fit[0]) if fit else 0.0
            if primary_fit > best_init_fitness:
                best_init_fitness = primary_fit
            logger.debug(
                "初始评估 [%s/%s] 耗时=%.2fs, fitness=%.4f, best=%.4f, 树高度=%s, 节点数=%s",
                i + 1,
                len(population),
                _ind_elapsed,
                primary_fit,
                best_init_fitness,
                ind.height,
                len(ind),
            )

            # 报告初始评估进度（初始评估 + 进化代数 = 总阶段数）
            if self.progress_callback and (i + 1) % 5 == 0:
                total_phases = self.n_generations + 1
                self.progress_callback(
                    0, total_phases, best_init_fitness, primary_fit
                )

        _init_eval_total = time.time() - _init_eval_start
        logger.info(f"初始评估完成: 共{len(population)}个体, 总耗时={_init_eval_total:.1f}s")

        # 终端使用统计：记录初始种群的终端分布
        self._record_terminal_usage(population)

        # Update Z-Score normalization stats from initial population
        self._update_zscore_stats()

        # Hall-of-Fame (stored as instance attr so evaluation can access it)
        hof_size = max(self.elite_size * 2, 10)
        halloffame = tools.HallOfFame(hof_size, similar=self._make_hof_similar())
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

        # 终端使用统计：进化结束后统计各基础因子进入 HOF 的次数（诊断死终端）
        terminal_usage = self._terminal_usage_report(halloffame)

        # Build result
        # 取消标记：_evolutionary_loop 检测到 _cancel_flag 后 break，
        # 但 mine_factors 仍会执行后续的结果构建代码（best_factors、验证等），
        # 以保存已发现的因子。cancelled 标记告诉 _run_mining 将状态设为 cancelled
        # 而非 completed（见 mining.py 中的 cancelled 处理分支）。
        cancelled = self._cancel_flag

        # 进化完成，进入结果验证阶段（向前端报告进度）
        n_hof = len(halloffame)
        if self.progress_callback:
            # 发送 gen = total_gen（= n_generations + 1）标记进化已完成，
            # 前端据此显示 "进化完成，正在验证候选因子..."
            self.progress_callback(
                self.n_generations + 1, self.n_generations + 1, best_init_fitness, best_init_fitness,
            )
        logger.info(f"进化循环完成，开始验证 {n_hof} 个候选因子...")

        # ---- 最终去重安全网：对 Hall of Fame 个体执行子表达式去重 ----
        # 即使进化过程中已执行 Phase 2.6，精英个体（elites）未经去重，
        # 且 GP 可能在后续代中重新进化出重复。此安全网确保最终输出干净。
        # 注意：fitness 保留原始值（近似），因子值从去重后的树重新计算。
        deduped_hof = []
        _n_final_deduped = 0
        for tree in halloffame:
            try:
                new_nodes, n_replaced = replace_duplicate_subtrees(
                    tree, self.pset
                )
                if new_nodes is not None and n_replaced > 0:
                    new_tree = gp.PrimitiveTree(new_nodes)
                    if len(new_tree) > 0:
                        _ = new_tree.height
                        gp.compile(new_tree, self.pset)
                        new_ind = type(tree)(new_tree)
                        new_ind.fitness.values = tree.fitness.values
                        deduped_hof.append(new_ind)
                        _n_final_deduped += n_replaced
                        continue
            except Exception:
                pass
            deduped_hof.append(tree)
        if _n_final_deduped > 0:
            logger.info(
                f"最终去重: 替换{_n_final_deduped}个重复子树"
            )

        _validate_start = time.time()
        best_factors = []
        _n_sympy_simplified = 0
        for i, tree in enumerate(deduped_hof):
            _ind_t0 = time.time()
            placeholder_expr = tree_to_placeholder_expr(tree)

            # SymPy 后置简化：对占位符表达式进行代数化简
            # 例如 add(factor_0, factor_0) → mul(2, factor_0)
            #      sub(factor_0, factor_0) → 0
            #      max(factor_0, factor_0) → factor_0
            # 这显著减少展开后的重复（如8个factor_0→2个）
            simplified_placeholder = simplify_gp_expression(placeholder_expr)
            was_simplified = simplified_placeholder != placeholder_expr

            # 始终使用简化后的占位符生成展开表达式
            # 即使 simplify_gp_expression 内部异常吞掉返回原表达式，
            # _placeholder_to_code(str(tree)) ≡ _convert_expression_to_code(tree)，
            # 行为等价，不会引入新问题
            actual_expr = self._placeholder_to_code(simplified_placeholder)

            # 生成可读的显示表达式（短变量名 + 定义映射）
            # 当基础因子代码较长时（如 Alpha101 因子），完整展开会导致
            # 同一代码重复多次，可读性极差。短变量名 F0, F1 解决此问题。
            display_expr, factor_defs = self._generate_display_expression(
                simplified_placeholder
            )

            if was_simplified:
                _n_sympy_simplified += 1
                logger.info(
                    f"  SymPy 简化候选 [{i+1}]: "
                    f"{len(placeholder_expr)}→{len(simplified_placeholder)} 字符"
                )

            # Extract primary fitness (IC-based)
            fitness_values = tree.fitness.values
            if self.use_nsga2:
                primary_fitness = float(fitness_values[0])
                complexity = (
                    float(fitness_values[1])
                    if len(fitness_values) > 1
                    else float(len(tree))
                )
            else:
                primary_fitness = float(fitness_values[0])
                complexity = float(len(tree))

            factor_info = {
                "rank": i + 1,
                "expression": actual_expr,
                "display_expression": display_expr,
                "factor_definitions": factor_defs,
                "placeholder_expression": placeholder_expr,
                "simplified_expression": simplified_placeholder if was_simplified else None,
                "fitness": primary_fitness,
                "complexity": complexity,
            }

            try:
                fv = self._compute_factor_expression(tree)
                # 防御性检查：release_memory() 会将 return_values 设为 None，
                # 且空 return_values 无法进行验证
                if fv is not None and self.return_values is not None and len(self.return_values) > 0:
                    # 单因子验证超时保护：validate_factor 内部包含
                    # IC/换手率/稳定性/相关性/前视偏差等多项重型检测，
                    # 任一项卡死都会阻塞整个验证流程（实测已卡 >5min）
                    #
                    # 注意：ThreadPoolExecutor 的 with 语句 __exit__ 会调用
                    # shutdown(wait=True)，即使 future.result(timeout=X) 抛出
                    # TimeoutError，仍会阻塞等待线程完成。
                    # 修复：不使用 with 语句，超时后直接 shutdown(wait=False)
                    # 放弃线程，让主流程继续。
                    _validation_timeout = 30.0  # 单因子验证超时 30s
                    def _do_validate():
                        return factor_validation_service.validate_factor(
                            factor_values=fv,
                            return_values=self.return_values,
                        )
                    _v_executor = ThreadPoolExecutor(max_workers=1)
                    _v_future = _v_executor.submit(_do_validate)
                    try:
                        validation = _v_future.result(timeout=_validation_timeout)
                        factor_info["validation"] = validation
                        # 将预计算的因子值传递给 _finalize_task，避免重复计算
                        # _finalize_task 中的 _unified_validate_factor 会检查此字段，
                        # 如果存在则跳过 factor_calculator.calculate() 重新计算
                        factor_info["_precomputed_factor_values"] = fv
                    except TimeoutError:
                        logger.warning(
                            f"  候选因子 [{i+1}/{n_hof}] 验证超时 (>{_validation_timeout:.0f}s)，跳过"
                        )
                    finally:
                        # wait=False: 不阻塞等待线程完成，让主流程继续
                        # cancel_futures=True: 取消尚未开始的任务（Python 3.9+）
                        _v_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.debug(f"因子验证失败: {e}")

            _ind_elapsed = time.time() - _ind_t0
            _simp_note = " [已简化]" if was_simplified else ""
            logger.info(
                f"  验证候选因子 [{i+1}/{n_hof}] "
                f"({_ind_elapsed:.1f}s){_simp_note}"
            )

            best_factors.append(factor_info)

        _validate_elapsed = time.time() - _validate_start
        if _n_sympy_simplified > 0:
            logger.info(
                f"SymPy 后置简化: {_n_sympy_simplified}/{n_hof} 个候选因子被简化"
            )
        logger.info(f"候选因子验证完成: {n_hof} 个，总耗时 {_validate_elapsed:.1f}s")

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
            "cancelled": cancelled,
            "best_factors": best_factors,
            "logbook": logbook,
            "final_population_size": len(population),
            "terminal_usage": terminal_usage,
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

            _gen_start = time.time()
            logger.info(f"=== Gen {gen}/{n_generations} 开始 ===")
            self._current_generation = gen
            self._refresh_stock_sample()

            # Phase 4: 因子值缓存跨代保留
            # 评估股票样本整轮固定（见 _refresh_stock_sample），同一棵树的
            # 因子值在所有代都相同，无需按代清空缓存。

            # ---- Elitism: copy top elite_size individuals unchanged ----
            if use_nsga2:
                elites = tools.selNSGA2(population, self.elite_size)
            else:
                elites = tools.selBest(population, self.elite_size)
            elites = list(map(self.toolbox.clone, elites))

            # ---- Selection ----
            offspring = self.toolbox.select(
                population, len(population) - self.elite_size
            )
            offspring = list(map(self.toolbox.clone, offspring))

            # ---- Crossover ----
            for i in range(1, len(offspring), 2):
                if random.random() < self.cx_prob:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(
                        offspring[i - 1], offspring[i]
                    )
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            # ---- Mutation ----
            for i in range(len(offspring)):
                if random.random() < self.mut_prob:
                    (offspring[i],) = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # ---- Phase 2.5: Online Simplification (SymPy) ----
            # 业界标准方案：在每次交叉/变异后，立即对新生成的个体做 SymPy 简化
            # 消除代数等价的冗余（如 add(a,sub(b,a))→b, mul(x,1)→x）
            # 参考: Javed & Gobet (2021) "On-the-fly simplification of GP models"
            # 注意：不透明函数（如 TSRANK/STD）的重复无法通过 SymPy 消除，
            #       由适应度中的子表达式重复惩罚（dup_penalty）抑制
            n_simplified = 0
            for i, ind in enumerate(offspring):
                if not ind.fitness.valid:
                    try:
                        original_str = str(ind)
                        simplified_str = simplify_gp_expression(original_str)
                        if simplified_str != original_str and simplified_str:
                            new_tree = gp.PrimitiveTree.from_string(
                                simplified_str, self.pset
                            )
                            # 验证树非空、结构完整、编译成功
                            if len(new_tree) == 0:
                                continue
                            # height 计算会检测结构完整性（缺少子节点会 IndexError）
                            _ = new_tree.height
                            # 验证所有节点的类型与 pset 一致
                            # SymPy 简化可能引入数值常量（如 0, 1.0），
                            # 这些常量的 .ret 类型（int/float）与 pset.ret（object）
                            # 不匹配，会导致后续变异时 gp.generate 找不到对应类型
                            # 的 primitive 而崩溃
                            if any(getattr(node, "ret", None) != self.pset.ret
                                   for node in new_tree):
                                continue
                            gp.compile(new_tree, self.pset)
                            # 用简化后的个体替换原始个体
                            offspring[i] = type(ind)(new_tree)
                            n_simplified += 1
                    except Exception:
                        pass  # 简化失败，保留原始个体
            if n_simplified > 0:
                logger.info(f"  Gen {gen}: 在线简化 {n_simplified} 个个体")

            # ---- Phase 2.6: Subtree Dedup Refactoring ----
            # 检测个体内部的重复子树（包括终端重复），将重复实例替换为不同终端/随机子树
            # 解决 SymPy 无法简化不透明函数（TSRANK/CORR/SUM）重复的问题
            # 参考: Poli & McPhee (2008) — intron removal via subtree replacement
            #
            # 重要：处理所有 offspring（不仅限于 not ind.fitness.valid）。
            # 未被交叉/变异的个体可能从初始种群继承了重复终端，
            # 跳过它们会导致重复一直存活到最终结果。
            # 修改后删除 fitness.values，使其在后续重评估步骤中被重新计算。
            n_deduped = 0
            n_dups_found = 0
            n_checked = 0
            n_errors = 0
            for i, ind in enumerate(offspring):
                n_checked += 1
                try:
                    new_nodes, n_replaced = replace_duplicate_subtrees(
                        ind, self.pset
                    )
                    n_dups_found += n_replaced
                    if new_nodes is not None and n_replaced > 0:
                        new_tree = gp.PrimitiveTree(new_nodes)
                        if len(new_tree) == 0:
                            n_errors += 1
                            continue
                        _ = new_tree.height
                        gp.compile(new_tree, self.pset)
                        offspring[i] = type(ind)(new_tree)
                        # 表达式已改变，fitness 不再有效，标记需要重评估
                        if ind.fitness.valid:
                            del offspring[i].fitness.values
                        n_deduped += n_replaced
                except Exception as e:
                    n_errors += 1
                    if n_errors <= 3:
                        logger.warning(
                            f"  Gen {gen}: 子表达式去重失败: {e}"
                        )
            logger.info(
                f"  Gen {gen}: 子表达式去重 检查{n_checked}个体, "
                f"发现{n_dups_found}重复, 替换{n_deduped}个, 错误{n_errors}"
            )

            # ---- Phase 3: Diversity protection – replace duplicates ----
            # 分层去重：
            #   第一层 zobrist_hash：O(1) 检测结构相同 + 交换律同构（add(a,b)=add(b,a)）
            #   第二层 sympy_canonical_key（默认关闭）：检测代数等价（add(a,sub(b,a))≡b）
            #   对比范围：offspring 内部 + 父代 population + 精英 elites
            use_sympy_dedup = self.use_sympy_dedup
            if diversity_protection:
                seen_zobrist: set = set()
                seen_sympy: set = set()

                # 预加载父代和精英的去重键
                for ind in list(population) + list(elites):
                    seen_zobrist.add(zobrist_hash(ind))
                    if use_sympy_dedup:
                        skey = self._sympy_canonical_key_with_timeout(
                            ind, timeout=3.0
                        )
                        if skey is not None:
                            seen_sympy.add(skey)

                n_struct_dup = 0
                n_algebraic_dup = 0
                for i, ind in enumerate(offspring):
                    zkey = zobrist_hash(ind)
                    if zkey in seen_zobrist:
                        # 结构重复（含交换律同构）
                        new_ind = self.toolbox.individual()
                        offspring[i] = new_ind
                        n_struct_dup += 1
                        seen_zobrist.add(zobrist_hash(new_ind))
                        if use_sympy_dedup:
                            skey = self._sympy_canonical_key_with_timeout(
                                new_ind, timeout=3.0
                            )
                            if skey is not None:
                                seen_sympy.add(skey)
                        continue

                    if not use_sympy_dedup:
                        seen_zobrist.add(zkey)
                        continue

                    # 深度去重：SymPy 规范形检测代数等价（带超时，避免大树卡死）
                    skey = self._sympy_canonical_key_with_timeout(ind, timeout=3.0)
                    if skey is not None and skey in seen_sympy:
                        new_ind = self.toolbox.individual()
                        offspring[i] = new_ind
                        n_algebraic_dup += 1
                        seen_zobrist.add(zobrist_hash(new_ind))
                        new_skey = self._sympy_canonical_key_with_timeout(
                            new_ind, timeout=3.0
                        )
                        if new_skey is not None:
                            seen_sympy.add(new_skey)
                    else:
                        seen_zobrist.add(zkey)
                        if skey is not None:
                            seen_sympy.add(skey)

                if n_struct_dup + n_algebraic_dup > 0:
                    logger.info(
                        f"  Gen {gen}: 去重 {n_struct_dup} 结构重复 + "
                        f"{n_algebraic_dup} 代数等价重复"
                    )

            # ---- Re-evaluate invalid individuals (serial, avoid nested ThreadPool deadlock) ----
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            if invalid:
                _eval_start = time.time()
                logger.info(
                    f"  Gen {gen}: 开始评估 {len(invalid)} 个无效个体..."
                )
                fitnesses = []
                for idx, ind in enumerate(invalid):
                    _ind_t0 = time.time()

                    # 代际超时保护：如果单个个体评估超过60秒，跳过
                    # 如果整个代评估超过300秒，强制终止
                    _gen_elapsed = time.time() - _eval_start
                    if _gen_elapsed > 300:
                        logger.warning(
                            f"  Gen {gen}: 代际评估超时（{_gen_elapsed:.0f}s > 300s），"
                            f"强制终止剩余 {len(invalid) - idx} 个个体"
                        )
                        break

                    # 同初始评估：逐个体串行评估（内部逐股已串行）
                    _ts_op_count = sum(1 for node in ind if hasattr(node, 'name') and
                                      node.name.startswith('ts_') and 'corr' in node.name)
                    if _ts_op_count >= 2:
                        logger.debug(
                            "  Gen %s 个体 [%s/%s] 跳过: %s个ts_corr嵌套",
                            gen + 1,
                            idx + 1,
                            len(invalid),
                            _ts_op_count,
                        )
                        fit = self._penalty_fitness() if self.use_nsga2 else (-1.0,)
                    else:
                        try:
                            # 单个个体评估超时保护（30秒）
                            # 使用单线程 ThreadPoolExecutor 作为超时包装器
                            _eval_executor = ThreadPoolExecutor(max_workers=1)
                            _eval_future = _eval_executor.submit(self.toolbox.evaluate, ind)
                            try:
                                fit = _eval_future.result(timeout=30.0)
                            except TimeoutError:
                                logger.warning(
                                    f"  Gen {gen+1} 个体 [{idx+1}/{len(invalid)}] 评估超时(30s)，"
                                    f"表达式={str(ind)[:80]}"
                                )
                                _eval_future.cancel()
                                fit = self._penalty_fitness() if self.use_nsga2 else (-1.0,)
                            finally:
                                _eval_executor.shutdown(wait=False)
                        except Exception as e:
                            logger.warning(
                                f"  Gen {gen+1} 个体 [{idx+1}/{len(invalid)}] 评估异常: {e}"
                            )
                            fit = self._penalty_fitness() if self.use_nsga2 else (-1.0,)

                    _ind_elapsed = time.time() - _ind_t0
                    logger.debug(
                        "  Gen %s 个体 [%s/%s] 评估完成 (%.2fs)",
                        gen + 1,
                        idx + 1,
                        len(invalid),
                        _ind_elapsed,
                    )
                    fitnesses.append(fit)
                _eval_elapsed = time.time() - _eval_start
                logger.info(
                    f"Gen {gen+1} 评估 {len(invalid)} 个个体，"
                    f"总耗时 {_eval_elapsed:.1f}s"
                )
                # 代际超时 break 时 fitnesses 可能短于 invalid，剩余个体的
                # fitness 为空元组，随后 stats.compile 取 values[0] 会抛
                # IndexError。缺失部分统一填惩罚值，保证种群 fitness 完整。
                _fallback_fit = (
                    self._penalty_fitness() if self.use_nsga2 else (-1.0,)
                )
                for idx, ind in enumerate(invalid):
                    ind.fitness.values = (
                        fitnesses[idx] if idx < len(fitnesses) else _fallback_fit
                    )

            # ---- Replace population: offspring + elites ----
            population[:] = offspring + elites

            # 终端使用统计：累计当代种群的终端分布
            self._record_terminal_usage(population)

            # Update Z-Score normalization stats from this generation's evaluations
            self._update_zscore_stats()

            halloffame.update(population)

            if logbook is not None:
                record = self.stats.compile(population)
                logbook.record(gen=gen, **record)

            # ---- Checkpoint: 每5代保存进化状态到数据库 ----
            # 保存失败不影响进化（_save_checkpoint 内部有 try/except）
            if gen % 5 == 0 or gen == n_generations:
                self._save_checkpoint(gen, n_generations, population, halloffame)

            if progress:
                record = (
                    self.stats.compile(population) if logbook is None else logbook[-1]
                )
                best_fit = (
                    float(record["max"]) if record.get("max") is not None else 0.0
                )
                avg_fit = float(record["avg"]) if record.get("avg") is not None else 0.0

                if self.progress_callback:
                    # total_phases = n_generations + 1（初始评估 + 进化代数）
                    # gen 从 1 开始，对应总阶段中的第 2 个阶段
                    self.progress_callback(gen, n_generations + 1, best_fit, avg_fit)

                logger.info(
                    f"Generation {gen}/{n_generations} - Best: {best_fit:.4f}, "
                    f"Avg: {avg_fit:.4f}, Elite: {self.elite_size}"
                )
                # 多样性观测（只读，不影响进化）：唯一结构数 / 树规模 / HOF 取舍
                _uniq = len({zobrist_hash(ind) for ind in population})
                _sizes = [len(ind) for ind in population]
                _hof_sizes = [len(ind) for ind in halloffame]
                _hof_ic = [float(ind.fitness.values[0]) for ind in halloffame]
                logger.info(
                    f"  多样性: 唯一结构 {_uniq}/{len(population)}, "
                    f"树节点 mean={float(np.mean(_sizes)):.1f} max={max(_sizes)}, "
                    f"HOF节点 mean={float(np.mean(_hof_sizes)):.1f} "
                    f"max={max(_hof_sizes)}, HOF_IC {min(_hof_ic):.4f}~{max(_hof_ic):.4f}"
                )
                _gen_elapsed = time.time() - _gen_start
                logger.info(f"=== Gen {gen}/{n_generations} 完成, 耗时 {_gen_elapsed:.1f}s ===")

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

        # Evaluate initial population (with per-individual logging)
        _init_eval_start = time.time()
        fitnesses = []
        for idx, ind in enumerate(population):
            _ind_t0 = time.time()
            try:
                # 单个个体评估超时保护（30秒）
                _eval_executor = ThreadPoolExecutor(max_workers=1)
                _eval_future = _eval_executor.submit(self.toolbox.evaluate, ind)
                try:
                    fit = _eval_future.result(timeout=30.0)
                except TimeoutError:
                    logger.warning(
                        f"  初始评估 [{idx+1}/{self.population_size}] 超时(30s)，"
                        f"表达式={str(ind)[:80]}"
                    )
                    _eval_future.cancel()
                    fit = (0.0,)
                finally:
                    _eval_executor.shutdown(wait=False)
                _ind_elapsed = time.time() - _ind_t0
                if (idx + 1) % 5 == 0 or idx == 0:
                    logger.info(
                        f"  初始评估 [{idx+1}/{self.population_size}] "
                        f"({_ind_elapsed:.2f}s/个体)"
                    )
                fitnesses.append(fit)
            except Exception as e:
                _ind_elapsed = time.time() - _ind_t0
                logger.warning(
                    f"  初始评估 [{idx+1}/{self.population_size}] 异常 "
                    f"({_ind_elapsed:.2f}s): {e}"
                )
                fitnesses.append(self._penalty_fitness())
        _init_eval_elapsed = time.time() - _init_eval_start
        logger.info(
            f"初始种群评估完成: {self.population_size} 个个体，"
            f"总耗时 {_init_eval_elapsed:.1f}s"
        )
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Update Z-Score normalization stats from initial population
        self._update_zscore_stats()

        halloffame = tools.HallOfFame(5, similar=self._make_hof_similar())
        halloffame.update(population)
        self._halloffame = halloffame

        # 委托通用进化循环（简化版：无日志、无NSGA2，但启用去重保护）
        self._evolutionary_loop(
            population,
            n_generations,
            halloffame,
            logbook=None,
            use_nsga2=False,
            diversity_protection=True,
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
    * ``diversity_penalty_coeff`` – penalty for similarity to HoF (default 0.02)
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
        **kwargs,
    )
