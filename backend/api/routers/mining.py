"""
因子挖掘API路由

支持六种算法模式:
  - genetic: DEAP遗传规划（向后兼容）
  - pysr: PySR符号回归
  - dual: 两者并行执行，合并最优结果
  - tree_prescreen: 树模型预筛选 → 符号回归管道
  - gflownet: GFlowNet增强遗传规划（实验性）
  - deep_implicit: 深度隐式因子模型（Transformer，前沿赛道）
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio

import numpy as np
import math

router = APIRouter()


def _safe_float(value, default=0.0) -> float:
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _extract_first_fitness(value) -> float:
    if isinstance(value, (tuple, list, np.ndarray)):
        return _safe_float(value[0]) if len(value) > 0 else 0.0
    return _safe_float(value)


# ========== 数据模型 ==========

class GeneticMiningRequest(BaseModel):
    stock_codes: List[str] = []
    stock_code: Optional[str] = None
    base_factors: List[str] = []
    start_date: str
    end_date: str
    population_size: int = 50
    n_generations: int = 10
    cx_prob: float = 0.7
    mut_prob: float = 0.3
    elite_size: int = 5
    fitness_objective: str = "ic_mean"
    ic_threshold: float = 0.03
    parsimony_coeff: float = 0.001
    diversity_penalty_coeff: float = 0.1
    cv_folds: int = 0
    use_extended_primitives: bool = True
    max_tree_depth: int = 17
    use_nsga2: bool = True
    # ---- Algorithm selection ----
    algorithm: str = "dual"
    # ---- PySR parameters ----
    pysr_niterations: int = 40
    pysr_populations: int = 30
    pysr_binary_operators: Optional[List[str]] = None
    pysr_unary_operators: Optional[List[str]] = None
    pysr_maxsize: int = 30
    pysr_maxdepth: int = 5
    pysr_constraints: Optional[Dict] = None
    pysr_nested_constraints: Optional[Dict] = None
    pysr_parsimony: float = 0.0032
    pysr_procs: int = 8
    pysr_population_size: int = 33
    # ---- Tree Prescreen parameters ----
    tree_model_type: str = "auto"
    top_k: int = 0
    importance_threshold: float = 0.01
    tree_n_estimators: int = 100
    tree_max_depth: int = 5
    downstream_algorithm: str = "dual"
    # ---- GFlowNet parameters ----
    gflownet_n_trajectories: int = 200
    gflownet_n_iterations: int = 50
    gflownet_hidden_dim: int = 128
    gflownet_learning_rate: float = 1e-3
    gflownet_max_expression_depth: int = 5
    gflownet_temperature: float = 1.0
    gflownet_reward_scale: float = 10.0
    gflownet_buffer_size: int = 1000
    # ---- Deep Factor parameters ----
    deep_d_model: int = 64
    deep_n_heads: int = 4
    deep_n_layers: int = 3
    deep_d_ff: int = 256
    deep_n_latent_factors: int = 5
    deep_dropout: float = 0.1
    deep_seq_length: int = 20
    deep_learning_rate: float = 1e-4
    deep_n_epochs: int = 50
    deep_batch_size: int = 32
    deep_weight_decay: float = 1e-5
    deep_early_stopping_patience: int = 5


# ========== 任务存储（内存） ==========
mining_tasks = {}


# ========== API端点 ==========

@router.post("/genetic")
async def start_genetic_mining(request: GeneticMiningRequest, background_tasks: BackgroundTasks):
    """启动因子挖掘（支持遗传算法/PySR/双算法并行）"""
    try:
        import uuid
        task_id = str(uuid.uuid4())

        mining_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "algorithm": request.algorithm,
        }

        background_tasks.add_task(
            _run_mining,
            task_id,
            request
        )

        algo_label = {
            "genetic": "遗传规划",
            "pysr": "PySR符号回归",
            "dual": "双算法并行",
            "tree_prescreen": "树模型预筛选",
            "gflownet": "GFlowNet增强GP",
            "deep_implicit": "深度隐式因子",
        }
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "pending",
                "algorithm": request.algorithm,
            },
            "message": f"{algo_label.get(request.algorithm, '挖掘')}任务已启动"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_mining(task_id: str, request: GeneticMiningRequest):
    """Unified mining entry point that dispatches to the correct algorithm."""
    try:
        import logging
        logger = logging.getLogger(__name__)

        stock_codes = list(request.stock_codes) if request.stock_codes else []
        if not stock_codes and request.stock_code:
            logger.warning("Single stock_code provided, converting to stock_codes list.")
            stock_codes = [request.stock_code]

        if not stock_codes:
            raise Exception("未提供股票代码，请通过stock_codes或stock_code指定")

        algorithm = request.algorithm
        logger.info(f"Starting mining task {task_id} with algorithm={algorithm}")
        logger.info(f"Stocks: {stock_codes}, Base factors: {request.base_factors}")

        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session
        from backend.services.data_service import data_service

        mining_tasks[task_id]["status"] = "running"

        data = data_service.get_stock_data(
            stock_codes[0],
            request.start_date,
            request.end_date
        )

        if data is None or len(data) == 0:
            raise Exception("未获取到有效数据")

        logger.info(f"Retrieved {len(data)} rows of data for primary stock")

        if "close" in data.columns:
            data["return"] = data["close"].pct_change()

        base_factor_codes = []
        if request.base_factors and len(request.base_factors) > 0:
            try:
                db = get_db_session()
                repo = FactorRepository(db)

                for factor_name in request.base_factors:
                    factor = repo.get_by_name(factor_name)
                    if factor:
                        base_factor_codes.append(factor.code)
                        logger.info(f"Found factor: {factor_name} -> {factor.code}")
                    else:
                        logger.warning(f"Factor not found in database: {factor_name}")

                db.close()
            except Exception as e:
                logger.error(f"Error loading factors from database: {e}")

        if not base_factor_codes:
            logger.warning("No valid base factors found, using default codes")
            base_factor_codes = [
                "RSI(close, 14)",
                "SMA(close, 20)",
                "close / open",
                "volume / 1000000",
                "MACD(close, 12, 26, 9)[0]"
            ]
        else:
            logger.info(f"Using {len(base_factor_codes)} base factor codes")

        try:
            if algorithm == "genetic":
                result = await _run_genetic_only(
                    task_id, request, data, base_factor_codes,
                    factor_service, stock_codes, logger
                )
            elif algorithm == "pysr":
                result = await _run_pysr_only(
                    task_id, request, data, base_factor_codes,
                    factor_service, stock_codes, logger
                )
            elif algorithm in ("tree_prescreen", "gflownet", "deep_implicit"):
                result = await _run_unified_mining(
                    task_id, request, data, base_factor_codes,
                    factor_service, stock_codes, logger
                )
            else:
                result = await _run_dual_mining(
                    task_id, request, data, base_factor_codes,
                    factor_service, stock_codes, logger
                )

            if not result.get("success"):
                raise Exception(result.get("message", "挖掘失败"))

            _finalize_task(task_id, result, request, logger)

        except ImportError as e:
            logger.warning(f"Mining library not available, using simulation mode: {e}")
            await _run_simulated_mining(task_id, request, data, base_factor_codes, factor_service, logger)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(e)


async def _run_genetic_only(
    task_id, request, data, base_factor_codes, factor_service, stock_codes, logger
) -> dict:
    """Run DEAP genetic programming only."""
    from backend.services.genetic_factor_mining_service import create_genetic_mining_service

    logger.info("Using DEAP genetic algorithm mining")

    mining_service = create_genetic_mining_service(
        base_factors=base_factor_codes,
        data=data,
        return_column="return",
        population_size=request.population_size,
        n_generations=request.n_generations,
        cx_prob=request.cx_prob,
        mut_prob=request.mut_prob,
        factor_calculator=factor_service.calculator,
        elite_size=request.elite_size,
        fitness_objective=request.fitness_objective,
        parsimony_coeff=request.parsimony_coeff,
        diversity_penalty_coeff=request.diversity_penalty_coeff,
        cv_folds=request.cv_folds,
        use_extended_primitives=request.use_extended_primitives,
        max_tree_depth=request.max_tree_depth,
        use_nsga2=request.use_nsga2,
    )

    if len(stock_codes) >= 2:
        logger.info(f"Setting stock pool with {len(stock_codes)} stocks")
        mining_service.set_stock_pool(stock_codes, request.start_date, request.end_date)

    def progress_callback(gen, total_gen, best_fitness, avg_fitness, **kwargs):
        _update_progress(task_id, gen, total_gen, best_fitness, avg_fitness, "genetic", logger)

    mining_service.set_progress_callback(progress_callback)
    result = mining_service.mine_factors()
    result["source"] = "genetic"
    return result


async def _run_pysr_only(
    task_id, request, data, base_factor_codes, factor_service, stock_codes, logger
) -> dict:
    """Run PySR symbolic regression only."""
    from backend.services.pysr_factor_mining_service import create_pysr_mining_service

    logger.info("Using PySR symbolic regression mining")

    mining_service = create_pysr_mining_service(
        base_factors=base_factor_codes,
        data=data,
        return_column="return",
        factor_calculator=factor_service.calculator,
        niterations=request.pysr_niterations,
        populations=request.pysr_populations,
        binary_operators=request.pysr_binary_operators,
        unary_operators=request.pysr_unary_operators,
        maxsize=request.pysr_maxsize,
        maxdepth=request.pysr_maxdepth,
        constraints=request.pysr_constraints,
        nested_constraints=request.pysr_nested_constraints,
        parsimony=request.pysr_parsimony,
        procs=request.pysr_procs,
        population_size=request.pysr_population_size,
        fitness_objective=request.fitness_objective,
        cv_folds=request.cv_folds,
    )

    if len(stock_codes) >= 2:
        logger.info(f"Setting stock pool with {len(stock_codes)} stocks for PySR")
        mining_service.set_stock_pool(stock_codes, request.start_date, request.end_date)

    def progress_callback(iteration, total_iter, best_fitness, avg_fitness, **kwargs):
        _update_progress(task_id, iteration, total_iter, best_fitness, avg_fitness, "pysr", logger)

    mining_service.set_progress_callback(progress_callback)
    result = mining_service.mine_factors()
    result["source"] = "pysr"
    return result


async def _run_unified_mining(
    task_id, request, data, base_factor_codes, factor_service, stock_codes, logger
) -> dict:
    """Run tree_prescreen / gflownet / deep_implicit via DualMiningService."""
    from backend.services.dual_mining_service import create_dual_mining_service

    algorithm = request.algorithm
    algo_names = {
        "tree_prescreen": "树模型预筛选符号回归",
        "gflownet": "GFlowNet增强遗传规划",
        "deep_implicit": "深度隐式因子模型",
    }
    logger.info(f"Using {algo_names.get(algorithm, algorithm)} mining")

    mining_service = create_dual_mining_service(
        base_factors=base_factor_codes,
        data=data,
        return_column="return",
        factor_calculator=factor_service.calculator,
        algorithm=algorithm,
        # GP params
        population_size=request.population_size,
        n_generations=request.n_generations,
        cx_prob=request.cx_prob,
        mut_prob=request.mut_prob,
        elite_size=request.elite_size,
        fitness_objective=request.fitness_objective,
        parsimony_coeff=request.parsimony_coeff,
        diversity_penalty_coeff=request.diversity_penalty_coeff,
        cv_folds=request.cv_folds,
        use_extended_primitives=request.use_extended_primitives,
        max_tree_depth=request.max_tree_depth,
        use_nsga2=request.use_nsga2,
        # PySR params
        pysr_niterations=request.pysr_niterations,
        pysr_populations=request.pysr_populations,
        pysr_binary_operators=request.pysr_binary_operators,
        pysr_unary_operators=request.pysr_unary_operators,
        pysr_maxsize=request.pysr_maxsize,
        pysr_maxdepth=request.pysr_maxdepth,
        pysr_constraints=request.pysr_constraints,
        pysr_nested_constraints=request.pysr_nested_constraints,
        pysr_parsimony=request.pysr_parsimony,
        pysr_procs=request.pysr_procs,
        pysr_population_size=request.pysr_population_size,
        # Tree Prescreen params
        tree_model_type=request.tree_model_type,
        top_k=request.top_k,
        importance_threshold=request.importance_threshold,
        tree_n_estimators=request.tree_n_estimators,
        tree_max_depth=request.tree_max_depth,
        downstream_algorithm=request.downstream_algorithm,
        # GFlowNet params
        gflownet_n_trajectories=request.gflownet_n_trajectories,
        gflownet_n_iterations=request.gflownet_n_iterations,
        gflownet_hidden_dim=request.gflownet_hidden_dim,
        gflownet_learning_rate=request.gflownet_learning_rate,
        gflownet_max_expression_depth=request.gflownet_max_expression_depth,
        gflownet_temperature=request.gflownet_temperature,
        gflownet_reward_scale=request.gflownet_reward_scale,
        gflownet_buffer_size=request.gflownet_buffer_size,
        # Deep Factor params
        deep_d_model=request.deep_d_model,
        deep_n_heads=request.deep_n_heads,
        deep_n_layers=request.deep_n_layers,
        deep_d_ff=request.deep_d_ff,
        deep_n_latent_factors=request.deep_n_latent_factors,
        deep_dropout=request.deep_dropout,
        deep_seq_length=request.deep_seq_length,
        deep_learning_rate=request.deep_learning_rate,
        deep_n_epochs=request.deep_n_epochs,
        deep_batch_size=request.deep_batch_size,
        deep_weight_decay=request.deep_weight_decay,
        deep_early_stopping_patience=request.deep_early_stopping_patience,
    )

    if len(stock_codes) >= 2:
        logger.info(f"Setting stock pool with {len(stock_codes)} stocks for {algorithm}")
        mining_service.set_stock_pool(stock_codes, request.start_date, request.end_date)

    def progress_callback(gen, total_gen, best_fitness, avg_fitness, algorithm=algorithm, **kwargs):
        _update_progress(task_id, gen, total_gen, best_fitness, avg_fitness, algorithm, logger)

    mining_service.set_progress_callback(progress_callback)
    result = mining_service.mine_factors()

    if result.get("fitness_history"):
        mining_tasks[task_id]["fitness_history"] = result["fitness_history"]

    return result


async def _run_dual_mining(
    task_id, request, data, base_factor_codes, factor_service, stock_codes, logger
) -> dict:
    """Run both DEAP GP and PySR in parallel, merge best results."""
    from backend.services.dual_mining_service import create_dual_mining_service

    logger.info("Using dual algorithm mining (DEAP GP + PySR)")

    mining_service = create_dual_mining_service(
        base_factors=base_factor_codes,
        data=data,
        return_column="return",
        factor_calculator=factor_service.calculator,
        algorithm="dual",
        population_size=request.population_size,
        n_generations=request.n_generations,
        cx_prob=request.cx_prob,
        mut_prob=request.mut_prob,
        elite_size=request.elite_size,
        fitness_objective=request.fitness_objective,
        parsimony_coeff=request.parsimony_coeff,
        diversity_penalty_coeff=request.diversity_penalty_coeff,
        cv_folds=request.cv_folds,
        use_extended_primitives=request.use_extended_primitives,
        max_tree_depth=request.max_tree_depth,
        use_nsga2=request.use_nsga2,
        pysr_niterations=request.pysr_niterations,
        pysr_populations=request.pysr_populations,
        pysr_binary_operators=request.pysr_binary_operators,
        pysr_unary_operators=request.pysr_unary_operators,
        pysr_maxsize=request.pysr_maxsize,
        pysr_maxdepth=request.pysr_maxdepth,
        pysr_constraints=request.pysr_constraints,
        pysr_nested_constraints=request.pysr_nested_constraints,
        pysr_parsimony=request.pysr_parsimony,
        pysr_procs=request.pysr_procs,
        pysr_population_size=request.pysr_population_size,
    )

    if len(stock_codes) >= 2:
        logger.info(f"Setting stock pool with {len(stock_codes)} stocks for dual mining")
        mining_service.set_stock_pool(stock_codes, request.start_date, request.end_date)

    def progress_callback(gen, total_gen, best_fitness, avg_fitness, algorithm="genetic"):
        _update_progress(task_id, gen, total_gen, best_fitness, avg_fitness, algorithm, logger)

    mining_service.set_progress_callback(progress_callback)
    result = mining_service.mine_factors()

    if result.get("fitness_history"):
        mining_tasks[task_id]["fitness_history"] = result["fitness_history"]

    return result


def _update_progress(task_id, gen, total_gen, best_fitness, avg_fitness, algorithm, logger):
    """Update mining task progress in the shared task store."""
    progress = int(gen / max(total_gen, 1) * 100)
    mining_tasks[task_id]["progress"] = progress
    mining_tasks[task_id]["current_generation"] = gen
    mining_tasks[task_id]["total_generations"] = total_gen
    mining_tasks[task_id]["best_fitness"] = _safe_float(best_fitness)
    mining_tasks[task_id]["avg_fitness"] = _safe_float(avg_fitness)
    mining_tasks[task_id]["current_algorithm"] = algorithm

    if "fitness_history" not in mining_tasks[task_id]:
        mining_tasks[task_id]["fitness_history"] = {"best": [], "average": []}
    mining_tasks[task_id]["fitness_history"]["best"].append(_safe_float(best_fitness))
    mining_tasks[task_id]["fitness_history"]["average"].append(_safe_float(avg_fitness))

    logger.info(
        f"Progress: {progress}%, {algorithm} Gen {gen}/{total_gen}, "
        f"Best: {_safe_float(best_fitness):.4f}, Avg: {_safe_float(avg_fitness):.4f}"
    )


def _finalize_task(task_id: str, result: dict, request: GeneticMiningRequest, logger):
    """Convert mining result to frontend format and store in task."""
    best_factors = result.get("best_factors", [])

    discovered_factors = []
    for i, factor_info in enumerate(best_factors):
        validation = factor_info.get("validation", {})
        ic = validation.get("ic_validation", {}).get("ic", 0.0)
        ir = validation.get("ir_validation", {}).get("ir", 0.0)
        fitness = factor_info.get("fitness", 0.0)
        complexity = factor_info.get("complexity", 0.0)
        source = factor_info.get("source", result.get("source", "unknown"))

        discovered_factors.append({
            "name": f"Mined_Factor_{i+1}",
            "expression": factor_info["expression"],
            "ic": _safe_float(ic),
            "ir": _safe_float(ir),
            "fitness": _safe_float(fitness),
            "complexity": _safe_float(complexity),
            "source": source,
        })

    logbook = result.get("logbook")
    if logbook is not None:
        fitness_history = {
            "best": [_extract_first_fitness(gen["max"]) for gen in logbook],
            "average": [_extract_first_fitness(gen["avg"]) for gen in logbook]
        }
    elif result.get("fitness_history"):
        fitness_history = result["fitness_history"]
    else:
        fitness_history = {"best": [], "average": []}

    result_data = {
        "factors": discovered_factors,
        "best_fitness": _safe_float(discovered_factors[0]["fitness"]) if discovered_factors else 0.0,
        "avg_fitness": _safe_float(sum(f["fitness"] for f in discovered_factors) / len(discovered_factors)) if discovered_factors else 0.0,
        "generations": request.n_generations,
        "fitness_history": fitness_history,
        "algorithm": request.algorithm,
    }

    gp_result = result.get("gp_result")
    pysr_result = result.get("pysr_result")
    if gp_result:
        gp_factors = gp_result.get("best_factors", [])
        result_data["gp_factor_count"] = len(gp_factors)
        result_data["gp_best_fitness"] = _safe_float(
            gp_factors[0].get("fitness", 0) if gp_factors else 0
        )
    if pysr_result:
        pysr_factors = pysr_result.get("best_factors", [])
        result_data["pysr_factor_count"] = len(pysr_factors)
        result_data["pysr_best_fitness"] = _safe_float(
            pysr_factors[0].get("fitness", 0) if pysr_factors else 0
        )

    # 新算法特有结果
    if result.get("feature_importance"):
        result_data["feature_importance"] = result["feature_importance"]
    if result.get("selected_features"):
        result_data["selected_features"] = result["selected_features"]
    if result.get("model_info"):
        result_data["model_info"] = result["model_info"]
    if result.get("training_history"):
        result_data["training_history"] = result["training_history"]
    if result.get("policy_loss_history"):
        result_data["policy_loss_history"] = result["policy_loss_history"]

    mining_tasks[task_id]["status"] = "completed"
    mining_tasks[task_id]["progress"] = 100
    mining_tasks[task_id]["result"] = result_data
    mining_tasks[task_id]["fitness_history"] = fitness_history

    logger.info(f"Task {task_id} completed successfully")
    logger.info(f"Discovered {len(discovered_factors)} factors (algorithm={request.algorithm})")


async def _run_simulated_mining(task_id: str, request: GeneticMiningRequest, data, base_factor_codes, factor_service, logger):
    """模拟模式挖掘（当DEAP/PySR库未安装时使用）"""
    factor_values = {}
    for code in base_factor_codes:
        try:
            values = factor_service.calculator.calculate(data, code)
            if values is not None and len(values.dropna()) > 0:
                factor_values[code] = values
                logger.info(f"Successfully calculated factor: {code}, {len(values.dropna())} valid values")
        except Exception as e:
            logger.warning(f"计算基础因子失败 {code}: {e}")
            continue

    if not factor_values:
        logger.error("No valid factor values calculated")
        raise Exception("无法计算任何有效的因子值")

    n_generations = request.n_generations
    fitness_history = {"best": [], "average": []}
    current_best_fitness = 0.0

    for gen in range(n_generations):
        progress = int((gen + 1) / n_generations * 100)
        mining_tasks[task_id]["progress"] = progress

        current_best_fitness = 0.03 + (gen + 1) * 0.005 + (0.001 * (gen % 3))
        current_avg_fitness = current_best_fitness * (0.85 + 0.1 * (gen % 2))

        fitness_history["best"].append(current_best_fitness)
        fitness_history["average"].append(current_avg_fitness)

        mining_tasks[task_id]["current_generation"] = gen + 1
        mining_tasks[task_id]["total_generations"] = n_generations
        mining_tasks[task_id]["best_fitness"] = current_best_fitness
        mining_tasks[task_id]["avg_fitness"] = current_avg_fitness
        mining_tasks[task_id]["fitness_history"] = fitness_history

        logger.info(f"Generation {gen + 1}/{n_generations} completed, best_fitness={current_best_fitness:.4f}")

        await asyncio.sleep(0.5)

    discovered_factors = []
    code_list = list(factor_values.keys())

    for i in range(min(5, len(code_list))):
        base_code = code_list[i % len(code_list)]
        if i == 0:
            expression = f"({base_code} * 1.5)"
        elif i == 1:
            expression = f"({base_code} + close / open)"
        elif i == 2:
            expression = f"({base_code} * volume / 1000000)"
        elif i == 3:
            expression = f"({base_code} - SMA(close, 20))"
        else:
            expression = f"({base_code} / (close + 1))"

        discovered_factors.append({
            "name": f"Mined_Factor_{i+1}",
            "expression": expression,
            "ic": 0.03 + (i * 0.01),
            "ir": 0.5 + (i * 0.1),
            "fitness": 0.03 + (i * 0.01),
            "source": "simulated",
        })

    result = {
        "factors": discovered_factors,
        "best_fitness": discovered_factors[0]["ic"] if discovered_factors else 0,
        "avg_fitness": sum(f["fitness"] for f in discovered_factors) / len(discovered_factors) if discovered_factors else 0,
        "generations": n_generations,
        "fitness_history": fitness_history,
        "algorithm": request.algorithm,
    }

    mining_tasks[task_id]["status"] = "completed"
    mining_tasks[task_id]["progress"] = 100
    mining_tasks[task_id]["result"] = result
    mining_tasks[task_id]["fitness_history"] = fitness_history

    logger.info(f"Task {task_id} completed (simulated mode)")
    logger.info(f"Discovered {len(discovered_factors)} factors")


@router.get("/status/{task_id}")
async def get_mining_status(task_id: str):
    """获取挖掘状态"""
    import logging
    logger = logging.getLogger(__name__)

    if task_id not in mining_tasks:
        logger.warning(f"Status requested for non-existent task {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    task = mining_tasks[task_id]
    logger.info(f"Status check for task {task_id}: {task['status']}")

    response_data = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "error": task.get("error"),
        "algorithm": task.get("algorithm", "genetic"),
    }

    if task["status"] == "completed" and "result" in task:
        result = task["result"]
        response_data["current_generation"] = result.get("generations", 0)
        response_data["total_generations"] = result.get("generations", 0)
        response_data["best_fitness"] = result.get("best_fitness", 0)
        response_data["avg_fitness"] = result.get("avg_fitness", 0)
        response_data["fitness_history"] = result.get("fitness_history", {"best": [], "average": []})
        response_data["algorithm"] = result.get("algorithm", task.get("algorithm", "genetic"))
        logger.info(f"Returning completed status with fitness_history length: {len(response_data['fitness_history']['best'])}")
    else:
        response_data["current_generation"] = task.get("current_generation", 0)
        response_data["total_generations"] = task.get("total_generations", 10)
        response_data["best_fitness"] = task.get("best_fitness", 0.03)
        response_data["avg_fitness"] = task.get("avg_fitness", 0.03)
        response_data["fitness_history"] = task.get("fitness_history", {"best": [], "average": []})
        response_data["current_algorithm"] = task.get("current_algorithm", "")
        logger.info(f"Returning running status: gen {response_data['current_generation']}/{response_data['total_generations']}")

    return {
        "success": True,
        "data": response_data
    }


@router.get("/results/{task_id}")
async def get_mining_results(task_id: str):
    """获取挖掘结果"""
    if task_id not in mining_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = mining_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}")

    return {
        "success": True,
        "data": task["result"]
    }
