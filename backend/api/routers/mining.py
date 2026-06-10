"""
因子挖掘API路由

支持五种算法模式:
  - genetic: DEAP遗传规划
  - pysr: PySR符号回归
  - tree_prescreen: 树模型预筛选 → 符号回归管道
  - gflownet: GFlowNet增强遗传规划（实验性）
  - deep_implicit: 深度隐式因子模型（Transformer，前沿赛道）
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio

import numpy as np
import math
import json
import time
import logging
from datetime import datetime

from backend.core.database import get_db
from backend.models.mining_task import MiningTaskModel
from backend.repositories.mining_task_repository import MiningTaskRepository

logger = logging.getLogger(__name__)

# 模块级导入：_unified_validate_factor 等模块级函数需要直接访问
from backend.services import factor_validation_service

router = APIRouter()


def _safe_float(value, default=None):
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
    algorithm: str = "genetic"
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
    downstream_algorithm: str = "genetic"
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
    # ---- Frequency ----
    freq: str = "D"
    period: Optional[str] = None


# ========== 任务存储（内存） ==========
MAX_TASKS = 100
TASK_TTL_SECONDS = 86400  # 24 hours
MAX_FITNESS_HISTORY_ENTRIES = 1000

mining_tasks = {}
# 挖掘服务实例引用（用于取消）
mining_services = {}


def _cleanup_old_tasks():
    """清理过期和超量的挖掘任务，防止内存泄漏"""
    now = time.time()
    # 1. 移除超过TTL的已完成/失败/取消任务
    expired_ids = []
    for task_id, task in mining_tasks.items():
        created_at = task.get("created_at", 0)
        if created_at and (now - created_at) > TASK_TTL_SECONDS:
            status = task.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                expired_ids.append(task_id)
    for task_id in expired_ids:
        mining_tasks.pop(task_id, None)
        mining_services.pop(task_id, None)

    # 2. 如果仍然超过MAX_TASKS，移除最旧的任务
    if len(mining_tasks) > MAX_TASKS:
        sorted_tasks = sorted(
            mining_tasks.items(),
            key=lambda x: x[1].get("created_at", 0)
        )
        excess = len(mining_tasks) - MAX_TASKS
        for task_id, _ in sorted_tasks[:excess]:
            mining_tasks.pop(task_id, None)
            mining_services.pop(task_id, None)


# ========== API端点 ==========

@router.post("/genetic")
async def start_genetic_mining(request: GeneticMiningRequest, background_tasks: BackgroundTasks):
    """启动因子挖掘（支持遗传算法/PySR/树预筛选/GFlowNet/深度隐式因子）"""
    try:
        import uuid
        task_id = str(uuid.uuid4())

        # 清理过期和超量任务，防止内存泄漏
        _cleanup_old_tasks()

        mining_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "algorithm": request.algorithm,
            "created_at": time.time(),
        }

        # 持久化到数据库
        try:
            with get_db() as db:
                repo = MiningTaskRepository(db)
                task_record = MiningTaskModel(
                    task_id=task_id,
                    status="pending",
                    algorithm=request.algorithm,
                    stock_codes=json.dumps(request.stock_codes or ([request.stock_code] if request.stock_code else [])),
                    base_factors=json.dumps(request.base_factors or []),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    freq=request.freq,
                    config=request.model_dump(),
                )
                repo.create(task_record)
        except Exception as e:
            logger.warning(f"持久化挖掘任务失败（不影响运行）: {e}")

        background_tasks.add_task(
            _run_mining,
            task_id,
            request
        )

        algo_label = {
            "genetic": "遗传规划",
            "pysr": "PySR符号回归",
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
        from backend.services.data_service import data_service

        mining_tasks[task_id]["status"] = "running"

        # 同步更新数据库状态（含started_at）
        _sync_task_to_db(task_id, status="running", started_at=datetime.now())

        if request.freq.upper() != "D":
            minute_period = (request.period or request.freq).lower().replace("min", "").replace("t", "")
            data = data_service.get_stock_minute_data(
                stock_codes[0],
                request.start_date,
                request.end_date,
                period=minute_period if minute_period.isdigit() else "5",
            )
        else:
            data = data_service.get_stock_data(
                stock_codes[0],
                request.start_date,
                request.end_date
            )

        if data is None or len(data) == 0:
            raise Exception("未获取到有效数据")

        logger.info(f"Retrieved {len(data)} rows of data for primary stock")

        if "close" in data.columns:
            data = data.copy()
            data["return"] = data["close"].pct_change()

        base_factor_codes = []
        if request.base_factors and len(request.base_factors) > 0:
            try:
                with get_db() as db:
                    repo = FactorRepository(db)

                    for factor_name in request.base_factors:
                        factor = repo.get_by_name(factor_name)
                        if factor:
                            base_factor_codes.append(factor.code)
                            logger.info(f"Found factor: {factor_name} -> {factor.code}")
                        else:
                            # Fuzzy match: try case-insensitive prefix match
                            all_factors = repo.get_all()
                            matched = None
                            name_lower = factor_name.lower()
                            prefix_lower = name_lower.replace(" ", "_")
                            prefix_matches = []
                            for f in all_factors:
                                if f.name.lower() == name_lower:
                                    matched = f
                                    break
                                if f.name.lower().startswith(prefix_lower):
                                    prefix_matches.append(f)
                            if not matched and prefix_matches:
                                # Choose the shortest name to get the most precise match (e.g. "rsi" prefers "rsi" over "rsi_volume")
                                prefix_matches.sort(key=lambda f: len(f.name))
                                matched = prefix_matches[0]
                            if matched:
                                base_factor_codes.append(matched.code)
                                logger.info(f"Found factor (fuzzy): {factor_name} -> {matched.name} -> {matched.code}")
                            else:
                                logger.warning(f"Factor not found in database: {factor_name}")
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
                logger.warning(f"Unknown algorithm '{algorithm}', falling back to genetic")
                result = await _run_genetic_only(
                    task_id, request, data, base_factor_codes,
                    factor_service, stock_codes, logger
                )

            if not result.get("success"):
                raise Exception(result.get("message", "挖掘失败"))

            _finalize_task(task_id, result, request, data, base_factor_codes,
                           factor_service.calculator, logger)

        except ImportError as e:
            logger.warning(f"Mining library not available, using simulation mode: {e}")
            await _run_simulated_mining(task_id, request, data, base_factor_codes, factor_service, logger)

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(e)
        # 同步失败状态到数据库
        _sync_task_to_db(task_id, status="failed", error=str(e))
    finally:
        # 清理已完成/失败/取消的过期任务，防止内存泄漏
        _cleanup_old_tasks()


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

    # 保存服务引用（用于取消，必须在mine_factors之前）
    mining_services[task_id] = mining_service

    result = await asyncio.to_thread(mining_service.mine_factors)
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

    # 保存服务引用（用于取消）
    mining_services[task_id] = mining_service

    result = await asyncio.to_thread(mining_service.mine_factors)
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

    # 保存服务引用（用于取消）
    mining_services[task_id] = mining_service

    result = await asyncio.to_thread(mining_service.mine_factors)

    # NOTE: Do NOT overwrite fitness_history from result here.
    # The progress callback already records normalized IC values per generation.
    # result["fitness_history"] may contain raw multi-objective tuples from logbook,
    # which would overwrite the correct callback data. _finalize_task handles this.

    return result


def _sync_task_to_db(task_id: str, **kwargs):
    """同步任务状态到数据库（静默失败，不影响挖掘流程）"""
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)
            repo.update_status(task_id, **kwargs)
    except Exception as e:
        logger.debug(f"同步任务状态到数据库失败（不影响运行）: {e}")


def _collect_process_info(result: dict, request: GeneticMiningRequest) -> dict:
    """收集算法特定的挖掘过程信息，用于前端展示"""
    algorithm = request.algorithm
    info = {
        "algorithm": algorithm,
        "algorithm_label": {
            "genetic": "遗传规划 (DEAP)",
            "pysr": "PySR符号回归",
            "tree_prescreen": "树模型预筛选",
            "gflownet": "GFlowNet增强GP",
            "deep_implicit": "深度隐式因子 (Transformer)",
        }.get(algorithm, algorithm),
    }

    if algorithm == "genetic":
        info.update({
            "population_size": request.population_size,
            "n_generations": request.n_generations,
            "elite_size": request.elite_size,
            "cx_prob": request.cx_prob,
            "mut_prob": request.mut_prob,
            "fitness_objective": request.fitness_objective,
            "use_nsga2": request.use_nsga2,
            "use_extended_primitives": request.use_extended_primitives,
            "cv_folds": request.cv_folds,
            "parsimony_coeff": request.parsimony_coeff,
            "diversity_penalty_coeff": request.diversity_penalty_coeff,
            "actual_generations": result.get("generations", request.n_generations),
        })
    elif algorithm == "pysr":
        info.update({
            "niterations": request.pysr_niterations,
            "populations": request.pysr_populations,
            "maxsize": request.pysr_maxsize,
            "maxdepth": request.pysr_maxdepth,
            "parsimony": request.pysr_parsimony,
            "procs": request.pysr_procs,
            "population_size": request.pysr_population_size,
            "equations_found": len(result.get("best_factors", [])),
        })
    elif algorithm == "tree_prescreen":
        info.update({
            "tree_model_type": request.tree_model_type,
            "top_k": request.top_k,
            "importance_threshold": request.importance_threshold,
            "tree_n_estimators": request.tree_n_estimators,
            "tree_max_depth": request.tree_max_depth,
            "downstream_algorithm": request.downstream_algorithm,
            "feature_importance": result.get("feature_importance"),
            "selected_features": result.get("selected_features"),
            "n_selected": len(result.get("selected_features", [])),
        })
    elif algorithm == "gflownet":
        info.update({
            "n_trajectories": request.gflownet_n_trajectories,
            "n_iterations": request.gflownet_n_iterations,
            "hidden_dim": request.gflownet_hidden_dim,
            "learning_rate": request.gflownet_learning_rate,
            "max_expression_depth": request.gflownet_max_expression_depth,
            "temperature": request.gflownet_temperature,
            "reward_scale": request.gflownet_reward_scale,
            "actual_iterations": result.get("generations", request.gflownet_n_iterations),
        })
        if result.get("policy_loss_history"):
            info["policy_loss_history"] = result["policy_loss_history"]
    elif algorithm == "deep_implicit":
        info.update({
            "d_model": request.deep_d_model,
            "n_heads": request.deep_n_heads,
            "n_layers": request.deep_n_layers,
            "d_ff": request.deep_d_ff,
            "n_latent_factors": request.deep_n_latent_factors,
            "dropout": request.deep_dropout,
            "seq_length": request.deep_seq_length,
            "learning_rate": request.deep_learning_rate,
            "n_epochs": request.deep_n_epochs,
            "batch_size": request.deep_batch_size,
            "early_stopping_patience": request.deep_early_stopping_patience,
            "actual_epochs": result.get("training_history", {}).get("train_loss", []).__len__() if result.get("training_history") else request.deep_n_epochs,
        })
        if result.get("training_history"):
            info["training_history"] = result["training_history"]
        if result.get("model_info"):
            info["model_info"] = result["model_info"]

    # 通用信息
    info["cancelled"] = result.get("cancelled", False)
    info["factors_found"] = len(result.get("best_factors", []))

    return info


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

    # 限制fitness_history长度，防止无限增长
    fh = mining_tasks[task_id]["fitness_history"]
    if len(fh["best"]) > MAX_FITNESS_HISTORY_ENTRIES:
        fh["best"] = fh["best"][-MAX_FITNESS_HISTORY_ENTRIES:]
        fh["average"] = fh["average"][-MAX_FITNESS_HISTORY_ENTRIES:]

    # 同步进度到数据库（每5代同步一次，减少IO）
    if gen % 5 == 0 or gen == total_gen:
        _sync_task_to_db(task_id, progress=progress, current_generation=gen,
                         total_generations=total_gen, best_fitness=_safe_float(best_fitness),
                         avg_fitness=_safe_float(avg_fitness),
                         fitness_history=mining_tasks[task_id]["fitness_history"])

    logger.info(
        f"Progress: {progress}%, {algorithm} Gen {gen}/{total_gen}, "
        f"Best: {_safe_float(best_fitness):.4f}, Avg: {_safe_float(avg_fitness):.4f}"
    )


def _unified_validate_factor(expression: str, factor_calculator, data, return_values) -> dict:
    """对所有算法的因子执行统一的验证评分。

    无论因子来自哪种算法（genetic/pysr/tree_prescreen/gflownet/deep_implicit），
    都使用同一套 factor_validation_service.validate_factor() 进行评估，
    确保跨算法的 score / ic / ir / overall_passed 具有可比性。

    Returns:
        dict with keys: ic, ir, fitness (score/100), validation_score,
                        overall_passed, turnover, stability, validation (full)
    """
    try:
        fv = factor_calculator.calculate(data, expression)
    except Exception as e:
        logger.debug(f"Unified validate: 无法计算表达式 '{expression[:60]}': {e}")
        return _empty_unified_result()

    if fv is None or len(fv.dropna()) < 10:
        return _empty_unified_result()

    fv = fv.replace([np.inf, -np.inf], np.nan).dropna()
    if len(fv) < 10 or fv.isna().all():
        return _empty_unified_result()

    if return_values is not None and len(return_values) > 0:
        try:
            validation = factor_validation_service.validate_factor(
                factor_values=fv,
                return_values=return_values,
                existing_factors=None,
            )
            ic_val = validation.get("ic_validation", {})
            ir_val = validation.get("ir_validation", {})
            ic = abs(float(ic_val.get("ic", 0.0)))
            ir_capped = float(ir_val.get("ir", 0.0))
            score = float(validation.get("score", 0.0))
            overall_passed = validation.get("overall_passed", False)
            turnover_val = validation.get("turnover_validation", {})
            stability_val = validation.get("stability_validation", {})

            return {
                "ic": ic,
                "ir": ir_capped,
                "fitness": score / 100.0,
                "validation_score": score,
                "overall_passed": overall_passed,
                "turnover": turnover_val.get("turnover"),
                "stability": stability_val.get("stability_score"),
                "validation": validation,
            }
        except Exception as e:
            logger.debug(f"Unified validate: 验证失败 '{expression[:60]}': {e}")
            return _empty_unified_result()

    return _empty_unified_result()


def _empty_unified_result() -> dict:
    """返回空的统一验证结果（当无法验证时使用）。"""
    return {
        "ic": 0.0,
        "ir": 0.0,
        "fitness": 0.0,
        "validation_score": 0.0,
        "overall_passed": False,
        "turnover": None,
        "stability": None,
        "validation": {},
    }


def _finalize_task(task_id: str, result: dict, request: GeneticMiningRequest,
                   data, base_factor_codes, factor_calculator, logger):
    """Convert mining result to frontend format and store in task.

    挖掘完成后，将因子暂存到 generated_factors 表，标记验证状态。
    只有 is_valid=True 的因子才允许保存到因子库。

    **统一验证**: 所有算法的因子都会通过同一个 factor_validation_service
    进行重验证，确保跨算法的 score/ic/ir/overall_passed 可比。
    各算法原始的 fitness 保留为 raw_fitness 供参考。
    """
    from backend.models.generated_factor import GeneratedFactorModel
    from backend.repositories.generated_factor_repository import GeneratedFactorRepository

    best_factors = result.get("best_factors", [])
    return_values = None
    if request.return_column and request.return_column in data.columns:
        return_values = data[request.return_column]
    elif "return" in data.columns:
        return_values = data["return"]

    discovered_factors = []

    with get_db() as db:
        repo = GeneratedFactorRepository(db)

        try:
            for i, factor_info in enumerate(best_factors):
                expression = factor_info["expression"]
                source = factor_info.get("source", result.get("source", "unknown"))
                complexity = factor_info.get("complexity", 0.0)

                # ---- 统一验证：用同一套标准重新打分 ----
                unified = _unified_validate_factor(
                    expression=expression,
                    factor_calculator=factor_calculator,
                    data=data,
                    return_values=return_values,
                )

                # 保留各算法原始 fitness 作为参考
                raw_fitness = factor_info.get("fitness", 0.0)

                # 统一后的指标（全部来自同一个 validate_factor）
                ic = unified["ic"]
                ir = unified["ir"]
                fitness = unified["fitness"]          # = validation_score / 100
                validation_score = unified["validation_score"]
                overall_passed = unified["overall_passed"]
                turnover = unified["turnover"]
                stability = unified["stability"]
                _full_validation = unified["validation"]

                # 如果统一验证完全无结果（如 deep_implicit 的隐式因子），回退到算法自带数据
                if validation_score == 0.0 and ic == 0.0 and ir == 0.0:
                    algo_validation = factor_info.get("validation", {})
                    ic = _safe_float(algo_validation.get("ic_validation", {}).get("ic", 0.0))
                    ir = _safe_float(algo_validation.get("ir_validation", {}).get("ir", 0.0))
                    fitness = _safe_float(raw_fitness)
                    validation_score = _safe_float(algo_validation.get("score", 0.0))
                    overall_passed = algo_validation.get("overall_passed", False)
                    _full_validation = algo_validation

                # 暂存到 generated_factors 表
                existing = repo.get_by_expression(expression)
                if existing:
                    existing.ic_value = _safe_float(ic)
                    existing.ir_value = _safe_float(ir)
                    existing.turnover_value = _safe_float(turnover) if turnover else None
                    existing.stability_score = _safe_float(stability) if stability else None
                    existing.validation_score = _safe_float(validation_score)
                    existing.is_valid = overall_passed
                    existing.generation_method = source
                    existing.complexity = str(complexity)
                    db.commit()
                    db.refresh(existing)
                    generated_id = existing.id
                else:
                    gen_factor = GeneratedFactorModel(
                        expression=expression,
                        generation_method=source,
                        ic_value=_safe_float(ic),
                        ir_value=_safe_float(ir),
                        turnover_value=_safe_float(turnover) if turnover else None,
                        stability_score=_safe_float(stability) if stability else None,
                        validation_score=_safe_float(validation_score),
                        is_valid=overall_passed,
                        is_saved=False,
                        complexity=str(complexity),
                    )
                    created = repo.create(gen_factor)
                    generated_id = created.id

                discovered_factors.append({
                    "name": f"Mined_Factor_{i+1}",
                    "expression": expression,
                    "ic": _safe_float(ic),
                    "ir": _safe_float(ir),
                    "fitness": _safe_float(fitness),           # 统一验证后的 score/100
                    "raw_fitness": _safe_float(raw_fitness),   # 算法原始 fitness（参考）
                    "complexity": _safe_float(complexity),
                    "source": source,
                    "overall_passed": overall_passed,
                    "validation_score": _safe_float(validation_score),  # 统一验证分数
                    "generated_factor_id": generated_id,
                })
        except Exception as e:
            logger.warning(f"保存挖掘结果到 generated_factors 表失败: {e}")
            # 降级：即使暂存失败，仍然返回结果给前端（兼容旧逻辑）
            for i, factor_info in enumerate(best_factors):
                expression = factor_info["expression"]
                source = factor_info.get("source", result.get("source", "unknown"))
                complexity = factor_info.get("complexity", 0.0)
                raw_fitness = factor_info.get("fitness", 0.0)

                # 降级时也尝试统一验证
                unified = _unified_validate_factor(
                    expression=expression,
                    factor_calculator=factor_calculator,
                    data=data,
                    return_values=return_values,
                )
                ic = unified["ic"]
                ir = unified["ir"]
                fitness = unified["fitness"]
                validation_score = unified["validation_score"]
                overall_passed = unified["overall_passed"]

                if validation_score == 0.0 and ic == 0.0:
                    algo_validation = factor_info.get("validation", {})
                    ic = _safe_float(algo_validation.get("ic_validation", {}).get("ic", 0.0))
                    ir = _safe_float(algo_validation.get("ir_validation", {}).get("ir", 0.0))
                    fitness = _safe_float(raw_fitness)
                    validation_score = _safe_float(algo_validation.get("score", 0.0))
                    overall_passed = algo_validation.get("overall_passed", False)

                discovered_factors.append({
                    "name": f"Mined_Factor_{i+1}",
                    "expression": expression,
                    "ic": _safe_float(ic),
                    "ir": _safe_float(ir),
                    "fitness": _safe_float(fitness),
                    "raw_fitness": _safe_float(raw_fitness),
                    "complexity": _safe_float(complexity),
                    "source": source,
                    "overall_passed": overall_passed,
                    "validation_score": _safe_float(validation_score),
                    "generated_factor_id": None,
                })

    # Prefer fitness_history from progress callback (already normalized IC values),
    # fallback to logbook extraction (may contain raw multi-objective tuples).
    fitness_history = mining_tasks[task_id].get("fitness_history")
    if fitness_history and len(fitness_history.get("best", [])) > 0:
        pass  # use progress callback data as-is
    else:
        logbook = result.get("logbook")
        if logbook is not None:
            fitness_history = {
                "best": [_extract_first_fitness(gen["max"]) for gen in logbook],
                "average": [_extract_first_fitness(gen["avg"]) for gen in logbook]
            }
        else:
            fitness_history = {"best": [], "average": []}

    # 限制fitness_history长度，防止无限增长
    if len(fitness_history.get("best", [])) > MAX_FITNESS_HISTORY_ENTRIES:
        fitness_history["best"] = fitness_history["best"][-MAX_FITNESS_HISTORY_ENTRIES:]
        fitness_history["average"] = fitness_history["average"][-MAX_FITNESS_HISTORY_ENTRIES:]

    result_data = {
        "factors": discovered_factors,
        "best_fitness": _safe_float(discovered_factors[0]["fitness"]) if discovered_factors else 0.0,
        "avg_fitness": _safe_float(sum(f["fitness"] for f in discovered_factors) / len(discovered_factors)) if discovered_factors else 0.0,
        "generations": request.n_generations,
        "fitness_history": fitness_history,
        "algorithm": request.algorithm,
    }

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

    # 收集算法特定的过程信息
    process_info = _collect_process_info(result, request)

    mining_tasks[task_id]["status"] = "completed"
    mining_tasks[task_id]["progress"] = 100
    mining_tasks[task_id]["result"] = result_data
    mining_tasks[task_id]["fitness_history"] = fitness_history

    # 持久化完成结果到数据库
    _sync_task_to_db(task_id, status="completed", progress=100, result=result_data,
                     fitness_history=fitness_history, best_fitness=result_data.get("best_fitness"),
                     avg_fitness=result_data.get("avg_fitness"), process_info=process_info)

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

        # 限制fitness_history长度，防止无限增长
        if len(fitness_history["best"]) > MAX_FITNESS_HISTORY_ENTRIES:
            fitness_history["best"] = fitness_history["best"][-MAX_FITNESS_HISTORY_ENTRIES:]
            fitness_history["average"] = fitness_history["average"][-MAX_FITNESS_HISTORY_ENTRIES:]

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
        "best_factors": discovered_factors,
        "best_fitness": discovered_factors[0]["ic"] if discovered_factors else 0,
        "avg_fitness": sum(f["fitness"] for f in discovered_factors) / len(discovered_factors) if discovered_factors else 0,
        "generations": n_generations,
        "fitness_history": fitness_history,
        "algorithm": request.algorithm,
        "source": "simulated",
    }

    # 调用 _finalize_task 完成统一验证、暂存和状态更新
    _finalize_task(task_id, result, request, data, base_factor_codes,
                   factor_service.calculator, logger)

    logger.info(f"Task {task_id} completed (simulated mode)")


@router.get("/status/{task_id}")
async def get_mining_status(task_id: str):
    """获取挖掘状态"""
    import logging
    logger = logging.getLogger(__name__)

    # 优先从内存获取
    if task_id in mining_tasks:
        task = mining_tasks[task_id]
        logger.info(f"Status check for task {task_id}: {task['status']}")

        # 尝试从数据库获取started_at
        started_at = None
        try:
            with get_db() as _db:
                _repo = MiningTaskRepository(_db)
                _rec = _repo.get_by_task_id(task_id)
                if _rec and _rec.started_at:
                    started_at = _rec.started_at.isoformat()
        except Exception as e:
            logger.debug(f"获取任务启动时间失败: {e}")

        response_data = {
            "status": task["status"],
            "progress": task.get("progress", 0),
            "error": task.get("error"),
            "algorithm": task.get("algorithm", "genetic"),
            "started_at": started_at,
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

    # 内存中没有，从数据库获取
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)
            task_record = repo.get_by_task_id(task_id)

            if not task_record:
                raise HTTPException(status_code=404, detail="任务不存在")

            response_data = {
                "task_id": task_id,
                "status": task_record.status,
                "progress": task_record.progress or 0,
                "error": task_record.error,
                "algorithm": task_record.algorithm,
                "started_at": task_record.started_at.isoformat() if task_record.started_at else None,
            }

            if task_record.status == "completed" and task_record.result:
                result = task_record.result
                response_data["current_generation"] = result.get("generations", task_record.current_generation or 0)
                response_data["total_generations"] = result.get("generations", task_record.total_generations or 0)
                response_data["best_fitness"] = result.get("best_fitness", task_record.best_fitness or 0)
                response_data["avg_fitness"] = result.get("avg_fitness", task_record.avg_fitness or 0)
                response_data["fitness_history"] = result.get("fitness_history", task_record.fitness_history or {"best": [], "average": []})
                response_data["algorithm"] = result.get("algorithm", task_record.algorithm)
            else:
                response_data["current_generation"] = task_record.current_generation or 0
                response_data["total_generations"] = task_record.total_generations or 0
                response_data["best_fitness"] = task_record.best_fitness or 0
                response_data["avg_fitness"] = task_record.avg_fitness or 0
                response_data["fitness_history"] = task_record.fitness_history or {"best": [], "average": []}

            return {
                "success": True,
                "data": response_data
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"从数据库获取任务状态失败: {e}")
        raise HTTPException(status_code=404, detail="任务不存在")


@router.get("/results/{task_id}")
async def get_mining_results(task_id: str):
    """获取挖掘结果"""
    # 优先从内存获取
    if task_id in mining_tasks:
        task = mining_tasks[task_id]
        if task["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}")
        result_data = task["result"]
        # 补充process_info
        try:
            with get_db() as _db:
                _repo = MiningTaskRepository(_db)
                _rec = _repo.get_by_task_id(task_id)
                if _rec and _rec.process_info:
                    result_data["process_info"] = _rec.process_info
        except Exception as e:
            logger.debug(f"获取任务过程信息失败: {e}")
        return {
            "success": True,
            "data": result_data
        }

    # 内存中没有，从数据库获取
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)
            task_record = repo.get_by_task_id(task_id)

            if not task_record:
                raise HTTPException(status_code=404, detail="任务不存在")

            if task_record.status != "completed":
                raise HTTPException(status_code=400, detail=f"任务尚未完成，当前状态: {task_record.status}")

            if not task_record.result:
                raise HTTPException(status_code=404, detail="任务结果不存在")

            result_data = task_record.result
            if task_record.process_info:
                result_data["process_info"] = task_record.process_info

            return {
                "success": True,
                "data": result_data
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"从数据库获取任务结果失败: {e}")
        raise HTTPException(status_code=404, detail="任务不存在")


@router.post("/cancel/{task_id}")
async def cancel_mining_task(task_id: str):
    """取消正在运行的挖掘任务"""
    # 调用服务的request_cancel方法
    service = mining_services.get(task_id)
    if service and hasattr(service, 'request_cancel'):
        service.request_cancel()

    # 更新内存和数据库状态
    if task_id in mining_tasks:
        mining_tasks[task_id]["status"] = "cancelled"

    _sync_task_to_db(task_id, status="cancelled", completed_at=datetime.now())

    return {
        "success": True,
        "message": "取消请求已发送"
    }


@router.get("/active")
async def get_active_tasks():
    """获取当前活跃的挖掘任务（pending/running），用于页面刷新后恢复状态"""
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)

            # 先检查内存中的活跃任务
            active_in_memory = []
            # 同时获取数据库记录以补充started_at等信息
            db_tasks_map = {}
            for task_record in (repo.get_active_tasks() or []):
                db_tasks_map[task_record.task_id] = task_record

            for tid, task in mining_tasks.items():
                if task.get("status") in ("pending", "running"):
                    task_info = {
                        "task_id": tid,
                        "status": task["status"],
                        "progress": task.get("progress", 0),
                        "algorithm": task.get("algorithm", "genetic"),
                        "current_generation": task.get("current_generation", 0),
                        "total_generations": task.get("total_generations", 0),
                        "best_fitness": task.get("best_fitness", 0),
                        "avg_fitness": task.get("avg_fitness", 0),
                        "fitness_history": task.get("fitness_history", {"best": [], "average": []}),
                    }
                    # 补充started_at
                    db_rec = db_tasks_map.get(tid)
                    if db_rec and db_rec.started_at:
                        task_info["started_at"] = db_rec.started_at.isoformat()
                    active_in_memory.append(task_info)

            # 再检查数据库中的活跃任务（服务重启后内存中没有）
            db_active = db_tasks_map.values()

            # 合并去重
            memory_task_ids = {t["task_id"] for t in active_in_memory}
            for task_record in db_active:
                if task_record.task_id not in memory_task_ids:
                    active_in_memory.append(repo.to_dict(task_record))

        return {
            "success": True,
            "data": active_in_memory
        }
    except Exception as e:
        logger.error(f"获取活跃任务失败: {e}")
        return {
            "success": True,
            "data": []
        }


@router.get("/history")
async def get_mining_history(limit: int = Query(default=20, ge=1, le=100),
                              offset: int = Query(default=0, ge=0)):
    """获取挖掘历史记录（分页）"""
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)

            total = repo.get_history_count()
            tasks = repo.get_history(limit=limit, offset=offset)

            items = [repo.to_summary_dict(t) for t in tasks]

            return {
                "success": True,
                "data": {
                    "items": items,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            }
    except Exception as e:
        logger.error(f"获取挖掘历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{task_id}")
async def get_mining_history_detail(task_id: str):
    """获取挖掘历史详情"""
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)
            task_record = repo.get_by_task_id(task_id)

            if not task_record:
                raise HTTPException(status_code=404, detail="任务不存在")

            return {
                "success": True,
                "data": repo.to_dict(task_record)
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取挖掘历史详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{task_id}")
async def delete_mining_history(task_id: str):
    """删除挖掘历史记录"""
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)
            task_record = repo.get_by_task_id(task_id)

            if not task_record:
                raise HTTPException(status_code=404, detail="任务不存在")

            repo.delete(task_record.id)

            return {
                "success": True,
                "message": "删除成功"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除挖掘历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
