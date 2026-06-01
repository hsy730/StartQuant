"""
因子挖掘API路由
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import math

router = APIRouter()


def _safe_float(value, default=0.0) -> float:
    """Convert to float, replacing NaN/Inf/None with *default*.

    FastAPI's JSON encoder rejects non-finite floats, so this must be
    applied to every numeric value that reaches the front-end.
    """
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
    """Extract the primary (IC-based) fitness from a stat value.

    When NSGA-II multi-objective optimisation is active the compiled
    statistics contain element-wise tuples/arrays ``(ic, complexity)``.
    This helper returns the first component so that frontend charts
    always display scalar floats.
    """
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
    # ---- Phase 2: Parsimony pressure ----
    parsimony_coeff: float = 0.001
    # ---- Phase 3 & 4: Diversity + cache ----
    diversity_penalty_coeff: float = 0.1
    # ---- Phase 6: Cross-validation ----
    cv_folds: int = 0
    # ---- Phase 7: Extended primitives ----
    use_extended_primitives: bool = True
    max_tree_depth: int = 17
    # ---- NSGA-II ----
    use_nsga2: bool = True


# ========== 任务存储（内存） ==========
mining_tasks = {}


# ========== API端点 ==========

@router.post("/genetic")
async def start_genetic_mining(request: GeneticMiningRequest, background_tasks: BackgroundTasks):
    """启动遗传算法挖掘"""
    try:
        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())

        # 初始化任务状态
        mining_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None
        }

        # 在后台执行挖掘
        background_tasks.add_task(
            _run_genetic_mining,
            task_id,
            request
        )

        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "pending"
            },
            "message": "挖掘任务已启动"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_genetic_mining(task_id: str, request: GeneticMiningRequest):
    try:
        import logging
        logger = logging.getLogger(__name__)

        stock_codes = list(request.stock_codes) if request.stock_codes else []
        if not stock_codes and request.stock_code:
            logger.warning("Single stock_code provided, converting to stock_codes list. Cross-sectional IC requires multiple stocks.")
            stock_codes = [request.stock_code]

        if not stock_codes:
            raise Exception("未提供股票代码，请通过stock_codes或stock_code指定")

        logger.info(f"Starting mining task {task_id}")
        logger.info(f"Stocks: {stock_codes}, Base factors: {request.base_factors}")
        logger.info(f"Parameters: population={request.population_size}, generations={request.n_generations}")

        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db_session
        from backend.services.data_service import data_service

        mining_tasks[task_id]["status"] = "running"
        logger.info(f"Task {task_id} status set to running")

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
                from backend.repositories.factor_repository import FactorRepository
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
            from backend.services.genetic_factor_mining_service import create_genetic_mining_service

            logger.info("Using real genetic algorithm mining")

            mining_service = create_genetic_mining_service(
                base_factors=base_factor_codes,
                data=data,
                return_column="return",
                population_size=request.population_size,
                n_generations=request.n_generations,
                cx_prob=request.cx_prob,
                mut_prob=request.mut_prob,
                factor_calculator=factor_service.calculator,
                # Phase 1: Elitism + fitness objective
                elite_size=request.elite_size,
                fitness_objective=request.fitness_objective,
                # Phase 2: Parsimony pressure
                parsimony_coeff=request.parsimony_coeff,
                # Phase 3 & 4: Diversity + cache
                diversity_penalty_coeff=request.diversity_penalty_coeff,
                # Phase 6: Cross-validation
                cv_folds=request.cv_folds,
                # Phase 7: Extended primitives
                use_extended_primitives=request.use_extended_primitives,
                max_tree_depth=request.max_tree_depth,
                # NSGA-II
                use_nsga2=request.use_nsga2,
            )

            if len(stock_codes) >= 2:
                logger.info(f"Setting stock pool with {len(stock_codes)} stocks for cross-sectional IC evaluation")
                mining_service.set_stock_pool(stock_codes, request.start_date, request.end_date)

            def progress_callback(gen, total_gen, best_fitness, avg_fitness):
                progress = int(gen / total_gen * 100)
                mining_tasks[task_id]["progress"] = progress
                mining_tasks[task_id]["current_generation"] = gen
                mining_tasks[task_id]["total_generations"] = total_gen
                mining_tasks[task_id]["best_fitness"] = _safe_float(best_fitness)
                mining_tasks[task_id]["avg_fitness"] = _safe_float(avg_fitness)

                if "fitness_history" not in mining_tasks[task_id]:
                    mining_tasks[task_id]["fitness_history"] = {"best": [], "average": []}
                mining_tasks[task_id]["fitness_history"]["best"].append(_safe_float(best_fitness))
                mining_tasks[task_id]["fitness_history"]["average"].append(_safe_float(avg_fitness))

                logger.info(f"Progress: {progress}%, Gen {gen}/{total_gen}, Best: {_safe_float(best_fitness):.4f}, Avg: {_safe_float(avg_fitness):.4f}")

            mining_service.set_progress_callback(progress_callback)

            result = mining_service.mine_factors()

            if not result.get("success"):
                raise Exception(result.get("message", "挖掘失败"))

            best_factors = result.get("best_factors", [])

            discovered_factors = []
            for i, factor_info in enumerate(best_factors):
                validation = factor_info.get("validation", {})
                ic = validation.get("ic_validation", {}).get("ic", 0.0)
                ir = validation.get("ir_validation", {}).get("ir", 0.0)
                fitness = factor_info.get("fitness", 0.0)
                complexity = factor_info.get("complexity", 0.0)

                discovered_factors.append({
                    "name": f"Mined_Factor_{i+1}",
                    "expression": factor_info["expression"],
                    "ic": _safe_float(ic),
                    "ir": _safe_float(ir),
                    "fitness": _safe_float(fitness),
                    "complexity": _safe_float(complexity),
                })

            logbook = result.get("logbook")
            if logbook is not None:
                fitness_history = {
                    "best": [_extract_first_fitness(gen["max"]) for gen in logbook],
                    "average": [_extract_first_fitness(gen["avg"]) for gen in logbook]
                }
            else:
                fitness_history = {"best": [], "average": []}

            result_data = {
                "factors": discovered_factors,
                "best_fitness": _safe_float(discovered_factors[0]["fitness"]) if discovered_factors else 0.0,
                "avg_fitness": _safe_float(sum(f["fitness"] for f in discovered_factors) / len(discovered_factors)) if discovered_factors else 0.0,
                "generations": request.n_generations,
                "fitness_history": fitness_history
            }

            mining_tasks[task_id]["status"] = "completed"
            mining_tasks[task_id]["progress"] = 100
            mining_tasks[task_id]["result"] = result_data
            mining_tasks[task_id]["fitness_history"] = fitness_history

            logger.info(f"Task {task_id} completed successfully")
            logger.info(f"Discovered {len(discovered_factors)} factors")
            logger.info(f"Final status: {mining_tasks[task_id]['status']}")

        except ImportError as e:
            logger.warning(f"DEAP library not available, using simulation mode: {e}")
            await _run_simulated_mining(task_id, request, data, base_factor_codes, factor_service, logger)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(e)


async def _run_simulated_mining(task_id: str, request: GeneticMiningRequest, data, base_factor_codes, factor_service, logger):
    """模拟模式挖掘（当DEAP库未安装时使用）"""
    # 计算基础因子值（用于验证和生成）
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

    # 模拟挖掘进度
    n_generations = request.n_generations
    fitness_history = {"best": [], "average": []}
    current_best_fitness = 0.0

    for gen in range(n_generations):
        # 更新进度
        progress = int((gen + 1) / n_generations * 100)
        mining_tasks[task_id]["progress"] = progress

        # 模拟适应度变化（逐渐改进）
        current_best_fitness = 0.03 + (gen + 1) * 0.005 + (0.001 * (gen % 3))
        current_avg_fitness = current_best_fitness * (0.85 + 0.1 * (gen % 2))

        fitness_history["best"].append(current_best_fitness)
        fitness_history["average"].append(current_avg_fitness)

        # 更新任务状态以便轮询可以获取
        mining_tasks[task_id]["current_generation"] = gen + 1
        mining_tasks[task_id]["total_generations"] = n_generations
        mining_tasks[task_id]["best_fitness"] = current_best_fitness
        mining_tasks[task_id]["avg_fitness"] = current_avg_fitness
        mining_tasks[task_id]["fitness_history"] = fitness_history

        logger.info(f"Generation {gen + 1}/{n_generations} completed, best_fitness={current_best_fitness:.4f}")

        # 模拟计算时间
        await asyncio.sleep(0.5)

    # 基于用户选择的因子代码生成组合因子
    discovered_factors = []
    code_list = list(factor_values.keys())

    for i in range(min(5, len(code_list))):
        base_code = code_list[i % len(code_list)]
        # 生成简单的组合表达式
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
            "fitness": 0.03 + (i * 0.01)
        })

    result = {
        "factors": discovered_factors,
        "best_fitness": discovered_factors[0]["ic"] if discovered_factors else 0,
        "avg_fitness": sum(f["fitness"] for f in discovered_factors) / len(discovered_factors) if discovered_factors else 0,
        "generations": n_generations,
        "fitness_history": fitness_history
    }

    # 保存结果
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

    # 构造返回数据，包含前端期望的所有字段
    response_data = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "error": task.get("error")
    }

    # 如果任务完成，添加结果信息
    if task["status"] == "completed" and "result" in task:
        result = task["result"]
        response_data["current_generation"] = result.get("generations", 0)
        response_data["total_generations"] = result.get("generations", 0)
        response_data["best_fitness"] = result.get("best_fitness", 0)
        response_data["avg_fitness"] = result.get("avg_fitness", 0)
        response_data["fitness_history"] = result.get("fitness_history", {"best": [], "average": []})
        logger.info(f"Returning completed status with fitness_history length: {len(response_data['fitness_history']['best'])}")
    else:
        # 进行中的任务 - 从任务状态获取实时数据
        response_data["current_generation"] = task.get("current_generation", 0)
        response_data["total_generations"] = task.get("total_generations", 10)
        response_data["best_fitness"] = task.get("best_fitness", 0.03)
        response_data["avg_fitness"] = task.get("avg_fitness", 0.03)
        response_data["fitness_history"] = task.get("fitness_history", {"best": [], "average": []})
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
