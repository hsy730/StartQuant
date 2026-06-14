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
import json
import time
import logging
from datetime import datetime

from backend.utils.serialization import sanitize_dict
from backend.utils.safe_math import safe_float as _safe_float
from backend.utils.safe_math import safe_divide
from backend.core.database import get_db
from backend.models.mining_task import MiningTaskModel
from backend.repositories.mining_task_repository import MiningTaskRepository

logger = logging.getLogger(__name__)

# 模块级导入：_unified_validate_factor 等模块级函数需要直接访问
from backend.services import factor_validation_service  # noqa: E402

router = APIRouter()


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
    # ---- Stock pool ----
    stock_pool_id: Optional[str] = None
    # ---- Return column ----
    return_column: Optional[str] = None


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
            mining_tasks.items(), key=lambda x: x[1].get("created_at", 0)
        )
        excess = len(mining_tasks) - MAX_TASKS
        for task_id, _ in sorted_tasks[:excess]:
            mining_tasks.pop(task_id, None)
            mining_services.pop(task_id, None)


# ========== API端点 ==========


@router.post("/genetic")
async def start_genetic_mining(
    request: GeneticMiningRequest, background_tasks: BackgroundTasks
):
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
            "total_generations": request.n_generations,
        }

        # 持久化到数据库
        try:
            with get_db() as db:
                repo = MiningTaskRepository(db)
                task_record = MiningTaskModel(
                    task_id=task_id,
                    status="pending",
                    algorithm=request.algorithm,
                    stock_codes=json.dumps(
                        request.stock_codes
                        or ([request.stock_code] if request.stock_code else [])
                    ),
                    base_factors=json.dumps(request.base_factors or []),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    freq=request.freq,
                    config=request.model_dump(),
                )
                repo.create(task_record)
        except Exception as e:
            logger.warning(f"持久化挖掘任务失败（不影响运行）: {e}")

        background_tasks.add_task(_run_mining, task_id, request)

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
            "message": f"{algo_label.get(request.algorithm, '挖掘')}任务已启动",
        }
    except Exception as e:
        logger.error(f"因子挖掘失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _resolve_stock_pool(pool_id: str) -> List[str]:
    """从股票池ID解析成分股代码列表（复用data路由的逻辑）"""
    from backend.api.routers.data import STOCK_POOLS, _call_akshare_with_timeout
    from backend.services.data_service import data_service
    import akshare as ak

    if pool_id not in STOCK_POOLS:
        raise Exception(f"股票池 {pool_id} 不存在")

    pool_info = STOCK_POOLS[pool_id]
    symbol = pool_info["symbol"]

    # 尝试从缓存获取
    cache_key = f"stock_pool_{pool_id}"
    cached = data_service.cache_service.get(cache_key)
    if cached is not None:
        logger.info(f"Stock pool {pool_id} resolved from cache, {len(cached)} stocks")
        return [s["code"] for s in cached]

    # 调用akshare获取成分股（3种策略，与data路由保持一致）
    df = None
    code_col = None

    ak_funcs = [
        (ak.index_stock_cons, "index_stock_cons"),
        (ak.index_stock_cons_csindex, "index_stock_cons_csindex"),
        (ak.index_stock_cons_weight_csindex, "index_stock_cons_weight_csindex"),
    ]

    for ak_func, func_name in ak_funcs:
        try:
            df = _call_akshare_with_timeout(ak_func, symbol=symbol)
            if df is not None and len(df) > 0:
                code_col = None
                for col in df.columns:
                    if "代码" in str(col) or "code" in str(col).lower():
                        code_col = col
                        break
                if code_col:
                    logger.info(f"{func_name} resolved {pool_id}, {len(df)} rows")
                    break
                else:
                    df = None  # 没有代码列，继续尝试下一个
        except Exception as e:
            logger.warning(f"{func_name} failed for {symbol}: {e}")

    if df is None or len(df) == 0 or code_col is None:
        raise Exception(f"无法获取股票池 {pool_id} 的成分股，所有数据源均失败")

    stock_codes = []
    limit = pool_info.get("limit")
    for _, row in df.iterrows():
        if limit and len(stock_codes) >= limit:
            break
        code = str(row[code_col]).strip()
        if not code.isdigit() or len(code) != 6:
            continue
        if code.startswith("6"):
            stock_codes.append(f"{code}.SH")
        elif code.startswith(("0", "3")):
            stock_codes.append(f"{code}.SZ")
        else:
            stock_codes.append(code)

    if not stock_codes:
        raise Exception(f"股票池 {pool_id} 解析后无有效股票代码")

    # 缓存结果（与data路由一致，缓存1天）
    stocks_cache = [{"code": c, "name": "", "short_code": c.split(".")[0]} for c in stock_codes]
    data_service.cache_service.set(cache_key, stocks_cache, ttl=24 * 60 * 60)

    logger.info(f"Stock pool {pool_id} resolved, {len(stock_codes)} stocks")
    return stock_codes


async def _run_mining(task_id: str, request: GeneticMiningRequest):
    """Unified mining entry point that dispatches to the correct algorithm."""
    try:
        stock_codes = list(request.stock_codes) if request.stock_codes else []
        if not stock_codes and request.stock_code:
            logger.warning(
                "Single stock_code provided, converting to stock_codes list."
            )
            stock_codes = [request.stock_code]

        # 如果提供了stock_pool_id，从后端获取成分股（避免前端单独调用akshare超时）
        if not stock_codes and request.stock_pool_id:
            logger.info(f"Resolving stock pool: {request.stock_pool_id}")
            stock_codes = await _resolve_stock_pool(request.stock_pool_id)

        if not stock_codes:
            raise Exception("未提供股票代码，请通过stock_codes、stock_code或stock_pool_id指定")

        algorithm = request.algorithm
        logger.info(f"Starting mining task {task_id} with algorithm={algorithm}")
        logger.info(f"Stocks: {stock_codes}, Base factors: {request.base_factors}")

        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.services.data_service import data_service

        mining_tasks[task_id]["status"] = "running"
        # 将 started_at 同时写入内存（避免 /status 端点每2秒查一次数据库）
        started_at_now = datetime.now().isoformat()
        mining_tasks[task_id]["started_at"] = started_at_now

        # 同步更新数据库状态（含started_at）
        _sync_task_to_db(task_id, status="running", started_at=datetime.now())

        if request.freq.upper() != "D":
            minute_period = (
                (request.period or request.freq)
                .lower()
                .replace("min", "")
                .replace("t", "")
            )
            # 在线程池中执行同步IO，避免阻塞事件循环
            data = await asyncio.to_thread(
                data_service.get_stock_minute_data,
                stock_codes[0],
                request.start_date,
                request.end_date,
                minute_period if minute_period.isdigit() else "5",
            )
        else:
            # 在线程池中执行同步IO，避免阻塞事件循环
            data = await asyncio.to_thread(
                data_service.get_stock_data,
                stock_codes[0],
                request.start_date,
                request.end_date,
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
                                # Choose the shortest name to get the most
                                # precise match (e.g. "rsi" prefers "rsi" over "rsi_volume")
                                prefix_matches.sort(key=lambda f: len(f.name))
                                matched = prefix_matches[0]
                            if matched:
                                base_factor_codes.append(matched.code)
                                logger.info(
                                    f"Found factor (fuzzy): {factor_name} -> {matched.name} -> {matched.code}"
                                )
                            else:
                                logger.warning(
                                    f"Factor not found in database: {factor_name}"
                                )
            except Exception as e:
                logger.error(f"Error loading factors from database: {e}")

        if not base_factor_codes:
            logger.warning("No valid base factors found, using default codes")
            base_factor_codes = [
                "RSI(close, 14)",
                "SMA(close, 20)",
                "close / open",
                "volume / 1000000",
                "MACD(close, 12, 26, 9)[0]",
            ]
        else:
            logger.info(f"Using {len(base_factor_codes)} base factor codes")

        try:
            # ---- 通过算法注册表创建服务实例 ----
            # 新增算法只需在 mining_algorithm_registry.py 中注册，
            # 无需修改此函数的任何代码
            from backend.services.mining_algorithm_registry import (
                create_algorithm,
            )

            service = create_algorithm(
                algorithm,
                task_id=task_id,
                request=request,
                data=data,
                base_factor_codes=base_factor_codes,
                factor_service=factor_service,
                stock_codes=stock_codes,
                logger=logger,
            )

            if service is None:
                # 未注册的算法回退到遗传算法
                logger.warning(
                    f"Unknown algorithm '{algorithm}', falling back to genetic"
                )
                service = create_algorithm(
                    "genetic",
                    task_id, request, data, base_factor_codes,
                    factor_service, stock_codes, logger,
                )

            # 设置股票池（注册表工厂已设置 progress_callback 和 _task_id）
            # set_stock_pool 内部有同步网络IO（get_multiple_stocks_data），
            # 必须在线程池中执行，避免阻塞事件循环
            if len(stock_codes) >= 2:
                await asyncio.to_thread(
                    service.set_stock_pool,
                    stock_codes=stock_codes,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )

            # 设置进度回调（统一入口，避免各算法工厂重复实现）
            def _progress_callback(gen, total_gen, best_fitness, avg_fitness, **kwargs):
                _update_progress(
                    task_id, gen, total_gen, best_fitness, avg_fitness,
                    algorithm, logger,
                )

            service.set_progress_callback(_progress_callback)

            # 设置 _task_id 用于 checkpoint 保存时关联挖掘任务。
            # 仅 API 路由层设置此属性，evolve_factor 等场景下为 None（不保存 checkpoint）。
            service._task_id = task_id

            # 注册到 mining_services（用于取消任务）
            mining_services[task_id] = service

            # mine_factors() 是同步阻塞函数，在线程池中执行避免阻塞事件循环
            result = await asyncio.to_thread(service.mine_factors)

            # 兼容 MiningResult 和 dict 两种返回格式
            from backend.services.mining_models import MiningResult
            if isinstance(result, MiningResult):
                result_success = result.success
                result_cancelled = result.cancelled
                result_message = result.error or "挖掘失败"
            else:
                result_success = result.get("success", False)
                result_cancelled = result.get("cancelled", False)
                result_message = result.get("message", "挖掘失败")

            if not result_success:
                raise Exception(result_message)

            # 处理取消：mine_factors 返回 cancelled=True 时，
            # 仍然保存已发现的因子（用户可能需要部分结果），但标记状态为 cancelled。
            # 不能直接跳过 _finalize_task，因为前端需要 result 数据展示已发现的因子。
            # 流程：_finalize_task 先设 status="completed" → 立即覆盖为 "cancelled"
            # _finalize_task 包含验证+存储等CPU/IO密集操作，在线程池中执行避免阻塞事件循环
            if result_cancelled:
                await asyncio.to_thread(
                    _finalize_task,
                    task_id, result, request, data,
                    base_factor_codes, factor_service.calculator, logger,
                )
                # 覆盖 _finalize_task 设置的 completed 状态为 cancelled
                # （_finalize_task 不知道任务被取消，总是设 completed）
                mining_tasks[task_id]["status"] = "cancelled"
                _sync_task_to_db(
                    task_id,
                    status="cancelled",
                    completed_at=datetime.now(),
                    result=mining_tasks[task_id].get("result"),
                )
                logger.info(f"Task {task_id} cancelled by user")
            else:
                await asyncio.to_thread(
                    _finalize_task,
                    task_id, result, request, data,
                    base_factor_codes, factor_service.calculator, logger,
                )

        except ImportError as e:
            logger.warning(f"Mining library not available, using simulation mode: {e}")
            await _run_simulated_mining(
                task_id, request, data, base_factor_codes, factor_service, logger
            )

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        mining_tasks[task_id]["status"] = "failed"
        mining_tasks[task_id]["error"] = str(e)
        # 同步失败状态到数据库
        _sync_task_to_db(task_id, status="failed", error=str(e))
    finally:
        # 立即释放服务实例内存（无论成功/失败/取消）
        # 必须在 finally 块中调用，因为 mine_factors() 在子线程中运行，
        # 取消只是设置 flag，不能在 cancel_mining_task() 中直接释放内存
        # （否则 mine_factors() 可能访问已释放的 self.data 导致崩溃）。
        service = mining_services.pop(task_id, None)
        if service and hasattr(service, "release_memory"):
            try:
                service.release_memory()
            except Exception as e:
                logger.debug(f"释放服务内存失败（不影响结果）: {e}")
        # 清理已完成/失败/取消的过期任务，防止内存泄漏
        _cleanup_old_tasks()


async def _run_genetic_only(
    task_id, request, data, base_factor_codes, factor_service, stock_codes, logger
) -> dict:
    """Run DEAP genetic programming only."""
    from backend.services.genetic_factor_mining_service import (
        create_genetic_mining_service,
    )

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
        _update_progress(
            task_id, gen, total_gen, best_fitness, avg_fitness, "genetic", logger
        )

    mining_service.set_progress_callback(progress_callback)

    # 保存服务引用（用于取消，必须在mine_factors之前）
    mining_services[task_id] = mining_service
    # 设置 _task_id 用于 checkpoint 保存时关联挖掘任务。
    # 仅 API 路由层设置此属性，evolve_factor 等场景下为 None（不保存 checkpoint）。
    mining_service._task_id = task_id

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
        _update_progress(
            task_id, iteration, total_iter, best_fitness, avg_fitness, "pysr", logger
        )

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
        logger.info(
            f"Setting stock pool with {len(stock_codes)} stocks for {algorithm}"
        )
        mining_service.set_stock_pool(stock_codes, request.start_date, request.end_date)

    def progress_callback(
        gen, total_gen, best_fitness, avg_fitness, algorithm=algorithm, **kwargs
    ):
        _update_progress(
            task_id, gen, total_gen, best_fitness, avg_fitness, algorithm, logger
        )

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
        info.update(
            {
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
            }
        )
    elif algorithm == "pysr":
        info.update(
            {
                "niterations": request.pysr_niterations,
                "populations": request.pysr_populations,
                "maxsize": request.pysr_maxsize,
                "maxdepth": request.pysr_maxdepth,
                "parsimony": request.pysr_parsimony,
                "procs": request.pysr_procs,
                "population_size": request.pysr_population_size,
                "equations_found": len(result.get("best_factors", [])),
            }
        )
    elif algorithm == "tree_prescreen":
        info.update(
            {
                "tree_model_type": request.tree_model_type,
                "top_k": request.top_k,
                "importance_threshold": request.importance_threshold,
                "tree_n_estimators": request.tree_n_estimators,
                "tree_max_depth": request.tree_max_depth,
                "downstream_algorithm": request.downstream_algorithm,
                "feature_importance": result.get("feature_importance"),
                "selected_features": result.get("selected_features"),
                "n_selected": len(result.get("selected_features", [])),
            }
        )
    elif algorithm == "gflownet":
        info.update(
            {
                "n_trajectories": request.gflownet_n_trajectories,
                "n_iterations": request.gflownet_n_iterations,
                "hidden_dim": request.gflownet_hidden_dim,
                "learning_rate": request.gflownet_learning_rate,
                "max_expression_depth": request.gflownet_max_expression_depth,
                "temperature": request.gflownet_temperature,
                "reward_scale": request.gflownet_reward_scale,
                "actual_iterations": result.get(
                    "generations", request.gflownet_n_iterations
                ),
            }
        )
        if result.get("policy_loss_history"):
            info["policy_loss_history"] = result["policy_loss_history"]
    elif algorithm == "deep_implicit":
        info.update(
            {
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
                "actual_epochs": (
                    result.get("training_history", {}).get("train_loss", []).__len__()
                    if result.get("training_history")
                    else request.deep_n_epochs
                ),
            }
        )
        if result.get("training_history"):
            info["training_history"] = result["training_history"]
        if result.get("model_info"):
            info["model_info"] = result["model_info"]

    # 通用信息
    info["cancelled"] = result.get("cancelled", False)
    info["factors_found"] = len(result.get("best_factors", []))

    return info


def _update_progress(
    task_id, gen, total_gen, best_fitness, avg_fitness, algorithm, logger
):
    """Update mining task progress in the shared task store.

    进度计算说明：
    - gen=0, total_gen=N+1: 初始种群评估阶段（占总进度的 1/(N+1)）
    - gen=1..N, total_gen=N+1: 进化代阶段

    初始评估阶段 progress 从 0% 逐步增长到 ~1/(N+1)*100%，
    进化阶段从 ~1/(N+1)*100% 增长到 100%。
    这样前端在初始评估期间也能看到进度变化，不会长时间停留在 0%。
    """
    progress = int(gen / max(total_gen, 1) * 100)
    mining_tasks[task_id]["progress"] = progress
    mining_tasks[task_id]["current_generation"] = gen
    mining_tasks[task_id]["total_generations"] = total_gen
    mining_tasks[task_id]["best_fitness"] = _safe_float(best_fitness)
    mining_tasks[task_id]["avg_fitness"] = _safe_float(avg_fitness)
    mining_tasks[task_id]["current_algorithm"] = algorithm

    if "fitness_history" not in mining_tasks[task_id]:
        mining_tasks[task_id]["fitness_history"] = {"best": [], "average": []}

    # 初始评估阶段（gen=0）不添加到 fitness_history，避免图表出现 "第0代" 数据点
    # 只有进化代（gen>=1）才记录到 fitness_history
    if gen >= 1:
        mining_tasks[task_id]["fitness_history"]["best"].append(_safe_float(best_fitness))
        mining_tasks[task_id]["fitness_history"]["average"].append(_safe_float(avg_fitness))

    # 限制fitness_history长度，防止无限增长
    fh = mining_tasks[task_id]["fitness_history"]
    if len(fh["best"]) > MAX_FITNESS_HISTORY_ENTRIES:
        fh["best"] = fh["best"][-MAX_FITNESS_HISTORY_ENTRIES:]
        fh["average"] = fh["average"][-MAX_FITNESS_HISTORY_ENTRIES:]

    # 同步进度到数据库（每5代同步一次，减少IO）
    if gen % 5 == 0 or gen == total_gen:
        _sync_task_to_db(
            task_id,
            progress=progress,
            current_generation=gen,
            total_generations=total_gen,
            best_fitness=_safe_float(best_fitness),
            avg_fitness=_safe_float(avg_fitness),
            fitness_history=mining_tasks[task_id]["fitness_history"],
        )

    bf = _safe_float(best_fitness, default=0.0)
    af = _safe_float(avg_fitness, default=0.0)
    phase_label = "初始评估" if gen == 0 else f"Gen {gen}"
    logger.info(
        f"Progress: {progress}%, {algorithm} {phase_label}/{total_gen}, "
        f"Best: {bf:.4f}, Avg: {af:.4f}"
    )


def _unified_validate_factor(
    expression: str,
    factor_calculator,
    data,
    return_values,
    precomputed_factor_values=None,
) -> dict:
    """对所有算法的因子执行统一的验证评分。

    无论因子来自哪种算法（genetic/pysr/tree_prescreen/gflownet/deep_implicit），
    都使用同一套 factor_validation_service.validate_factor() 进行评估，
    确保跨算法的 score / ic / ir / overall_passed 具有可比性。

    Args:
        precomputed_factor_values: 预计算的因子值 pd.Series。
            如果提供，跳过 factor_calculator.calculate() 重新计算，
            直接使用此值进行验证。遗传算法的 mine_factors 已经计算过
            因子值并做了验证，此处复用可避免重复计算（节省约50%耗时）。

    Returns:
        dict with keys: ic, ir, fitness (score/100), validation_score,
                        overall_passed, turnover, stability, validation (full)
    """
    # 优先使用预计算的因子值，避免重复调用 factor_calculator.calculate()
    if precomputed_factor_values is not None:
        fv = precomputed_factor_values
    else:
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
            ic = abs(float(ic_val.get("ic")) if ic_val.get("ic") is not None else 0.0)
            ir_capped = float(ir_val.get("ir")) if ir_val.get("ir") is not None else 0.0
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
        "ic": None,
        "ir": None,
        "fitness": None,
        "validation_score": None,
        "overall_passed": False,
        "turnover": None,
        "stability": None,
        "validation": {},
    }


def _finalize_task(
    task_id: str,
    result: dict,
    request: GeneticMiningRequest,
    data,
    base_factor_codes,
    factor_calculator,
    logger,
):
    """Convert mining result to frontend format and store in task.

    挖掘完成后，将因子暂存到 generated_factors 表，标记验证状态。
    只有 is_valid=True 的因子才允许保存到因子库。

    **架构**: 挖掘与分析解耦
    - 挖掘算法只负责发现因子表达式（返回 MiningResult）
    - FactorAnalyzer 负责统一验证、评分、存储
    - 新增挖掘算法无需修改此函数
    """
    from backend.services.factor_analyzer import FactorAnalyzer
    from backend.services.mining_models import MiningResult

    # 将旧格式 dict 转为 MiningResult（兼容过渡期，新算法可直接返回 MiningResult）
    if isinstance(result, MiningResult):
        mining_result = result
    else:
        mining_result = MiningResult.from_legacy_dict(result)

    # 提取 return_values
    return_values = None
    if request.return_column and request.return_column in data.columns:
        return_values = data[request.return_column]
    elif "return" in data.columns:
        return_values = data["return"]

    # ---- 通过 FactorAnalyzer 统一验证和存储 ----
    # 所有算法的因子都走同一套验证标准，确保跨算法可比
    analyzer = FactorAnalyzer(factor_calculator, data, return_values)
    result_data = analyzer.analyze(mining_result, source=request.algorithm)

    # 将 discovered_factors 重命名为 factors（前端兼容）
    if "discovered_factors" in result_data:
        result_data["factors"] = result_data.pop("discovered_factors")

    # Prefer fitness_history from progress callback (already normalized IC values),
    # fallback to logbook extraction (may contain raw multi-objective tuples).
    fitness_history = mining_tasks[task_id].get("fitness_history")
    if fitness_history and len(fitness_history.get("best", [])) > 0:
        pass  # use progress callback data as-is
    else:
        # 兼容旧格式 dict 和新格式 MiningResult
        logbook = None
        if isinstance(result, dict):
            logbook = result.get("logbook")
        elif isinstance(result, MiningResult):
            logbook = result.algorithm_metadata.get("logbook")

        if logbook is not None:
            fitness_history = {
                "best": [_extract_first_fitness(gen["max"]) for gen in logbook],
                "average": [_extract_first_fitness(gen["avg"]) for gen in logbook],
            }
        else:
            fitness_history = mining_result.fitness_history

    # 限制fitness_history长度，防止无限增长
    if len(fitness_history.get("best", [])) > MAX_FITNESS_HISTORY_ENTRIES:
        fitness_history["best"] = fitness_history["best"][-MAX_FITNESS_HISTORY_ENTRIES:]
        fitness_history["average"] = fitness_history["average"][
            -MAX_FITNESS_HISTORY_ENTRIES:
        ]

    # 补充 result_data 中的通用字段
    factors = result_data.get("factors", [])
    result_data.update({
        "best_fitness": _safe_float(factors[0]["fitness"])
        if factors
        else 0.0,
        "avg_fitness": _safe_float(result_data.get("avg_fitness", 0.0)),
        "generations": request.n_generations,
        "fitness_history": fitness_history,
        "algorithm": request.algorithm,
    })

    # 透传算法特有元数据（从 MiningResult.algorithm_metadata 中获取）
    # 新算法只需在 algorithm_metadata 中添加字段，无需修改此函数
    algo_meta = mining_result.algorithm_metadata
    for key in ("feature_importance", "selected_features", "model_info",
                "training_history", "policy_loss_history", "equations"):
        if key in algo_meta:
            result_data[key] = algo_meta[key]

    # 收集算法特定的过程信息
    # 兼容旧格式 dict 和新格式 MiningResult
    if isinstance(result, MiningResult):
        result_dict = result.to_dict()
    else:
        result_dict = result
    process_info = _collect_process_info(result_dict, request)

    mining_tasks[task_id]["status"] = "completed"
    mining_tasks[task_id]["progress"] = 100
    mining_tasks[task_id]["result"] = result_data
    mining_tasks[task_id]["fitness_history"] = fitness_history

    # 持久化完成结果到数据库
    _sync_task_to_db(
        task_id,
        status="completed",
        progress=100,
        result=result_data,
        fitness_history=fitness_history,
        best_fitness=result_data.get("best_fitness"),
        avg_fitness=result_data.get("avg_fitness"),
        process_info=process_info,
    )

    logger.info(f"Task {task_id} completed successfully")
    logger.info(
        f"Discovered {len(factors)} factors (algorithm={request.algorithm})"
    )


async def _run_simulated_mining(
    task_id: str,
    request: GeneticMiningRequest,
    data,
    base_factor_codes,
    factor_service,
    logger,
):
    """模拟模式挖掘（当DEAP/PySR库未安装时使用）"""
    factor_values = {}
    for code in base_factor_codes:
        try:
            values = factor_service.calculator.calculate(data, code)
            if values is not None and len(values.dropna()) > 0:
                factor_values[code] = values
                logger.info(
                    f"Successfully calculated factor: {code}, {len(values.dropna())} valid values"
                )
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
            fitness_history["best"] = fitness_history["best"][
                -MAX_FITNESS_HISTORY_ENTRIES:
            ]
            fitness_history["average"] = fitness_history["average"][
                -MAX_FITNESS_HISTORY_ENTRIES:
            ]

        mining_tasks[task_id]["current_generation"] = gen + 1
        mining_tasks[task_id]["total_generations"] = n_generations
        mining_tasks[task_id]["best_fitness"] = current_best_fitness
        mining_tasks[task_id]["avg_fitness"] = current_avg_fitness
        mining_tasks[task_id]["fitness_history"] = fitness_history

        logger.info(
            f"Generation {gen + 1}/{n_generations} completed, best_fitness={current_best_fitness:.4f}"
        )

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

        discovered_factors.append(
            {
                "name": f"Mined_Factor_{i + 1}",
                "expression": expression,
                "ic": 0.03 + (i * 0.01),
                "ir": 0.5 + (i * 0.1),
                "fitness": 0.03 + (i * 0.01),
                "source": "simulated",
            }
        )

    result = {
        "best_factors": discovered_factors,
        "best_fitness": discovered_factors[0]["ic"] if discovered_factors else 0,
        "avg_fitness": (
            sum(f["fitness"] for f in discovered_factors) / len(discovered_factors)
            if discovered_factors
            else 0
        ),
        "generations": n_generations,
        "fitness_history": fitness_history,
        "algorithm": request.algorithm,
        "source": "simulated",
    }

    # 调用 _finalize_task 完成统一验证、暂存和状态更新
    _finalize_task(
        task_id,
        result,
        request,
        data,
        base_factor_codes,
        factor_service.calculator,
        logger,
    )

    logger.info(f"Task {task_id} completed (simulated mode)")


@router.get("/status/{task_id}")
async def get_mining_status(task_id: str):
    """获取挖掘状态"""

    # 优先从内存获取
    if task_id in mining_tasks:
        task = mining_tasks[task_id]
        logger.debug(f"Status check for task {task_id}: {task['status']}")

        # 从内存中的 task 字典获取 started_at（避免每2秒查一次数据库）
        # started_at 在 _run_mining 启动时写入 mining_tasks[task_id]
        started_at = task.get("started_at")

        response_data = {
            "status": task["status"],
            "progress": task.get("progress", 0),
            "error": task.get("error"),
            "algorithm": task.get("algorithm", "genetic"),
            "started_at": started_at,
        }

        if task["status"] == "completed" and "result" in task:
            result = task["result"]
            # 优先使用 _update_progress 设置的 total_generations（= n_generations + 1），
            # 回退到 result["generations"]（= n_generations）
            task_total_gen = task.get("total_generations")
            result_generations = result.get("generations", 0)
            final_total_gen = task_total_gen if task_total_gen else result_generations
            logger.debug(
                f"Completed task total_gen: task={task_total_gen}, result={result_generations}, final={final_total_gen}"
            )
            response_data["current_generation"] = final_total_gen
            response_data["total_generations"] = final_total_gen
            response_data["best_fitness"] = result.get("best_fitness", 0)
            response_data["avg_fitness"] = result.get("avg_fitness", 0)
            response_data["fitness_history"] = result.get(
                "fitness_history", {"best": [], "average": []}
            )
            response_data["algorithm"] = result.get(
                "algorithm", task.get("algorithm", "genetic")
            )
            logger.info(
                f"Returning completed status with fitness_history length: "
                f"{len(response_data['fitness_history']['best'])}"
            )
        else:
            response_data["current_generation"] = task.get("current_generation", 0)
            response_data["total_generations"] = task.get("total_generations", 0)
            response_data["best_fitness"] = task.get("best_fitness", 0.03)
            response_data["avg_fitness"] = task.get("avg_fitness", 0.03)
            response_data["fitness_history"] = task.get(
                "fitness_history", {"best": [], "average": []}
            )
            response_data["current_algorithm"] = task.get("current_algorithm", "")
            logger.info(
                f"Returning running status: gen "
                f"{response_data['current_generation']}/"
                f"{response_data['total_generations']}"
            )

        return sanitize_dict({"success": True, "data": response_data})

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
                "started_at": task_record.started_at.isoformat()
                if task_record.started_at
                else None,
            }

            if task_record.status == "completed" and task_record.result:
                result = task_record.result
                response_data["current_generation"] = (
                    task_record.total_generations or result.get("generations", 0)
                )
                response_data["total_generations"] = (
                    task_record.total_generations or result.get("generations", 0)
                )
                response_data["best_fitness"] = result.get(
                    "best_fitness", task_record.best_fitness or 0
                )
                response_data["avg_fitness"] = result.get(
                    "avg_fitness", task_record.avg_fitness or 0
                )
                response_data["fitness_history"] = result.get(
                    "fitness_history",
                    task_record.fitness_history or {"best": [], "average": []},
                )
                response_data["algorithm"] = result.get(
                    "algorithm", task_record.algorithm
                )
            else:
                response_data["current_generation"] = (
                    task_record.current_generation or 0
                )
                response_data["total_generations"] = task_record.total_generations or 0
                response_data["best_fitness"] = task_record.best_fitness or 0
                response_data["avg_fitness"] = task_record.avg_fitness or 0
                response_data["fitness_history"] = task_record.fitness_history or {
                    "best": [],
                    "average": [],
                }

            return sanitize_dict({"success": True, "data": response_data})
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
            raise HTTPException(
                status_code=400, detail=f"任务尚未完成，当前状态: {task['status']}"
            )
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
        return sanitize_dict({"success": True, "data": result_data})

    # 内存中没有，从数据库获取
    try:
        with get_db() as db:
            repo = MiningTaskRepository(db)
            task_record = repo.get_by_task_id(task_id)

            if not task_record:
                raise HTTPException(status_code=404, detail="任务不存在")

            if task_record.status != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"任务尚未完成，当前状态: {task_record.status}",
                )

            if not task_record.result:
                raise HTTPException(status_code=404, detail="任务结果不存在")

            result_data = task_record.result
            if task_record.process_info:
                result_data["process_info"] = task_record.process_info

            return sanitize_dict({"success": True, "data": result_data})
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
    if service and hasattr(service, "request_cancel"):
        service.request_cancel()

    # 更新内存和数据库状态
    if task_id in mining_tasks:
        mining_tasks[task_id]["status"] = "cancelled"

    _sync_task_to_db(task_id, status="cancelled", completed_at=datetime.now())

    return {"success": True, "message": "取消请求已发送"}


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
            for task_record in repo.get_active_tasks() or []:
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
                        "fitness_history": task.get(
                            "fitness_history", {"best": [], "average": []}
                        ),
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

        return {"success": True, "data": active_in_memory}
    except Exception as e:
        logger.error(f"获取活跃任务失败: {e}")
        return {"success": True, "data": []}


@router.get("/history")
async def get_mining_history(
    limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0)
):
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
                },
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

            return {"success": True, "data": repo.to_dict(task_record)}
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

            return {"success": True, "message": "删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除挖掘历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume/{task_id}")
async def resume_mining(task_id: str, background_tasks: BackgroundTasks):
    """从最近的检查点恢复挖掘任务

    查找指定任务的最新checkpoint，创建新任务继续进化。
    新任务继承原任务的配置和已完成的代数。
    """
    try:
        from backend.repositories.mining_checkpoint_repository import (
            MiningCheckpointRepository,
        )

        # 查找最新checkpoint
        with get_db() as db:
            repo = MiningCheckpointRepository(db)
            checkpoint = repo.get_latest(task_id)

        if not checkpoint:
            raise HTTPException(
                status_code=404,
                detail=f"任务 {task_id} 没有可用的检查点",
            )

        # 获取原任务配置
        with get_db() as db:
            task_repo = MiningTaskRepository(db)
            original_task = task_repo.get_by_task_id(task_id)

        if not original_task or not original_task.config:
            raise HTTPException(
                status_code=404,
                detail=f"原任务 {task_id} 的配置不存在",
            )

        # 从原任务配置创建新请求
        config = original_task.config
        remaining_gens = checkpoint.total_generations - checkpoint.generation

        if remaining_gens <= 0:
            raise HTTPException(
                status_code=400,
                detail="原任务已完成所有代数，无需恢复",
            )

        # 创建新任务（减少代数为剩余代数）
        request = GeneticMiningRequest(**config)
        request.n_generations = remaining_gens

        import uuid

        new_task_id = str(uuid.uuid4())

        _cleanup_old_tasks()

        mining_tasks[new_task_id] = {
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "algorithm": request.algorithm,
            "created_at": time.time(),
            "total_generations": request.n_generations,
            "resumed_from": task_id,
            "resumed_generation": checkpoint.generation,
        }

        # 持久化到数据库
        try:
            with get_db() as db:
                task_repo = MiningTaskRepository(db)
                task_record = MiningTaskModel(
                    task_id=new_task_id,
                    status="pending",
                    algorithm=request.algorithm,
                    stock_codes=json.dumps(
                        request.stock_codes
                        or ([request.stock_code] if request.stock_code else [])
                    ),
                    base_factors=json.dumps(request.base_factors or []),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    freq=request.freq,
                    config=request.model_dump(),
                )
                task_repo.create(task_record)
        except Exception as e:
            logger.warning(f"持久化恢复任务失败（不影响运行）: {e}")

        background_tasks.add_task(_run_mining, new_task_id, request)

        return {
            "success": True,
            "data": {
                "task_id": new_task_id,
                "status": "pending",
                "resumed_from": task_id,
                "resumed_generation": checkpoint.generation,
                "remaining_generations": remaining_gens,
            },
            "message": f"从第{checkpoint.generation}代检查点恢复，剩余{remaining_gens}代",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复挖掘任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints/{task_id}")
async def get_checkpoints(task_id: str):
    """获取指定任务的检查点列表"""
    try:
        from backend.repositories.mining_checkpoint_repository import (
            MiningCheckpointRepository,
        )

        with get_db() as db:
            repo = MiningCheckpointRepository(db)
            checkpoints = repo.get_by_task_id(task_id)

            return {
                "success": True,
                "data": [
                    {
                        "id": cp.id,
                        "generation": cp.generation,
                        "total_generations": cp.total_generations,
                        "created_at": (
                            cp.created_at.isoformat() if cp.created_at else None
                        ),
                    }
                    for cp in checkpoints
                ],
            }
    except Exception as e:
        logger.error(f"获取检查点失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
