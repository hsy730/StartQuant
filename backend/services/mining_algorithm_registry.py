"""
挖掘算法注册表 — 新增算法只需注册，无需修改 _run_mining

使用方式：
    # 注册新算法
    register_algorithm("my_algo", MyAlgoFactory)

    # 创建算法实例
    service = create_algorithm("my_algo", task_id, request, data, ...)

    # 设置回调和股票池（由 _run_mining 统一处理）
    service.set_progress_callback(...)
    service.set_stock_pool(...)
    service._task_id = task_id

    # 执行挖掘
    result = service.mine_factors()

新增挖掘算法的步骤：
1. 继承 BaseMiningService
2. 实现 mine_factors() -> MiningResult（推荐）或 dict（兼容旧格式）
3. 在此文件中注册：register_algorithm("my_algo", MyAlgoFactory)
4. 完成 — 无需修改 mining.py 的任何代码
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# 算法工厂注册表：algorithm_name -> factory_function
_ALGORITHM_REGISTRY: Dict[str, Callable] = {}


def register_algorithm(name: str, factory: Callable) -> None:
    """注册挖掘算法工厂函数

    Args:
        name: 算法名称（如 "genetic", "pysr", "gflownet"）
        factory: 工厂函数，签名为 (task_id, request, data, base_factor_codes,
                 factor_service, stock_codes, logger) -> BaseMiningService
    """
    _ALGORITHM_REGISTRY[name] = factory
    logger.debug(f"Registered mining algorithm: {name}")


def create_algorithm(name: str, **kwargs) -> Optional[Any]:
    """创建算法实例

    Args:
        name: 算法名称
        **kwargs: 传递给工厂函数的关键字参数
            (task_id, request, data, base_factor_codes,
             factor_service, stock_codes, logger)

    Returns:
        算法服务实例，如果算法未注册则返回 None
    """
    factory = _ALGORITHM_REGISTRY.get(name)
    if factory is None:
        logger.warning(f"Unknown mining algorithm: {name}")
        return None
    return factory(**kwargs)


def get_registered_algorithms() -> Dict[str, Callable]:
    """获取所有已注册的算法"""
    return dict(_ALGORITHM_REGISTRY)


def is_algorithm_registered(name: str) -> bool:
    """检查算法是否已注册"""
    return name in _ALGORITHM_REGISTRY


# ============================================================
# 内置算法注册（延迟导入避免循环依赖）
# ============================================================

def _register_builtin_algorithms():
    """注册内置挖掘算法

    工厂函数只负责创建服务实例，不设置 progress_callback 和 _task_id。
    这些由 _run_mining 统一设置，避免循环依赖。
    """

    # 遗传算法
    def genetic_factory(task_id, request, data, base_factor_codes,
                        factor_service, stock_codes, logger):
        from backend.services.genetic_factor_mining_service import (
            create_genetic_mining_service,
        )
        return create_genetic_mining_service(
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
            max_base_factors=request.max_base_factors,
        )

    # PySR 符号回归
    def pysr_factory(task_id, request, data, base_factor_codes,
                     factor_service, stock_codes, logger):
        from backend.services.pysr_factor_mining_service import PySRFactorMiningService
        return PySRFactorMiningService(
            base_factors=base_factor_codes,
            data=data,
            return_column="return",
            factor_calculator=factor_service.calculator,
            niterations=request.n_generations,
            populations=request.population_size or 5,
        )

    # 统一调度（tree_prescreen / gflownet / deep_implicit）
    def unified_factory(task_id, request, data, base_factor_codes,
                        factor_service, stock_codes, logger):
        from backend.services.dual_mining_service import DualMiningService
        return DualMiningService(
            base_factors=base_factor_codes,
            data=data,
            return_column="return",
            factor_calculator=factor_service.calculator,
            algorithm=request.algorithm,
            n_generations=request.n_generations,
            population_size=request.population_size,
            # Tree Prescreen
            tree_model_type=getattr(request, "tree_model_type", "auto"),
            top_k=getattr(request, "top_k", 0),
            importance_threshold=getattr(request, "importance_threshold", 0.01),
            tree_n_estimators=getattr(request, "tree_n_estimators", 100),
            tree_max_depth=getattr(request, "tree_max_depth", 5),
            downstream_algorithm=getattr(request, "downstream_algorithm", "genetic"),
            # GFlowNet
            gflownet_n_trajectories=getattr(request, "gflownet_n_trajectories", 200),
            gflownet_n_iterations=getattr(request, "gflownet_n_iterations", 50),
            gflownet_hidden_dim=getattr(request, "gflownet_hidden_dim", 128),
            gflownet_learning_rate=getattr(request, "gflownet_learning_rate", 1e-3),
            gflownet_max_expression_depth=getattr(request, "gflownet_max_expression_depth", 5),
            gflownet_temperature=getattr(request, "gflownet_temperature", 1.0),
            gflownet_reward_scale=getattr(request, "gflownet_reward_scale", 10.0),
            gflownet_buffer_size=getattr(request, "gflownet_buffer_size", 1000),
            # Deep Factor
            deep_d_model=getattr(request, "deep_d_model", 64),
            deep_n_heads=getattr(request, "deep_n_heads", 4),
            deep_n_layers=getattr(request, "deep_n_layers", 3),
            deep_d_ff=getattr(request, "deep_d_ff", 256),
            deep_n_latent_factors=getattr(request, "deep_n_latent_factors", 5),
            deep_dropout=getattr(request, "deep_dropout", 0.1),
            deep_seq_length=getattr(request, "deep_seq_length", 20),
            deep_learning_rate=getattr(request, "deep_learning_rate", 1e-4),
            deep_n_epochs=getattr(request, "deep_n_epochs", 50),
            deep_batch_size=getattr(request, "deep_batch_size", 32),
            deep_weight_decay=getattr(request, "deep_weight_decay", 1e-5),
            deep_early_stopping_patience=getattr(request, "deep_early_stopping_patience", 5),
        )

    register_algorithm("genetic", genetic_factory)
    register_algorithm("pysr", pysr_factory)
    register_algorithm("tree_prescreen", unified_factory)
    register_algorithm("gflownet", unified_factory)
    register_algorithm("deep_implicit", unified_factory)


# 模块加载时自动注册内置算法
_register_builtin_algorithms()
