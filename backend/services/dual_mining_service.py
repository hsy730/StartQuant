"""
统一因子挖掘服务

支持五种算法模式:
  - "genetic": DEAP遗传规划
  - "pysr": PySR符号回归
  - "tree_prescreen": 树模型预筛选 → 符号回归管道
  - "gflownet": GFlowNet增强遗传规划（实验性）
  - "deep_implicit": 深度隐式因子模型（Transformer，前沿赛道）
"""

import logging
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

from backend.services.genetic_factor_mining_service import (  # noqa: E402
    GeneticFactorMiningService,
    create_genetic_mining_service,
    DEAP_AVAILABLE,
)
from backend.services.pysr_factor_mining_service import (  # noqa: E402
    PySRFactorMiningService,
    create_pysr_mining_service,
    PYSR_AVAILABLE,
)

# 新算法模块（可选依赖）
try:
    from backend.services.tree_prescreen_mining_service import (  # noqa: F401
        TreePrescreenMiningService,
        create_tree_prescreen_mining_service,
        TREE_PRESCREEN_AVAILABLE,
    )
except ImportError:
    TREE_PRESCREEN_AVAILABLE = False

try:
    from backend.services.gflownet_mining_service import (  # noqa: F401
        GFlowNetMiningService,
        create_gflownet_mining_service,
        GFLOWNET_AVAILABLE,
    )
except ImportError:
    GFLOWNET_AVAILABLE = False

try:
    from backend.services.deep_factor_mining_service import (  # noqa: F401
        DeepFactorMiningService,
        create_deep_factor_mining_service,
        DEEP_FACTOR_AVAILABLE,
    )
except ImportError:
    DEEP_FACTOR_AVAILABLE = False


class DualMiningService:
    """统一因子挖掘服务

    支持五种算法模式，统一接口调度：
    - genetic: DEAP遗传规划
    - pysr: PySR符号回归
    - tree_prescreen: 树模型预筛选 → 符号回归管道
    - gflownet: GFlowNet增强遗传规划
    - deep_implicit: 深度隐式因子模型
    """

    # 支持的算法模式
    SUPPORTED_ALGORITHMS = {
        "genetic",
        "pysr",
        "tree_prescreen",
        "gflownet",
        "deep_implicit",
    }

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        factor_calculator=None,
        max_eval_stocks: int = 50,
        algorithm: str = "genetic",
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
        # ---- Tree Prescreen parameters ----
        tree_model_type: str = "auto",
        top_k: int = 0,
        importance_threshold: float = 0.01,
        tree_n_estimators: int = 100,
        tree_max_depth: int = 5,
        downstream_algorithm: str = "genetic",
        # ---- GFlowNet parameters ----
        gflownet_n_trajectories: int = 200,
        gflownet_n_iterations: int = 50,
        gflownet_hidden_dim: int = 128,
        gflownet_learning_rate: float = 1e-3,
        gflownet_max_expression_depth: int = 5,
        gflownet_temperature: float = 1.0,
        gflownet_reward_scale: float = 10.0,
        gflownet_buffer_size: int = 1000,
        # ---- Deep Factor parameters ----
        deep_d_model: int = 64,
        deep_n_heads: int = 4,
        deep_n_layers: int = 3,
        deep_d_ff: int = 256,
        deep_n_latent_factors: int = 5,
        deep_dropout: float = 0.1,
        deep_seq_length: int = 20,
        deep_learning_rate: float = 1e-4,
        deep_n_epochs: int = 50,
        deep_batch_size: int = 32,
        deep_weight_decay: float = 1e-5,
        deep_early_stopping_patience: int = 5,
    ):
        self.base_factor_codes = base_factors
        self.data = data.copy() if data is not None else None
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

        self._tree_prescreen_params = dict(
            tree_model_type=tree_model_type,
            top_k=top_k,
            importance_threshold=importance_threshold,
            tree_n_estimators=tree_n_estimators,
            tree_max_depth=tree_max_depth,
            downstream_algorithm=downstream_algorithm,
            fitness_objective=fitness_objective,
            cv_folds=cv_folds,
        )

        self._gflownet_params = dict(
            n_trajectories=gflownet_n_trajectories,
            n_iterations=gflownet_n_iterations,
            hidden_dim=gflownet_hidden_dim,
            learning_rate=gflownet_learning_rate,
            max_expression_depth=gflownet_max_expression_depth,
            temperature=gflownet_temperature,
            reward_scale=gflownet_reward_scale,
            buffer_size=gflownet_buffer_size,
            fitness_objective=fitness_objective,
            cv_folds=cv_folds,
        )

        self._deep_factor_params = dict(
            d_model=deep_d_model,
            n_heads=deep_n_heads,
            n_layers=deep_n_layers,
            d_ff=deep_d_ff,
            n_latent_factors=deep_n_latent_factors,
            dropout=deep_dropout,
            seq_length=deep_seq_length,
            learning_rate=deep_learning_rate,
            n_epochs=deep_n_epochs,
            batch_size=deep_batch_size,
            weight_decay=deep_weight_decay,
            early_stopping_patience=deep_early_stopping_patience,
            fitness_objective=fitness_objective,
            cv_folds=cv_folds,
        )

        self.progress_callback = None
        self._gp_service: Optional[GeneticFactorMiningService] = None
        self._pysr_service: Optional[PySRFactorMiningService] = None
        self._tree_prescreen_service = None
        self._gflownet_service = None
        self._deep_factor_service = None
        # 股票池信息（在set_stock_pool时保存，用于子服务创建后传递）
        self._stock_codes: Optional[List[str]] = None
        self._stock_start_date: Optional[str] = None
        self._stock_end_date: Optional[str] = None

    def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
        # 保存股票池信息，以便子服务创建时传递
        self._stock_codes = stock_codes
        self._stock_start_date = start_date
        self._stock_end_date = end_date
        # 如果子服务已存在，直接传递
        if self._gp_service is not None:
            self._gp_service.set_stock_pool(stock_codes, start_date, end_date)
        if self._pysr_service is not None:
            self._pysr_service.set_stock_pool(stock_codes, start_date, end_date)
        if self._tree_prescreen_service is not None:
            self._tree_prescreen_service.set_stock_pool(
                stock_codes, start_date, end_date
            )
        if self._gflownet_service is not None:
            self._gflownet_service.set_stock_pool(stock_codes, start_date, end_date)
        if self._deep_factor_service is not None:
            self._deep_factor_service.set_stock_pool(stock_codes, start_date, end_date)

    def _pass_stock_pool_to_service(self, service):
        """将保存的股票池信息传递给新创建的子服务"""
        if self._stock_codes is not None and service is not None:
            service.set_stock_pool(
                self._stock_codes, self._stock_start_date, self._stock_end_date
            )

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def request_cancel(self):
        """请求取消挖掘任务，转发到当前活跃的子服务"""
        if self._gp_service is not None:
            self._gp_service.request_cancel()
        if self._pysr_service is not None:
            self._pysr_service.request_cancel()
        if self._tree_prescreen_service is not None:
            self._tree_prescreen_service.request_cancel()
        if self._gflownet_service is not None:
            self._gflownet_service.request_cancel()
        if self._deep_factor_service is not None:
            self._deep_factor_service.request_cancel()
        logger.info("DualMiningService: 已转发取消请求到子服务")

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
        self._pass_stock_pool_to_service(self._gp_service)

        if self.progress_callback:

            def gp_progress(gen, total_gen, best_fitness, avg_fitness):
                self.progress_callback(
                    gen, total_gen, best_fitness, avg_fitness, algorithm="genetic"
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
        self._pass_stock_pool_to_service(self._pysr_service)

        if self.progress_callback:

            def pysr_progress(iteration, total_iter, best_fitness, avg_fitness):
                self.progress_callback(
                    iteration, total_iter, best_fitness, avg_fitness, algorithm="pysr"
                )

            self._pysr_service.set_progress_callback(pysr_progress)

        result = self._pysr_service.mine_factors()
        result["source"] = "pysr"
        return result

    def mine_factors(self) -> Dict:
        """Execute factor mining using the configured algorithm mode.

        Modes:
        - ``"genetic"``: DEAP GP only
        - ``"pysr"``: PySR only
        - ``"tree_prescreen"``: tree model pre-screening → symbolic regression
        - ``"gflownet"``: GFlowNet-enhanced genetic programming
        - ``"deep_implicit"``: Transformer-based deep implicit factors
        """
        if self.algorithm == "genetic":
            return self._run_genetic()
        elif self.algorithm == "pysr":
            return self._run_pysr()
        elif self.algorithm == "tree_prescreen":
            return self._run_tree_prescreen()
        elif self.algorithm == "gflownet":
            return self._run_gflownet()
        elif self.algorithm == "deep_implicit":
            return self._run_deep_implicit()
        else:
            logger.warning(
                f"Unknown algorithm mode '{self.algorithm}', falling back to genetic"
            )
            return self._run_genetic()

    def _run_tree_prescreen(self) -> Dict:
        """Run tree model pre-screening → symbolic regression pipeline."""
        if not TREE_PRESCREEN_AVAILABLE:
            return {
                "success": False,
                "message": "树模型预筛选不可用（需安装 lightgbm 或 xgboost）",
                "best_factors": [],
            }

        logger.info("启动树模型预筛选符号回归管道...")

        # 构造TreePrescreenMiningService所需的参数（使用公开API参数名）
        merged_kwargs = dict(self._gp_params)
        # PySR参数需要pysr_前缀（_pysr_params存的是内部名，需映射回公开名）
        _pysr_name_map = {
            "niterations": "pysr_niterations",
            "populations": "pysr_populations",
            "binary_operators": "pysr_binary_operators",
            "unary_operators": "pysr_unary_operators",
            "maxsize": "pysr_maxsize",
            "maxdepth": "pysr_maxdepth",
            "constraints": "pysr_constraints",
            "nested_constraints": "pysr_nested_constraints",
            "parsimony": "pysr_parsimony",
            "procs": "pysr_procs",
            "population_size": "pysr_population_size",
        }
        for internal_name, public_name in _pysr_name_map.items():
            if internal_name in self._pysr_params:
                merged_kwargs[public_name] = self._pysr_params[internal_name]
        # 树模型参数名映射（DualMiningService内部名 -> TreePrescreenMiningService公开名）
        _tree_name_map = {
            "tree_model_type": "tree_model",
            "tree_n_estimators": "n_estimators",
            "tree_max_depth": "max_depth",
        }
        for internal_name, public_name in _tree_name_map.items():
            if internal_name in self._tree_prescreen_params:
                merged_kwargs[public_name] = self._tree_prescreen_params[internal_name]
        # 直接透传的树参数
        for key in ("top_k", "importance_threshold", "downstream_algorithm"):
            if key in self._tree_prescreen_params:
                merged_kwargs[key] = self._tree_prescreen_params[key]

        self._tree_prescreen_service = create_tree_prescreen_mining_service(
            base_factors=self.base_factor_codes,
            data=self.data,
            return_column=self.return_column,
            factor_calculator=self.factor_calculator,
            max_eval_stocks=self.max_eval_stocks,
            **merged_kwargs,
        )

        if self.progress_callback:

            def tp_progress(iteration, total_iter, best_fitness, avg_fitness, **kwargs):
                # tree_prescreen 的 _report_progress 可能传字符串 phase 名（如 "feature_importance"），
                # 而 mining.py 的 _update_progress 期望数字，此处过滤非数值调用
                if isinstance(iteration, (int, float)):
                    self.progress_callback(
                        iteration,
                        total_iter,
                        best_fitness,
                        avg_fitness,
                        algorithm="tree_prescreen",
                    )

            self._tree_prescreen_service.set_progress_callback(tp_progress)

        result = self._tree_prescreen_service.mine_factors()
        result["source"] = "tree_prescreen"
        return result

    def _run_gflownet(self) -> Dict:
        """Run GFlowNet-enhanced genetic programming."""
        if not GFLOWNET_AVAILABLE:
            return {
                "success": False,
                "message": "GFlowNet不可用（需安装 torch）",
                "best_factors": [],
            }

        logger.info("启动GFlowNet增强遗传规划...")

        self._gflownet_service = create_gflownet_mining_service(
            base_factors=self.base_factor_codes,
            data=self.data,
            return_column=self.return_column,
            factor_calculator=self.factor_calculator,
            max_eval_stocks=self.max_eval_stocks,
            **self._gflownet_params,
        )
        self._pass_stock_pool_to_service(self._gflownet_service)

        if self.progress_callback:

            def gfn_progress(
                iteration, total_iter, best_fitness, avg_fitness, **kwargs
            ):
                self.progress_callback(
                    iteration,
                    total_iter,
                    best_fitness,
                    avg_fitness,
                    algorithm="gflownet",
                )

            self._gflownet_service.set_progress_callback(gfn_progress)

        result = self._gflownet_service.mine_factors()
        result["source"] = "gflownet"
        return result

    def _run_deep_implicit(self) -> Dict:
        """Run deep implicit factor model (Transformer)."""
        if not DEEP_FACTOR_AVAILABLE:
            return {
                "success": False,
                "message": "深度隐式因子模型不可用（需安装 torch）",
                "best_factors": [],
            }

        logger.info("启动深度隐式因子模型 (Transformer)...")

        # 只传递 DeepFactorMiningService 接受的参数
        _deep_allowed = {
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "n_latent_factors",
            "dropout",
            "seq_length",
            "learning_rate",
            "n_epochs",
            "batch_size",
            "weight_decay",
            "early_stopping_patience",
            "sparsity_coeff",
        }
        filtered_deep = {
            k: v for k, v in self._deep_factor_params.items() if k in _deep_allowed
        }

        self._deep_factor_service = create_deep_factor_mining_service(
            base_factors=self.base_factor_codes,
            data=self.data,
            return_column=self.return_column,
            factor_calculator=self.factor_calculator,
            max_eval_stocks=self.max_eval_stocks,
            **filtered_deep,
        )
        self._pass_stock_pool_to_service(self._deep_factor_service)

        if self.progress_callback:

            def deep_progress(epoch, total_epochs, best_fitness, avg_fitness, **kwargs):
                self.progress_callback(
                    epoch,
                    total_epochs,
                    best_fitness,
                    avg_fitness,
                    algorithm="deep_implicit",
                )

            self._deep_factor_service.set_progress_callback(deep_progress)

        result = self._deep_factor_service.mine_factors()
        result["source"] = "deep_implicit"
        return result


def create_dual_mining_service(
    base_factors: List[str], data: pd.DataFrame, factor_calculator=None, **kwargs
) -> DualMiningService:
    """Create a configured :class:`DualMiningService` instance.

    Accepted keyword arguments include all algorithm parameters,
    plus:

    * ``algorithm`` – "genetic" / "pysr" / "tree_prescreen" /
      "gflownet" / "deep_implicit" (default "genetic")
    """
    return DualMiningService(
        base_factors=base_factors,
        data=data,
        factor_calculator=factor_calculator,
        **kwargs,
    )
