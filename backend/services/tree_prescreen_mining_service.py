"""
树模型预筛选符号回归因子挖掘服务

核心思路：
  原始特征（数百个）→ LightGBM/XGBoost 特征重要性 → Top-K 选择 → GP/PySR 符号回归 → 可解释公式

这是业界最实用、最主流的高维特征空间符号回归方法：
1. 树模型快速筛选：利用 GBDT 的特征重要性（gain）快速定位有效特征
2. 符号回归精炼：在低维空间上运行 GP/PySR，搜索可解释的因子表达式
3. 两阶段协同：大幅降低符号回归的搜索空间，提升效率和结果质量

优势：
- 解决"维度灾难"：数百个特征直接做符号回归几乎不可能收敛
- 保留可解释性：最终输出是公式而非黑箱
- 业界最佳实践：BigQuant/WorldQuant 等平台均采用类似流程
"""
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# ---- 依赖检测 ----
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

TREE_PRESCREEN_AVAILABLE = LGB_AVAILABLE or XGB_AVAILABLE

if not TREE_PRESCREEN_AVAILABLE:
    logger.warning("LightGBM 和 XGBoost 均未安装，树模型预筛选功能不可用。请运行: pip install lightgbm 或 pip install xgboost")

from backend.services.base_mining_service import BaseMiningService
# ---- 下游服务延迟导入 ----


class TreePrescreenMiningService(BaseMiningService):
    """树模型预筛选 + 符号回归因子挖掘服务

    Two-stage pipeline:
      Phase 1: 树模型特征重要性计算（进度 0-30%）
      Phase 2: 下游符号回归（进度 30-100%）

    支持的下游算法：
      - "genetic": DEAP 遗传规划（默认）
      - "pysr": PySR 符号回归
    """

    _service_name = "树模型预筛选"

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        factor_calculator=None,
        max_eval_stocks: int = 50,
        # ---- 树模型参数 ----
        tree_model: str = "auto",          # "auto" / "lightgbm" / "xgboost"
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        importance_type: str = "gain",     # "gain" / "split"
        top_k: int = 0,                   # 0 = 自动
        importance_threshold: float = 0.01,
        # ---- 下游符号回归参数 ----
        downstream_algorithm: str = "genetic",
        # DEAP GP 参数
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
        # PySR 参数
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
        if not TREE_PRESCREEN_AVAILABLE:
            raise ImportError("LightGBM 和 XGBoost 均未安装，请运行: pip install lightgbm 或 pip install xgboost")

        super().__init__(
            base_factors=base_factors,
            data=data,
            return_column=return_column,
            factor_calculator=factor_calculator,
            max_eval_stocks=max_eval_stocks,
            fitness_objective=fitness_objective,
            cv_folds=cv_folds,
        )

        # 树模型参数
        self.tree_model = tree_model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.importance_type = importance_type
        self.top_k = top_k
        self.importance_threshold = importance_threshold

        # 下游算法
        self.downstream_algorithm = downstream_algorithm

        # DEAP GP 参数
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

        # PySR 参数
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

        # 结果缓存
        self._feature_importance: Optional[Dict[str, float]] = None
        self._selected_features: Optional[List[str]] = None

        self._gp_service = None
        self._pysr_service = None

    # ------------------------------------------------------------------
    # 进度回调（重写以同时取消下游子服务）
    # ------------------------------------------------------------------

    def request_cancel(self):
        """请求取消挖掘任务"""
        self._cancel_flag = True
        # 同时取消下游子服务
        if self._gp_service is not None:
            self._gp_service.request_cancel()
        if self._pysr_service is not None:
            self._pysr_service.request_cancel()
        logger.info("收到取消请求")

    def _report_progress(self, phase: str, current: float, total: float, message: str = ""):
        """内部进度报告（映射到 0-100% 全局进度）"""
        if self.progress_callback is None:
            return

        if phase == "feature_importance":
            # Phase 1: 0-30%
            global_progress = 0.0 + (current / max(total, 1)) * 30.0
        else:
            # Phase 2: 30-100%
            global_progress = 30.0 + (current / max(total, 1)) * 70.0

        self.progress_callback(phase, global_progress, 100.0, message)

    # ------------------------------------------------------------------
    # Phase 1: 特征重要性计算
    # ------------------------------------------------------------------

    def _resolve_tree_model(self) -> str:
        """解析实际使用的树模型类型"""
        if self.tree_model == "lightgbm":
            if LGB_AVAILABLE:
                return "lightgbm"
            logger.warning("LightGBM 不可用，尝试回退到 XGBoost")
            if XGB_AVAILABLE:
                return "xgboost"
            raise ImportError("LightGBM 和 XGBoost 均不可用")

        if self.tree_model == "xgboost":
            if XGB_AVAILABLE:
                return "xgboost"
            logger.warning("XGBoost 不可用，尝试回退到 LightGBM")
            if LGB_AVAILABLE:
                return "lightgbm"
            raise ImportError("XGBoost 和 LightGBM 均不可用")

        # auto: 优先 LightGBM（更快），回退 XGBoost
        if LGB_AVAILABLE:
            return "lightgbm"
        if XGB_AVAILABLE:
            return "xgboost"
        raise ImportError("LightGBM 和 XGBoost 均不可用")

    def _build_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """构建特征矩阵和目标变量

        Returns:
            (X_df, y, feature_names) 特征DataFrame、目标Series、特征名列表
        """
        feature_names = []
        series_list = []

        for var_name in sorted(self.base_factor_values.keys()):
            info = self.base_factor_values[var_name]
            feature_names.append(var_name)
            series_list.append(info["values"])

        if not series_list:
            raise ValueError("没有有效的基础因子，无法构建特征矩阵")

        X_df = pd.DataFrame({name: s for name, s in zip(feature_names, series_list)})

        if self.return_values is None:
            raise ValueError("没有收益率数据，无法构建目标变量")

        y = self.return_values

        # 对齐索引
        combined = X_df.copy()
        combined["__target__"] = y
        combined = combined.dropna()

        if len(combined) < 50:
            raise ValueError(f"有效数据点不足({len(combined)})，至少需要 50 个")

        X_df = combined[feature_names]
        y = combined["__target__"]

        return X_df, y, feature_names

    def _compute_feature_importance(self) -> Tuple[Dict[str, float], List[str]]:
        """计算特征重要性并选择 Top-K 特征

        Returns:
            (importance_dict, selected_features) 重要性字典和选中的特征列表
        """
        model_type = self._resolve_tree_model()
        logger.info(f"开始计算特征重要性, 使用 {model_type}...")

        self._report_progress("feature_importance", 0, 3, f"构建特征矩阵 ({model_type})")

        X_df, y, feature_names = self._build_feature_matrix()
        n_features = len(feature_names)

        logger.info(f"特征矩阵: {X_df.shape[0]} 样本 x {n_features} 特征")

        self._report_progress("feature_importance", 1, 3, "训练树模型")

        try:
            if model_type == "lightgbm":
                importance_dict = self._train_lgb_feature_importance(X_df, y, feature_names)
            else:
                importance_dict = self._train_xgb_feature_importance(X_df, y, feature_names)
        except Exception as e:
            logger.warning(f"树模型训练失败: {e}，回退使用全部特征")
            # 回退：使用所有特征
            importance_dict = {name: 1.0 / n_features for name in feature_names}
            selected = feature_names
            self._feature_importance = importance_dict
            self._selected_features = selected
            return importance_dict, selected

        self._report_progress("feature_importance", 2, 3, "选择 Top-K 特征")

        # 过滤低重要性特征
        filtered = {
            name: imp for name, imp in importance_dict.items()
            if imp >= self.importance_threshold
        }

        if not filtered:
            logger.warning(
                f"所有特征重要性均低于阈值 {self.importance_threshold}，"
                "回退使用全部特征"
            )
            filtered = importance_dict

        # 确定 Top-K
        effective_k = self.top_k if self.top_k > 0 else min(10, n_features)
        effective_k = min(effective_k, len(filtered))

        # 按重要性降序排列，取 Top-K
        sorted_features = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in sorted_features[:effective_k]]
        _selected_importance = {name: imp for name, imp in sorted_features[:effective_k]}

        self._feature_importance = importance_dict
        self._selected_features = selected

        logger.info(
            f"特征重要性计算完成: {n_features} → {len(selected)} 个特征被选中 (Top-{effective_k})"
        )
        for name, imp in sorted_features[:effective_k]:
            factor_code = self.base_factor_values.get(name, {}).get("code", name)
            logger.info(f"  {factor_code}: importance={imp:.6f}")

        self._report_progress("feature_importance", 3, 3, "特征选择完成")

        return importance_dict, selected

    def _train_lgb_feature_importance(
        self, X_df: pd.DataFrame, y: pd.Series, feature_names: List[str]
    ) -> Dict[str, float]:
        """使用 LightGBM 计算特征重要性"""
        train_data = lgb.Dataset(X_df.values, label=y.values, feature_name=feature_names)

        params = {
            "objective": "regression",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "verbose": -1,
            "seed": 42,
            "num_threads": 1,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
        )

        raw_importance = model.feature_importance(importance_type=self.importance_type)
        total = raw_importance.sum()

        if total == 0:
            return {name: 0.0 for name in feature_names}

        importance_dict = {
            name: float(imp / total)
            for name, imp in zip(feature_names, raw_importance)
        }

        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _train_xgb_feature_importance(
        self, X_df: pd.DataFrame, y: pd.Series, feature_names: List[str]
    ) -> Dict[str, float]:
        """使用 XGBoost 计算特征重要性"""
        dtrain = xgb.DMatrix(X_df.values, label=y.values, feature_names=feature_names)

        params = {
            "objective": "reg:squarederror",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "verbosity": 0,
            "seed": 42,
            "nthread": 1,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )

        raw_importance = model.get_score(importance_type=self.importance_type)

        # XGBoost 可能不返回所有特征（重要性为0的被省略），补全
        importance_dict = {}
        for name in feature_names:
            importance_dict[name] = float(raw_importance.get(name, 0.0))

        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {name: imp / total for name, imp in importance_dict.items()}

        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Phase 2: 下游符号回归
    # ------------------------------------------------------------------

    def _get_selected_factor_codes(self, selected_features: List[str]) -> List[str]:
        """将选中的特征变量名映射回原始因子代码"""
        factor_codes = []
        for var_name in selected_features:
            info = self.base_factor_values.get(var_name)
            if info is not None:
                factor_codes.append(info["code"])
        return factor_codes

    def _run_downstream_genetic(self, selected_factor_codes: List[str]) -> Dict:
        """运行 DEAP 遗传规划下游"""
        from backend.services.genetic_factor_mining_service import (
            create_genetic_mining_service,
            DEAP_AVAILABLE,
        )

        if not DEAP_AVAILABLE:
            logger.warning("DEAP 不可用，跳过遗传规划")
            return {"success": False, "message": "DEAP库未安装", "best_factors": []}

        logger.info(f"启动下游 DEAP 遗传规划, 使用 {len(selected_factor_codes)} 个预筛选特征...")

        service = create_genetic_mining_service(
            base_factors=selected_factor_codes,
            data=self.data,
            return_column=self.return_column,
            factor_calculator=self.factor_calculator,
            max_eval_stocks=self.max_eval_stocks,
            **self._gp_params,
        )

        # 传递股票池
        if self.stock_codes:
            service.set_stock_pool(
                self.stock_codes,
                list(self.stock_pool_data.values())[0].index[0].strftime("%Y-%m-%d") if self.stock_pool_data else "",
                list(self.stock_pool_data.values())[0].index[-1].strftime("%Y-%m-%d") if self.stock_pool_data else "",
            )

        # 设置进度回调（映射到 30-100%）
        if self.progress_callback:
            def gp_progress(gen, total_gen, best_fitness, avg_fitness):
                self._report_progress(
                    "symbolic_regression",
                    gen,
                    total_gen,
                    f"GP Gen {gen}/{total_gen}, best={best_fitness:.4f}"
                )
            service.set_progress_callback(gp_progress)

        result = service.mine_factors()
        result["source"] = "tree_prescreen/genetic"
        return result

    def _run_downstream_pysr(self, selected_factor_codes: List[str]) -> Dict:
        """运行 PySR 符号回归下游"""
        from backend.services.pysr_factor_mining_service import (
            create_pysr_mining_service,
            PYSR_AVAILABLE,
        )

        if not PYSR_AVAILABLE:
            logger.warning("PySR 不可用，跳过符号回归")
            return {"success": False, "message": "PySR库未安装", "best_factors": []}

        logger.info(f"启动下游 PySR 符号回归, 使用 {len(selected_factor_codes)} 个预筛选特征...")

        service = create_pysr_mining_service(
            base_factors=selected_factor_codes,
            data=self.data,
            return_column=self.return_column,
            factor_calculator=self.factor_calculator,
            max_eval_stocks=self.max_eval_stocks,
            **self._pysr_params,
        )

        # 传递股票池
        if self.stock_codes:
            service.set_stock_pool(
                self.stock_codes,
                list(self.stock_pool_data.values())[0].index[0].strftime("%Y-%m-%d") if self.stock_pool_data else "",
                list(self.stock_pool_data.values())[0].index[-1].strftime("%Y-%m-%d") if self.stock_pool_data else "",
            )

        # 设置进度回调
        if self.progress_callback:
            def pysr_progress(iteration, total_iter, best_fitness, avg_fitness):
                self._report_progress(
                    "symbolic_regression",
                    iteration,
                    total_iter,
                    f"PySR iter {iteration}/{total_iter}"
                )
            service.set_progress_callback(pysr_progress)

        result = service.mine_factors()
        result["source"] = "tree_prescreen/pysr"
        return result

    # ------------------------------------------------------------------
    # 结果合并与格式化
    # ------------------------------------------------------------------

    def _format_result(
        self,
        downstream_result: Dict,
        feature_importance: Dict[str, float],
        selected_features: List[str],
    ) -> Dict:
        """格式化最终输出结果

        统一输出格式，与 GeneticFactorMiningService / PySRFactorMiningService 保持一致
        """
        best_factors = downstream_result.get("best_factors", [])

        # 将特征变量名映射回因子代码
        selected_factor_codes = self._get_selected_factor_codes(selected_features)

        # 特征重要性也映射回因子代码
        importance_with_codes = {}
        for var_name, imp in feature_importance.items():
            info = self.base_factor_values.get(var_name)
            if info is not None:
                importance_with_codes[info["code"]] = imp

        # 构建适应度历史
        fitness_history = downstream_result.get("fitness_history", {"best": [], "average": []})
        if not isinstance(fitness_history, dict):
            fitness_history = {"best": [], "average": []}

        # 添加树模型预筛选阶段的适应度占位
        phase1_best = [0.0] * 3  # Phase 1 占 3 步
        phase1_avg = [0.0] * 3

        downstream_best = fitness_history.get("best", [])
        downstream_avg = fitness_history.get("average", [])

        merged_best = phase1_best + downstream_best
        merged_avg = phase1_avg + downstream_avg

        return {
            "success": downstream_result.get("success", True),
            "best_factors": best_factors,
            "feature_importance": importance_with_codes,
            "selected_features": selected_factor_codes,
            "fitness_history": {
                "best": merged_best,
                "average": merged_avg,
            },
            "downstream_result": downstream_result,
            "source": downstream_result.get("source", "tree_prescreen"),
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def mine_factors(self) -> Dict:
        """执行树模型预筛选 + 符号回归因子挖掘

        两阶段流程：
          Phase 1 (0-30%): 树模型特征重要性计算 → Top-K 选择
          Phase 2 (30-100%): 下游符号回归 (GP / PySR / Dual)

        Returns:
            dict with keys: success, best_factors, feature_importance,
            selected_features, fitness_history, downstream_result, source
        """
        if not TREE_PRESCREEN_AVAILABLE:
            return {
                "success": False,
                "message": "LightGBM 和 XGBoost 均未安装",
                "best_factors": [],
                "feature_importance": {},
                "selected_features": [],
                "fitness_history": {"best": [], "average": []},
            }

        logger.info("=" * 60)
        logger.info("开始树模型预筛选符号回归因子挖掘")
        logger.info(
            f"基础因子数: {len(self.base_factor_codes)}, "
            f"树模型: {self.tree_model}, "
            f"下游算法: {self.downstream_algorithm}"
        )
        logger.info("=" * 60)

        # ---- Phase 1: 特征重要性 ----
        try:
            feature_importance, selected_features = self._compute_feature_importance()
        except Exception as e:
            logger.error(f"特征重要性计算失败: {e}", exc_info=True)
            # 回退：使用全部特征
            feature_importance = {
                name: 1.0 / max(len(self.base_factor_values), 1)
                for name in self.base_factor_values
            }
            selected_features = list(self.base_factor_values.keys())
            logger.warning(f"回退使用全部 {len(selected_features)} 个特征")

        selected_factor_codes = self._get_selected_factor_codes(selected_features)

        if not selected_factor_codes:
            logger.warning("没有有效的预筛选特征，无法执行下游符号回归")
            return {
                "success": True,
                "best_factors": [],
                "feature_importance": feature_importance,
                "selected_features": [],
                "fitness_history": {"best": [], "average": []},
                "source": "tree_prescreen",
            }

        logger.info(
            f"Phase 1 完成: {len(self.base_factor_codes)} → {len(selected_factor_codes)} 个特征"
        )

        # 取消检查
        if self._cancel_flag:
            logger.info("树模型预筛选在Phase 1后被用户取消")
            return {
                "success": True,
                "best_factors": [],
                "feature_importance": feature_importance,
                "selected_features": selected_features,
                "fitness_history": {"best": [], "average": []},
                "source": "tree_prescreen",
                "cancelled": True,
            }

        # ---- Phase 2: 下游符号回归 ----
        logger.info(f"Phase 2: 启动下游 {self.downstream_algorithm} 符号回归...")

        try:
            if self.downstream_algorithm == "genetic":
                downstream_result = self._run_downstream_genetic(selected_factor_codes)
            elif self.downstream_algorithm == "pysr":
                downstream_result = self._run_downstream_pysr(selected_factor_codes)
            else:
                logger.warning(
                    f"未知下游算法 '{self.downstream_algorithm}'，回退到 genetic"
                )
                downstream_result = self._run_downstream_genetic(selected_factor_codes)
        except Exception as e:
            logger.error(f"下游符号回归失败: {e}", exc_info=True)
            downstream_result = {
                "success": False,
                "message": str(e),
                "best_factors": [],
            }

        # ---- 格式化结果 ----
        result = self._format_result(downstream_result, feature_importance, selected_features)

        # ---- 日志汇总 ----
        n_factors = len(result.get("best_factors", []))
        logger.info("=" * 60)
        logger.info(f"树模型预筛选挖掘完成: 发现 {n_factors} 个因子")
        if n_factors > 0:
            top = result["best_factors"][0]
            fitness_val = top.get('fitness')
            complexity_val = top.get('complexity')
            logger.info(
                f"  Top-1: expression={top.get('expression', 'N/A')}, "
                f"fitness={fitness_val:.4f if fitness_val is not None else 'N/A'}, "
                f"complexity={complexity_val:.1f if complexity_val is not None else 'N/A'}"
            )
        logger.info("=" * 60)

        return result


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------

def create_tree_prescreen_mining_service(
    base_factors: List[str],
    data: pd.DataFrame,
    factor_calculator=None,
    **kwargs
) -> TreePrescreenMiningService:
    """创建配置好的 TreePrescreenMiningService 实例

    接受的关键字参数（转发到构造函数）：

    树模型参数:
    * ``tree_model`` – "auto" / "lightgbm" / "xgboost" (默认 "auto")
    * ``n_estimators`` – 树模型迭代次数 (默认 100)
    * ``max_depth`` – 树模型最大深度 (默认 5)
    * ``learning_rate`` – 学习率 (默认 0.1)
    * ``importance_type`` – 重要性类型 "gain" / "split" (默认 "gain")
    * ``top_k`` – 选中特征数，0=自动 (默认 0)
    * ``importance_threshold`` – 最低重要性阈值 (默认 0.01)

    下游算法参数:
    * ``downstream_algorithm`` – "genetic" / "pysr" (默认 "genetic")
    * DEAP GP 参数: population_size, n_generations, cx_prob, mut_prob 等
    * PySR 参数: pysr_niterations, pysr_populations 等
    """
    return TreePrescreenMiningService(
        base_factors=base_factors,
        data=data,
        factor_calculator=factor_calculator,
        **kwargs
    )
