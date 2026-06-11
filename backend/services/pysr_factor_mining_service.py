"""
PySR符号回归因子挖掘服务

使用PySR (Miles Cranmer) 进行符号回归，相比DEAP GP:
- Julia后端多线程并行，速度提升10-100x
- 内置复杂度控制: maxsize, maxdepth, constraints, nested_constraints
- Curriculum Learning: 从简单方程逐步增加复杂度
- Hall of Fame多样性保证: 自动去重
- 多目标优化内建: Pareto frontier直接输出
- Feynman benchmark: 59% solve rate (vs gplearn/DEAP 20%)
"""
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pysr
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    logger.warning("PySR库未安装，符号回归功能将不可用。请运行: pip install pysr")

from backend.services.base_mining_service import BaseMiningService
from backend.utils.safe_math import safe_divide
from backend.services.factor_validation_service import factor_validation_service
from backend.services.alphalens_analysis_service import alphalens_analysis_service


_PYSR_OPERATOR_MAP = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "neg": "-",
    "abs": "abs",
    "log": "log",
    "sqrt": "sqrt",
}


class PySRFactorMiningService(BaseMiningService):
    """PySR符号回归因子挖掘服务

    Uses PySR's Julia-backed symbolic regression engine to discover
    factor expressions.  The service mirrors the interface of
    :class:`GeneticFactorMiningService` so that both can be used
    interchangeably or in parallel via :class:`DualMiningService`.
    """

    _service_name = "PySR"

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        factor_calculator=None,
        max_eval_stocks: int = 50,
        # ---- PySR core parameters ----
        niterations: int = 40,
        populations: int = 30,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        maxsize: int = 30,
        maxdepth: int = 5,
        constraints: Optional[Dict] = None,
        nested_constraints: Optional[Dict] = None,
        parsimony: float = 0.0032,
        procs: int = 8,
        population_size: int = 33,
        # ---- Fitness ----
        fitness_objective: str = "ic_mean",
        # ---- Cross-validation ----
        cv_folds: int = 0,
    ):
        if not PYSR_AVAILABLE:
            raise ImportError("PySR库未安装，请运行: pip install pysr")

        super().__init__(
            base_factors=base_factors,
            data=data,
            return_column=return_column,
            factor_calculator=factor_calculator,
            max_eval_stocks=max_eval_stocks,
            fitness_objective=fitness_objective,
            cv_folds=cv_folds,
            naming_pattern="x{i}",
        )

        self.niterations = niterations
        self.populations = populations
        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        self.unary_operators = unary_operators or ["neg", "abs", "log", "sqrt"]
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.constraints = constraints or {"/": (-1, 9)}
        self.nested_constraints = nested_constraints
        self.parsimony = parsimony
        self.procs = procs
        self.population_size = population_size

        self._current_iteration = 0
        self._total_iterations = niterations

    def _build_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build the feature matrix X and target vector y for PySR.

        Returns (X, y, feature_names) where X is a 2-D numpy array with
        rows = time steps and columns = base factors, y is the return
        series, and feature_names lists the column labels.
        """
        feature_names = []
        series_list = []

        for var_name in sorted(self.base_factor_values.keys()):
            info = self.base_factor_values[var_name]
            feature_names.append(var_name)
            series_list.append(info["values"])

        if not series_list:
            raise ValueError("No valid base factors to build feature matrix")

        combined = pd.DataFrame({name: s for name, s in zip(feature_names, series_list)})

        if self.return_values is not None:
            combined["__target__"] = self.return_values
        else:
            raise ValueError("No return values available for target")

        combined = combined.dropna()
        if len(combined) < 50:
            raise ValueError(f"Not enough valid data points ({len(combined)}), need at least 50")

        X = combined[feature_names].values
        y = combined["__target__"].values

        return X, y, feature_names

    def _build_cross_sectional_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """Build feature matrix from cross-sectional (multi-stock) data.

        Returns (X, y, feature_names, stock_factor_map) where
        stock_factor_map maps stock_code -> {var_name: pd.Series}.
        """
        feature_names = sorted(
            self.stock_pool_base_factor_values.get(
                self._sampled_stock_codes[0], {}
            ).keys()
        ) if self._sampled_stock_codes else []

        if not feature_names:
            return self._build_feature_matrix()

        all_X = []
        all_y = []
        stock_factor_map = {}

        for code in self._sampled_stock_codes:
            base_factors = self.stock_pool_base_factor_values.get(code, {})
            ret = self.stock_pool_return_values.get(code)
            if ret is None:
                continue

            stock_data = {}
            for var_name in feature_names:
                info = base_factors.get(var_name)
                if info is not None:
                    stock_data[var_name] = info["values"]

            if len(stock_data) != len(feature_names):
                continue

            df = pd.DataFrame(stock_data)
            df["__target__"] = ret
            df = df.dropna()

            if len(df) < 20:
                continue

            all_X.append(df[feature_names].values)
            all_y.append(df["__target__"].values)
            stock_factor_map[code] = stock_data

        if not all_X:
            return self._build_feature_matrix()

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        return X, y, feature_names, stock_factor_map

    def _pysr_expr_to_factor_code(self, expr_str: str, feature_names: List[str]) -> str:
        """Convert a PySR expression string to a factor code string.

        Replaces x0, x1, ... with the actual factor codes and maps PySR
        operator names to the project's factor expression format.
        """
        result = expr_str

        for var_name in sorted(feature_names, key=len, reverse=True):
            _idx = int(var_name.replace("x", ""))
            factor_code = self.base_factor_values.get(var_name, {}).get("code", var_name)
            result = result.replace(var_name, f"({factor_code})")

        result = result.replace("np.", "")

        return result

    def _evaluate_pysr_factor(self, expr_callable, stock_factor_map: Dict) -> Tuple[float, dict]:
        """Evaluate a PySR-discovered factor across stocks using IC.

        Returns (fitness, validation_dict) where validation uses the same
        format as factor_validation_service.validate_factor() for consistency.

        In cross-sectional mode (multi-stock), uses alphalens IC/IR which is
        the correct approach for factor evaluation. Falls back to single-stock
        validation only when alphalens is unavailable.
        """
        factor_values_dict: Dict[str, pd.Series] = {}

        for code, stock_data in stock_factor_map.items():
            try:
                ordered_values = []
                for var_name in sorted(stock_data.keys()):
                    ordered_values.append(stock_data[var_name].values)

                X_stock = np.column_stack(ordered_values)
                fv_array = expr_callable(X_stock)
                ret = self.stock_pool_return_values.get(code)
                if ret is None:
                    continue

                idx = ret.index[:len(fv_array)]
                fv = pd.Series(fv_array, index=idx)
                fv = fv.replace([np.inf, -np.inf], np.nan)
                if fv.notna().sum() < 10:
                    continue

                factor_values_dict[code] = fv.dropna()
            except Exception as e:
                logger.debug(f"PySR factor eval failed for {code}: {e}")
                continue

        logger.info(f"[PySR Eval] Built factor_values_dict with {len(factor_values_dict)} stocks")

        if len(factor_values_dict) < 2:
            return 0.0, {}

        fitness = 0.0
        validation = {}
        best_fv = None
        best_ret = None
        alphalens_success = False

        # 提前设置 best_fv/best_ret 以便回退使用
        if factor_values_dict:
            first_code = list(factor_values_dict.keys())[0]
            best_fv = factor_values_dict.get(first_code)
            best_ret = self.stock_pool_return_values.get(first_code)

        logger.info("[PySR Eval] Attempting alphalens cross-sectional analysis...")

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

            if factor_data is not None and not factor_data.empty:
                logger.info("[PySR Eval] Factor data prepared, calling analyze_ic...")
                ic_results = alphalens_analysis_service.analyze_ic(factor_data)
                logger.info(f"[PySR Eval] IC results: {list(ic_results.keys())}")
                if "error" not in ic_results:
                    fitness = self._route_fitness(ic_results)

                    # 正确提取 IC/IR：从 pearson_ic 或 spearman_ic 的第一个周期中提取
                    ic_mean_val = 0.0
                    ir_val = 0.0

                    for ic_type in ["pearson_ic", "spearman_ic"]:
                        ic_type_data = ic_results.get(ic_type, {})
                        if ic_type_data:
                            first_period = list(ic_type_data.keys())[0] if ic_type_data else None
                            if first_period:
                                period_stats = ic_type_data[first_period]
                                ic_mean_val = float(period_stats.get("mean_ic", 0))
                                ir_raw = period_stats.get("ir")
                                ir_val = float(ir_raw) if ir_raw is not None else 0.0
                                break

                    _stability = float(ic_results.get("stability", 0))

                    ir_capped = min(ir_val, 5.0)

                    validation = {
                        "ic_validation": {
                            "ic": abs(ic_mean_val),
                            "passed": abs(ic_mean_val) >= 0.02,
                        },
                        "ir_validation": {
                            "ir": ir_capped,
                            "passed": ir_capped >= 0.3,
                        },
                        "score": max(0.0, fitness * 100),
                        "_raw_ic_mean": ic_mean_val,
                        "_raw_ir": ir_val,
                        "_alphalens": True,
                    }
                    alphalens_success = True
                    logger.info(f"[PySR Eval] ✅ Alphalens SUCCESS: IC={ic_mean_val:.4f}, IR={ir_val:.4f} (capped={ir_capped:.4f})")
        except Exception as e:
            logger.warning(f"[PySR Eval] ❌ Alphalens FAILED: {e}")

        if not alphalens_success:
            logger.info("[PySR Eval] Attempting fallback to single-stock validation...")
            logger.info(f"[PySR Eval] best_fv={'available' if best_fv is not None else 'None'}, best_ret={'available' if best_ret is not None else 'None'}")

            if best_fv is not None and best_ret is not None and self.return_values is not None:
                try:
                    validation = factor_validation_service.validate_factor(
                        factor_values=best_fv,
                        return_values=best_ret,
                        existing_factors=None,
                    )
                    logger.info(f"[PySR Eval] ✅ Fallback validation SUCCESS: {list(validation.keys())}")
                except Exception as e:
                    logger.debug(f"PySR validation failed: {e}")
                    validation = {}
            else:
                logger.warning("[PySR Eval] ⚠️ No fallback possible: best_fv or best_ret is None")

        logger.info(f"[PySR Eval] Returning: fitness={fitness:.4f}, validation_keys={list(validation.keys())}")
        return fitness, validation

    def _evaluate_pysr_factor_single(self, expr_callable) -> Tuple[float, dict]:
        """Evaluate a PySR-discovered factor on single stock using validation score.

        Returns (fitness, validation_dict).
        """
        feature_names = sorted(self.base_factor_values.keys())
        ordered_values = []
        for var_name in feature_names:
            info = self.base_factor_values[var_name]
            ordered_values.append(info["values"].values)

        X = np.column_stack(ordered_values)
        fv_array = expr_callable(X)

        idx = self.base_factor_values[feature_names[0]]["values"].index[:len(fv_array)]
        fv = pd.Series(fv_array, index=idx)
        fv = fv.replace([np.inf, -np.inf], np.nan)

        if fv.notna().sum() < 10:
            return 0.0, {}

        if self.return_values is not None:
            validation = factor_validation_service.validate_factor(
                factor_values=fv,
                return_values=self.return_values,
                existing_factors=None,
            )
            fitness = validation.get("score", 0) / 100.0
            return fitness, validation
        else:
            # 无收益率数据时，使用变异系数(CV)作为代理适应度
            cv_value = safe_divide(float(fv.std()), abs(float(fv.mean())), default=None)
            if np.isfinite(cv_value) and abs(fv.mean()) > 1e-8:
                fitness = cv_value
            else:
                # 均值接近0或CV无效时，无法评估，设为0避免误导
                logger.warning("无收益率数据且CV代理无效，适应度设为0")
                fitness = 0.0
            return fitness, {}

    def _route_fitness(self, ic_results: dict, factor_values_dict=None) -> float:
        """Select the fitness value according to ``self.fitness_objective``.

        For ``combined``, returns a raw weighted score as a placeholder; the actual
        Z-Score normalization is applied post-hoc in ``_apply_batch_zscore`` using
        only the filtered best_factors subset.

        For other objectives, returns the raw metric directly.
        """
        # Call super to update zscore stats (collects IC/IR values for _update_zscore_stats)
        super()._route_fitness(ic_results, factor_values_dict)

        best_ic, best_ir = self._extract_best_ic_ir(ic_results)

        if self.fitness_objective == "ir_ratio":
            return best_ir
        elif self.fitness_objective == "sharpe":
            return best_ir
        elif self.fitness_objective == "combined":
            # Placeholder: raw weighted score (will be re-ranked by batch Z-Score)
            return 0.6 * best_ic + 0.4 * best_ir
        else:
            return best_ic

    def cleanup(self):
        """释放PySR模型和Julia子进程资源"""
        try:
            if hasattr(self, '_pysr_model') and self._pysr_model is not None:
                # PySR doesn't have explicit cleanup, but we can release the reference
                self._pysr_model = None
            self._halloffame = None
        except Exception as e:
            logger.debug(f"PySR cleanup error: {e}")

    def mine_factors(self) -> Dict:
        """Execute PySR-based symbolic regression factor mining.

        Returns
        -------
        dict with keys: ``success``, ``best_factors``, ``equations``,
        ``source``
        """
        if not PYSR_AVAILABLE:
            return {"success": False, "message": "PySR库未安装", "best_factors": []}

        logger.info("开始PySR符号回归因子挖掘...")
        logger.info(
            f"参数: niterations={self.niterations}, populations={self.populations}, "
            f"maxsize={self.maxsize}, maxdepth={self.maxdepth}, parsimony={self.parsimony}"
        )

        try:
            use_cross_sectional = len(self.stock_pool_data) >= 2

            if use_cross_sectional:
                X, y, feature_names, stock_factor_map = self._build_cross_sectional_matrix()
            else:
                X, y, feature_names = self._build_feature_matrix()
                stock_factor_map = None

            logger.info(f"Feature matrix shape: {X.shape}, target shape: {y.shape}")

            self._pysr_model = pysr.PySRRegressor(
                niterations=self.niterations,
                populations=self.populations,
                binary_operators=self.binary_operators,
                unary_operators=self.unary_operators,
                maxsize=self.maxsize,
                maxdepth=self.maxdepth,
                constraints=self.constraints,
                nested_constraints=self.nested_constraints,
                parsimony=self.parsimony,
                procs=self.procs,
                population_size=self.population_size,
                progress=True,
                temp_equation_file=True,
                verbosity=1,
                random_state=42,
            )

            self._pysr_model.fit(X, y)
            self._current_iteration = self.niterations

            # 取消检查（PySR的fit无法中断，但可以在后处理前停止）
            if self._cancel_flag:
                logger.info("PySR fit完成后，用户已取消，跳过结果处理")
                return {"success": True, "best_factors": [], "equations": None, "source": "pysr", "cancelled": True}

            if self.progress_callback:
                self.progress_callback(self.niterations, self.niterations, 0.0, 0.0)

            equations_df = self._pysr_model.equations_
            if equations_df is None or len(equations_df) == 0:
                logger.warning("PySR未发现有效方程")
                return {"success": True, "best_factors": [], "equations": None, "source": "pysr"}

            best_factors = []
            for idx, row in equations_df.iterrows():
                sympy_expr = str(row.get("sympy_format", row.get("equation", "")))
                complexity = int(row.get("complexity", len(sympy_expr)))
                loss = float(row.get("loss", 1.0))
                score = float(row.get("score", 0.0))

                factor_code = self._pysr_expr_to_factor_code(sympy_expr, feature_names)

                fitness = 0.0
                validation = {}

                try:
                    expr_callable = row.get("lambda_format")
                    if expr_callable is not None:
                        if use_cross_sectional and stock_factor_map:
                            fitness, validation = self._evaluate_pysr_factor(expr_callable, stock_factor_map)
                        else:
                            fitness, validation = self._evaluate_pysr_factor_single(expr_callable)
                except Exception as e:
                    logger.warning(f"[Mine Factors] ❌ Equation {idx} evaluation FAILED: {e}")
                    import traceback
                    logger.warning(f"[Mine Factors] Traceback: {traceback.format_exc()}")

                factor_info = {
                    "rank": idx + 1,
                    "expression": factor_code,
                    "placeholder_expression": sympy_expr,
                    "fitness": fitness,
                    "complexity": float(complexity),
                    "pysr_loss": loss,
                    "pysr_score": score,
                    "source": "pysr",
                }

                if validation:
                    factor_info["validation"] = validation

                best_factors.append(factor_info)

            def _sort_key(f):
                v = f.get("validation", {})
                if v and isinstance(v, dict):
                    return v.get("score", f.get("fitness", 0))
                return f.get("fitness", 0)

            best_factors.sort(key=_sort_key, reverse=True)

            # Post-hoc batch Z-Score normalization for combined objective
            if self.fitness_objective == "combined":
                best_factors = self._apply_batch_zscore(best_factors)

            for i, fi in enumerate(best_factors):
                fi["rank"] = i + 1

            logger.info(f"PySR discovered {len(best_factors)} factor expressions")

            return {
                "success": True,
                "best_factors": best_factors,
                "equations": equations_df[[col for col in equations_df.columns if equations_df[col].apply(lambda x: not callable(x)).all()]].to_dict() if equations_df is not None else {},
                "source": "pysr",
            }

        except Exception as e:
            logger.error(f"PySR mining failed: {e}", exc_info=True)
            return {"success": False, "message": str(e), "best_factors": []}


def create_pysr_mining_service(
    base_factors: List[str],
    data: pd.DataFrame,
    factor_calculator=None,
    **kwargs
) -> PySRFactorMiningService:
    """Create a configured :class:`PySRFactorMiningService` instance.

    Accepted keyword arguments (forwarded to the constructor):

    * ``niterations`` – number of PySR iterations (default 40)
    * ``populations`` – number of parallel populations (default 30)
    * ``binary_operators`` – list of binary operators (default ["+","-","*","/"])
    * ``unary_operators`` – list of unary operators (default ["neg","abs","log","sqrt"])
    * ``maxsize`` – max expression size (default 30)
    * ``maxdepth`` – max expression depth (default 5)
    * ``constraints`` – operator constraints dict (default {"/": (-1, 9)})
    * ``nested_constraints`` – nested operator constraints (default None)
    * ``parsimony`` – parsimony coefficient (default 0.0032)
    * ``procs`` – number of Julia processes (default 8)
    * ``population_size`` – individuals per population (default 33)
    * ``fitness_objective`` – ic_mean / ir_ratio / sharpe / combined (default ic_mean)
    * ``cv_folds`` – cross-validation folds (default 0)
    """
    return PySRFactorMiningService(
        base_factors=base_factors,
        data=data,
        factor_calculator=factor_calculator,
        **kwargs
    )
