"""
因子挖掘服务公共基类

抽取4个挖掘服务（遗传规划、GFlowNet、PySR、树模型预筛选）的公共逻辑：
- 基础因子预计算
- 股票池管理
- 进度控制
- 交叉验证惩罚
- 适应度路由
- Z-Score归一化
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

from backend.services.data_service import data_service  # noqa: E402
from backend.utils.safe_math import safe_divide, safe_ir  # noqa: E402


class BaseMiningService(ABC):
    """因子挖掘服务公共基类

    提供所有挖掘服务共享的基础设施：
    - 基础因子预计算与缓存
    - 股票池设置与评估样本刷新
    - 进度回调与取消控制
    - 适应度路由（IC/IR/Sharpe/Combined）
    - 交叉验证过拟合惩罚
    - 代际/批量 Z-Score 归一化

    子类应覆盖 ``_service_name`` 类属性以自定义日志前缀，
    并实现 ``mine_factors()`` 抽象方法。
    """

    # 子类应覆盖此属性以自定义日志前缀
    _service_name: str = "Mining"

    # Z-Score 先验冷启动常量（基于量化因子领域知识）
    # 注意：_PRIOR_IC_MEAN 是贝叶斯估计先验值，不是验证通过阈值（IC_PASS_THRESHOLD=0.02）
    _PRIOR_IC_MEAN = 0.03
    _PRIOR_IC_STD = 0.02
    _PRIOR_IR_MEAN = 0.5
    _PRIOR_IR_STD = 0.3

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        factor_calculator=None,
        max_eval_stocks: int = 50,
        fitness_objective: str = "ic_mean",
        cv_folds: int = 0,
        naming_pattern: str = "factor_{i}",
    ):
        self.base_factor_codes = base_factors
        self.data = data.copy() if data is not None else None
        self.return_column = return_column
        self.factor_calculator = factor_calculator
        self.max_eval_stocks = max_eval_stocks
        self.fitness_objective = fitness_objective
        self.cv_folds = cv_folds
        self.naming_pattern = naming_pattern

        self.return_values = (
            data[return_column].copy()
            if data is not None and return_column in data.columns
            else None
        )

        # 股票池
        self.stock_codes: List[str] = []
        self.stock_pool_data: Dict[str, pd.DataFrame] = {}
        self.stock_pool_return_values: Dict[str, pd.Series] = {}
        self.stock_pool_base_factor_values: Dict[str, dict] = {}
        self._sampled_stock_codes: List[str] = []

        # 预计算基础因子值
        self.base_factor_values: Dict[str, dict] = {}
        self._precompute_base_factors()

        # 进度控制
        self.progress_callback = None
        self._cancel_flag = False

        # Z-Score 归一化状态
        self._gen_ic_values: List[float] = []
        self._gen_ir_values: List[float] = []
        self._zscore_ic_mean: float = self._PRIOR_IC_MEAN
        self._zscore_ic_std: float = self._PRIOR_IC_STD
        self._zscore_ir_mean: float = self._PRIOR_IR_MEAN
        self._zscore_ir_std: float = self._PRIOR_IR_STD
        self._has_zscore_stats: bool = True

    # ------------------------------------------------------------------
    # 变量命名
    # ------------------------------------------------------------------

    def _make_var_name(self, index: int) -> str:
        """根据 naming_pattern 生成基础因子变量名

        默认 pattern "factor_{i}" 生成 factor_0, factor_1, ...
        PySR 使用 "x{i}" 生成 x0, x1, ...
        """
        return self.naming_pattern.format(i=index)

    # ------------------------------------------------------------------
    # 基础因子预计算
    # ------------------------------------------------------------------

    def _precompute_base_factors(self):
        """预计算基础因子值"""
        if self.factor_calculator is None:
            from backend.services.factor_service import factor_service

            self.factor_calculator = factor_service.calculator

        logger.info(
            f"[{self._service_name}] 预计算 {len(self.base_factor_codes)} 个基础因子..."
        )

        for i, factor_code in enumerate(self.base_factor_codes):
            try:
                fv = self.factor_calculator.calculate(self.data, factor_code)
                if fv is not None and len(fv.dropna()) > 0:
                    var_name = self._make_var_name(i)
                    self.base_factor_values[var_name] = {
                        "code": factor_code,
                        "values": fv,
                    }
                    logger.info(
                        f"  [{i + 1}/{len(self.base_factor_codes)}] {factor_code}: {len(fv.dropna())} 个有效值"
                    )
                else:
                    logger.warning(
                        f"  [{i + 1}/{len(self.base_factor_codes)}] {factor_code}: 计算失败或无有效值"
                    )
            except Exception as e:
                logger.warning(
                    f"  [{i + 1}/{len(self.base_factor_codes)}] {factor_code}: 计算出错 - {e}"
                )

        logger.info(
            f"[{self._service_name}] 成功预计算 {len(self.base_factor_values)} 个基础因子"
        )

    # ------------------------------------------------------------------
    # 股票池管理
    # ------------------------------------------------------------------

    def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
        """设置股票池用于截面IC评估"""
        self.stock_codes = stock_codes
        raw_data = data_service.get_multiple_stocks_data(
            stock_codes, start_date, end_date
        )
        # 规则3：防御性copy，避免就地修改data_service返回的DataFrame
        self.stock_pool_data = {k: v.copy() for k, v in raw_data.items()}

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
                        var_name = self._make_var_name(i)
                        stock_base_factors[var_name] = {
                            "code": factor_code,
                            "values": fv,
                        }
                except Exception as e:
                    logger.warning(
                        f"Stock {code} factor {factor_code} compute error: {e}"
                    )
            self.stock_pool_base_factor_values[code] = stock_base_factors

        self._refresh_stock_sample()
        logger.info(
            f"[{self._service_name}] 股票池已设置: {len(self.stock_pool_data)} 只股票, "
            f"评估样本={len(self._sampled_stock_codes)}"
        )

    def _refresh_stock_sample(self):
        """刷新评估样本（随机抽样）"""
        available = list(self.stock_pool_base_factor_values.keys())
        if len(available) <= self.max_eval_stocks:
            self._sampled_stock_codes = available
        else:
            self._sampled_stock_codes = random.sample(available, self.max_eval_stocks)

    # ------------------------------------------------------------------
    # 进度控制
    # ------------------------------------------------------------------

    def set_progress_callback(self, callback):
        """设置进度回调函数

        Args:
            callback: 签名为 callback(iteration, total_iterations, best_fitness, avg_fitness)
        """
        self.progress_callback = callback

    def request_cancel(self):
        """请求取消挖掘任务"""
        self._cancel_flag = True
        logger.info("收到取消请求，将在当前迭代结束后停止")

    # ------------------------------------------------------------------
    # 交叉验证过拟合惩罚
    # ------------------------------------------------------------------

    def _cv_penalty(self, factor_values_dict: Dict[str, pd.Series]) -> float:
        """计算交叉验证过拟合惩罚

        将时间序列分为 cv_folds 段，计算每段IC，
        返回 1.0 - (min_fold_ic / max_fold_ic)，值域 [0, 1]。
        IC一致的因子惩罚≈0，IC不稳定的因子惩罚→1。
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
                    ic_result = spearmanr(segment["factor"], segment["return"])
                    ic = ic_result[0]
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
    # 适应度路由
    # ------------------------------------------------------------------

    def _extract_best_ic_ir(self, ic_results: dict) -> Tuple[float, float]:
        """从IC分析结果中提取最优IC和IR

        Returns:
            (best_ic, best_ir)
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
                ir = safe_ir(float(mean_ic), float(std_ic), default=None)
                if mean_ic > best_ic:
                    best_ic = mean_ic
                if ir is not None and ir > best_ir:
                    best_ir = ir

        return best_ic, best_ir

    def _route_fitness(
        self,
        ic_results: dict,
        factor_values_dict: Optional[Dict[str, pd.Series]] = None,
    ) -> float:
        """根据 fitness_objective 选择适应度值

        支持的目标:
        - ic_mean: 最优绝对均值IC（默认）
        - ir_ratio: IC均值/IC标准差（信息比率）
        - sharpe: 类Sharpe比率（用IR代理）
        - combined: Z-Score归一化后的IC和IR加权组合

        对于 combined 模式，IC和IR先通过代际Z-Score归一化，
        使得60/40权重在不同量纲下仍然有效。

        公式:
            z_ic = clip((IC - μ_ic) / (σ_ic + ε), -3, 3)
            z_ir = clip((IR - μ_ir) / (σ_ir + ε), -3, 3)
            Norm(IC) = (z_ic + 3) / 6   → maps [-3σ, +3σ] to [0, 1]
            Norm(IR) = (z_ir + 3) / 6
            combined  = 0.6 * Norm(IC) + 0.4 * Norm(IR)

        统计量从前一代/迭代收集，通过 _update_zscore_stats() 更新。
        第一代使用先验冷启动值。
        """
        best_ic, best_ir = self._extract_best_ic_ir(ic_results)

        # 收集原始IC/IR用于代际Z-Score计算
        self._gen_ic_values.append(best_ic)
        self._gen_ir_values.append(best_ir)

        if self.fitness_objective == "ir_ratio":
            return best_ir
        elif self.fitness_objective == "sharpe":
            return best_ir
        elif self.fitness_objective == "combined":
            # Z-Score归一化使用上一代的统计量（含先验冷启动）
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
            # 映射 [-3, 3] → [0, 1]
            norm_ic = (z_ic + 3.0) / 6.0
            norm_ir = (z_ir + 3.0) / 6.0
            return 0.6 * norm_ic + 0.4 * norm_ir
        else:  # ic_mean (default)
            return best_ic

    # ------------------------------------------------------------------
    # Z-Score 归一化
    # ------------------------------------------------------------------

    def _update_zscore_stats(self):
        """从当前代/迭代收集的IC/IR值计算Z-Score归一化统计量

        要求至少5个有效值才计算稳定统计量。
        应用σ下界保护: max(σ, max(0.01*μ, 0.005)) 防止种群收敛时Z-Score爆炸。
        计算后清空收集列表，为下一代做准备。
        """
        valid_ic = [v for v in self._gen_ic_values if v > 1e-10]
        valid_ir = [v for v in self._gen_ir_values if v > 1e-10]

        if len(valid_ic) >= 5 and len(valid_ir) >= 5:
            ic_mean = float(np.mean(valid_ic))
            ic_std = float(np.std(valid_ic))
            ir_mean = float(np.mean(valid_ir))
            ir_std = float(np.std(valid_ir))

            # σ下界保护: 防止Z-Score爆炸
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

        # 清空为下一代做准备
        self._gen_ic_values = []
        self._gen_ir_values = []

    def _apply_batch_zscore(self, best_factors: List[Dict]) -> List[Dict]:
        """对combined适应度分数进行事后批量Z-Score归一化

        在所有方程评估完成后，从收集的IC/IR值计算Z-Score统计量，
        重新计算combined分数并重新排名。

        σ下界保护: max(σ, max(0.01*μ, 0.005)) 防止所有方程IC/IR相近时Z-Score爆炸。
        """
        if not best_factors or self.fitness_objective != "combined":
            return best_factors

        # 从best_factors中动态收集原始IC/IR（仅筛选后的子集）
        valid_ic = []
        valid_ir = []
        for factor_info in best_factors:
            validation = factor_info.get("validation", {})
            if not isinstance(validation, dict):
                continue
            raw_ic_val = validation.get("_raw_ic_mean")
            raw_ir_val = validation.get("_raw_ir")
            raw_ic = abs(raw_ic_val) if raw_ic_val is not None else 0.0
            raw_ir = abs(raw_ir_val) if raw_ir_val is not None else 0.0
            if raw_ic > 1e-10 and np.isfinite(raw_ic):
                valid_ic.append(raw_ic)
            if raw_ir > 1e-10 and np.isfinite(raw_ir):
                valid_ir.append(raw_ir)

        if len(valid_ic) < 2 or len(valid_ir) < 2:
            logger.warning(
                "Too few valid IC/IR values for batch Z-Score, keeping raw scores"
            )
            return best_factors

        ic_mean = float(np.mean(valid_ic))
        ic_std = float(np.std(valid_ic))
        ir_mean = float(np.mean(valid_ir))
        ir_std = float(np.std(valid_ir))

        # σ下界保护
        ic_std = max(ic_std, max(0.01 * ic_mean, 0.005))
        ir_std = max(ir_std, max(0.01 * ir_mean, 0.005))

        logger.info(
            f"Batch Z-Score stats: IC μ={ic_mean:.4f} σ={ic_std:.4f}, "
            f"IR μ={ir_mean:.4f} σ={ir_std:.4f} (from {len(valid_ic)} equations)"
        )

        # 使用Z-Score归一化重新计算combined适应度
        for factor_info in best_factors:
            validation = factor_info.get("validation", {})
            if not isinstance(validation, dict):
                continue

            raw_ic_val = validation.get("_raw_ic_mean")
            raw_ir_val = validation.get("_raw_ir")
            raw_ic = abs(raw_ic_val) if raw_ic_val is not None else 0.0
            raw_ir = abs(raw_ir_val) if raw_ir_val is not None else 0.0

            z_ic = max(
                -3.0,
                min(
                    safe_divide(float(raw_ic - ic_mean), float(ic_std), default=0.0),
                    3.0,
                ),
            )
            z_ir = max(
                -3.0,
                min(
                    safe_divide(float(raw_ir - ir_mean), float(ir_std), default=0.0),
                    3.0,
                ),
            )
            norm_ic = (z_ic + 3.0) / 6.0
            norm_ir = (z_ir + 3.0) / 6.0
            factor_info["fitness"] = 0.6 * norm_ic + 0.4 * norm_ir

            # 更新validation score以匹配
            if "score" in validation:
                validation["score"] = max(0.0, min(factor_info["fitness"] * 100, 100.0))

        # 按更新后的适应度重新排名
        best_factors.sort(key=lambda f: f.get("fitness", 0), reverse=True)
        for i, fi in enumerate(best_factors):
            fi["rank"] = i + 1

        return best_factors

    # ------------------------------------------------------------------
    # 内存管理
    # ------------------------------------------------------------------

    def release_memory(self):
        """释放大对象占用的内存，在挖掘任务完成后调用

        设计意图：
        - 挖掘任务完成后，服务实例仍被 mining_services 字典持有引用，
          直到 _cleanup_old_tasks() 清理（最长24小时TTL）。
        - 如果不主动释放，每个完成的任务实例会持续占用 15-20MB 内存，
          100个并发任务可累积 1.5GB。
        - 调用此方法后，服务实例不可复用（data/return_values 被设为 None）。
        - 调用方：mining.py _run_mining() 的 finally 块。
        """
        self.stock_pool_data = {}
        self.stock_pool_base_factor_values = {}
        self.stock_pool_return_values = {}
        self.data = None  # 注意：设为 None 后 _compute_factor_expression 等方法不可用
        self.return_values = None
        self.base_factor_values = {}
        self._sampled_stock_codes = []
        logger.info(f"[{self._service_name}] 内存已释放")

    # ------------------------------------------------------------------
    # 抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def mine_factors(self) -> Dict:
        """执行因子挖掘（子类必须实现）"""
        ...
