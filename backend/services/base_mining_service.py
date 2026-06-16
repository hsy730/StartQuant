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
import os
import random
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

from backend.services.data_service import data_service  # noqa: E402
from backend.utils.safe_math import safe_divide, safe_ir  # noqa: E402

# ======================================================================
# 基础因子值 LRU 缓存（模块级，所有挖掘服务实例共享）
# ======================================================================
# 缓存同一 (date_range, factor_code) 的预计算结果，
# 避免同一次运行中相同股票+日期+因子的重复计算。
# key = (index_start, index_end, row_count, factor_code)
# value = pd.Series（因子值）

from collections import OrderedDict

_BASE_FACTOR_CACHE: OrderedDict[tuple, "pd.Series"] = OrderedDict()
_BASE_FACTOR_CACHE_MAX_SIZE = 256  # 覆盖 ~50股 × 5因子 + 余量

_cache_hits = 0
_cache_misses = 0


def _make_data_fingerprint(df: pd.DataFrame) -> str:
    """生成 DataFrame 的轻量指纹。

    对于 OHLCV 金融时序数据，相同的日期范围 + 相同行数 → 相同的确定性因子计算结果。
    """
    if df is None or len(df) == 0:
        return "empty"
    idx = df.index
    return f"{idx[0]}|{idx[-1]}|{len(idx)}"


def _get_cached_factor(
    df: pd.DataFrame, factor_code: str, calculator,
) -> Optional["pd.Series"]:
    """查询或计算并缓存基础因子值。

    Returns:
        因子值 Series（缓存命中的副本），或 None（计算失败/无有效值）
    """
    global _cache_hits, _cache_misses
    fp = (_make_data_fingerprint(df), factor_code)

    if fp in _BASE_FACTOR_CACHE:
        _BASE_FACTOR_CACHE.move_to_end(fp)  # 标记为最近使用
        _cache_hits += 1
        return _BASE_FACTOR_CACHE[fp].copy()  # 返回副本防止调用方就地修改

    # 缓存未命中，执行实际计算
    _cache_misses += 1
    result = calculator.calculate(df, factor_code)

    if result is not None and len(result.dropna()) > 0:
        _BASE_FACTOR_CACHE[fp] = result.copy()
        # LRU 淘汰
        while len(_BASE_FACTOR_CACHE) > _BASE_FACTOR_CACHE_MAX_SIZE:
            _BASE_FACTOR_CACHE.popitem(last=False)

    return result


def clear_base_factor_cache() -> None:
    """清空基础因子缓存（测试或内存压力时调用）"""
    global _cache_hits, _cache_misses
    _BASE_FACTOR_CACHE.clear()
    _cache_hits = 0
    _cache_misses = 0


def get_base_factor_cache_stats() -> dict:
    """返回缓存统计信息（用于监控和日志）"""
    return {
        "size": len(_BASE_FACTOR_CACHE),
        "max_size": _BASE_FACTOR_CACHE_MAX_SIZE,
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": (
            round(_cache_hits / max(_cache_hits + _cache_misses, 1), 4)
            if (_cache_hits + _cache_misses) > 0
            else 0.0
        ),
    }


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
        max_base_factors: int = 30,
    ):
        self.base_factor_codes = base_factors
        self.max_base_factors = max_base_factors

        # 因子池随机抽样：当因子数超过 max_base_factors 时，随机选取子集
        if max_base_factors > 0 and len(base_factors) > max_base_factors:
            self.base_factor_codes = random.sample(base_factors, max_base_factors)
            logger.info(
                f"[{self._service_name}] 因子池抽样: {len(base_factors)} → {max_base_factors} 个因子"
            )
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

        # 进度控制（必须在 _precompute_base_factors 之前初始化）
        self.progress_callback = None
        self._cancel_flag = False

        # 预计算基础因子值
        self.base_factor_values: Dict[str, dict] = {}
        self._precompute_base_factors()

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
        """预计算基础因子值（多线程并行，支持取消）"""
        if self.factor_calculator is None:
            from backend.services.factor_service import factor_service

            self.factor_calculator = factor_service.calculator

        n_factors = len(self.base_factor_codes)
        logger.info(
            f"[{self._service_name}] 预计算 {n_factors} 个基础因子..."
        )

        if n_factors == 0:
            logger.info(f"[{self._service_name}] 无基础因子，跳过预计算")
            return

        max_workers = min(n_factors, os.cpu_count() or 4)

        def _compute_one(idx_and_code):
            i, factor_code = idx_and_code
            # 检查取消标志
            if self._cancel_flag:
                return (i, "cancelled", factor_code, None, None, 0.0)
            t0 = time.time()
            try:
                fv = _get_cached_factor(self.data, factor_code, self.factor_calculator)
                elapsed = time.time() - t0
                if fv is not None and len(fv.dropna()) > 0:
                    var_name = self._make_var_name(i)
                    return (i, "ok", factor_code, var_name, fv, elapsed)
                else:
                    return (i, "empty", factor_code, None, None, elapsed)
            except Exception as e:
                elapsed = time.time() - t0
                return (i, "error", factor_code, None, None, elapsed, str(e))

        t_total_start = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_one, (i, fc)): (i, fc)
                for i, fc in enumerate(self.base_factor_codes)
            }
            for future in as_completed(futures):
                # 检查取消：如果已取消，不再等待剩余 future
                if self._cancel_flag:
                    logger.warning(
                        f"[{self._service_name}] 预计算被取消，已完成 {completed}/{n_factors}"
                    )
                    break
                result = future.result()
                completed += 1
                status = result[1]
                factor_code = result[2]
                elapsed = result[5]

                if status == "ok":
                    _, _, _, var_name, fv, _ = result
                    self.base_factor_values[var_name] = {
                        "code": factor_code,
                        "values": fv,
                    }
                    logger.info(
                        f"  [{completed}/{n_factors}] {factor_code}: "
                        f"{len(fv.dropna())} 个有效值 ({elapsed:.2f}s)"
                    )
                elif status == "empty":
                    logger.warning(
                        f"  [{completed}/{n_factors}] {factor_code}: "
                        f"无有效值 ({elapsed:.2f}s)"
                    )
                elif status == "cancelled":
                    logger.warning(
                        f"  [{completed}/{n_factors}] {factor_code}: 已取消"
                    )
                else:  # error
                    err_msg = result[6] if len(result) > 6 else "?"
                    logger.warning(
                        f"  [{completed}/{n_factors}] {factor_code}: "
                        f"计算出错 - {err_msg} ({elapsed:.2f}s)"
                    )

        total_elapsed = time.time() - t_total_start
        cache_stats = get_base_factor_cache_stats()
        logger.info(
            f"[{self._service_name}] 成功预计算 {len(self.base_factor_values)}/{n_factors} 个基础因子"
            f"（耗时 {total_elapsed:.1f}s，线程数={max_workers}，"
            f"缓存命中={cache_stats['hits']}，未命中={cache_stats['misses']}）"
        )

    # ------------------------------------------------------------------
    # 股票池管理
    # ------------------------------------------------------------------

    def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
        """设置股票池用于截面IC评估（多线程并行，支持取消）"""
        self.stock_codes = stock_codes
        raw_data = data_service.get_multiple_stocks_data(
            stock_codes, start_date, end_date
        )
        # 规则3：防御性copy，避免就地修改data_service返回的DataFrame
        self.stock_pool_data = {k: v.copy() for k, v in raw_data.items()}

        n_stocks = len(self.stock_pool_data)
        n_factors = len(self.base_factor_codes)
        logger.info(
            f"[{self._service_name}] 预计算股票池因子: {n_stocks} 只股票 × {n_factors} 个因子"
        )

        if self.factor_calculator is None:
            from backend.services.factor_service import factor_service

            self.factor_calculator = factor_service.calculator

        # 每只股票的因子计算独立，并行处理
        # 线程数基于股票数（而非因子数），因为每只股票的计算是独立的
        factor_workers = min(n_stocks, os.cpu_count() or 4, 8)

        if n_stocks == 0:
            logger.info(f"[{self._service_name}] 无股票数据，跳过股票池预计算")
            return

        def _compute_stock_factors(args):
            code, df = args
            if self._cancel_flag:
                return code, None, {}
            if "close" in df.columns:
                df["return"] = df["close"].pct_change().shift(-1)

            return_val = (
                df[self.return_column] if self.return_column in df.columns else None
            )

            stock_base_factors = {}
            for i, factor_code in enumerate(self.base_factor_codes):
                if self._cancel_flag:
                    break
                try:
                    fv = _get_cached_factor(df, factor_code, self.factor_calculator)
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

            return code, return_val, stock_base_factors

        t_start = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=factor_workers) as executor:
            futures = {
                executor.submit(_compute_stock_factors, (code, df)): code
                for code, df in self.stock_pool_data.items()
            }
            for future in as_completed(futures):
                if self._cancel_flag:
                    logger.warning(
                        f"[{self._service_name}] 股票池预计算被取消，已完成 {completed}/{n_stocks}"
                    )
                    break
                code, return_val, stock_base_factors = future.result()
                completed += 1
                self.stock_pool_return_values[code] = return_val
                self.stock_pool_base_factor_values[code] = stock_base_factors

                if completed % 10 == 0 or completed == n_stocks:
                    logger.info(
                        f"[{self._service_name}] 预计算进度: {completed}/{n_stocks} 只股票完成"
                    )

        elapsed = time.time() - t_start
        cache_stats = get_base_factor_cache_stats()
        logger.info(
            f"[{self._service_name}] 股票池预计算完成: {n_stocks} 只股票 × {n_factors} 个因子 "
            f"（耗时 {elapsed:.1f}s，缓存命中={cache_stats['hits']}，未命中={cache_stats['misses']}）"
        )

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
