"""
因子编排器（Factor Orchestrator）— AlphaMiner 式一键验证流水线

将分散的因子研究步骤编排为一条自动化流水线：
  因子计算 → 未来函数检测 → IC/IR 分析 → Alphalens 全量分析 → 分组回测 → Tear Sheet 报告

设计原则：
- 零配置即可运行（Smart Default）
- 每个阶段可独立开关
- 早期失败快速终止（如未来函数检测不通过则跳过后续）
- 结构化输出，兼容前端渲染和 Markdown 报告

对比 BigQuant AlphaMiner:
  AlphaMiner: SQL 表达式输入 → 内置引擎执行 → 固定格式报告输出
  FactorHub: 公式表达式输入 → 可插拔服务编排 → JSON + Markdown 双模式输出
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """流水线状态枚举"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"  # 该阶段通过
    WARNING = "warning"  # 通过但有警告
    FAILED = "failed"  # 该阶段失败
    SKIPPED = "skipped"  # 跳过（前置条件未满足）
    REJECTED = "rejected"  # 被拒绝（如未来函数检测不通过）


@dataclass
class PipelineStageResult:
    """单个流水线阶段的执行结果"""

    stage_name: str  # 阶段名称
    status: PipelineStatus  # 执行状态
    duration_seconds: float = 0.0  # 执行耗时
    result: Optional[Dict] = None  # 阶段输出数据
    error: Optional[str] = None  # 错误信息
    warnings: List[str] = field(default_factory=list)  # 警告列表

    @property
    def passed(self) -> bool:
        return self.status in (PipelineStatus.PASSED, PipelineStatus.WARNING)


@dataclass
class OrchestratorConfig:
    """编排器配置"""

    # 各阶段开关
    enable_lookahead_detection: bool = True  # 未来函数检测
    enable_ic_analysis: bool = True  # IC/IR 分析
    enable_alphalens: bool = True  # Alphalens 全量分析
    enable_quantile_backtest: bool = True  # 分组回测
    enable_tear_sheet: bool = True  # Tear Sheet 报告
    enable_shap_analysis: bool = False  # SHAP 分析（默认关闭，较耗时）

    # 失败策略
    fail_fast_on_bias: bool = True  # 检测到未来函数立即终止
    ic_threshold: float = 0.02  # IC 最低阈值（低于此值标记 WARNING）

    # 输出控制
    include_raw_data: bool = False  # 是否在结果中包含原始数据
    include_intermediate_results: bool = True  # 包含中间结果


class FactorOrchestrator:
    """
    因子编排器 — AlphaMiner 一键验证流水线的 FactorHub 实现

    使用方式：
        orchestrator = FactorOrchestrator()
        result = orchestrator.validate(
            expression="RSI(close, 14) / SMA(volume, 20)",
            stock_codes=["000001", "600036"],
            start_date="2023-01-01",
            end_date="2024-01-01",
        )
        logger.info(result["status"])          # "PASSED" / "REJECTED" / "PARTIAL"
        logger.info(result["report_markdown"])  # 完整 Markdown 报告
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()

    def validate(
        self,
        expression: str,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        factor_name: Optional[str] = None,
        benchmark: int = 0,  # 上证指数
        existing_factors: Optional[Dict[str, pd.Series]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的一键因子验证流水线

        Args:
            expression: 因子表达式（如 "RSI(close, 14) * volume / MA(volume, 20)"）
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            factor_name: 因子名称（默认从表达式生成）
            benchmark: 基准指数代码
            existing_factors: 已有因子字典（用于相关性检测）
            extra_context: 额外上下文（传递给各子模块）

        Returns:
            完整的验证结果字典，包含：
            - status: 总体状态 ("PASSED" / "REJECTED" / "PARTIAL" / "ERROR")
            - stages: 各阶段详细结果
            - summary: 综合评分和结论
            - report_markdown: Markdown 格式报告
            - report_structured: 结构化 JSON 数据（供前端渲染）
        """
        pipeline_start = time.time()
        _factor_name = factor_name or self._derive_factor_name(expression)

        logger.info(
            f"[FactorOrchestrator] 开始一键验证: [{_factor_name}] "
            f"表达式={expression[:60]}, "
            f"股票数={len(stock_codes)}, "
            f"时间范围={start_date}~{end_date}"
        )

        result = {
            "metadata": {
                "factor_name": _factor_name,
                "expression": expression,
                "stock_codes": stock_codes,
                "start_date": start_date,
                "end_date": end_date,
                "started_at": datetime.now().isoformat(),
                "config": {
                    "enable_lookahead_detection": self.config.enable_lookahead_detection,
                    "enable_ic_analysis": self.config.enable_ic_analysis,
                    "enable_alphalens": self.config.enable_alphalens,
                    "enable_quantile_backtest": self.config.enable_quantile_backtest,
                    "enable_tear_sheet": self.config.enable_tear_sheet,
                    "fail_fast_on_bias": self.config.fail_fast_on_bias,
                },
            },
            "status": "PENDING",
            "stages": {},
            "summary": None,
            "report_markdown": "",
            "report_structured": {},
            "total_duration": 0.0,
        }

        shared_data = {}  # 阶段间共享的数据

        try:
            # ===== Stage 0: 因子数据准备 =====
            stage0 = self._stage_compute_factor(
                expression, stock_codes, start_date, end_date, _factor_name
            )
            result["stages"]["compute_factor"] = stage0
            if not stage0.passed:
                result["status"] = "ERROR"
                result["summary"] = {"verdict": "ERROR", "reason": stage0.error}
                return self._finalize(result, pipeline_start)

            shared_data["factor_data"] = stage0.result.get("factor_data", {})
            shared_data["factor_name"] = _factor_name
            shared_data["stock_codes"] = stock_codes
            shared_data["price_df"] = stage0.result.get("price_df")

            # ===== Stage 1: 未来函数检测 =====
            if self.config.enable_lookahead_detection:
                stage1 = self._stage_lookahead_detection(shared_data)
                result["stages"]["lookahead_detection"] = stage1

                if (
                    stage1.status == PipelineStatus.REJECTED
                    and self.config.fail_fast_on_bias
                ):
                    result["status"] = "REJECTED"
                    result["summary"] = {
                        "verdict": "REJECTED",
                        "reason": "未来函数检测未通过",
                        "bias_detail": stage1.result,
                    }
                    return self._finalize(result, pipeline_start)
            else:
                result["stages"]["lookahead_detection"] = PipelineStageResult(
                    stage_name="lookahead_detection",
                    status=PipelineStatus.SKIPPED,
                    error="已禁用",
                )

            # ===== Stage 2: IC/IR 分析 =====
            if self.config.enable_ic_analysis:
                stage2 = self._stage_ic_analysis(shared_data, existing_factors)
                result["stages"]["ic_analysis"] = stage2
                shared_data["ic_result"] = stage2.result
            else:
                result["stages"]["ic_analysis"] = PipelineStageResult(
                    stage_name="ic_analysis",
                    status=PipelineStatus.SKIPPED,
                    error="已禁用",
                )

            # ===== Stage 3: Alphalens 全量分析 =====
            if self.config.enable_alphalens:
                stage3 = self._stage_alphalens_analysis(shared_data)
                result["stages"]["alphalens_analysis"] = stage3
                shared_data["alpha_result"] = stage3.result
            else:
                result["stages"]["alphalens_analysis"] = PipelineStageResult(
                    stage_name="alphalens_analysis",
                    status=PipelineStatus.SKIPPED,
                    error="已禁用",
                )

            # ===== Stage 4: 分组回测 =====
            if self.config.enable_quantile_backtest:
                stage4 = self._stage_quantile_backtest(shared_data)
                result["stages"]["quantile_backtest"] = stage4
                shared_data["backtest_result"] = stage4.result
            else:
                result["stages"]["quantile_backtest"] = PipelineStageResult(
                    stage_name="quantile_backtest",
                    status=PipelineStatus.SKIPPED,
                    error="已禁用",
                )

            # ===== Stage 5: Tear Sheet 报告 =====
            if self.config.enable_tear_sheet:
                stage5 = self._stage_tear_sheet(shared_data)
                result["stages"]["tear_sheet"] = stage5
            else:
                result["stages"]["tear_sheet"] = PipelineStageResult(
                    stage_name="tear_sheet",
                    status=PipelineStatus.SKIPPED,
                    error="已禁用",
                )

            # ===== Stage 6: SHAP 分析（可选）=====
            if self.config.enable_shap_analysis:
                stage6 = self._stage_shap_analysis(shared_data)
                result["stages"]["shap_analysis"] = stage6
            else:
                result["stages"]["shap_analysis"] = PipelineStageResult(
                    stage_name="shap_analysis",
                    status=PipelineStatus.SKIPPED,
                    error="已禁用",
                )

            # ===== 汇总判定 =====
            result["status"], result["summary"] = self._determine_overall_status(
                result["stages"]
            )
            result["report_markdown"] = self._generate_markdown_report(
                result, shared_data
            )
            result["report_structured"] = self._generate_structured_report(
                result, shared_data
            )

        except Exception as e:
            logger.error(f"[FactorOrchestrator] 流水线异常: {e}", exc_info=True)
            result["status"] = "ERROR"
            result["summary"] = {"verdict": "ERROR", "reason": str(e)}

        return self._finalize(result, pipeline_start)

    def batch_validate(
        self,
        expressions: List[str],
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        批量验证多个因子表达式

        Args:
            expressions: 因子表达式列表
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            parallel: 是否并行执行

        Returns:
            批量验证结果，包含每个因子的独立结果和汇总对比表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        summary_rows = []

        if parallel and len(expressions) > 1:
            # 并行执行
            with ThreadPoolExecutor(max_workers=min(len(expressions), 4)) as executor:
                future_to_expr = {
                    executor.submit(
                        self.validate,
                        expression=expr,
                        stock_codes=stock_codes,
                        start_date=start_date,
                        end_date=end_date,
                        factor_name=self._derive_factor_name(expr),
                    ): expr
                    for expr in expressions
                }
                for future in as_completed(future_to_expr):
                    expr = future_to_expr[future]
                    factor_name = self._derive_factor_name(expr)
                    try:
                        result = future.result()
                        results[factor_name] = result
                        ic_stage = result.get("stages", {}).get("ic_analysis")
                        ic_result = (
                            ic_stage.result
                            if isinstance(ic_stage, PipelineStageResult)
                            and ic_stage.result
                            else {}
                        )
                        bias_stage = result.get("stages", {}).get("lookahead_detection")
                        bias_result = (
                            bias_stage.result
                            if isinstance(bias_stage, PipelineStageResult)
                            and bias_stage.result
                            else {}
                        )
                        summary_rows.append(
                            {
                                "factor_name": factor_name,
                                "expression": expr[:50],
                                "status": result["status"],
                                "score": result.get("summary", {}).get(
                                    "overall_score", 0
                                ),
                                "ic_mean": ic_result.get("ic_mean"),
                                "ir": ic_result.get("ir"),
                                "risk_level": bias_result.get("risk_level", "N/A"),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"并行验证失败: {expr}, 错误: {e}")
                        results[factor_name] = {"status": "ERROR", "error": str(e)}
                        summary_rows.append(
                            {
                                "factor_name": factor_name,
                                "expression": expr[:50],
                                "status": "ERROR",
                                "score": 0,
                                "ic_mean": 0,
                                "ir": 0,
                                "risk_level": "ERROR",
                            }
                        )
        else:
            # 顺序执行
            for expr in expressions:
                factor_name = self._derive_factor_name(expr)
                try:
                    result = self.validate(
                        expression=expr,
                        stock_codes=stock_codes,
                        start_date=start_date,
                        end_date=end_date,
                        factor_name=factor_name,
                    )
                    results[factor_name] = result
                    ic_stage = result.get("stages", {}).get("ic_analysis")
                    ic_result = (
                        ic_stage.result
                        if isinstance(ic_stage, PipelineStageResult) and ic_stage.result
                        else {}
                    )
                    bias_stage = result.get("stages", {}).get("lookahead_detection")
                    bias_result = (
                        bias_stage.result
                        if isinstance(bias_stage, PipelineStageResult)
                        and bias_stage.result
                        else {}
                    )
                    summary_rows.append(
                        {
                            "factor_name": factor_name,
                            "expression": expr[:50],
                            "status": result["status"],
                            "score": result.get("summary", {}).get("overall_score", 0),
                            "ic_mean": ic_result.get("ic_mean")
                            if ic_result.get("ic_mean") is not None
                            else 0,
                            "ir": ic_result.get("ir"),
                            "risk_level": bias_result.get("risk_level", "N/A"),
                        }
                    )
                except Exception as e:
                    results[factor_name] = {"status": "ERROR", "error": str(e)}
                    summary_rows.append(
                        {
                            "factor_name": factor_name,
                            "expression": expr[:50],
                            "status": "ERROR",
                            "score": 0,
                            "ic_mean": 0,
                            "ir": 0,
                            "risk_level": "ERROR",
                        }
                    )

        # 按综合得分排序
        summary_rows.sort(key=lambda x: x["score"], reverse=True)

        return {
            "individual_results": results,
            "comparison_table": summary_rows,
            "total_factors": len(expressions),
            "passed_count": sum(1 for r in summary_rows if r["status"] == "PASSED"),
            "rejected_count": sum(1 for r in summary_rows if r["status"] == "REJECTED"),
        }

    # ==================== 各 Stage 实现 ====================

    def _stage_compute_factor(
        self,
        expression: str,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        factor_name: str,
    ) -> PipelineStageResult:
        """Stage 0: 计算因子值"""
        t0 = time.time()
        try:
            from backend.services.factor_service import factor_service
            from backend.services.data_service import data_service

            factor_data = {}
            price_records = {}

            for stock_code in stock_codes:
                try:
                    df = data_service.get_stock_data(stock_code, start_date, end_date)
                    if df is None or len(df) == 0:
                        continue

                    df = df.copy()
                    fv = factor_service.calculator.calculate(df, expression)
                    if fv is not None and len(fv.dropna()) > 10:
                        df[factor_name] = fv
                        factor_data[stock_code] = df
                        price_records[stock_code] = df["close"].dropna()
                except Exception as e:
                    logger.warning(f"股票 {stock_code} 因子计算失败: {e}")
                    continue

            if len(factor_data) == 0:
                return PipelineStageResult(
                    stage_name="compute_factor",
                    status=PipelineStatus.FAILED,
                    duration_seconds=time.time() - t0,
                    error=f"所有 {len(stock_codes)} 只股票的因子计算均失败或数据不足",
                )

            price_df = pd.DataFrame(price_records) if price_records else None

            return PipelineStageResult(
                stage_name="compute_factor",
                status=PipelineStatus.PASSED,
                duration_seconds=time.time() - t0,
                result={
                    "factor_data": factor_data,
                    "n_stocks": len(factor_data),
                    "n_total_bars": sum(len(df) for df in factor_data.values()),
                    "price_df": price_df,
                },
            )
        except Exception as e:
            logger.error(f"因子计算阶段失败: {e}", exc_info=True)
            return PipelineStageResult(
                stage_name="compute_factor",
                status=PipelineStatus.FAILED,
                duration_seconds=time.time() - t0,
                error=str(e),
            )

    def _stage_lookahead_detection(self, shared_data: Dict) -> PipelineStageResult:
        """Stage 1: 未来函数检测"""
        t0 = time.time()
        try:
            from backend.services.lookahead_bias_detector import lookahead_bias_detector

            factor_data = shared_data["factor_data"]
            factor_name = shared_data["factor_name"]

            # 收集多股票的因子值和收益
            if len(factor_data) >= 2:
                # 多股票：使用横截面检测，需构造 factor_df 和 return_df
                factor_records = []
                return_records = []
                for stock_code, df in factor_data.items():
                    df_copy = df.copy()
                    if factor_name not in df_copy.columns:
                        continue
                    if "close" not in df_copy.columns:
                        continue
                    # 计算未来收益率
                    df_copy["return"] = df_copy["close"].pct_change().shift(-1)
                    for date_idx, row in df_copy.iterrows():
                        if pd.notna(row.get(factor_name)):
                            factor_records.append(
                                {
                                    "date": date_idx,
                                    "stock_code": stock_code,
                                    factor_name: row[factor_name],
                                }
                            )
                        if pd.notna(row.get("return")):
                            return_records.append(
                                {
                                    "date": date_idx,
                                    "stock_code": stock_code,
                                    "return": row["return"],
                                }
                            )

                if len(factor_records) < 50 or len(return_records) < 50:
                    return PipelineStageResult(
                        stage_name="lookahead_detection",
                        status=PipelineStatus.WARNING,
                        duration_seconds=time.time() - t0,
                        result={
                            "has_bias": False,
                            "risk_level": "unknown",
                            "reason": "横截面数据不足",
                        },
                        warnings=["横截面数据不足50条，无法可靠检测未来函数"],
                    )

                factor_df = pd.DataFrame(factor_records)
                return_df = pd.DataFrame(return_records)

                detection_result = lookahead_bias_detector.detect_cross_sectional(
                    factor_df=factor_df,
                    return_df=return_df,
                    factor_name=factor_name,
                )
            else:
                # 单股票：使用时序检测
                all_factor_vals = []
                all_return_vals = []
                for stock_code, df in factor_data.items():
                    if factor_name in df.columns and "close" in df.columns:
                        fv = df[factor_name].dropna()
                        ret = df["close"].pct_change(1).shift(-1).dropna()
                        common = fv.index.intersection(ret.index)
                        if len(common) >= 20:
                            all_factor_vals.extend(fv.loc[common].tolist())
                            all_return_vals.extend(ret.loc[common].tolist())

                if len(all_factor_vals) < 30:
                    return PipelineStageResult(
                        stage_name="lookahead_detection",
                        status=PipelineStatus.WARNING,
                        duration_seconds=time.time() - t0,
                        result={
                            "has_bias": False,
                            "risk_level": "unknown",
                            "reason": "样本不足",
                        },
                        warnings=["样本数不足30，无法可靠检测未来函数"],
                    )

                detection_result = lookahead_bias_detector.detect(
                    factor_values=pd.Series(all_factor_vals),
                    return_values=pd.Series(all_return_vals),
                    factor_name=factor_name,
                )

            status = (
                PipelineStatus.REJECTED
                if detection_result.has_bias
                else (
                    PipelineStatus.WARNING
                    if detection_result.risk_level.value in ("medium", "low")
                    else PipelineStatus.PASSED
                )
            )

            return PipelineStageResult(
                stage_name="lookahead_detection",
                status=status,
                duration_seconds=time.time() - t0,
                result={
                    "has_bias": detection_result.has_bias,
                    "risk_level": detection_result.risk_level.value,
                    "risk_score": detection_result.risk_score,
                    "summary": detection_result.summary,
                    "recommendations": detection_result.recommendations,
                    "checks_count": len(detection_result.checks),
                    "failed_checks": sum(
                        1 for c in detection_result.checks if not c.passed
                    ),
                },
                warnings=detection_result.recommendations
                if not detection_result.has_bias
                else [],
            )
        except Exception as e:
            logger.warning(f"未来函数检测阶段异常: {e}")
            return PipelineStageResult(
                stage_name="lookahead_detection",
                status=PipelineStatus.WARNING,
                duration_seconds=time.time() - t0,
                error=str(e),
                warnings=[f"检测异常: {e}"],
            )

    def _stage_ic_analysis(
        self, shared_data: Dict, existing_factors: Optional[Dict]
    ) -> PipelineStageResult:
        """Stage 2: IC/IR 分析"""
        t0 = time.time()
        try:
            from backend.services.analysis_service import analysis_service

            factor_data = shared_data["factor_data"]
            factor_name = shared_data["factor_name"]
            stock_codes = list(factor_data.keys())

            ic_ir_result = analysis_service.calculate_ic_ir(
                factor_data, [factor_name], stock_codes
            )

            # calculate_ic_ir 返回 {"ic_stats": {factor_key: {...}}, "monthly_ic": ..., "rolling_ir": ...}
            # 多股票模式下 key 格式为 {factor_name}_{ic_type}_{period}，单股票模式为 factor_name
            ic_stats = ic_ir_result.get("ic_stats", {})
            factor_ic = ic_stats.get(factor_name, {})
            # 多股票模式：取第一个包含该因子名的key
            if not factor_ic:
                for key, stats in ic_stats.items():
                    if factor_name in key or key == factor_name:
                        factor_ic = stats
                        break

            ic_mean_val = factor_ic.get("IC均值")
            ic_mean = float(ic_mean_val) if ic_mean_val is not None else None
            ir = float(factor_ic["IR"]) if factor_ic.get("IR") is not None else None

            # 判定是否通过
            if ic_mean is not None:
                status = (
                    PipelineStatus.PASSED
                    if abs(ic_mean) >= self.config.ic_threshold
                    else PipelineStatus.WARNING
                )
            else:
                status = PipelineStatus.WARNING  # IC不可计算时标记为警告

            warnings = []
            if ic_mean is not None and abs(ic_mean) < self.config.ic_threshold:
                warnings.append(
                    f"IC均值({ic_mean:.4f})低于阈值({self.config.ic_threshold})"
                )
            if ic_mean is not None and abs(ic_mean) > 0.15:
                warnings.append(
                    f"IC均值({ic_mean:.4f})异常偏高，请核查是否存在未来函数"
                )

            return PipelineStageResult(
                stage_name="ic_analysis",
                status=status,
                duration_seconds=time.time() - t0,
                result={
                    "ic_mean": ic_mean,
                    "ir": ir,
                    "rank_ic": float(factor_ic.get("Rank_IC均值"))
                    if factor_ic.get("Rank_IC均值") is not None
                    else None,
                    "full_result": ic_ir_result
                    if self.config.include_intermediate_results
                    else None,
                },
                warnings=warnings,
            )
        except Exception as e:
            logger.warning(f"IC分析阶段异常: {e}")
            return PipelineStageResult(
                stage_name="ic_analysis",
                status=PipelineStatus.FAILED,
                duration_seconds=time.time() - t0,
                error=str(e),
            )

    def _stage_alphalens_analysis(self, shared_data: Dict) -> PipelineStageResult:
        """Stage 3: Alphalens 全量分析"""
        t0 = time.time()
        try:
            from backend.services.alphalens_analysis_service import (
                alphalens_analysis_service,
            )

            factor_data = shared_data["factor_data"]
            factor_name = shared_data["factor_name"]
            price_df = shared_data.get("price_df")

            if price_df is None or len(price_df) == 0:
                return PipelineStageResult(
                    stage_name="alphalens_analysis",
                    status=PipelineStatus.WARNING,
                    duration_seconds=time.time() - t0,
                    error="无价格数据",
                )

            # 构造 factor_values_dict
            factor_values_dict = {}
            for stock_code, df in factor_data.items():
                if factor_name in df.columns:
                    series = df[factor_name].dropna()
                    if len(series) > 0:
                        factor_values_dict[stock_code] = series

            if len(factor_values_dict) < 2:
                return PipelineStageResult(
                    stage_name="alphalens_analysis",
                    status=PipelineStatus.WARNING,
                    duration_seconds=time.time() - t0,
                    warnings=[
                        f"有效股票数({len(factor_values_dict)})不足，横截面分析可能不可靠"
                    ],
                    result=None,
                )

            alpha_result = alphalens_analysis_service.full_analysis(
                factor_values_dict=factor_values_dict,
                pricing_df=price_df,
                periods=(1, 5, 10),
                quantiles=5,
            )

            return PipelineStageResult(
                stage_name="alphalens_analysis",
                status=PipelineStatus.PASSED,
                duration_seconds=time.time() - t0,
                result=(
                    alpha_result
                    if self.config.include_intermediate_results
                    else self._summarize_alpha(alpha_result)
                ),
            )
        except Exception as e:
            logger.warning(f"Alphalens分析阶段异常: {e}")
            return PipelineStageResult(
                stage_name="alphalens_analysis",
                status=PipelineStatus.FAILED,
                duration_seconds=time.time() - t0,
                error=str(e),
            )

    def _stage_quantile_backtest(self, shared_data: Dict) -> PipelineStageResult:
        """Stage 4: 分组回测"""
        t0 = time.time()
        try:
            from backend.services.factor_return_analysis_service import (
                factor_return_analysis_service,
            )

            factor_data = shared_data["factor_data"]
            factor_name = shared_data["factor_name"]

            quantile_result = factor_return_analysis_service.calculate_quantile_returns(
                factor_data=factor_data,
                factor_name=factor_name,
                price_column="close",
            )

            if "error" in quantile_result:
                return PipelineStageResult(
                    stage_name="quantile_backtest",
                    status=PipelineStatus.FAILED,
                    duration_seconds=time.time() - t0,
                    error=quantile_result["error"],
                )

            # 同时获取累计收益
            cumulative_result = (
                factor_return_analysis_service.calculate_cumulative_returns(
                    factor_data=factor_data,
                    factor_name=factor_name,
                    price_column="close",
                    long_short=True,
                )
            )

            return PipelineStageResult(
                stage_name="quantile_backtest",
                status=PipelineStatus.PASSED,
                duration_seconds=time.time() - t0,
                result={
                    "quantile_returns": quantile_result,
                    "cumulative_returns": cumulative_result
                    if "error" not in cumulative_result
                    else None,
                },
            )
        except Exception as e:
            logger.warning(f"分组回测阶段异常: {e}")
            return PipelineStageResult(
                stage_name="quantile_backtest",
                status=PipelineStatus.FAILED,
                duration_seconds=time.time() - t0,
                error=str(e),
            )

    def _stage_tear_sheet(self, shared_data: Dict) -> PipelineStageResult:
        """Stage 5: Tear Sheet 报告生成"""
        t0 = time.time()
        try:
            from backend.services.tear_sheet_service import TearSheetService

            factor_data = shared_data["factor_data"]
            factor_name = shared_data["factor_name"]

            tear_sheet_service = TearSheetService()
            report = tear_sheet_service.create_full_tear_sheet(
                factor_data=factor_data,
                factor_name=factor_name,
                price_column="close",
            )

            return PipelineStageResult(
                stage_name="tear_sheet",
                status=PipelineStatus.PASSED,
                duration_seconds=time.time() - t0,
                result=report,
            )
        except Exception as e:
            logger.warning(f"Tear Sheet生成阶段异常: {e}")
            return PipelineStageResult(
                stage_name="tear_sheet",
                status=PipelineStatus.FAILED,
                duration_seconds=time.time() - t0,
                error=str(e),
            )

    def _stage_shap_analysis(self, shared_data: Dict) -> PipelineStageResult:
        """Stage 6: SHAP 因子归因分析（可选，较耗时）"""
        t0 = time.time()
        try:
            from backend.services.analysis_service import analysis_service

            factor_data = shared_data["factor_data"]
            factor_name = shared_data["factor_name"]
            stock_codes = list(factor_data.keys())

            shap_result = analysis_service.run_shap_analysis(
                factor_data=factor_data,
                factor_names=[factor_name],
                stock_codes=stock_codes,
                target_column="future_return_1",
            )

            return PipelineStageResult(
                stage_name="shap_analysis",
                status=PipelineStatus.PASSED,
                duration_seconds=time.time() - t0,
                result=shap_result,
            )
        except Exception as e:
            logger.warning(f"SHAP分析阶段异常: {e}")
            return PipelineStageResult(
                stage_name="shap_analysis",
                status=PipelineStatus.FAILED,
                duration_seconds=time.time() - t0,
                error=str(e),
            )

    # ==================== 汇总 & 报告 ====================

    def _determine_overall_status(
        self, stages: Dict[str, PipelineStageResult]
    ) -> Tuple[str, Dict]:
        """根据各阶段结果确定总体状态"""
        has_rejected = any(s.status == PipelineStatus.REJECTED for s in stages.values())
        has_failed = any(s.status == PipelineStatus.FAILED for s in stages.values())
        has_warning = any(s.status == PipelineStatus.WARNING for s in stages.values())
        n_passed = sum(1 for s in stages.values() if s.passed)
        n_total = len(stages)

        if has_rejected:
            verdict = "REJECTED"
            reason = "存在致命问题（如未来函数）"
        elif has_failed:
            verdict = "PARTIAL"
            reason = f"{n_passed}/{n_total} 阶段通过，部分阶段失败"
        elif has_warning:
            verdict = "PASSED"
            reason = f"{n_passed}/{n_total} 阶段通过，部分阶段有警告"
        else:
            verdict = "PASSED"
            reason = f"全部 {n_passed}/{n_total} 阶段通过"

        # 计算综合评分
        score = self._calculate_overall_score(stages)

        return verdict, {
            "verdict": verdict,
            "reason": reason,
            "overall_score": score,
            "stages_passed": n_passed,
            "stages_total": n_total,
        }

    def _calculate_overall_score(self, stages: Dict[str, PipelineStageResult]) -> float:
        """综合评分 (0-100)"""
        score = 50.0  # 基础分

        stage_weights = {
            "lookahead_detection": 25,
            "ic_analysis": 25,
            "alphalens_analysis": 20,
            "quantile_backtest": 20,
            "tear_sheet": 10,
        }

        for name, stage in stages.items():
            w = stage_weights.get(name, 5)
            if stage.status == PipelineStatus.PASSED:
                score += w
            elif stage.status == PipelineStatus.WARNING:
                score += w * 0.5
            elif stage.status == PipelineStatus.FAILED:
                score -= w * 0.3
            elif stage.status == PipelineStatus.REJECTED:
                score -= w * 2

        return max(0.0, min(100.0, round(score, 1)))

    def _generate_markdown_report(self, result: Dict, shared_data: Dict) -> str:
        """生成 Markdown 格式的验证报告"""
        meta = result["metadata"]
        stages = result["stages"]
        summary = result.get("summary", {})

        lines = [
            f"# 因子验证报告: {meta['factor_name']}",
            "",
            f"> **验证时间**: {meta['started_at']} | "
            f"**表达式**: `{meta['expression']}` | "
            f"**股票池**: {len(meta['stock_codes'])}只 | "
            f"**范围**: {meta['start_date']} ~ {meta['end_date']}",
            "",
            "---",
            "",
            f"## 总体结论: {summary.get('verdict', 'UNKNOWN')}",
            "",
            f"- **综合评分**: {summary.get('overall_score', 0):.1f}/100",
            f"- **通过情况**: {summary.get('stages_passed', 0)}/{summary.get('stages_total', 0)} 阶段",
            f"- **原因**: {summary.get('reason', '')}",
            "",
        ]

        # 各阶段详情
        lines.append("---\n\n## 各阶段详情\n")

        stage_labels = {
            "compute_factor": "因子计算",
            "lookahead_detection": "未来函数检测",
            "ic_analysis": "IC/IR 分析",
            "alphalens_analysis": "Alphalens 全量分析",
            "quantile_backtest": "分组回测",
            "tear_sheet": "Tear Sheet 报告",
            "shap_analysis": "SHAP 归因分析",
        }
        status_emoji = {
            PipelineStatus.PASSED: ":white_check_mark:",
            PipelineStatus.WARNING: ":warning:",
            PipelineStatus.FAILED: ":x:",
            PipelineStatus.REJECTED: ":no_entry:",
            PipelineStatus.SKIPPED: ":fast_forward:",
        }

        for stage_name, stage in stages.items():
            label = stage_labels.get(stage_name, stage_name)
            emoji = status_emoji.get(stage.status, "")
            dur = f"{stage.duration_seconds:.2f}s"

            lines.append(f"### {emoji} {label} (`{stage.status.value}`)")
            lines.append(f"- 状态: `{stage.status.value}` | 耗时: {dur}")

            if stage.error:
                lines.append(f"- 错误: {stage.error}")
            if stage.warnings:
                for w in stage.warnings:
                    lines.append(f"- :warning: {w}")
            if stage.result and isinstance(stage.result, dict):
                # 显示关键指标
                keys_to_show = [
                    "has_bias",
                    "risk_level",
                    "risk_score",
                    "ic_mean",
                    "ir",
                    "rank_ic",
                    "n_stocks",
                ]
                shown = [
                    f"{k}={v}"
                    for k, v in stage.result.items()
                    if k in keys_to_show and v is not None
                ]
                if shown:
                    lines.append(f"- 关键指标: {', '.join(shown)}")
            lines.append("")

        # 改进建议汇总
        all_recommendations = []
        bias_stage = stages.get("lookahead_detection")
        if bias_stage and bias_stage.result and isinstance(bias_stage.result, dict):
            recs = bias_stage.result.get("recommendations", [])
            all_recommendations.extend(recs)
        ic_stage = stages.get("ic_analysis")
        if ic_stage and ic_stage.warnings:
            all_recommendations.extend(ic_stage.warnings)

        if all_recommendations:
            lines.append("---\n\n## 改进建议\n")
            for rec in set(all_recommendations):
                lines.append(f"- {rec}")
            lines.append("")

        lines.append("---\n\n*报告由 FactorOrchestrator 自动生成*")
        return "\n".join(lines)

    def _generate_structured_report(self, result: Dict, shared_data: Dict) -> Dict:
        """生成结构化报告（供前端渲染）"""
        structured = {
            "verdict": result.get("summary", {}).get("verdict", "UNKNOWN"),
            "score": result.get("summary", {}).get("overall_score", 0),
            "stages": {},
        }

        for name, stage in result.get("stages", {}).items():
            structured["stages"][name] = {
                "status": stage.status.value,
                "duration": round(stage.duration_seconds, 2),
                "warnings": stage.warnings,
                "has_error": stage.error is not None,
            }
            # 提取关键数值供前端图表使用
            if stage.result and isinstance(stage.result, dict):
                numeric_keys = [
                    "ic_mean",
                    "ir",
                    "rank_ic",
                    "risk_score",
                    "has_bias",
                    "n_stocks",
                ]
                structured["stages"][name]["metrics"] = {
                    k: v
                    for k, v in stage.result.items()
                    if k in numeric_keys and isinstance(v, (int, float, bool))
                }

        return structured

    # ==================== 辅助方法 ====================

    def _finalize(self, result: Dict, start_time: float) -> Dict:
        """添加最终元信息"""
        result["total_duration"] = round(time.time() - start_time, 2)
        result["metadata"]["completed_at"] = datetime.now().isoformat()
        return result

    @staticmethod
    def _derive_factor_name(expression: str) -> str:
        """从表达式派生因子名称"""
        # 取前40字符，替换特殊字符
        name = (
            expression.replace("*", "_x_")
            .replace("/", "_div_")
            .replace("(", "")
            .replace(")", "")[:40]
        )
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        return name.strip("_") or "unnamed_factor"

    @staticmethod
    def _summarize_alpha(alpha_result: Dict) -> Dict:
        """精简 Alphalens 结果以减少体积"""
        if not alpha_result:
            return {}
        summary = {}
        for key in ("ic_analysis", "returns_analysis", "turnover_analysis"):
            if key in alpha_result:
                section = alpha_result[key]
                if isinstance(section, dict):
                    summary[key] = {
                        k: v
                        for k, v in section.items()
                        if k
                        in (
                            "mean_ic",
                            "std_ic",
                            "ir",
                            "mean_return_q1",
                            "mean_return_q5",
                        )
                        or "ic_" in k.lower()
                    }
        return summary


# ==================== 全局实例 ====================

factor_orchestrator = FactorOrchestrator()
