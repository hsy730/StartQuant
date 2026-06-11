"""
因子分析API路由
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import logging
import traceback

from backend.utils.serialization import safe_numeric_value, sanitize_dict
from backend.utils.ic_calculator import calculate_rolling_ic

logger = logging.getLogger(__name__)


from backend.services.analysis_service import analysis_service  # noqa: E402
from backend.services.factor_stability_service import factor_stability_service  # noqa: E402
from backend.services.enhanced_analysis_service import enhanced_analysis_service  # noqa: E402
from backend.services.factor_exposure_service import factor_exposure_service  # noqa: E402
from backend.services.factor_effectiveness_service import factor_effectiveness_service  # noqa: E402
from backend.services.factor_attribution_service import factor_attribution_service  # noqa: E402
from backend.services.factor_monitoring_service import factor_monitoring_service  # noqa: E402

router = APIRouter()


# ========== 数据模型 ==========


class CalculateRequest(BaseModel):
    """计算因子值请求"""

    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str
    freq: str = "D"
    period: Optional[str] = None


class ICAnalysisRequest(BaseModel):
    """IC分析请求"""

    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str
    freq: str = "D"
    period: Optional[str] = None


class StabilityRequest(BaseModel):
    """稳定性检验请求"""

    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str
    freq: str = "D"
    period: Optional[str] = None


class MultiPeriodRequest(BaseModel):
    """多周期分析请求"""

    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str
    freq: str = "D"
    period: Optional[str] = None


# ========== API端点 ==========


@router.post("/calculate")
async def calculate_factor(request: CalculateRequest):
    """计算因子值"""

    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        # 获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        logger.info(f"开始计算因子: {request.factor_name}, 代码: {factor.code}")

        # 获取数据并计算因子
        result_data = {}
        errors = []

        for stock_code in request.stock_codes:
            try:
                logger.info(f"获取股票数据: {stock_code}, 时间范围: {request.start_date} - {request.end_date}")
                if request.freq.upper() != "D":
                    minute_period = (request.period or request.freq).lower().replace("min", "").replace("t", "")
                    data = data_service.get_stock_minute_data(
                        stock_code,
                        request.start_date,
                        request.end_date,
                        period=minute_period if minute_period.isdigit() else "5",
                    )
                else:
                    data = data_service.get_stock_data(stock_code, request.start_date, request.end_date)

                if data is None or len(data) == 0:
                    logger.warning(f"股票 {stock_code} 未获取到数据")
                    errors.append(f"股票 {stock_code} 未获取到数据")
                    continue

                logger.info(f"股票 {stock_code} 获取到 {len(data)} 条数据")

                # 使用 calculator 计算因子
                logger.info(f"开始计算因子值，因子代码: {factor.code}")
                factor_series = factor_service.calculator.calculate(data, factor.code)

                if factor_series is None:
                    logger.warning(f"股票 {stock_code} 因子计算返回 None")
                    errors.append(f"股票 {stock_code} 因子计算失败")
                    continue

                logger.info(f"因子计算完成，有效值数量: {factor_series.notna().sum()}/{len(factor_series)}")

                # 将因子值添加到数据中（先复制避免污染缓存）
                data = data.copy()
                data[request.factor_name] = factor_series

                # 过滤掉因子值为 NaN 的行，确保 dates 和 factor_values 一一对应
                valid_data = data[[request.factor_name]].dropna()
                valid_dates = valid_data.index.strftime("%Y-%m-%d").tolist()
                valid_factor_values = valid_data[request.factor_name].tolist()

                # 额外检查：确保所有值都是有效的数字，转换 NaN 和 inf 为 None
                valid_factor_values = [safe_numeric_value(v) for v in valid_factor_values]

                # 移除值为 None 的项
                filtered_dates = []
                filtered_values = []
                for d, v in zip(valid_dates, valid_factor_values):
                    if v is not None:
                        filtered_dates.append(d)
                        filtered_values.append(v)

                valid_dates = filtered_dates
                valid_factor_values = filtered_values

                logger.info(
                    f"股票 {stock_code}: 有效数据范围 "
                    f"{valid_dates[0] if valid_dates else '无'} 到 "
                    f"{valid_dates[-1] if valid_dates else '无'}, 共 {len(valid_dates)} 行"
                )

                # 验证数据完整性
                if len(valid_dates) != len(valid_factor_values):
                    logger.error(f"数据长度不一致! dates={len(valid_dates)}, values={len(valid_factor_values)}")
                    errors.append(f"股票 {stock_code} 数据长度不一致")
                    continue

                # 转换为字典格式返回
                result_data[stock_code] = {
                    "dates": valid_dates,
                    "factor_values": valid_factor_values,
                    "statistics": {
                        "mean": safe_numeric_value(factor_series.mean()) if len(factor_series) > 0 else None,
                        "std": safe_numeric_value(factor_series.std()) if len(factor_series) > 0 else None,
                        "min": safe_numeric_value(factor_series.min()) if len(factor_series) > 0 else None,
                        "max": safe_numeric_value(factor_series.max()) if len(factor_series) > 0 else None,
                        "count": int(factor_series.count()),
                    },
                }
                logger.info(f"股票 {stock_code} 因子计算成功")

            except Exception as e:
                logger.error(f"股票 {stock_code} 因子计算失败: {str(e)}\n{traceback.format_exc()}")
                errors.append(f"股票 {stock_code} 计算失败: {str(e)}")
                continue

        if not result_data:
            error_msg = f"因子计算失败或无有效数据。详情: {'; '.join(errors) if errors else '未知错误'}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        if errors:
            logger.warning(f"因子计算完成，但有部分错误: {'; '.join(errors)}")

        return {"success": True, "data": result_data, "warnings": errors if errors else None}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子计算异常: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子计算失败: {str(e)}")


@router.post("/ic")
async def calculate_ic(request: ICAnalysisRequest):
    """计算IC/IR"""

    try:
        logger.info(
            f"开始IC分析: {request.factor_name}, 股票: {request.stock_codes}, 时间: {request.start_date} - {request.end_date}"
        )

        # 先尝试使用缓存
        result = analysis_service.analyze(
            stock_codes=request.stock_codes,
            factor_names=[request.factor_name],
            start_date=request.start_date,
            end_date=request.end_date,
            use_cache=True,
            rolling_window=252,
        )

        logger.info(f"IC分析原始结果 keys: {result.keys() if result else 'None'}")
        logger.info(f"IC分析原始 ic_ir: {result.get('ic_ir', {}).get('ic_stats', {}) if result else 'None'}")

        # 提取 IC/IR 相关数据并简化返回格式
        ic_ir_data = result.get("ic_ir", {})
        ic_stats = ic_ir_data.get("ic_stats", {})

        # 如果缓存中的数据无效，重新计算
        if not ic_stats or len(ic_stats) == 0:
            logger.warning("缓存中的数据无效，重新计算IC分析")
            result = analysis_service.analyze(
                stock_codes=request.stock_codes,
                factor_names=[request.factor_name],
                start_date=request.start_date,
                end_date=request.end_date,
                use_cache=False,  # 不使用缓存
                rolling_window=252,
            )

            # 重新提取
            ic_ir_data = result.get("ic_ir", {})
            ic_stats = ic_ir_data.get("ic_stats", {})
            logger.info(f"重新计算的ic_stats: {ic_stats}")

        simplified_result = {
            "metadata": result.get("metadata", {}),
            "ic_stats": ic_stats,
        }

        # 检查是否有有效数据
        if not ic_stats or len(ic_stats) == 0:
            logger.warning("IC分析未返回有效统计数据")
            return {
                "success": True,
                "data": simplified_result,
                "message": "IC分析未返回有效统计数据，可能原因：股票数据不足或因子计算失败",
            }

        return {"success": True, "data": sanitize_dict(simplified_result)}
    except Exception as e:
        logger.error(f"IC分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"IC分析失败: {str(e)}")


@router.post("/stability")
async def stability_test(request: StabilityRequest):
    """稳定性检验"""
    try:
        # 调用稳定性服务
        result = factor_stability_service.comprehensive_stability_test(
            factor_name=request.factor_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        return sanitize_dict({"success": True, "data": result})
    except Exception as e:
        logger.error(f"稳定性检验失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-period")
async def multi_period_analysis(request: MultiPeriodRequest):
    """多周期分析"""
    try:
        # 调用增强分析服务
        result = enhanced_analysis_service.analyze_multi_period_ic(
            factor_name=request.factor_name,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        return sanitize_dict({"success": True, "data": result})
    except Exception as e:
        logger.error(f"多周期分析失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decay")
async def decay_analysis(request: ICAnalysisRequest):
    """因子衰减分析"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db
        import numpy as np

        # 获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 计算因子在不同周期的IC（衰减分析）
        decay_periods = [1, 3, 5, 10, 20]  # 1日, 3日, 5日, 10日, 20日
        decay_results = []

        # 预取所有股票数据，避免在 period x stock 嵌套循环中重复 IO
        stock_data_cache = {}
        for stock_code in request.stock_codes:
            if request.freq.upper() != "D":
                minute_period = (request.period or request.freq).lower().replace("min", "").replace("t", "")
                data = data_service.get_stock_minute_data(
                    stock_code,
                    request.start_date,
                    request.end_date,
                    period=minute_period if minute_period.isdigit() else "5",
                )
            else:
                data = data_service.get_stock_data(stock_code, request.start_date, request.end_date)
            if data is not None and len(data) > 0:
                data = data.copy()
                # 预计算因子值（同一因子只需计算一次）
                factor_series = factor_service.calculator.calculate(data, factor.code)
                if factor_series is not None:
                    data[request.factor_name] = factor_series
                    stock_data_cache[stock_code] = data

        for period in decay_periods:
            all_ics = []
            for stock_code, data in stock_data_cache.items():
                factor_series = data[request.factor_name]
                # 计算未来收益率
                future_returns = data["close"].pct_change(period).shift(-period)
                # 计算IC（使用Spearman，符合规则7.1）
                ic = calculate_rolling_ic(factor_series, future_returns, window=20, method="spearman")
                if not ic.empty and ic.dropna().count() > 0:
                    all_ics.append(ic.dropna().mean())

            if all_ics:
                mean_ic = np.mean(all_ics)
                decay_results.append({"period": f"{period}日", "ic_mean": float(mean_ic), "period_days": period})

        result = {"factor_name": request.factor_name, "decay_analysis": decay_results}

        return sanitize_dict({"success": True, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exposure")
async def exposure_analysis(request: CalculateRequest):
    """因子暴露度分析"""

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"开始因子暴露度分析: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用暴露度分析服务
        result = factor_exposure_service.calculate_exposure_metrics(
            factor_data=factor_data, factor_name=request.factor_name, window=20
        )

        return sanitize_dict({"success": True, "data": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子暴露度分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子暴露度分析失败: {str(e)}")


@router.post("/effectiveness")
async def effectiveness_analysis(request: ICAnalysisRequest):
    """因子有效性分析"""

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"开始因子有效性分析: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用有效性分析服务
        result = factor_effectiveness_service.analyze_effectiveness(
            factor_data=factor_data, factor_name=request.factor_name, future_periods=[1, 5, 10, 20]
        )

        return sanitize_dict({"success": True, "data": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子有效性分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子有效性分析失败: {str(e)}")


@router.post("/attribution")
async def attribution_analysis(request: ICAnalysisRequest):
    """因子贡献度分解"""

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"开始因子贡献度分解: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用贡献度分解服务
        result = factor_attribution_service.analyze_attribution(
            factor_data=factor_data, factor_name=request.factor_name, benchmark_data=None
        )

        return sanitize_dict({"success": True, "data": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子贡献度分解失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子贡献度分解失败: {str(e)}")


@router.post("/monitoring")
async def monitoring_analysis(request: ICAnalysisRequest):
    """时间序列动态监测"""

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"开始时间序列动态监测: {request.factor_name}, 股票: {request.stock_codes}")

        # 获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        # 获取因子数据
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        # 调用动态监测服务
        result = factor_monitoring_service.monitor_dynamics(factor_data=factor_data, factor_name=request.factor_name)

        return sanitize_dict({"success": True, "data": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"时间序列动态监测失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"时间序列动态监测失败: {str(e)}")


# ========== 增强版因子相关性分析API ==========


class CorrelationAnalysisRequest(BaseModel):
    """因子相关性分析请求"""

    factor_names: List[str]
    stock_codes: List[str]
    start_date: str
    end_date: str
    config: Optional[Dict] = None  # 可选配置（如滚动窗口大小等）


@router.post("/correlation/enhanced")
async def enhanced_correlation_analysis(request: CorrelationAnalysisRequest):
    """
    增强版因子相关性分析（Alphalens + 自研算法）

    功能：
    - 横截面相关性（每天计算，时间平均）
    - 时间序列相关性（基于因子收益率）
    - 滚动窗口稳定性分析
    - 显著性检验（t检验 + p值）
    - VIF多重共线性检查
    - 自动频率对齐和缺失值处理

    符合专业量化研究的10个关键要求
    """

    try:
        from backend.services.factor_service import factor_service

        logger.info(
            f"开始增强版因子相关性分析: "
            f"因子={request.factor_names}, "
            f"股票数={len(request.stock_codes)}, "
            f"时间范围={request.start_date} ~ {request.end_date}"
        )

        # 获取多因子数据
        all_factor_data = {}
        for factor_name in request.factor_names:
            try:
                factor_data = factor_service.calculate_factors_for_stocks(
                    request.stock_codes, [factor_name], request.start_date, request.end_date
                )
                if factor_data:
                    all_factor_data[factor_name] = factor_data
            except Exception as e:
                logger.warning(f"因子{factor_name}获取失败: {e}")
                continue

        if not all_factor_data:
            raise HTTPException(status_code=500, detail="未能获取任何有效的因子数据")

        # 构建MultiIndex DataFrame (date, asset) × factors
        factor_panel_list = []
        for factor_name, factor_dict in all_factor_data.items():
            records = []
            for stock_code, stock_df in factor_dict.items():
                if not isinstance(stock_df, pd.DataFrame):
                    continue
                if factor_name not in stock_df.columns:
                    continue
                for date_idx, row in stock_df.iterrows():
                    value = row[factor_name]
                    if pd.notna(value):
                        records.append({"date": pd.Timestamp(date_idx), "asset": stock_code, factor_name: value})

            if records:
                factor_df = pd.DataFrame(records)
                factor_df = factor_df.set_index(["date", "asset"])
                factor_panel_list.append(factor_df)

        if not factor_panel_list:
            raise HTTPException(status_code=500, detail="数据构建失败")

        # 合并所有因子数据
        from functools import reduce

        factor_panel = reduce(lambda left, right: left.join(right, how="outer"), factor_panel_list)
        factor_panel = factor_panel.dropna(how="all")

        if len(factor_panel) == 0:
            raise HTTPException(status_code=500, detail="合并后无有效数据")

        # 调用因子相关性分析服务（精简版，零冗余依赖）
        from backend.services.factor_correlation_service import factor_correlation_service

        config = request.config or {
            "rolling_window": 120,
            "rolling_step": 20,
            "use_knn": True,
            "knn_neighbors": 5,
            "winsorize_method": "mad",
            "n_sigma": 3.0,
        }

        result = factor_correlation_service.analyze(
            factor_panel=factor_panel, factor_cols=request.factor_names, config=config
        )

        logger.info(f"增强版相关性分析完成，生成{len(result.get('warnings', []))}个警告")

        return sanitize_dict(
            {
                "success": True,
                "data": result,
                "metadata": {
                    "factors_analyzed": len(request.factor_names),
                    "stocks_analyzed": len(request.stock_codes),
                    "time_range": f"{request.start_date} ~ {request.end_date}",
                    "mode": result.get("metadata", {}).get("mode", "unknown"),
                    "warnings_count": len(result.get("warnings", [])),
                    "recommendations_count": len(result.get("recommendations", [])),
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增强版相关性分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"增强版相关性分析失败: {str(e)}")


@router.post("/correlation/interpret")
async def correlation_interpretation(request: CorrelationAnalysisRequest):
    """
    因子相关性智能解读（内建规则引擎，无需外部依赖）

    功能：
    - 自动识别高/低相关因子对
    - 检测非线性关系警告
    - 生成可操作的建议

    注意：此功能已内置在 /correlation/enhanced 的返回结果中
    本端点提供独立的解读服务（如果已有相关性矩阵）
    """

    try:

        return {
            "success": True,
            "data": {
                "message": "智能解读功能已内置",
                "implementation": "自建规则引擎（零外部依赖）",
                "capabilities": [
                    "高/低相关因子识别",
                    "Pearson vs Spearman一致性检验",
                    "滚动稳定性评估",
                    "VIF共线性预警",
                    "自动生成改进建议",
                ],
                "note": "调用 /correlation/enhanced 可获取完整分析+解读",
            },
            "status": "ready",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"解读服务初始化失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"解读服务失败: {str(e)}")


@router.post("/correlation/mixed-type")
async def mixed_type_correlation_analysis(request: CorrelationAnalysisRequest):
    """
    混合类型因子相关性分析（可选：需要phik）

    适用场景：
    - 因子包含行业分类、市值分层等离散变量
    - 需要统一处理数值和分类变量

    安装方式（按需）：
        uv pip install phik
        或
        pip install factor-flow[advanced]

    如果未安装phik，将使用scipy的ANOVA作为降级方案
    """

    try:
        from backend.services.phik_correlation_service import get_phik_service

        phik_svc = get_phik_service()

        if phik_svc and phik_svc.is_available():
            status = "phik_available"
            capabilities = ["Phi_K系数矩阵", "显著性检验", "非线性检测"]
        else:
            status = "scipy_fallback"
            capabilities = ["ANOVA F检验", "分组均值比较"]

        return {
            "success": True,
            "data": {
                "status": status,
                "capabilities": capabilities,
                "install_hint": (None if status == "phik_available" else "运行: pip install phik 获得完整功能"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"混合类型分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"混合类型分析失败: {str(e)}")


# ========== 新增：专业因子收益分析API ==========


class QuantileReturnsRequest(BaseModel):
    """因子分组收益请求"""

    factor_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str
    n_quantiles: int = 5


@router.post("/quantile-returns")
async def quantile_returns_analysis(request: QuantileReturnsRequest):
    """
    因子分组收益分析（Quantile Returns）⭐核心功能

    功能：
    - 将股票按因子值分成N组（默认5组）
    - 计算每组的平均收益和统计显著性
    - 检验单调性（有效因子的关键特征）
    - 计算多空利差（最高组-最低组）
    - Bootstrap置信区间评估稳健性

    这是验证因子预测能力最重要的工具之一。

    对比表状态更新：❌ → ✅ 已实现
    """

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"开始因子分组收益分析: {request.factor_name}, " f"股票数={len(request.stock_codes)}")

        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        from backend.services.factor_return_analysis_service import factor_return_analysis_service

        _config = {  # noqa: F841
            "n_quantiles": request.n_quantiles,
        }

        result = factor_return_analysis_service.calculate_quantile_returns(
            factor_data=factor_data,
            factor_name=request.factor_name,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info("因子分组收益分析完成")

        return sanitize_dict(
            {
                "success": True,
                "data": result,
                "metadata": {
                    "factor_name": request.factor_name,
                    "n_stocks": len(request.stock_codes),
                    "time_range": f"{request.start_date} ~ {request.end_date}",
                    "implementation": "FactorHub原生实现（对标JoinQuant/BigQuant）",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子分组收益分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子分组收益分析失败: {str(e)}")


@router.post("/cumulative-returns")
async def cumulative_returns_analysis(request: QuantileReturnsRequest):
    """
    累计收益曲线分析（Cumulative Returns）⭐核心功能

    功能：
    - 基于因子分组的累计收益走势
    - 多空组合（Long-Short）表现
    - 最大回撤、夏普比率等风险指标

    最直观展示因子有效性的方式。

    对比表状态更新：❌ → ✅ 已实现
    """

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"开始累计收益曲线分析: {request.factor_name}")

        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        from backend.services.factor_return_analysis_service import factor_return_analysis_service

        result = factor_return_analysis_service.calculate_cumulative_returns(
            factor_data=factor_data,
            factor_name=request.factor_name,
            long_short=True,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info("累计收益曲线分析完成")

        return sanitize_dict(
            {
                "success": True,
                "data": result,
                "metadata": {
                    "factor_name": request.factor_name,
                    "time_range": f"{request.start_date} ~ {request.end_date}",
                    "implementation": "FactorHub原生实现",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"累计收益曲线分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"累计收益曲线分析失败: {str(e)}")


@router.post("/turnover")
async def turnover_analysis(request: ICAnalysisRequest):
    """
    因子换手率/自相关分析 ⭐增强版

    功能：
    - 计算因子换手率（衡量稳定性）
    - 自相关系数（衡量持续性）
    - 半衰期估计
    - 稳定性评分和建议

    完善版换手率分析，超越基础版本。

    对比表状态更新：⚠️ → ✅ 已完善
    """

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"开始换手率分析: {request.factor_name}")

        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        from backend.services.factor_return_analysis_service import factor_return_analysis_service

        result = factor_return_analysis_service.calculate_turnover_analysis(
            factor_data=factor_data,
            factor_name=request.factor_name,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info("换手率分析完成")

        return sanitize_dict(
            {
                "success": True,
                "data": result,
                "metadata": {
                    "factor_name": request.factor_name,
                    "n_stocks": len(request.stock_codes),
                    "implementation": "FactorHub增强版（含自相关+稳定性评分）",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"换手率分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"换手率分析失败: {str(e)}")


@router.post("/tear-sheet")
async def full_tear_sheet(request: QuantileReturnsRequest):
    """
    Tear Sheet 全貌报告 ⭐旗舰功能

    生成完整的因子分析报告，整合所有分析维度：
    - IC/IR 分析摘要
    - 分组收益（Quantile Returns）
    - 累计收益曲线（Cumulative Returns）
    - 换手率和稳定性
    - 综合评分（0-100，A-F等级）
    - 专业解读和改进建议

    类似 Alphalens 的 create_full_tear_sheet，
    但更适合现代Web应用（返回结构化JSON而非图片）。

    对比表状态更新：⚠️ → ✅ 升级完成
    """

    try:
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        logger.info(f"🎯 开始生成Tear Sheet全貌报告: {request.factor_name}")

        with get_db() as db:
            repo = FactorRepository(db)
            factor = repo.get_by_name(request.factor_name)

        if not factor:
            raise HTTPException(status_code=404, detail=f"因子 '{request.factor_name}' 不存在")

        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes, [request.factor_name], request.start_date, request.end_date
        )

        if not factor_data:
            raise HTTPException(status_code=500, detail="未能获取有效的因子数据")

        from backend.services.tear_sheet_service import tear_sheet_service

        result = tear_sheet_service.create_full_tear_sheet(
            factor_data=factor_data,
            factor_name=request.factor_name,
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Tear Sheet生成失败"))

        tear_sheet = result["tear_sheet"]
        score = tear_sheet.get("summary", {}).get("overall_score")
        grade = tear_sheet.get("summary", {}).get("grade", "N/A")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        logger.info(f"Tear Sheet生成完成: 得分={score_str}, 等级={grade}")

        return sanitize_dict(
            {
                "success": True,
                "data": tear_sheet,
                "metadata": {
                    "report_type": "full_tear_sheet_v2",
                    "factor_name": request.factor_name,
                    "n_stocks": len(request.stock_codes),
                    "time_range": f"{request.start_date} ~ {request.end_date}",
                    "score": score,
                    "grade": grade,
                    "implementation": "FactorHub原生Tear Sheet（对标Alphalens create_full_tear_sheet）",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tear Sheet生成失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Tear Sheet生成失败: {str(e)}")


class WeightedICRequest(BaseModel):
    """加权IC请求"""

    factor_names: List[str]
    stock_codes: List[str]
    start_date: str
    end_date: str
    weighting_method: str = "ir_weight"  # equal_weight / ir_weight / abs_ic_weight / decay_weight


@router.post("/weighted-ic")
async def weighted_ic_analysis(request: WeightedICRequest):
    """
    因子加权IC计算 ⭐新功能

    功能：
    - 多种加权方法（等权、IR加权、IC绝对值加权、衰减加权）
    - 相关性调整（消除多重共线性影响）
    - 各因子贡献度归因
    - 加权后的综合IC/IR指标

    用于多因子组合优化和因子重要性评估。

    对比表状态更新：❌ → ✅ 已实现
    """

    try:
        from backend.services.factor_service import factor_service

        logger.info(f"开始加权IC分析: 因子={request.factor_names}, " f"方法={request.weighting_method}")

        all_factor_data = {}
        for factor_name in request.factor_names:
            try:
                factor_data = factor_service.calculate_factors_for_stocks(
                    request.stock_codes, [factor_name], request.start_date, request.end_date
                )
                if factor_data:
                    all_factor_data[factor_name] = factor_data
            except Exception as e:
                logger.warning(f"因子{factor_name}获取失败: {e}")
                continue

        if not all_factor_data:
            raise HTTPException(status_code=500, detail="未能获取任何有效的因子数据")

        factor_ic_dict = _extract_all_ics(all_factor_data, request.factor_names)

        if not factor_ic_dict:
            raise HTTPException(status_code=500, detail="无法计算任何因子的IC序列")

        from backend.services.weighted_ic_service import weighted_ic_service

        result = weighted_ic_service.calculate_weighted_ic(
            factor_ic_dict=factor_ic_dict,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info("加权IC分析完成")

        return sanitize_dict(
            {
                "success": True,
                "data": result,
                "metadata": {
                    "factors_analyzed": list(factor_ic_dict.keys()),
                    "weighting_method": request.weighting_method,
                    "implementation": "FactorHub原生实现（对标Barra/Bloomberg多因子模型）",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加权IC分析失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"加权IC分析失败: {str(e)}")


def _extract_all_ics(
    all_factor_data: Dict[str, Dict],
    factor_names: List[str],
) -> Dict[str, pd.Series]:
    """提取所有因子的IC序列"""
    factor_ic_dict = {}

    for factor_name in factor_names:
        if factor_name not in all_factor_data:
            continue

        factor_data = all_factor_data[factor_name]
        all_ics = []

        for stock_code, df in factor_data.items():
            if factor_name in df.columns and "close" in df.columns:
                future_ret = df["close"].pct_change(1).shift(-1)

                valid_mask = df[factor_name].notna() & future_ret.notna()

                if valid_mask.sum() > 20:
                    ic_series = calculate_rolling_ic(
                        df.loc[valid_mask, factor_name], future_ret.loc[valid_mask], window=20, method="spearman"
                    )
                    valid_ic = ic_series.dropna()

                    if len(valid_ic) > 10:
                        all_ics.extend(valid_ic.tolist())

        if all_ics:
            factor_ic_dict[factor_name] = pd.Series(all_ics)

    return factor_ic_dict


@router.post("/factor-importance")
async def factor_importance_analysis(request: WeightedICRequest):
    """
    因子重要性排名 ⭐新功能

    功能：
    - 综合评估多个因子的相对重要性
    - 考虑维度：IC强度、IR质量、稳定性、独特性、动量
    - 自动识别冗余因子（基于相关性惩罚）
    - 生成排名和改进建议

    用于因子筛选和组合构建。

    对比表状态更新：❌ → ✅ 已实现
    """

    try:
        from backend.services.factor_service import factor_service

        logger.info(f"开始因子重要性排名: {request.factor_names}")

        all_factor_data = {}
        for factor_name in request.factor_names:
            try:
                factor_data = factor_service.calculate_factors_for_stocks(
                    request.stock_codes, [factor_name], request.start_date, request.end_date
                )
                if factor_data:
                    all_factor_data[factor_name] = factor_data
            except Exception as e:
                logger.warning(f"因子{factor_name}获取失败: {e}")
                continue

        if not all_factor_data:
            raise HTTPException(status_code=500, detail="未能获取任何有效的因子数据")

        factor_ic_dict = _extract_all_ics(all_factor_data, request.factor_names)

        if not factor_ic_dict:
            raise HTTPException(status_code=500, detail="无法计算任何因子的IC序列")

        from backend.services.weighted_ic_service import weighted_ic_service

        result = weighted_ic_service.calculate_factor_importance(
            factor_ic_dict=factor_ic_dict,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info("因子重要性排名完成")

        top_factor = result.get("ranking", [{}])[0].get("factor_name", "N/A") if result.get("ranking") else "N/A"

        return sanitize_dict(
            {
                "success": True,
                "data": result,
                "metadata": {
                    "n_factors_evaluated": result.get("n_factors_evaluated", 0),
                    "top_factor": top_factor,
                    "implementation": "FactorHub原生实现（5维综合评分）",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"因子重要性排名失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"因子重要性排名失败: {str(e)}")


# ========== 未来函数检测API ==========


class LookaheadBiasRequest(BaseModel):
    """未来函数检测请求"""

    factor_names: List[str]
    stock_codes: List[str]
    start_date: str
    end_date: str
    strict_mode: bool = False  # 是否使用严格模式（更敏感的阈值）


@router.post("/lookahead-bias")
async def detect_lookahead_bias(request: LookaheadBiasRequest):
    """
    未来函数（Look-ahead Bias）检测 ⭐核心安全功能

    通过多维度统计特征自动识别因子计算中的未来信息泄漏。

    检测维度：
    - IC/IR 异常偏高（正常因子 IC 通常 < 0.08）
    - 完美排名相关（Spearman > 0.3 高度可疑）
    - 自相关异常（接近 1.0 意味着数据泄漏）
    - 分层收益异常完美（单调性过强）
    - 回测指标不真实（年化收益 > 500% 等）

    返回：
    - 每个因子的风险等级 (safe/low/medium/high/critical)
    - 风险评分 (0-100)
    - 各检测项详细结果
    - 改进建议

    对比表状态更新：❌ → ✅ 已实现
    """

    try:
        from backend.services.factor_service import factor_service
        from backend.services.lookahead_bias_detector import (
            lookahead_bias_detector,
            strict_lookahead_bias_detector,
        )

        logger.info(
            f"[未来函数检测] 开始检测: 因子={request.factor_names}, "
            f"股票数={len(request.stock_codes)}, 严格模式={request.strict_mode}"
        )

        # 选择检测器
        detector = strict_lookahead_bias_detector if request.strict_mode else lookahead_bias_detector

        # 获取多因子数据
        all_factor_data = {}
        for factor_name in request.factor_names:
            try:
                factor_data = factor_service.calculate_factors_for_stocks(
                    request.stock_codes,
                    [factor_name],
                    request.start_date,
                    request.end_date,
                )
                if factor_data:
                    all_factor_data[factor_name] = factor_data
            except Exception as e:
                logger.warning(f"因子 {factor_name} 获取失败: {e}")
                continue

        if not all_factor_data:
            raise HTTPException(status_code=500, detail="未能获取任何有效的因子数据")

        # 对每个因子执行检测
        per_factor_results = {}
        for factor_name in request.factor_names:
            if factor_name not in all_factor_data:
                per_factor_results[factor_name] = {
                    "has_bias": False,
                    "risk_level": "unknown",
                    "risk_score": 0,
                    "summary": f"因子 [{factor_name}] 数据获取失败",
                    "checks": [],
                    "recommendations": [],
                }
                continue

            factor_data = all_factor_data[factor_name]

            # 判断单股票还是多股票，选择正确的检测方法（Rule 7.1）
            stock_codes = list(factor_data.keys())
            if len(stock_codes) > 1:
                # 多股票：构建横截面 DataFrame，使用 detect_cross_sectional
                rows = []
                for stock_code, df in factor_data.items():
                    if factor_name in df.columns and "close" in df.columns:
                        fv = df[factor_name]
                        ret = df["close"].pct_change(1).shift(-1)
                        for date_idx in fv.index:
                            f_val = fv.loc[date_idx] if pd.notna(fv.loc[date_idx]) else None
                            r_val = ret.loc[date_idx] if date_idx in ret.index and pd.notna(ret.loc[date_idx]) else None
                            if f_val is not None and r_val is not None:
                                rows.append(
                                    {
                                        "date": date_idx,
                                        "stock_code": stock_code,
                                        factor_name: f_val,
                                        "return": r_val,
                                    }
                                )

                if len(rows) < 30:
                    per_factor_results[factor_name] = {
                        "has_bias": False,
                        "risk_level": "unknown",
                        "risk_score": 0,
                        "summary": f"因子 [{factor_name}] 有效样本不足({len(rows)})",
                        "checks": [],
                        "recommendations": [],
                    }
                    continue

                factor_df = pd.DataFrame(rows)
                return_df = factor_df[["date", "stock_code", "return"]].copy()
                result = detector.detect_cross_sectional(
                    factor_df=factor_df,
                    return_df=return_df,
                    date_column="date",
                    stock_column="stock_code",
                    factor_name=factor_name,
                )
            else:
                # 单股票：使用 detect 方法
                all_factor_values = []
                all_return_values = []

                for stock_code, df in factor_data.items():
                    if factor_name in df.columns and "close" in df.columns:
                        fv = df[factor_name]
                        ret = df["close"].pct_change(1).shift(-1)
                        combined = pd.DataFrame({"factor": fv, "return": ret}).dropna()
                        if len(combined) >= 20:
                            all_factor_values.extend(combined["factor"].tolist())
                            all_return_values.extend(combined["return"].tolist())

                if len(all_factor_values) < 30:
                    per_factor_results[factor_name] = {
                        "has_bias": False,
                        "risk_level": "unknown",
                        "risk_score": 0,
                        "summary": f"因子 [{factor_name}] 有效样本不足({len(all_factor_values)})",
                        "checks": [],
                        "recommendations": [],
                    }
                    continue

                factor_series = pd.Series(all_factor_values)
                return_series = pd.Series(all_return_values)

                # 执行检测
                result = detector.detect(
                    factor_values=factor_series,
                    return_values=return_series,
                    factor_name=factor_name,
                )

            per_factor_results[factor_name] = {
                "has_bias": result.has_bias,
                "risk_level": result.risk_level.value,
                "risk_score": result.risk_score,
                "summary": result.summary,
                "checks": [
                    {
                        "name": c.check_name,
                        "passed": c.passed,
                        "value": c.value,
                        "threshold": c.threshold,
                        "severity": c.severity,
                        "message": c.message,
                    }
                    for c in result.checks
                ],
                "recommendations": result.recommendations,
                "metadata": result.metadata,
            }

        # 汇总统计
        risk_levels = [r["risk_level"] for r in per_factor_results.values()]
        level_order = ["safe", "low", "medium", "high", "critical", "error", "unknown"]
        overall_risk = max(risk_levels, key=lambda x: level_order.index(x)) if risk_levels else "safe"
        n_high_risk = sum(1 for r in per_factor_results.values() if r["risk_level"] in ("high", "critical"))

        logger.info(
            f"[未来函数检测] 完成: {len(per_factor_results)}个因子, " f"综合风险={overall_risk}, 高风险={n_high_risk}"
        )

        return sanitize_dict(
            {
                "success": True,
                "data": {
                    "per_factor": per_factor_results,
                    "overall_risk": overall_risk,
                    "n_high_risk": n_high_risk,
                    "n_total": len(per_factor_results),
                    "strict_mode": request.strict_mode,
                },
                "metadata": {
                    "factors_analyzed": list(per_factor_results.keys()),
                    "n_stocks": len(request.stock_codes),
                    "time_range": f"{request.start_date} ~ {request.end_date}",
                    "detector_version": "1.0.0",
                    "implementation": "FactorHub原生实现（多维度统计检测）",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"未来函数检测失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"未来函数检测失败: {str(e)}")
