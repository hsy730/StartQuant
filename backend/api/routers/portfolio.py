"""
组合分析API路由
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from backend.utils.serialization import safe_numeric_value, sanitize_dict
from backend.utils.safe_math import safe_divide, safe_ir, safe_series_divide
from backend.utils.ic_calculator import calculate_rolling_ic
from scipy.stats import spearmanr
from backend.utils.return_calculator import calculate_future_return
from backend.services.portfolio_analysis_service import portfolio_analysis_service
from backend.services.weight_optimizer_service import WeightOptimizer

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== 数据模型 ==========


class OptimizeWeightsRequest(BaseModel):
    """权重优化请求"""

    stock_code: str
    factors: List[str]
    start_date: str
    end_date: str
    method: str = "equal_weight"
    rebalance_freq: str = "monthly"


class CompositeScoreRequest(BaseModel):
    """计算综合得分请求"""

    stock_code: str
    factors: List[str]
    start_date: str
    end_date: str


class CompareMethodsRequest(BaseModel):
    """对比权重方法请求"""

    stock_code: str
    factors: List[str]
    start_date: str
    end_date: str
    methods: List[str] = ["equal_weight", "ic_weight"]


# ========== API端点 ==========


@router.post("/optimize-weights")
async def optimize_weights(request: OptimizeWeightsRequest):
    """优化权重"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db
        import pandas as pd
        import numpy as np

        # 获取股票数据
        stock_data = data_service.get_stock_data(request.stock_code, request.start_date, request.end_date)

        if stock_data is None or len(stock_data) == 0:
            raise HTTPException(status_code=404, detail="未获取到数据")

        # 从数据库获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor_defs = {}
            for factor_name in request.factors:
                factor = repo.get_by_name(factor_name)
                if factor:
                    factor_defs[factor_name] = factor

        if not factor_defs:
            raise HTTPException(status_code=400, detail="未找到任何有效的因子定义")

        # 计算所有因子的值
        factor_values = {}
        for factor_name, factor_def in factor_defs.items():
            try:
                values = factor_service.calculator.calculate(stock_data.copy(), factor_def.code)
                if values is not None and len(values.dropna()) > 0:
                    factor_values[factor_name] = values
            except Exception as e:
                logger.warning(f"因子 {factor_name} 计算失败: {e}")
                continue

        if not factor_values:
            raise HTTPException(status_code=400, detail="没有有效的因子数据")

        # 计算收益率序列（使用次日收益率用于IC计算，IC衡量因子对未来收益的预测力）
        returns = calculate_future_return(stock_data)  # 次日收益率（预测目标）

        # 根据方法计算权重（委托WeightOptimizer统一入口，消除代码重复）
        optimizer = WeightOptimizer()
        result_weights = optimizer.calculate_weights(
            factor_values, request.factors, method=request.method, returns=returns
        )
        weights = result_weights["weights"]

        # 归一化权重
        total_weight = sum(weights.values())
        logger.debug(f"归一化前总权重: {total_weight:.4f}")
        weights = {k: safe_divide(v, total_weight, default=0.0) for k, v in weights.items()}
        logger.debug(f"归一化后权重: {weights}")
        logger.debug(f"最终权重总和: {sum(weights.values()):.4f}")

        # 计算组合因子值和性能指标
        # 构建DataFrame用于计算，使用 stock_data 的索引
        factor_df = pd.DataFrame(index=stock_data.index)

        for factor_name, values in factor_values.items():
            factor_df[factor_name] = values

        # 计算加权组合因子（NaN不填充为0，符合规则7.7）
        weighted_sum = pd.Series(0.0, index=factor_df.index)
        weight_sum = pd.Series(0.0, index=factor_df.index)
        for factor_name, weight in weights.items():
            if factor_name in factor_df.columns:
                valid = factor_df[factor_name].notna()
                weighted_sum[valid] += factor_df.loc[valid, factor_name] * weight
                weight_sum[valid] += weight
        weighted_factor = safe_series_divide(weighted_sum, weight_sum, fill_value=np.nan)

        weighted_factor = weighted_factor.dropna()

        # 对齐数据 - 使用共同的索引
        common_index = weighted_factor.index.intersection(returns.index)

        if len(common_index) < 3:
            raise HTTPException(status_code=400, detail=f"有效数据点太少（{len(common_index)}个），无法计算组合指标")

        aligned_factor = weighted_factor.loc[common_index]
        aligned_returns = returns.loc[common_index]

        # 移除 NaN 值
        valid_mask = ~(aligned_factor.isna() | aligned_returns.isna())
        aligned_factor = aligned_factor[valid_mask]
        aligned_returns = aligned_returns[valid_mask]

        if len(aligned_factor) > 3:
            # 计算组合IC（使用Spearman，符合规则7.1）
            portfolio_ic_result = spearmanr(aligned_factor, aligned_returns)
            portfolio_ic = float(portfolio_ic_result[0]) if not np.isnan(portfolio_ic_result[0]) else None

            # 计算组合收益率（因子的平均收益）
            portfolio_return = aligned_returns.mean()

            # 计算组合IR (IC均值 / IC标准差)（使用Spearman，符合规则7.1/7.30）
            ic_series = calculate_rolling_ic(aligned_factor, aligned_returns, window=20, method="spearman")
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            portfolio_ir = safe_ir(float(ic_mean), float(ic_std), default=None)
        else:
            portfolio_ic = None
            portfolio_return = None
            portfolio_ir = None

        result = {
            "weights": weights,
            "method": request.method,
            "factors": request.factors,
            "stock_code": request.stock_code,
            "metrics": {
                "return": safe_numeric_value(portfolio_return),
                "ic": safe_numeric_value(portfolio_ic),
                "ir": safe_numeric_value(portfolio_ir),
            },
        }

        # 计算综合得分（使用优化后的权重）
        try:
            composite_score_result = portfolio_analysis_service.calculate_combined_factor_score(
                factor_data=factor_values, weights=weights, normalize=True
            )

            # 转换为列表格式
            if hasattr(composite_score_result, "index"):
                composite_score = {
                    "dates": composite_score_result.index.astype(str).tolist(),
                    "values": composite_score_result.values.tolist(),
                }
            else:
                composite_score = {"values": list(composite_score_result)}

            # 计算统计指标
            values = composite_score.get("values", [])
            if len(values) > 0:
                import numpy as np

                composite_stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            else:
                composite_stats = {}

            # 转换 numpy 类型
            composite_score = sanitize_dict(composite_score)
            composite_stats = sanitize_dict(composite_stats)

        except Exception as e:
            logger.warning(f"计算综合得分失败: {e}")
            composite_score = None
            composite_stats = {}

        # 添加综合得分到结果中
        result["composite_score"] = composite_score
        result["composite_stats"] = composite_stats

        # 转换 numpy 类型为 Python 原生类型，以避免 JSON 序列化错误
        result = sanitize_dict(result)

        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/composite-score")
async def calculate_composite_score(request: CompositeScoreRequest):
    """计算综合得分"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db

        # 获取股票数据
        stock_data = data_service.get_stock_data(request.stock_code, request.start_date, request.end_date)

        if stock_data is None or len(stock_data) == 0:
            raise HTTPException(status_code=404, detail="未获取到数据")

        # 从数据库获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor_defs = {}
            for factor_name in request.factors:
                factor = repo.get_by_name(factor_name)
                if factor:
                    factor_defs[factor_name] = factor

        if not factor_defs:
            raise HTTPException(status_code=400, detail="未找到任何有效的因子定义")

        # 计算所有因子的值
        factor_data = {}
        for factor_name, factor_def in factor_defs.items():
            try:
                values = factor_service.calculator.calculate(stock_data.copy(), factor_def.code)
                if values is not None:
                    factor_data[factor_name] = values
            except Exception as e:
                logger.warning(f"计算因子 {factor_name} 失败: {e}")
                continue

        if not factor_data:
            raise HTTPException(status_code=400, detail="没有有效的因子数据")

        # 使用等权重（简化）
        weights = {f: 1.0 / len(factor_data) for f in factor_data.keys()}

        # 调用综合得分计算
        result = portfolio_analysis_service.calculate_combined_factor_score(
            factor_data=factor_data, weights=weights, normalize=True
        )

        # 转换为列表
        if hasattr(result, "index"):
            score_list = {"dates": result.index.astype(str).tolist(), "values": result.values.tolist()}
        else:
            score_list = {"values": list(result)}

        # 转换 numpy 类型为 Python 原生类型，以避免 JSON 序列化错误
        score_list = sanitize_dict(score_list)

        return {"success": True, "data": score_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-methods")
async def compare_weight_methods(request: CompareMethodsRequest):
    """对比权重方法 - 基于IC/IR指标评估不同权重优化方法的效果"""
    try:
        from backend.services.data_service import data_service
        from backend.services.factor_service import factor_service
        from backend.repositories.factor_repository import FactorRepository
        from backend.core.database import get_db
        import pandas as pd
        import numpy as np

        # 获取股票数据
        stock_data = data_service.get_stock_data(request.stock_code, request.start_date, request.end_date)

        if stock_data is None or len(stock_data) == 0:
            raise HTTPException(status_code=404, detail="未获取到数据")

        # 从数据库获取因子定义
        with get_db() as db:
            repo = FactorRepository(db)
            factor_defs = {}
            for factor_name in request.factors:
                factor = repo.get_by_name(factor_name)
                if factor:
                    factor_defs[factor_name] = factor

        if not factor_defs:
            raise HTTPException(status_code=400, detail="未找到任何有效的因子定义")

        # 计算所有因子的值
        factor_data = {}
        for factor_name, factor_def in factor_defs.items():
            try:
                values = factor_service.calculator.calculate(stock_data.copy(), factor_def.code)
                if values is not None and len(values.dropna()) > 0:
                    factor_data[factor_name] = values
            except Exception as e:
                logger.warning(f"计算因子 {factor_name} 失败: {e}")
                continue

        if not factor_data:
            raise HTTPException(status_code=400, detail="没有有效的因子数据")

        # 计算次日收益率（用于IC计算，IC衡量因子对未来收益的预测力）
        returns = calculate_future_return(stock_data)  # 次日收益率（预测目标）

        # 对每种权重方法进行测试（委托WeightOptimizer统一入口，消除代码重复）
        optimizer = WeightOptimizer()
        results = {}

        # 构建因子DataFrame（循环外一次性构建，避免重复创建）
        factor_df = pd.DataFrame(index=stock_data.index)
        for factor_name, values in factor_data.items():
            factor_df[factor_name] = values

        for method in request.methods:
            try:
                # 1. 计算该方法的因子权重
                result_weights = optimizer.calculate_weights(
                    factor_data, request.factors, method=method, returns=returns
                )
                weights = result_weights["weights"]

                # 2. 构建加权组合因子（factor_df已在循环外构建）

                # 计算加权组合（NaN不填充为0，符合规则7.7）
                weighted_sum = pd.Series(0.0, index=factor_df.index)
                weight_sum = pd.Series(0.0, index=factor_df.index)
                for factor_name, weight in weights.items():
                    if factor_name in factor_df.columns:
                        valid = factor_df[factor_name].notna()
                        weighted_sum[valid] += factor_df.loc[valid, factor_name] * weight
                        weight_sum[valid] += weight
                weighted_factor = safe_series_divide(weighted_sum, weight_sum, fill_value=np.nan)

                weighted_factor = weighted_factor.dropna()

                # 3. 计算组合因子的IC/IR统计
                aligned = pd.DataFrame({"factor": weighted_factor, "returns": returns}).dropna()

                if len(aligned) >= 20:
                    # 计算IC时间序列（使用Spearman，符合规则7.1/7.30）
                    ic_series = calculate_rolling_ic(
                        aligned["factor"], aligned["returns"], window=20, method="spearman"
                    )

                    # 计算统计指标
                    ic_mean = ic_series.mean()
                    ic_std = ic_series.std()
                    ir = safe_ir(float(ic_mean), float(ic_std), default=None)

                    # 注意：此处计算的是IC代理指标，非真实投资组合收益率，
                    # 无法直接使用 empyrical（需要真实收益率序列）。
                    # 如有真实组合收益率，应通过 risk_metrics.calculate_risk_metrics() 计算。
                    results[method] = {
                        "ic_mean": float(ic_mean),
                        "ic_std": float(ic_std),
                        "ir": float(ir) if ir is not None else None,
                        "ic_annualized": float(ic_mean * 252),  # 年化IC（非真实收益率）
                        "ic_volatility_annualized": float(ic_std * np.sqrt(252)),  # 年化IC标准差（非真实波动率）
                        "information_ratio": float(ir) if ir is not None else None,  # IR（信息比率，非夏普比率）
                    }
                else:
                    results[method] = {
                        "ic_mean": None,
                        "ic_std": None,
                        "ir": None,
                        "ic_annualized": None,
                        "ic_volatility_annualized": None,
                        "information_ratio": None,
                    }

            except Exception as e:
                logger.warning(f"方法 {method} 计算失败: {e}")
                import traceback

                logger.debug(traceback.format_exc())
                results[method] = {
                    "ic_mean": None,
                    "ic_std": None,
                    "ir": None,
                    "ic_annualized": None,
                    "ic_volatility_annualized": None,
                    "information_ratio": None,
                }

        return sanitize_dict({"success": True, "data": {"results": results}})
    except Exception as e:
        import traceback

        logger.error(f"方法对比失败: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
