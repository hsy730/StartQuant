"""
数据预处理配置API

提供：
1. 智能参数推荐接口
2. 配置验证接口
3. 预设模板查询接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging

router = APIRouter(prefix="/api/preprocessing", tags=["数据预处理"])
logger = logging.getLogger(__name__)


class PreprocessingConfigRequest(BaseModel):
    """预处理配置请求"""
    stock_codes: List[str] = Field(..., description="股票代码列表")
    factor_names: List[str] = Field(..., description="因子名称列表")
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD")
    
    # 可选的用户覆盖参数（如果提供则使用，否则使用智能推荐）
    mode: str = Field("smart", description="模式: smart/auto/custom")
    user_config: Optional[Dict[str, Any]] = Field(None, description="用户自定义配置（覆盖智能推荐）")


class SmartRecommendationResponse(BaseModel):
    """智能推荐响应"""
    success: bool
    data: Dict[str, Any]


@router.post("/recommend", response_model=SmartRecommendationResponse)
async def get_smart_recommendation(request: PreprocessingConfigRequest):
    """
    获取智能预处理参数推荐
    
    根据输入的股票池和因子特征，自动推荐最优的去极值/中性化/标准化参数
    
    示例请求:
    ```json
    {
        "stock_codes": ["300001.SZ", "300002.SZ", "600036.SH"],
        "factor_names": ["momentum_20", "volatility"],
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "mode": "smart"
    }
    ```
    """
    try:
        from backend.services.factor_service import factor_service
        from backend.services.smart_preprocessing_detector import smart_detector
        
        logger.info(f"获取智能推荐: {len(request.stock_codes)}只股票, {request.factor_names}")
        
        # 1. 获取因子数据（用于分析特征）
        factor_data = factor_service.calculate_factors_for_stocks(
            request.stock_codes,
            request.factor_names,
            request.start_date,
            request.end_date,
            rolling_window=None,  # 不需要标准化，要原始数据
        )
        
        if not factor_data:
            raise HTTPException(status_code=400, detail="无法获取因子数据")
        
        # 2. 智能分析并生成推荐
        recommendation = smart_detector.recommend_config(
            factor_data=factor_data,
            factor_names=request.factor_names,
            user_preference=None,
        )
        
        # 3. 如果用户提供了自定义配置，进行合并
        final_config = recommendation.config_dict.copy()
        if request.user_config:
            final_config.update(request.user_config)
            logger.info("用户提供了自定义配置，将覆盖智能推荐")
        
        # 4. 生成人类可读的报告
        report = smart_detector.get_config_summary(recommendation)
        
        return {
            "success": True,
            "data": {
                "recommended_config": recommendation.config_dict,
                "final_config": final_config,  # 合并后的最终配置
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "warnings": recommendation.warnings,
                "data_characteristics": {
                    "market_board": recommendation.data_characteristics.market_board.value,
                    "n_stocks": recommendation.data_characteristics.n_stocks,
                    "n_dates": recommendation.data_characteristics.n_dates,
                    "factor_volatility": round(recommendation.data_characteristics.factor_volatility, 4),
                    "is_fat_tailed": recommendation.data_characteristics.is_fat_tailed,
                    "outlier_ratio": round(recommendation.data_characteristics.outlier_ratio, 4),
                    "n_industries": recommendation.data_characteristics.n_industries,
                    "min_industry_size": recommendation.data_characteristics.min_industry_size,
                },
                "report": report,
                "presets": _get_available_presets(),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"智能推荐失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"智能推荐失败: {str(e)}")


@router.get("/presets")
async def get_available_presets():
    """获取可用的预设配置模板"""
    return {
        "success": True,
        "data": _get_available_presets()
    }


def _get_available_presets() -> Dict[str, Any]:
    """获取预设配置列表"""
    return {
        "smart": {
            "name": "🤖 智能推荐（默认）",
            "description": "根据数据特征自动选择最优参数",
            "icon": "auto",
            "suitable_for": ["大多数场景", "快速探索", "初学者"],
        },
        "conservative": {
            "name": "🛡️ 保守型",
            "description": "强去极值 + 双重中性化，适合严谨研究",
            "config": {
                "winsorize_method": "mad",
                "winsorize_n_sigma": 2.5,
                "enable_market_cap_neutralization": True,
                "enable_industry_neutralization": True,
                "standardize_method": "zscore",
                "handle_missing": "fill_median",
                "min_samples": 20,
            },
            "icon": "shield",
            "suitable_for": ["正式研究报告", "监管报送", "风险控制"],
        },
        "aggressive": {
            "name": "🚀 激进型",
            "description": "弱去极值 + 无中性化，保留更多信号",
            "config": {
                "winsorize_method": "percentile",
                "winsorize_limits": [0.005, 0.995],
                "enable_market_cap_neutralization": False,
                "enable_industry_neutralization": False,
                "standardize_method": "zscore",
                "handle_missing": "fill_zero",
                "min_samples": 5,
            },
            "icon": "rocket",
            "suitable_for": ["探索性分析", "发现新Alpha", "高频策略"],
        },
        "ml_model": {
            "name": "🤖 ML专用",
            "description": "适合机器学习模型训练的数据预处理",
            "config": {
                "winsorize_method": "mad",
                "winsorize_n_sigma": 3.0,
                "enable_market_cap_neutralization": True,
                "enable_industry_neutralization": True,
                "standardize_method": "zscore",
                "handle_missing": "fill_median",
                "min_samples": 15,
            },
            "icon": "machine_learning",
            "suitable_for": ["XGBoost/LightGBM", "神经网络", "特征工程"],
        },
        "chinext_optimized": {
            "name": "📈 创业板优化",
            "description": "针对创业板高波动特性的专用配置",
            "config": {
                "winsorize_method": "mad",
                "winsorize_n_sigma": 2.8,  # 更严格
                "enable_market_cap_neutralization": True,
                "enable_industry_neutralization": True,
                "standardize_method": "zscore",
                "handle_missing": "fill_median",
                "min_samples": 15,
            },
            "icon": "stock",
            "suitable_for": ["创业板选股", "成长股策略", "高Beta因子"],
        },
    }


@router.post("/validate")
async def validate_config(config: Dict[str, Any]):
    """验证用户自定义配置的合理性"""
    warnings = []
    suggestions = []
    
    # 检查去极值参数
    if config.get("winsorize_n_sigma", 3.0) < 2.0:
        warnings.append("⚠️ 去极值强度过低(<2σ)，可能保留过多噪声")
    elif config.get("winsorize_n_sigma", 3.0) > 5.0:
        warnings.append("⚠️ 去极值强度过高(>5σ)，可能丢失有效信号")
    
    # 检查中性化组合
    mc_neutral = config.get("enable_market_cap_neutralization", False)
    ind_neutral = config.get("enable_industry_neutralization", False)
    
    if not mc_neutral and not ind_neutral:
        suggestions.append("💡 建议至少启用一种中性化以消除已知风险因子")
    
    # 检查标准化方法
    std_method = config.get("standardize_method", "zscore")
    if std_method == "rank" and config.get("enable_industry_neutralization"):
        warnings.append("⚠️ Rank标准化与行业中性化同时使用可能导致信息损失")
    
    return {
        "success": True,
        "data": {
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "suggestions": suggestions,
            "risk_level": "low" if len(warnings) <= 1 else ("medium" if len(warnings) <= 2 else "high"),
        }
    }
