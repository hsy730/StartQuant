"""
因子编排 & 排序学习 & 模型管理 API 路由

三大新功能端点：
1. /api/orchestrator/*  — AlphaMiner 式一键验证流水线
2. /api/stock-ranker/*  — StockRanker 式 GBDT 排序学习
3. /api/models/*         — ML 模型注册中心 CRUD

对比表状态更新：
- AlphaMiner: ❌ → ✅ (通过 FactorOrchestrator)
- StockRanker: ❌ → ✅ (通过 StockRankerService)
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import traceback

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# =====================================================================
#  1. Factor Orchestrator API（AlphaMiner 替代）
# =====================================================================

class OrchestratorValidateRequest(BaseModel):
    """一键验证请求"""
    expression: str                          # 因子表达式
    stock_codes: List[str]                   # 股票代码列表
    start_date: str                          # 开始日期
    end_date: str                            # 结束日期
    factor_name: Optional[str] = None        # 因子名称
    enable_lookahead_detection: bool = True  # 未来函数检测
    enable_ic_analysis: bool = True          # IC/IR 分析
    enable_alphalens: bool = True            # Alphalens 分析
    enable_quantile_backtest: bool = True    # 分组回测
    enable_tear_sheet: bool = True           # Tear Sheet 报告
    fail_fast_on_bias: bool = True           # 发现未来函数立即终止


class OrchestratorBatchRequest(BaseModel):
    """批量验证请求"""
    expressions: List[str]
    stock_codes: List[str]
    start_date: str
    end_date: str
    parallel: bool = False


@router.post("/orchestrator/validate")
async def orchestrator_validate(request: OrchestratorValidateRequest):
    """
    一键因子验证 ⭐ AlphaMiner 替代功能

    将以下步骤自动串联为一条流水线：
      因子计算 → 未来函数检测 → IC/IR分析 → Alphalens全量分析 → 分组回测 → Tear Sheet报告

    输入：一个因子表达式 + 股票池 + 时间范围
    输出：完整的验证报告（JSON + Markdown）

    对比 BigQuant AlphaMiner:
      AlphaMiner: SQL 表达式 → 固定流程 → 固定格式输出
      本服务: 公式表达式 → 可配置流程 → JSON + Markdown 双模式 + 未来函数检测
    """
    try:
        from backend.services.factor_orchestrator_service import (
            FactorOrchestrator,
            OrchestratorConfig,
            factor_orchestrator,
        )

        config = OrchestratorConfig(
            enable_lookahead_detection=request.enable_lookahead_detection,
            enable_ic_analysis=request.enable_ic_analysis,
            enable_alphalens=request.enable_alphalens,
            enable_quantile_backtest=request.enable_quantile_backtest,
            enable_tear_sheet=request.enable_tear_sheet,
            fail_fast_on_bias=request.fail_fast_on_bias,
        )

        orchestrator = FactorOrchestrator(config=config)

        result = orchestrator.validate(
            expression=request.expression,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            factor_name=request.factor_name,
        )

        return {
            "success": result["status"] != "ERROR",
            "data": result,
        }

    except Exception as e:
        logger.error(f"一键验证失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"一键验证失败: {str(e)}")


@router.post("/orchestrator/batch-validate")
async def orchestrator_batch_validate(request: OrchestratorBatchRequest):
    """批量验证多个因子表达式，返回对比表"""
    try:
        from backend.services.factor_orchestrator_service import (
            FactorOrchestrator,
            factor_orchestrator,
        )

        orchestrator = FactorOrchestrator()
        result = orchestrator.batch_validate(
            expressions=request.expressions,
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            parallel=request.parallel,
        )

        return {"success": True, "data": result}

    except Exception as e:
        logger.error(f"批量验证失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"批量验证失败: {str(e)}")


# =====================================================================
#  2. StockRanker API（StockRanker 替代）
# =====================================================================

class RankerTrainRequest(BaseModel):
    """排序模型训练请求"""
    model_name: str = "stock_ranker"
    feature_cols: Optional[List[str]] = None   # 特征列（None=自动检测）
    label_col: str = "forward_return_5d"
    date_col: str = "date"
    group_col: str = "date"
    objective: str = "rank:ndcg"
    learning_rate: float = 0.05
    max_depth: int = 6
    n_estimators: int = 200
    validation_split: float = 0.2
    tags: Optional[List[str]] = None
    enable_bias_check: bool = True


class RankerPredictRequest(BaseModel):
    """排序预测请求"""
    model_id: str
    top_n: int = 50
    feature_cols: Optional[List[str]] = None
    features: List[Dict[str, Any]] = []  # 特征数据，每条记录为一个样本的特征字典


@router.post("/stock-ranker/train")
async def stock_ranker_train(request: RankerTrainRequest):
    """
    训练 GBDT 排序模型 ⭐ StockRanker 替代功能

    基于 XGBoost Ranking 目标训练 Learning-to-Rank 模型。
    支持 rank:pairwise / rank:ndcg / rank:map 三种目标函数。

    对比 BigQuant StockRanker:
      StockRanker: 封闭式 GBDT，内置特征工程
      本服务: 开放式 XGBoost Ranking，用户自定义特征，
             支持 SHAP 解释 + 未来函数检测 + 预测→回测闭环
    """
    try:
        from backend.services.stock_ranker_service import (
            stock_ranker_service,
            XGB_AVAILABLE,
            RankTrainingConfig,
            RankObjective,
        )

        if not XGB_AVAILABLE or stock_ranker_service is None:
            raise HTTPException(
                status_code=503,
                detail="XGBoost 未安装，StockRanker 服务不可用。请执行: pip install xgboost",
            )

        from backend.services.data_service import data_service

        # 构造特征数据（示例：使用基础因子作为特征）
        # 实际生产中应从数据库/因子库中获取预计算的特征矩阵
        logger.info(f"[StockRanker API] 开始训练: model={request.model_name}, objective={request.objective}")

        config = RankTrainingConfig(
            objective=request.objective,
            learning_rate=request.learning_rate,
            max_depth=request.max_depth,
            n_estimators=request.n_estimators,
        )

        # 注意：实际调用时需要前端传入完整的特征 DataFrame
        # 这里返回配置确认和说明
        return {
            "success": True,
            "data": {
                "message": "StockRanker 训练接口就绪。请通过 /api/stock-ranker/train-with-data 端点传入完整数据。",
                "config_used": {
                    "objective": config.objective,
                    "learning_rate": config.learning_rate,
                    "max_depth": config.max_depth,
                    "n_estimators": config.n_estimators,
                },
                "supported_objectives": [o.value for o in RankObjective],
                "requirements": {
                    "feature_df": "DataFrame with date, stock_code columns and numeric feature columns",
                    "label_col": request.label_col,
                    "format": "long format: one row per (date, stock) pair",
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"StockRanker 训练准备失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class RankerTrainWithDataRequest(BaseModel):
    """带数据的排序模型训练请求"""
    feature_data: List[Dict[str, Any]] = []
    label_col: str = "label"
    training_config: Optional[Dict[str, Any]] = None


@router.post("/stock-ranker/train-with-data")
async def stock_ranker_train_with_data(request: RankerTrainWithDataRequest):
    """
    带数据的排序模型训练（完整版）

    Request body 应包含：
    - feature_data: List[Dict] 或 DataFrame-like 的特征数据
    - label_col: 标签列名
    - training_config: 训练参数
    """
    try:
        from backend.services.stock_ranker_service import (
            stock_ranker_service,
            XGB_AVAILABLE,
            RankTrainingConfig,
        )
        import pandas as pd

        if not XGB_AVAILABLE or stock_ranker_service is None:
            raise HTTPException(status_code=503, detail="XGBoost 未安装")

        # 从请求中提取数据
        raw_data = request.feature_data
        if not raw_data:
            raise HTTPException(status_code=400, detail="feature_data 不能为空")

        df = pd.DataFrame(raw_data)
        label_col = request.label_col
        tc = request.training_config or {}

        config = RankTrainingConfig(
            objective=tc.get("objective", "rank:ndcg"),
            learning_rate=tc.get("learning_rate", 0.05),
            max_depth=tc.get("max_depth", 6),
            n_estimators=tc.get("n_estimators", 200),
        )

        result = stock_ranker_service.train(
            feature_df=df,
            label_col=label_col,
            config=config,
            model_name=tc.get("model_name", "stock_ranker"),
            tags=tc.get("tags"),
            enable_bias_check=tc.get("enable_bias_check", True),
        )

        return {
            "success": True,
            "data": {
                "model_id": result.model_id,
                "status": result.status.value,
                "n_samples": result.n_samples,
                "n_features": result.n_features,
                "duration_seconds": round(result.duration_seconds, 2),
                "training_metrics": result.training_metrics,
                "feature_importance": dict(list(result.feature_importance.items())[:20]),
                "train_period": result.train_period,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"StockRanker 训练失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")


@router.post("/stock-ranker/predict")
async def stock_ranker_predict(request: RankerPredictRequest):
    """使用已训练的排序模型进行预测"""
    try:
        from backend.services.stock_ranker_service import (
            stock_ranker_service,
            XGB_AVAILABLE,
        )
        import pandas as pd

        if not XGB_AVAILABLE or stock_ranker_service is None:
            raise HTTPException(status_code=503, detail="XGBoost 未安装")

        raw_data = request.features
        if isinstance(raw_data, list) and len(raw_data) > 0:
            features = pd.DataFrame(raw_data)
        else:
            raise HTTPException(status_code=400, detail="需要提供 features 数据")

        result = stock_ranker_service.predict(
            model_id=request.model_id,
            features=features,
            feature_cols=request.feature_cols,
            top_n=request.top_n,
        )

        return {
            "success": True,
            "data": {
                "model_id": request.model_id,
                "top_n_stocks": result.top_n_stocks.to_dict(orient="records"),
                "metrics": result.metrics,
                "generated_at": result.generated_at,
            },
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"模型不存在: {e}")
    except Exception as e:
        logger.error(f"StockRanker 预测失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/stock-ranker/models")
async def stock_ranker_list_models():
    """列出所有已训练的排序模型"""
    try:
        from backend.services.stock_ranker_service import (
            stock_ranker_service,
            XGB_AVAILABLE,
        )

        if not XGB_AVAILABLE or stock_ranker_service is None:
            raise HTTPException(status_code=503, detail="XGBoost 未安装")

        models = stock_ranker_service.list_models()
        return {"success": True, "data": models}

    except Exception as e:
        logger.error(f"列出模型失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stock-ranker/models/{model_id}")
async def stock_ranker_delete_model(model_id: str):
    """删除指定模型"""
    try:
        from backend.services.stock_ranker_service import (
            stock_ranker_service,
            XGB_AVAILABLE,
        )

        if not XGB_AVAILABLE or stock_ranker_service is None:
            raise HTTPException(status_code=503, detail="XGBoost 未安装")

        success = stock_ranker_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="模型不存在或删除失败")

        return {"success": True, "message": f"模型 {model_id} 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock-ranker/explain/{model_id}")
async def stock_ranker_explain(model_id: str):
    """获取模型的 SHAP 解释和特征重要性"""
    try:
        from backend.services.stock_ranker_service import (
            stock_ranker_service,
            XGB_AVAILABLE,
        )

        if not XGB_AVAILABLE or stock_ranker_service is None:
            raise HTTPException(status_code=503, detail="XGBoost 未安装")

        explanation = stock_ranker_service.explain_model(model_id)
        return {"success": True, "data": explanation}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
    except Exception as e:
        logger.error(f"模型解释失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
#  3. Model Registry API（ML 模型管理）
# =====================================================================

@router.get("/models")
async def list_models(
    framework: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 100,
):
    """列出所有已注册的 ML 模型"""
    try:
        from backend.services.model_registry import model_registry

        models = model_registry.list_models(
            framework=framework,
            stage=stage,
            limit=limit,
        )
        stats = model_registry.get_statistics()

        return {
            "success": True,
            "data": {
                "models": models,
                "statistics": stats,
            },
        }
    except Exception as e:
        logger.error(f"列出模型失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/metadata")
async def get_model_metadata(model_id: str):
    """获取模型元数据"""
    try:
        from backend.services.model_registry import model_registry

        meta = model_registry.get_metadata(model_id)
        if not meta:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        return {"success": True, "data": meta}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取元数据失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """获取指定模型的版本历史"""
    try:
        from backend.services.model_registry import model_registry

        versions = model_registry.get_version_history(model_name)
        return {"success": True, "data": versions}
    except Exception as e:
        logger.error(f"获取版本历史失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/promote")
async def promote_model(model_id: str, body: Dict[str, str]):
    """将模型提升到新的生命周期阶段"""
    try:
        from backend.services.model_registry import model_registry

        target_stage = body.get("stage", "staging")
        success = model_registry.promote(model_id, target_stage)
        if not success:
            raise HTTPException(status_code=404, detail="模型不存在或提升失败")

        return {
            "success": True,
            "message": f"模型 {model_id} 已提升至 {target_stage}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型提升失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """删除指定模型"""
    try:
        from backend.services.model_registry import model_registry

        success = model_registry.delete(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="模型不存在或删除失败")

        return {"success": True, "message": f"模型 {model_id} 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/stats")
async def registry_statistics():
    """获取模型注册中心的统计信息"""
    try:
        from backend.services.model_registry import model_registry

        stats = model_registry.get_statistics()
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error(f"统计信息获取失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
