"""
FastAPI主应用
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import sys
import logging
from pathlib import Path
import numpy as np
import json

logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.database import init_db  # noqa: E402
from backend.core.settings import settings  # noqa: E402
from backend.core.logging_config import setup_logging  # noqa: E402
from backend.services.factor_service import factor_service  # noqa: E402
# 确保所有模型在 init_db() 前被导入，以便 create_all() 创建对应表
import backend.models.mining_checkpoint  # noqa: F401

# 导入路由
from .routers import (  # noqa: E402
    factors,
    analysis,
    mining,
    portfolio,
    backtest,
    data,
    orchestrator,
    preprocessing_api,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 初始化日志系统（控制台 + 文件双输出）
    setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)

    # 启动时初始化
    logger.info("启动FastAPI服务...")
    init_db()
    factor_service.load_preset_factors()
    app.json_encoder = NumpyJSONEncoder

    # 恢复中断任务：将上次重启前未完成的任务标记为 aborted
    try:
        from backend.core.database import get_db
        from backend.repositories.mining_task_repository import MiningTaskRepository

        with get_db() as db:
            repo = MiningTaskRepository(db)
            aborted_count = repo.abort_interrupted_tasks()
            if aborted_count > 0:
                logger.warning(f"启动恢复: 已将 {aborted_count} 个中断任务标记为 aborted")
    except Exception as e:
        logger.warning(f"启动恢复中断任务时出错（非致命）: {e}")

    # 确保内存中的挖掘状态是干净的（新进程应该为空，但 reload 场景下可能有残留）
    try:
        from .routers import mining as mining_router
        n_stale = len([t for t in mining_router.mining_tasks.values() if t.get('status') in ('running', 'pending')])
        if n_stale > 0:
            logger.warning(f"启动检测: 发现 {n_stale} 个残留运行中任务，正在清理...")
            mining_router.shutdown_all_mining_tasks()
            mining_router.mining_tasks.clear()
            mining_router.mining_services.clear()
            logger.info("启动检测: 残留任务已清理完毕，内存状态已重置")
    except Exception as e:
        logger.warning(f"启动清理残留任务时出错（非致命）: {e}")

    logger.info("数据库和预置因子加载完成")

    yield

    # 关闭时清理：强制取消运行中的挖掘任务，避免线程死锁
    logger.info("关闭FastAPI服务...")
    try:
        from .routers.mining import shutdown_all_mining_tasks
        shutdown_all_mining_tasks()
    except Exception as e:
        logger.warning(f"挖掘任务清理时出错（非致命）: {e}")


# 自定义JSON编码器来处理numpy浮点数值
class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# 配置JSON编码器
def jsonable_encoder_with_numpy(obj, *args, **kwargs):
    """处理numpy类型的JSON编码器"""
    try:
        return jsonable_encoder(
            obj,
            *args,
            **kwargs,
            custom_serializer=lambda x: NumpyJSONEncoder().default(x),
        )
    except (TypeError, ValueError):
        return jsonable_encoder(obj, *args, **kwargs)


# 创建FastAPI应用
app = FastAPI(
    title="FactorFlow API",
    description="股票因子分析系统 REST API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=JSONResponse,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],  # 允许的前端来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================
# Static File Serving (for production)
# ============================================
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "react-antd" / "dist"

if FRONTEND_DIST.exists():
    logger.info(f"Serving static files from: {FRONTEND_DIST}")
    # Mount static assets directory (js, css, images, etc.)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.exception_handler(404)
async def spa_fallback(request, exc):
    """SPA fallback - return index.html for 404 errors (non-API routes)"""
    # Only handle non-API routes for HTML requests
    if FRONTEND_DIST.exists() and not request.url.path.startswith(
        ("/api", "/docs", "/redoc", "/openapi.json")
    ):
        return FileResponse(FRONTEND_DIST / "index.html")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})


# 注册路由
app.include_router(factors.router, prefix="/api/factors", tags=["因子管理"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["因子分析"])
app.include_router(mining.router, prefix="/api/mining", tags=["因子挖掘"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["组合分析"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["策略回测"])
app.include_router(data.router, prefix="/api/data", tags=["数据管理"])
app.include_router(
    orchestrator.router, prefix="/api", tags=["编排器/排序学习/模型管理"]
)
# preprocessing_api 自带 prefix="/api/preprocessing"，无需额外前缀
app.include_router(preprocessing_api.router)


@app.get("/api")
async def api_root():
    """API 根路径"""
    return {
        "message": "FactorFlow API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# 全局异常处理
# 覆盖FastAPI的默认JSON响应编码器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"请求错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": str(exc), "detail": "服务器内部错误"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
