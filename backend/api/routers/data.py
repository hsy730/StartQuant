"""
数据管理API路由
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.data_service import data_service
from backend.utils.serialization import sanitize_dict

logger = logging.getLogger(__name__)

AKSHARE_TIMEOUT = 30  # akshare API调用超时时间（秒）


def _call_akshare_with_timeout(func, *args, **kwargs):
    """带超时的akshare API调用"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=AKSHARE_TIMEOUT)
        except FuturesTimeoutError:
            logger.warning(f"akshare API调用超时: {func.__name__}")
            return None


router = APIRouter()


# ========== 预设股票池定义 ==========

STOCK_POOLS = {
    "sz50": {
        "name": "上证50成分股",
        "description": "上证50指数成分股（50只），超大盘蓝筹代表，速度最快",
        "symbol": "000016",
    },
    "cyb50": {
        "name": "创业板50成分股",
        "description": "创业板50指数成分股（50只），创业板龙头代表，速度最快",
        "symbol": "399673",
    },
    "hs300_sample": {
        "name": "沪深300成分股（前50只）",
        "description": "沪深300前50只成分股，大盘蓝筹代表，快速体验横截面IC",
        "symbol": "000300",
        "limit": 50,
    },
    "hs300": {
        "name": "沪深300成分股（全部）",
        "description": "沪深300全部300只成分股，大盘蓝筹代表，耗时较长",
        "symbol": "000300",
    },
    "zz500_sample": {
        "name": "中证500成分股（前50只）",
        "description": "中证500前50只成分股，中盘成长代表，快速体验横截面IC",
        "symbol": "000905",
        "limit": 50,
    },
    "zz500": {
        "name": "中证500成分股（全部）",
        "description": "中证500全部500只成分股，中盘成长代表，耗时较长",
        "symbol": "000905",
    },
    "zz1000_sample": {
        "name": "中证1000成分股（前50只）",
        "description": "中证1000前50只成分股，小盘价值代表，快速体验横截面IC",
        "symbol": "000852",
        "limit": 50,
    },
    "zz1000": {
        "name": "中证1000成分股（全部）",
        "description": "中证1000全部1000只成分股，小盘价值代表，耗时很长",
        "symbol": "000852",
    },
}


# ========== 数据模型 ==========


class StockDataRequest(BaseModel):
    """获取股票数据请求"""

    code: str
    start_date: str
    end_date: str


# ========== API端点 ==========


@router.get("/stock/{code}")
async def get_stock_data(code: str, start_date: str, end_date: str):
    """
    获取股票数据

    参数:
    - code: 股票代码
    - start_date: 开始日期 (YYYY-MM-DD)
    - end_date: 结束日期 (YYYY-MM-DD)
    """
    try:
        data = data_service.get_stock_data(
            stock_code=code, start_date=start_date, end_date=end_date
        )

        if data is None or len(data) == 0:
            raise HTTPException(status_code=404, detail="未获取到数据")

        # 转换为JSON格式
        data_dict = {
            "index": data.index.astype(str).tolist(),
            "columns": data.columns.tolist(),
            "data": data.values.tolist(),
        }

        return sanitize_dict({"success": True, "data": data_dict})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock-pools")
async def get_stock_pools():
    """获取可用的预设股票池列表"""
    pools = []
    for pool_id, pool_info in STOCK_POOLS.items():
        pools.append(
            {
                "id": pool_id,
                "name": pool_info["name"],
                "description": pool_info["description"],
            }
        )
    return {"success": True, "data": pools}


@router.get("/stock-pools/{pool_id}/stocks")
async def get_stock_pool_stocks(pool_id: str):
    """
    获取指定股票池的成分股列表

    参数:
    - pool_id: 股票池ID (hs300, zz500, zz1000, sz50, cyb50)
    """
    import akshare as ak

    if pool_id not in STOCK_POOLS:
        raise HTTPException(status_code=404, detail=f"股票池 {pool_id} 不存在")

    pool_info = STOCK_POOLS[pool_id]
    symbol = pool_info["symbol"]

    try:
        # 尝试从缓存获取
        cache_key = f"stock_pool_{pool_id}"
        cached = data_service.cache_service.get(cache_key)
        if cached is not None:
            return {"success": True, "data": cached, "cached": True}

        # 策略1: 使用 index_stock_cons（支持所有指数代码，包括深证）
        df = None
        code_col = None
        name_col = None

        try:
            df = _call_akshare_with_timeout(ak.index_stock_cons, symbol=symbol)
            if df is not None and len(df) > 0:
                # 列名: 品种代码, 品种名称, 纳入日期
                for col in df.columns:
                    if "代码" in str(col) or "code" in str(col).lower():
                        code_col = col
                    if "名称" in str(col) or "name" in str(col).lower():
                        name_col = col
                logger.info(
                    f"index_stock_cons 成功获取 {symbol}, {len(df)} 条, 列: {df.columns.tolist()}"
                )
        except Exception as e:
            logger.warning(f"index_stock_cons 获取 {symbol} 失败: {e}")

        # 策略2: fallback 使用 index_stock_cons_csindex
        if df is None or len(df) == 0:
            try:
                df = _call_akshare_with_timeout(
                    ak.index_stock_cons_csindex, symbol=symbol
                )
                if df is not None and len(df) > 0:
                    code_col = None
                    name_col = None
                    for col in df.columns:
                        if "代码" in str(col) or "code" in str(col).lower():
                            code_col = col
                        if "名称" in str(col) or "name" in str(col).lower():
                            name_col = col
                    logger.info(
                        f"index_stock_cons_csindex 成功获取 {symbol}, {len(df)} 条"
                    )
            except Exception as e:
                logger.warning(f"index_stock_cons_csindex 获取 {symbol} 失败: {e}")

        # 策略3: fallback 使用 index_stock_cons_weight_csindex
        if df is None or len(df) == 0:
            try:
                df = _call_akshare_with_timeout(
                    ak.index_stock_cons_weight_csindex, symbol=symbol
                )
                if df is not None and len(df) > 0:
                    code_col = None
                    name_col = None
                    for col in df.columns:
                        if "代码" in str(col) or "code" in str(col).lower():
                            code_col = col
                        if "名称" in str(col) or "name" in str(col).lower():
                            name_col = col
                    logger.info(
                        f"index_stock_cons_weight_csindex 成功获取 {symbol}, {len(df)} 条"
                    )
            except Exception as e:
                logger.warning(
                    f"index_stock_cons_weight_csindex 获取 {symbol} 失败: {e}"
                )

        if df is None or len(df) == 0:
            raise HTTPException(
                status_code=500, detail=f"所有数据源均无法获取指数 {symbol} 的成分股"
            )

        # 如果没找到代码列，使用第一列
        if code_col is None:
            code_col = df.columns[0]
        if name_col is None and len(df.columns) > 1:
            name_col = df.columns[1]

        stocks = []
        limit = pool_info.get("limit")  # 限制数量（如 _sample 池只取前N只）
        for _, row in df.iterrows():
            if limit and len(stocks) >= limit:
                break
            code = str(row[code_col]).strip()
            # 跳过非纯数字的代码（如指数代码 000300 等）
            if not code.isdigit() or len(code) != 6:
                continue
            # 标准化股票代码：添加 .SH / .SZ 后缀
            if code.startswith("6"):
                full_code = f"{code}.SH"
            elif code.startswith(("0", "3")):
                full_code = f"{code}.SZ"
            else:
                full_code = code

            name = str(row[name_col]).strip() if name_col else ""
            stocks.append({"code": full_code, "name": name, "short_code": code})

        # 缓存1天（成分股变化不频繁）
        data_service.cache_service.set(cache_key, stocks, ttl=24 * 60 * 60)

        return {"success": True, "data": stocks, "count": len(stocks)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票池成分股失败: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计"""
    try:
        stats = data_service.get_cache_stats()
        return {"success": True, "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/cleanup")
async def cleanup_cache():
    """清理过期缓存"""
    try:
        cleaned = data_service.cleanup_cache()
        return {
            "success": True,
            "data": {"cleaned_count": cleaned},
            "message": f"已清理 {cleaned} 个过期缓存",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """清空全部缓存"""
    try:
        cleared = data_service.clear_cache()
        return {
            "success": True,
            "data": {"cleared_count": cleared},
            "message": f"已清空 {cleared} 个缓存",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
