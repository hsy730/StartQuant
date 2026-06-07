"""
A股板块识别工具

统一提供板块枚举、识别函数和板块自适应参数查询。
消除 data_service、smart_preprocessing_detector、smart_slippage_detector 中的重复定义。
"""
from enum import Enum
from typing import Dict, Optional


class MarketBoard(Enum):
    """A股市场板块枚举"""
    MAIN = "main"           # 主板 (60xxxx, 00xxxx)
    CHINEXT = "chinext"     # 创业板 (30xxxx)
    STAR = "star"           # 科创板 (68xxxx)
    BSE = "bse"             # 北交所 (8xxxx, 4xxxx)
    UNKNOWN = "unknown"     # 未知板块


def detect_market_board(stock_code: str) -> MarketBoard:
    """
    根据股票代码识别所属市场板块

    Args:
        stock_code: 股票代码，如 "600519"、"000001"、"300750" 等

    Returns:
        MarketBoard 枚举值
    """
    if not stock_code or not isinstance(stock_code, str):
        return MarketBoard.UNKNOWN

    code = stock_code.strip()

    if code.startswith("60"):
        return MarketBoard.MAIN      # 沪市主板
    elif code.startswith("00"):
        return MarketBoard.MAIN      # 深市主板
    elif code.startswith("30"):
        return MarketBoard.CHINEXT   # 创业板
    elif code.startswith("68"):
        return MarketBoard.STAR      # 科创板
    elif code.startswith("8") or code.startswith("4"):
        return MarketBoard.BSE       # 北交所
    else:
        return MarketBoard.UNKNOWN


def get_board_n_sigma(board: MarketBoard) -> float:
    """
    获取板块自适应的 MAD 去极值 n_sigma 参数

    根据项目规则：
    - 主板: n_sigma = 3.0（标准波动）
    - 创业板: n_sigma = 2.8（高波动，收紧20%）
    - 科创板: n_sigma = 2.7（更高波动，收紧25%）
    - 北交所: n_sigma = 2.5（最高波动，收紧33%）

    Args:
        board: 市场板块

    Returns:
        n_sigma 参数值
    """
    board_config = {
        MarketBoard.MAIN: 3.0,
        MarketBoard.CHINEXT: 2.8,
        MarketBoard.STAR: 2.7,
        MarketBoard.BSE: 2.5,
        MarketBoard.UNKNOWN: 3.0,  # 未知板块使用主板默认值
    }
    return board_config.get(board, 3.0)


def get_board_slippage_multiplier(board: MarketBoard) -> float:
    """
    获取板块自适应的滑点乘数

    Args:
        board: 市场板块

    Returns:
        滑点乘数
    """
    slippage_config = {
        MarketBoard.MAIN: 1.0,
        MarketBoard.CHINEXT: 1.2,
        MarketBoard.STAR: 1.5,
        MarketBoard.BSE: 2.0,
        MarketBoard.UNKNOWN: 1.0,
    }
    return slippage_config.get(board, 1.0)
