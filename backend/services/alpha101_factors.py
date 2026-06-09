# NOTE: Alpha101因子公式中的除零保护统一使用 .replace(0, np.nan)，
# 将不可计算值转为NaN而非极大值，与项目规范一致。
# np 在因子表达式执行上下文中可用。
from typing import Dict, List


def get_alpha101_factors() -> Dict[str, List[Dict]]:
    return {
        "Alpha101-动量反转": [
            {
                "name": "alpha001",
                "code": "TSRANK(SIGNEDPOWER(IF(RETURNS(close) < 0, STD(RETURNS(close), 20), close), 2), 5) - 0.5",
                "description": "Alpha#1: 条件波动率/价格的时序排名（负收益时用波动率替代价格）",
            },
            {
                "name": "alpha002",
                "code": "-1 * CORR(DELTA(np.log(volume), 2), (close - open) / open.replace(0, np.nan), 6)",
                "description": "Alpha#2: 量变化与日内收益率的负相关",
            },
            {
                "name": "alpha007",
                "code": "IF(SMA(close * volume, timeperiod=20) < volume, -1 * TSRANK(np.abs(DELTA(close, 7)), 60) * SIGN(DELTA(close, 7)), -1)",
                "description": "Alpha#7: 放量时的周度价格变化时序排名",
            },
            {
                "name": "alpha008",
                "code": "-1 * (SUM(open, 5) * SUM(RETURNS(close), 5) - REF(SUM(open, 5) * SUM(RETURNS(close), 5), 10))",
                "description": "Alpha#8: 开盘价收益动量差值",
            },
            {
                "name": "alpha009",
                "code": "IF(LLV(DELTA(close, 1), 5) > 0, DELTA(close, 1), IF(HHV(DELTA(close, 1), 5) < 0, DELTA(close, 1), -1 * DELTA(close, 1)))",
                "description": "Alpha#9: 条件性日收益方向（持续上涨看多，持续下跌看空，震荡反转）",
            },
            {
                "name": "alpha019",
                "code": "-1 * SIGN((close - REF(close, 7)) + DELTA(close, 7)) * (1 + SUM(RETURNS(close), 250))",
                "description": "Alpha#19: 周度价格变化方向与年累计收益的交互",
            },
            {
                "name": "alpha020",
                "code": "-1 * (open - REF(high, 1)) * (open - REF(close, 1)) * (open - REF(low, 1))",
                "description": "Alpha#20: 开盘缺口三因子乘积反转",
            },
            {
                "name": "alpha033",
                "code": "-1 * (1 - open / close.replace(0, np.nan))",
                "description": "Alpha#33: 日内阴线强度反转",
            },
            {
                "name": "alpha034",
                "code": "(1 - STD(RETURNS(close), 2) / STD(RETURNS(close), 5).replace(0, np.nan)) + (1 - DELTA(close, 1))",
                "description": "Alpha#34: 波动率比率与价格动量组合",
            },
            {
                "name": "alpha038",
                "code": "-1 * TSRANK(DELTA(close, 5), 20)",
                "description": "Alpha#38: 5日价格变化的时序排名反转",
            },
            {
                "name": "alpha046",
                "code": "-1 * CORR(RETURNS(close), SMA(volume, timeperiod=10), 5)",
                "description": "Alpha#46: 收益率与均量负相关",
            },
            {
                "name": "alpha050",
                "code": "-1 * TSRANK(DELTA(close, 3) / close.replace(0, np.nan), 3)",
                "description": "Alpha#50: 3日收益率时序排名反转",
            },
            {
                "name": "alpha051",
                "code": "-1 * TSRANK(DELTA(close, 1), 5)",
                "description": "Alpha#51: 日价格变化时序排名反转",
            },
            {
                "name": "alpha071",
                "code": "(close - SMA(close, timeperiod=24)) / SMA(close, timeperiod=24).replace(0, np.nan) * 100",
                "description": "Alpha#71: 价格偏离24日均线百分比",
            },
            {
                "name": "alpha084",
                "code": "SUM(close > REF(close, 1), 5) / 5 - 0.5",
                "description": "Alpha#84: 近5日上涨天数占比偏离",
            },
            {
                "name": "alpha088",
                "code": "(close - DELTA(close, 1)) / close.replace(0, np.nan)",
                "description": "Alpha#88: 昨日收盘占今日收盘比",
            },
        ],
        "Alpha101-量价关系": [
            {
                "name": "alpha003",
                "code": "-1 * CORR(open, volume, 10)",
                "description": "Alpha#3: 开盘价与成交量负相关",
            },
            {
                "name": "alpha012",
                "code": "SIGN(DELTA(volume, 1)) * (-1 * DELTA(close, 1))",
                "description": "Alpha#12: 量增价跌/量减价涨信号",
            },
            {
                "name": "alpha014",
                "code": "-1 * DELTA(RETURNS(close), 3) * CORR(open, volume, 10)",
                "description": "Alpha#14: 收益动量变化与量价相关交互",
            },
            {
                "name": "alpha015",
                "code": "-1 * SUM(CORR(high, volume, 3), 3)",
                "description": "Alpha#15: 高价与成交量滚动相关求和",
            },
            {
                "name": "alpha037",
                "code": "-1 * CORR(SMA(close, timeperiod=10), SMA(volume, timeperiod=10), 7)",
                "description": "Alpha#37: 均价与均量负相关",
            },
            {
                "name": "alpha044",
                "code": "-1 * CORR(HHV(high, 5), volume, 5)",
                "description": "Alpha#44: 近期最高价与成交量负相关",
            },
            {
                "name": "alpha045",
                "code": "-1 * CORR(DELTA(close, 5), DELTA(volume, 5), 5)",
                "description": "Alpha#45: 价格变化与成交量变化负相关",
            },
            {
                "name": "alpha047",
                "code": "-1 * CORR(RETURNS(close), volume, 5)",
                "description": "Alpha#47: 收益率与成交量负相关",
            },
            {
                "name": "alpha049",
                "code": "-1 * CORR(DELTA(close, 1), DELTA(volume, 1), 5)",
                "description": "Alpha#49: 日价格变化与日成交量变化负相关",
            },
            {
                "name": "alpha052",
                "code": "-1 * CORR(RETURNS(close), DELTA(volume, 2), 6)",
                "description": "Alpha#52: 收益率与2日量变化负相关",
            },
            {
                "name": "alpha053",
                "code": "-1 * DELTA(((close - low - (high - close)) / (close - low).replace(0, np.nan)) * volume, 1)",
                "description": "Alpha#53: 日内价格位置加权成交量变化",
            },
            {
                "name": "alpha055",
                "code": "-1 * CORR(DELTA(close, 1) / close.replace(0, np.nan), volume, 5)",
                "description": "Alpha#55: 日收益率与成交量负相关",
            },
            {
                "name": "alpha057",
                "code": "-1 * CORR(RETURNS(close), DELTA(volume, 1), 5)",
                "description": "Alpha#57: 收益率与日量变化负相关",
            },
            {
                "name": "alpha058",
                "code": "-1 * CORR(DELTA(close, 1), volume, 5)",
                "description": "Alpha#58: 日价格变化与成交量负相关",
            },
            {
                "name": "alpha062",
                "code": "-1 * CORR(close, volume, 10)",
                "description": "Alpha#62: 收盘价与成交量负相关",
            },
            {
                "name": "alpha063",
                "code": "-1 * CORR(SMA(close, timeperiod=5), SMA(volume, timeperiod=5), 10)",
                "description": "Alpha#63: 5日均价与5日均量负相关",
            },
        ],
        "Alpha101-波动率": [
            {
                "name": "alpha004",
                "code": "-1 * TSRANK(low, 9)",
                "description": "Alpha#4: 最低价时序排名反转",
            },
            {
                "name": "alpha018",
                "code": "-1 * (STD(np.abs(close - open), 5) + (close - open) + CORR(close, open, 10))",
                "description": "Alpha#18: 日内波幅标准差+方向+相关综合信号",
            },
            {
                "name": "alpha022",
                "code": "-1 * DELTA(CORR(high, volume, 5), 5) * STD(close, 20)",
                "description": "Alpha#22: 量价相关变化乘以波动率",
            },
            {
                "name": "alpha026",
                "code": "-1 * HHV(CORR(TSRANK(volume, 5), TSRANK(high, 5), 5), 3)",
                "description": "Alpha#26: 量价时序排名相关的近期最大值反转",
            },
            {
                "name": "alpha039",
                "code": "-1 * TSRANK(STD(RETURNS(close), 5), 20)",
                "description": "Alpha#39: 收益波动率时序排名反转",
            },
            {
                "name": "alpha040",
                "code": "-1 * STD(high, 10) * CORR(high, volume, 10)",
                "description": "Alpha#40: 高价波动率与量价相关乘积反转",
            },
            {
                "name": "alpha041",
                "code": "(high * low) ** 0.5 - (high + low + close) / 3",
                "description": "Alpha#41: 几何均价与典型价格之差",
            },
            {
                "name": "alpha074",
                "code": "TSRANK(CORR(SUM((close - open) ** 2, 5), SUM(volume, 5), 5), 5)",
                "description": "Alpha#74: 日内波幅与成交量相关的时序排名",
            },
            {
                "name": "alpha036",
                "code": "2.21 * CORR(close - open, DELTA(volume, 1), 15)",
                "description": "Alpha#36: 日内收益与量变化相关（15日窗口）",
            },
            {
                "name": "alpha095",
                "code": "STD(RETURNS(close), 20) / STD(RETURNS(close), 5).replace(0, np.nan)",
                "description": "Alpha#95: 长短期波动率比率",
            },
        ],
        "Alpha101-趋势强度": [
            {
                "name": "alpha017",
                "code": "-1 * TSRANK(close, 10) * DELTA(DELTA(close, 1), 1) * TSRANK(volume / SMA(close * volume, timeperiod=20).replace(0, np.nan), 5)",
                "description": "Alpha#17: 价格时序排名×加速度×量比三因子",
            },
            {
                "name": "alpha021",
                "code": "IF(AVE(close, 8) + STD(close, 8) < AVE(close, 2), -1, IF(AVE(close, 2) < AVE(close, 8) - STD(close, 8), 1, IF(volume / SMA(close * volume, timeperiod=20).replace(0, np.nan) >= 1, 1, -1)))",
                "description": "Alpha#21: 均线布林带+量比综合趋势信号",
            },
            {
                "name": "alpha023",
                "code": "IF(AVE(high, 20) < high, -1 * DELTA(high, 2), 0)",
                "description": "Alpha#23: 突破20日均价时的价格动量反转",
            },
            {
                "name": "alpha028",
                "code": "SCALE(CORR(SMA(close * volume, timeperiod=20), low, 5) + (high + low) / 2 - close)",
                "description": "Alpha#28: 均额与低价相关+价格偏离中值",
            },
            {
                "name": "alpha030",
                "code": "(1.0 - (SIGN(close - REF(close, 1)) + SIGN(REF(close, 1) - REF(close, 2)) + SIGN(REF(close, 2) - REF(close, 3)))) * SUM(volume, 5) / SUM(volume, 20).replace(0, np.nan)",
                "description": "Alpha#30: 连续方向反转×短期量比",
            },
            {
                "name": "alpha031",
                "code": "DECAY_LINEAR(-1 * DELTA(close, 10), 10) + (-1 * DELTA(close, 3))",
                "description": "Alpha#31: 衰减加权周度动量+短期动量反转",
            },
            {
                "name": "alpha032",
                "code": "SCALE(AVE(close, 7) - close) + 20 * SCALE(CORR((high + low + close) / 3, REF(close, 5), 230))",
                "description": "Alpha#32: 短期均值回归+长期量价相关",
            },
            {
                "name": "alpha035",
                "code": "TSRANK(volume, 32) * (1 - TSRANK(close + high - low, 16)) * (1 - TSRANK(RETURNS(close), 32))",
                "description": "Alpha#35: 量排名×价格范围排名×收益排名三因子",
            },
            {
                "name": "alpha043",
                "code": "TSRANK(volume / SMA(close * volume, timeperiod=20).replace(0, np.nan), 20) * TSRANK(-1 * DELTA(close, 7), 8)",
                "description": "Alpha#43: 量比时序排名×周度动量反转排名",
            },
            {
                "name": "alpha048",
                "code": "-1 * TSRANK(CORR(SMA(close, timeperiod=10), SMA(volume, timeperiod=10), 7), 4)",
                "description": "Alpha#48: 均价均量相关的时序排名反转",
            },
            {
                "name": "alpha068",
                "code": "-1 * TSRANK(CORR(high, SMA(close * volume, timeperiod=15), 9), 14)",
                "description": "Alpha#68: 高价与均额相关的时序排名反转",
            },
            {
                "name": "alpha072",
                "code": "(AVE(close, 12) - AVE(close, 26)) / AVE(close, 26).replace(0, np.nan)",
                "description": "Alpha#72: 12/26日均线偏离度（类MACD）",
            },
            {
                "name": "alpha073",
                "code": "TSRANK(DELTA(close, 1), 20) / TSRANK(volume, 20).replace(0, np.nan)",
                "description": "Alpha#73: 价格变化排名与成交量排名之比",
            },
        ],
        "Alpha101-综合信号": [
            {
                "name": "alpha024",
                "code": "IF(np.abs(DELTA(AVE(close, 100), 100) / REF(close, 100).replace(0, np.nan)) <= 0.05, -1 * (close - LLV(close, 100)), -1 * DELTA(close, 3))",
                "description": "Alpha#24: 横盘时用价格位置，趋势时用短期动量",
            },
            {
                "name": "alpha054",
                "code": "-1 * (low - close) ** 2 * volume / ((low - high) ** 2).replace(0, np.nan)",
                "description": "Alpha#54: 低价偏离度加权成交量反转",
            },
            {
                "name": "alpha077",
                "code": "AVE(close, 7) - close",
                "description": "Alpha#77: 7日均值回归信号",
            },
            {
                "name": "alpha078",
                "code": "SCALE(CORR(SUM((close - open) ** 2, 5), SUM(volume, 5), 5))",
                "description": "Alpha#78: 日内波幅与成交量相关的标准化信号",
            },
            {
                "name": "alpha085",
                "code": "-1 * TSRANK(CORR(close, volume, 10), 5)",
                "description": "Alpha#85: 量价相关的时序排名反转",
            },
            {
                "name": "alpha087",
                "code": "-1 * CORR(RETURNS(close), SMA(volume, timeperiod=20), 6)",
                "description": "Alpha#87: 收益率与20日均量负相关",
            },
            {
                "name": "alpha098",
                "code": "-1 * CORR(RETURNS(close), SMA(volume, timeperiod=5), 10)",
                "description": "Alpha#98: 收益率与5日均量负相关（10日窗口）",
            },
            {
                "name": "alpha100",
                "code": "SCALE(-1 * DELTA(close, 3))",
                "description": "Alpha#100: 3日价格变化标准化反转",
            },
            {
                "name": "alpha101",
                "code": "(close - open) / (high - low).replace(0, np.nan) * volume",
                "description": "Alpha#101: 日内方向强度加权成交量",
            },
        ],
    }
