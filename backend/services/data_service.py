"""
数据服务模块 - 股票数据获取与缓存

Mask-First设计：在数据加载阶段即构建可交易性掩码(tradable_mask)，
确保所有下游计算（滚动窗口、相关系数、排名）都不会被涨跌停价格污染。

核心原理：
- 涨停/跌停价格不可交易，但会污染rolling/corr等窗口计算
- 即便事后删除这些行，计算时已经被污染
- 解决方案：在数据加载时就标记不可交易日，让所有算子接收并传递mask
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import akshare as ak

from backend.core.market_board import MarketBoard, detect_market_board
from backend.core.settings import settings
from backend.services.cache_service import cache_service
from backend.services.data_preprocessing_service import data_preprocessing_service

logger = logging.getLogger(__name__)


@dataclass
class TradableMaskConfig:
    """可交易性掩码配置"""
    check_limit_up: bool = True           # 是否检测涨停
    check_limit_down: bool = True         # 是否检测跌停
    check_suspended: bool = True          # 是否检测停牌
    check_new_stock: bool = True          # 是否过滤新股（上市<250天）
    new_stock_threshold: int = 250         # 新股上市天数阈值
    volume_threshold: float = 0.0         # 成交量阈值（<=此值视为停牌）


class DataService:
    """数据服务类 - 负责股票数据获取和缓存（Mask-First设计）"""

    def __init__(self):
        self.cache_dir = settings.AKSHARE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_service = cache_service
        self.preprocessing = data_preprocessing_service
        
        # 市场板块配置（涨跌幅限制）
        self._board_config = {
            MarketBoard.STAR: {
                "price_limit": 0.20,      # ±20%
                "description": "科创板市场",
            },
            MarketBoard.CHINEXT: {
                "price_limit": 0.20,      # ±20%
                "description": "创业板市场",
            },
            MarketBoard.BSE: {
                "price_limit": 0.30,      # ±30%
                "description": "北交所市场",
            },
            MarketBoard.MAIN: {
                "price_limit": 0.10,      # ±10%
                "description": "主板市场",
            },
        }

    def _get_cache_key(self, stock_code: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _get_cache_path(self, stock_code: str, start_date: str, end_date: str) -> Path:
        """生成缓存文件路径（保留向后兼容）"""
        cache_hash = self._get_cache_key(stock_code, start_date, end_date)
        return self.cache_dir / f"{cache_hash}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从智能缓存加载数据"""
        return self.cache_service.get(cache_key)

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str, ttl: Optional[int] = None) -> None:
        """保存数据到智能缓存"""
        if ttl is None:
            ttl = settings.CACHE_DEFAULT_TTL
        self.cache_service.set(cache_key, data, ttl=ttl)

    def get_stock_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取股票历史数据

        Args:
            stock_code: 股票代码，如 "000001" 或 "000001.SZ"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            use_cache: 是否使用缓存

        Returns:
            包含OHLCV数据的DataFrame
        """
        # 标准化股票代码
        stock_code = self._normalize_stock_code(stock_code)

        # 检查智能缓存
        if use_cache and settings.AKSHARE_CACHE_ENABLED:
            cache_key = self._get_cache_key(stock_code, start_date, end_date)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        # 从 akshare 获取数据
        try:
            if stock_code.endswith(".SH"):
                symbol = "sh" + stock_code.replace(".SH", "")
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",  # 前复权
                )
            elif stock_code.endswith(".SZ"):
                symbol = "sz" + stock_code.replace(".SZ", "")
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
            elif stock_code.endswith(".BJ"):
                # 北交所股票
                symbol = "bj" + stock_code.replace(".BJ", "")
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
            else:
                # 尝试自动识别
                # 添加市场前缀
                if stock_code.startswith("6"):
                    symbol = "sh" + stock_code
                elif stock_code.startswith(("0", "3")):
                    symbol = "sz" + stock_code
                elif stock_code.startswith(("4", "8")):
                    symbol = "bj" + stock_code
                else:
                    symbol = stock_code

                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )

            # 标准化列名
            df = self._standardize_columns(df)

            # ✅ 数据预处理（Mask-First设计 - 在此阶段构建tradable_mask）
            df = self._preprocess_data(df, stock_code=stock_code)

            # 保存到智能缓存
            if use_cache and settings.AKSHARE_CACHE_ENABLED:
                cache_key = self._get_cache_key(stock_code, start_date, end_date)
                self._save_to_cache(df, cache_key)

            return df

        except Exception as e:
            raise ValueError(f"获取股票 {stock_code} 数据失败: {e}")

    def get_stock_minute_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        period: str = "5",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        股票分钟级历史数据（基于东方财富）

        Args:
            stock_code: 股票代码，如 "000001" 或 "000001.SZ"
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，格式 "YYYY-MM-DD"
            period: 分钟周期，支持 "1"/"5"/"15"/"30"/"60"
            use_cache: 是否使用缓存

        Returns:
            包含分钟级OHLCV数据的DataFrame
        """
        stock_code = self._normalize_stock_code(stock_code)

        if use_cache and settings.AKSHARE_CACHE_ENABLED:
            cache_key = f"{self._get_cache_key(stock_code, start_date, end_date)}_min_{period}"
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        try:
            pure_code = stock_code.replace(".SH", "").replace(".SZ", "")
            df = ak.stock_zh_a_hist_min_em(
                symbol=pure_code,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            # 标准化列名
            df = self._standardize_columns(df)

            # 保存缓存（分钟级数据较短TTL）
            if use_cache and settings.AKSHARE_CACHE_ENABLED:
                cache_key = f"{self._get_cache_key(stock_code, start_date, end_date)}_min_{period}"
                self._save_to_cache(df, cache_key, ttl=2 * 60 * 60)  # 2小时TTL

            return df
        except Exception as e:
            raise ValueError(f"获取股票 {stock_code} 分钟级数据失败: {e}")

    def _normalize_stock_code(self, code: str) -> str:
        """标准化股票代码格式"""
        code = code.strip().upper()
        if not code.endswith((".SH", ".SZ", ".BJ")):
            # 自动判断上海、深圳或北交所
            if code.startswith("6"):
                return f"{code}.SH"
            elif code.startswith(("0", "3")):
                return f"{code}.SZ"
            elif code.startswith(("4", "8")):
                return f"{code}.BJ"
        return code

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame列名"""
        # akshare 返回的列名映射
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }

        df = df.rename(columns=column_mapping)

        # 确保日期列是datetime类型
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # 确保数值列是正确的类型
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_index()

    def get_multiple_stocks_data(
        self,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        max_workers: int = 15,
    ) -> dict[str, pd.DataFrame]:
        """
        获取多个股票的数据（并行）

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            max_workers: 并行线程数

        Returns:
            字典，key为股票代码，value为对应的DataFrame
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        result = {}

        def _fetch_one(code):
            try:
                return code, self.get_stock_data(code, start_date, end_date, use_cache)
            except Exception as e:
                logger.warning(f"获取股票 {code} 数据失败: {e}")
                return code, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, code): code for code in stock_codes}
            for future in as_completed(futures):
                code, df = future.result()
                if df is not None:
                    result[code] = df

        return result

    def _identify_market_board(self, stock_code: str) -> MarketBoard:
        """
        识别股票所属市场板块

        Args:
            stock_code: 股票代码（纯数字，如"000001"）

        Returns:
            MarketBoard枚举值
        """
        # 提取纯数字代码
        pure_code = stock_code.replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()

        board = detect_market_board(pure_code)

        if board == MarketBoard.UNKNOWN:
            logger.warning(f"无法识别股票 {stock_code} 的市场板块，默认为主板")
            return MarketBoard.MAIN

        return board

    def _detect_price_limits(
        self,
        df: pd.DataFrame,
        stock_code: str,
        config: Optional[TradableMaskConfig] = None
    ) -> pd.DataFrame:
        """
        检测涨跌停并构建可交易性掩码（Mask-First核心方法）

        这是解决A股涨跌停污染问题的关键：
        1. 计算理论涨跌停价格
        2. 检测实际是否触及涨跌停
        3. 标记停牌日
        4. 构建tradable_mask列供所有下游算子使用

        Args:
            df: OHLCV数据框（必须包含open/high/low/close/volume列）
            stock_code: 股票代码
            config: 掩码配置（可选）

        Returns:
            添加了以下列的DataFrame:
            - is_limit_up: 是否涨停（bool）
            - is_limit_down: 是否跌停（bool）
            - is_suspended: 是否停牌（bool）
            - tradable_mask: 可交易性掩码（bool，True=可交易）
        """
        if config is None:
            config = TradableMaskConfig()

        df = df.copy()

        # 确保必要的列存在
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

        # 1. 识别市场板块并获取涨跌幅限制
        board = self._identify_market_board(stock_code)
        price_limit = self._board_config[board]["price_limit"]

        # 2. 计算前一日收盘价（用于计算涨跌停价格）
        df["prev_close"] = df["close"].shift(1)

        # 3. 计算理论涨跌停价格
        df["limit_up_price"] = (df["prev_close"] * (1 + price_limit)).round(2)
        df["limit_down_price"] = (df["prev_close"] * (1 - price_limit)).round(2)

        # 4. 检测涨停
        if config.check_limit_up:
            # 一字涨停：最高价>=涨停价 且 最低价≈最高价（全天封死，允许浮点误差）
            df["is_limit_up"] = (
                (df["high"] >= df["limit_up_price"]) &
                np.isclose(df["low"], df["high"], atol=0.001)
            ).fillna(False)
            
            # 额外检测：收盘价==涨停价（触及涨停）
            df["touched_limit_up"] = (
                df["close"] >= df["limit_up_price"] * 0.999  # 允许0.1%误差
            ).fillna(False)
        else:
            df["is_limit_up"] = False
            df["touched_limit_up"] = False

        # 5. 检测跌停
        if config.check_limit_down:
            # 一字跌停：最低价<=跌停价 且 最低价≈最高价（允许浮点误差）
            df["is_limit_down"] = (
                (df["low"] <= df["limit_down_price"]) &
                np.isclose(df["low"], df["high"], atol=0.001)
            ).fillna(False)
            
            # 触及跌停
            df["touched_limit_down"] = (
                df["close"] <= df["limit_down_price"] * 1.001  # 允许0.1%误差
            ).fillna(False)
        else:
            df["is_limit_down"] = False
            df["touched_limit_down"] = False

        # 6. 检测停牌（成交量异常低或缺失）
        if config.check_suspended:
            df["is_suspended"] = (
                (df["volume"] <= config.volume_threshold) |
                df["volume"].isna()
            ).fillna(True)  # 缺失成交量视为停牌
        else:
            df["is_suspended"] = False

        # 7. 检测新股（可选：上市不足N天）
        # TODO: 当前实现假设数据框第一行即为上市日，这对于非完整历史数据是不准确的。
        # 应通过akshare获取真实上市日期（如 ak.stock_ipo_info()）来计算距上市日的天数。
        if config.check_new_stock and len(df) > 0:
            # 假设第一行就是上市日（简化处理，存在局限性）
            days_since_listing = (df.index - df.index[0]).days
            df["is_new_stock"] = days_since_listing < config.new_stock_threshold
        else:
            df["is_new_stock"] = False

        # 8. 构建核心的可交易性掩码（Mask-First的灵魂！）
        mask_conditions = []

        if config.check_limit_up:
            # 排除涨停日（买不进去）- 包括一字涨停和触及涨停
            mask_conditions.append(~df["is_limit_up"])
            mask_conditions.append(~df["touched_limit_up"])

        if config.check_limit_down:
            # 排除跌停日（卖不出来）- 包括一字跌停和触及跌停
            mask_conditions.append(~df["is_limit_down"])
            mask_conditions.append(~df["touched_limit_down"])

        if config.check_suspended:
            # 排除停牌日（无法交易）
            mask_conditions.append(~df["is_suspended"])

        if config.check_new_stock:
            # 排除新股（波动异常）
            mask_conditions.append(~df["is_new_stock"])

        # 所有条件都必须满足（AND逻辑）
        if mask_conditions:
            df["tradable_mask"] = pd.concat(mask_conditions, axis=1).all(axis=1)
        else:
            df["tradable_mask"] = True

        # 统计信息
        total_days = len(df)
        tradable_days = df["tradable_mask"].sum()
        limit_up_days = df["is_limit_up"].sum()
        limit_down_days = df["is_limit_down"].sum()
        suspended_days = df["is_suspended"].sum()

        logger.info(
            f"📊 Mask-First统计 [{stock_code}] | "
            f"总天数: {total_days} | "
            f"可交易: {tradable_days} ({tradable_days/total_days*100:.1f}%) | "
            f"涨停: {limit_up_days} ({limit_up_days/total_days*100:.1f}%) | "
            f"跌停: {limit_down_days} ({limit_down_days/total_days*100:.1f}%) | "
            f"停牌: {suspended_days} ({suspended_days/total_days*100:.1f}%) | "
            f"市场板块: {board.value}"
        )

        # 清理中间列（只保留关键列）
        columns_to_keep = [
            "open", "high", "low", "close", "volume",
            "amount", "pct_change", "turnover",  # 原始OHLCV
            "is_limit_up", "is_limit_down", "is_suspended",  # 状态标记
            "tradable_mask"  # 核心掩码
        ]
        
        # 保留存在的列
        final_columns = [col for col in columns_to_keep if col in df.columns]
        # 也保留其他可能存在的列
        other_columns = [col for col in df.columns if col not in final_columns and not col.startswith(("limit_", "prev_", "touched_", "is_new"))]
        
        return df[final_columns + other_columns]

    def _preprocess_data(self, df: pd.DataFrame, stock_code: str = "") -> pd.DataFrame:
        """
        数据预处理（增强版：集成Mask-First设计）

        处理顺序（符合业界标准）：
        Step 0: 构建tradable_mask ← 新增！最关键的一步
        Step 1: 缺失值填充
        Step 2: 异常值检测与处理
        """
        # ✅ Step 0: Mask-First - 构建可交易性掩码
        if stock_code:
            try:
                df = self._detect_price_limits(df, stock_code)
                logger.info(f"✅ Mask-First: 已为 {stock_code} 构建tradable_mask")
            except Exception as e:
                logger.warning(f"⚠️ Mask-First构建失败: {e}，将使用默认全True掩码")
                df["tradable_mask"] = True
                df["is_limit_up"] = False
                df["is_limit_down"] = False
                df["is_suspended"] = False
        else:
            # 无股票代码时，默认全部可交易（向后兼容）
            df["tradable_mask"] = True
            df["is_limit_up"] = False
            df["is_limit_down"] = False
            df["is_suspended"] = False

        # Step 1: 缺失值填充
        if settings.DATA_FILL_MISSING:
            df = df.ffill()

        # Step 2: 异常值检测与处理（使用MAD法，比3σ更抗异常值）
        if settings.DATA_OUTLIER_DETECTION:
            price_columns = ["open", "high", "low", "close"]
            window = 20
            n_sigma = settings.DATA_OUTLIER_N_SIGMA
            for col in price_columns:
                if col not in df.columns:
                    continue
                rolling_median = df[col].rolling(window=window, min_periods=1).median()
                # TODO: 滚动MAD的当前实现是近似值——每个偏差值使用的是不同滚动窗口的中位数，
                # 而非同一个窗口中位数。严格MAD应为：对同一窗口计算median，
                # 再计算 |x - median| 的median。向量化实现标准滚动MAD较复杂，
                # 当前近似在大多数场景下误差可接受，后续应考虑使用更精确的实现。
                mad = (df[col] - rolling_median).abs().rolling(window=window, min_periods=1).median() * 1.4826
                mad = mad.replace(0, float("nan")).ffill().bfill()
                lower_bound = rolling_median - n_sigma * mad
                upper_bound = rolling_median + n_sigma * mad
                df.loc[df[col] < lower_bound, col] = lower_bound[df[col] < lower_bound]
                df.loc[df[col] > upper_bound, col] = upper_bound[df[col] > upper_bound]

        return df

    def get_industry_classification(self, stock_codes: List[str]) -> Dict[str, str]:
        cache_key = "industry_classification_sw"
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return {code: cached[code] for code in stock_codes if code in cached}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        industry_map: Dict[str, str] = {}
        try:
            industry_df = ak.stock_board_industry_name_em()
            industry_names = industry_df["板块名称"].tolist()
        except Exception as e:
            raise ValueError(f"获取申万行业列表失败: {e}")

        def _fetch_one_industry(name):
            local_map = {}
            try:
                cons_df = ak.stock_board_industry_cons_em(symbol=name)
                if "代码" in cons_df.columns:
                    for _, row in cons_df.iterrows():
                        code = str(row["代码"]).strip()
                        local_map[code] = name
            except Exception as e:
                logger.debug(f"获取行业 {name} 成分股失败: {e}")
            return local_map

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_one_industry, name): name for name in industry_names}
            for future in as_completed(futures):
                local_map = future.result()
                industry_map.update(local_map)

        if industry_map:
            self._save_to_cache(industry_map, cache_key, ttl=30 * 24 * 60 * 60)

        return {code: industry_map.get(code, "") for code in stock_codes}

    def get_market_cap_data(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        stock_code = self._normalize_stock_code(stock_code)
        cache_key = self._get_cache_key(stock_code + "_mktcap", start_date, end_date)

        if settings.AKSHARE_CACHE_ENABLED:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        df = self.get_stock_data(stock_code, start_date, end_date, use_cache=True)
        df = df.copy()

        total_shares = None
        try:
            pure_code = stock_code.replace(".SH", "").replace(".SZ", "")
            info_df = ak.stock_individual_info_em(symbol=pure_code)
            for _, row in info_df.iterrows():
                if "总股本" in str(row.iloc[0]):
                    total_shares = float(row.iloc[1])
                    break
        except Exception as e:
            logger.debug(f"获取股票 {stock_code} 总股本信息失败: {e}")

        if total_shares is not None and "close" in df.columns:
            df["market_cap"] = df["close"] * total_shares

        if settings.AKSHARE_CACHE_ENABLED:
            self._save_to_cache(df, cache_key)

        return df

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return self.cache_service.get_stats()

    def cleanup_cache(self) -> int:
        """清理过期缓存"""
        return self.cache_service.cleanup_expired()

    def clear_cache(self) -> int:
        """清空所有缓存"""
        return self.cache_service.clear_all()

    def incremental_update(
        self,
        stock_code: str,
        existing_df: pd.DataFrame,
        end_date: str,
    ) -> pd.DataFrame:
        """
        增量更新股票数据

        Args:
            stock_code: 股票代码
            existing_df: 现有的数据框
            end_date: 新的结束日期

        Returns:
            更新后的数据框
        """
        # 获取现有数据的最后日期
        last_date = existing_df.index.max()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # 如果新日期在现有数据之前，直接返回现有数据
        if start_date > end_date:
            return existing_df

        # 获取新数据
        new_df = self.get_stock_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
        )

        # 增量合并
        combined_df = self.preprocessing.incremental_update(
            existing_df=existing_df,
            new_df=new_df,
        )

        return combined_df


# 全局数据服务实例
data_service = DataService()
