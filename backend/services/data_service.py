"""
数据服务模块 - 股票数据获取与缓存
"""
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import akshare as ak

from backend.core.settings import settings
from backend.services.cache_service import cache_service
from backend.services.data_preprocessing_service import data_preprocessing_service


class DataService:
    """数据服务类 - 负责股票数据获取和缓存"""

    def __init__(self):
        self.cache_dir = settings.AKSHARE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_service = cache_service
        self.preprocessing = data_preprocessing_service

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
            else:
                # 尝试自动识别
                # 添加市场前缀
                if stock_code.startswith("6"):
                    symbol = "sh" + stock_code
                elif stock_code.startswith(("0", "3")):
                    symbol = "sz" + stock_code
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

            # 数据预处理
            df = self._preprocess_data(df)

            # 保存到智能缓存
            if use_cache and settings.AKSHARE_CACHE_ENABLED:
                cache_key = self._get_cache_key(stock_code, start_date, end_date)
                self._save_to_cache(df, cache_key)

            return df

        except Exception as e:
            raise ValueError(f"获取股票 {stock_code} 数据失败: {e}")

    def _normalize_stock_code(self, code: str) -> str:
        """标准化股票代码格式"""
        code = code.strip().upper()
        if not code.endswith((".SH", ".SZ")):
            # 自动判断上海或深圳
            if code.startswith("6"):
                return f"{code}.SH"
            elif code.startswith(("0", "3")):
                return f"{code}.SZ"
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
                print(f"Warning: 获取股票 {code} 数据失败: {e}")
                return code, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, code): code for code in stock_codes}
            for future in as_completed(futures):
                code, df = future.result()
                if df is not None:
                    result[code] = df

        return result

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if settings.DATA_FILL_MISSING:
            df = df.ffill()

        if settings.DATA_OUTLIER_DETECTION:
            price_columns = ["open", "high", "low", "close"]
            window = 20
            n_sigma = settings.DATA_OUTLIER_N_SIGMA
            for col in price_columns:
                if col not in df.columns:
                    continue
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                rolling_std = rolling_std.replace(0, float("nan")).ffill().bfill()
                lower_bound = rolling_mean - n_sigma * rolling_std
                upper_bound = rolling_mean + n_sigma * rolling_std
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
            except Exception:
                pass
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

        total_shares = None
        try:
            pure_code = stock_code.replace(".SH", "").replace(".SZ", "")
            info_df = ak.stock_individual_info_em(symbol=pure_code)
            for _, row in info_df.iterrows():
                if "总股本" in str(row.iloc[0]):
                    total_shares = float(row.iloc[1])
                    break
        except Exception:
            pass

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
