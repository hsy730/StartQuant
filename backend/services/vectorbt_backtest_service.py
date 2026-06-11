"""
基于 vectorbt 的回测服务
更准确、更高效的回测引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import traceback
import gc

import vectorbt as vbt

from backend.services.risk_metrics import (
    calculate_risk_metrics,
    calculate_volatility as _calc_volatility,
    _empty_metrics,
)
from backend.services.smart_slippage_detector import smart_slippage_detector, SlippageRecommendation
from backend.utils.return_calculator import calculate_future_return
from backend.utils.safe_math import safe_divide, safe_series_divide
from backend.utils.ic_calculator import calculate_rolling_ic

logger = logging.getLogger(__name__)


# 频率配置映射表
_FREQ_CONFIG = {
    "D": {
        "vbt_freq": "D",
        "annual_bars": 252,
        "rolling_window": 252,
        "rolling_min_periods": 150,
        "description": "日频",
    },
    "5min": {
        "vbt_freq": "5T",
        "annual_bars": 252 * 48,  # 每交易日48根5分钟K线(4h/5min)
        "rolling_window": 252 * 48,
        "rolling_min_periods": 150 * 48,
        "description": "5分钟",
    },
    "15min": {
        "vbt_freq": "15T",
        "annual_bars": 252 * 16,
        "rolling_window": 252 * 16,
        "rolling_min_periods": 150 * 16,
        "description": "15分钟",
    },
    "30min": {
        "vbt_freq": "30T",
        "annual_bars": 252 * 8,
        "rolling_window": 252 * 8,
        "rolling_min_periods": 150 * 8,
        "description": "30分钟",
    },
    "60min": {
        "vbt_freq": "1H",
        "annual_bars": 252 * 4,
        "rolling_window": 252 * 4,
        "rolling_min_periods": 150 * 4,
        "description": "60分钟",
    },
}


def _get_freq_config(freq: str) -> Dict:
    """获取频率配置，不支持的频率则回退到日频"""
    return _FREQ_CONFIG.get(freq if freq else "", _FREQ_CONFIG["D"])


class VectorBTBacktestService:
    """基于 vectorbt 的回测服务"""

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        slippage: float = 0.0,
        slippage_mode: str = "custom",  # "smart" 或 "custom"
    ):
        """
        初始化回测服务

        Args:
            initial_capital: 初始资金，默认100万
            commission_rate: 手续费率，默认万三
            slippage: 滑点率，默认0（不考虑滑点）
            slippage_mode: 滑点模式
                         - "smart": 使用智能检测器自动推荐
                         - "custom": 使用用户指定的滑点值
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.slippage_mode = slippage_mode
        self._slippage_recommendation: Optional[SlippageRecommendation] = None

    def set_smart_slippage(
        self,
        stock_codes: List[str],
        strategy_turnover: float = 12.0,
        market_data: Optional[pd.DataFrame] = None,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        user_preference: Optional[str] = None,
    ) -> SlippageRecommendation:
        """
        使用智能检测器设置滑点参数

        Args:
            stock_codes: 股票代码列表
            strategy_turnover: 策略年化换手率（倍数/年）
            market_data: 市场数据（可选）
            price_data: 价格数据（可选）
            user_preference: 用户偏好 ("conservative"/"aggressive"/None)

        Returns:
            滑点推荐结果
        """
        logger.info(f"启动智能滑点检测: {len(stock_codes)}只股票, 换手率{strategy_turnover:.1f}倍/年")

        self._slippage_recommendation = smart_slippage_detector.recommend_slippage(
            stock_codes=stock_codes,
            strategy_turnover=strategy_turnover,
            market_data=market_data,
            price_data=price_data,
            user_preference=user_preference,
        )

        # 应用推荐的滑点值
        self.slippage = self._slippage_recommendation.recommended_slippage
        self.slippage_mode = "smart"

        logger.info(
            f"智能滑点推荐完成: {self.slippage*100:.3f}% "
            f"(置信度{self._slippage_recommendation.confidence*100:.0f}%)"
        )

        return self._slippage_recommendation

    def get_slippage_info(self) -> Dict:
        """获取当前滑点配置信息"""
        if self._slippage_recommendation and self.slippage_mode == "smart":
            return {
                "mode": "smart",
                "slippage": self.slippage,
                "recommendation": {
                    "recommended": self._slippage_recommendation.recommended_slippage,
                    "conservative": self._slippage_recommendation.conservative_slippage,
                    "aggressive": self._slippage_recommendation.aggressive_slippage,
                    "confidence": self._slippage_recommendation.confidence,
                    "reasoning": self._slippage_recommendation.reasoning,
                    "sensitivity": self._slippage_recommendation.sensitivity_analysis,
                },
            }
        else:
            return {
                "mode": "custom",
                "slippage": self.slippage,
            }

    # ==================== 内存优化 & 分块计算 ====================

    @staticmethod
    def _to_memory_efficient(df: pd.DataFrame) -> pd.DataFrame:
        """下转型数据类型以降低内存占用（float64→float32, int64→int32）"""
        df = df.copy()
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == np.float64:
                df[col] = df[col].astype(np.float32)
            elif col_type == np.int64:
                df[col] = df[col].astype(np.int32)
        return df

    @staticmethod
    def _auto_chunk_config(freq: str, total_bars: int) -> Tuple[int, int]:
        """
        根据频率和数据量自动确定分块大小和重叠大小

        Returns:
            (chunk_size, overlap_size)
        """
        fc = _get_freq_config(freq)
        # 每个 chunk 的目标内存占用 ~200MB（对应约 5000 bar 的完整数据）
        bars_per_chunk = {
            "D": 500,
            "60min": 1000,
            "30min": 2000,
            "15min": 4000,
            "5min": 5000,
        }
        chunk_size = bars_per_chunk.get(freq.lower(), 5000)
        # 重叠区 = 滚动窗口，保证每个 chunk 的 rolling 计算有足够历史
        overlap_size = fc["rolling_window"]
        return chunk_size, overlap_size

    @staticmethod
    def _split_chunks(
        df: pd.DataFrame,
        chunk_size: int,
        overlap_size: int,
    ) -> List[Tuple[pd.DataFrame, int, int]]:
        """
        将 DataFrame 切分为重叠的分块

        Returns:
            [(chunk_df, chunk_start_idx, chunk_end_idx), ...]
        """
        n = len(df)
        chunks = []
        start = 0
        while start < n:
            end = min(start + chunk_size + overlap_size, n)
            chunk_df = df.iloc[start:end]
            chunks.append((chunk_df, start, end))
            start += chunk_size
        return chunks

    def _run_vbt_core(
        self,
        df: pd.DataFrame,
        factor_name: str,
        percentile: int,
        direction: str,
        n_quantiles: int,
        shares_per_trade: int,
        use_tradable_mask: bool,
        fc: Dict,
        skip_copy: bool = False,
    ) -> Dict:
        """
        共享的 VectorBT 核心回测逻辑：因子排名 → 信号生成 → 回测执行

        _run_single_chunk 和 single_factor_backtest 的公共实现，
        避免两处维护相同逻辑导致漂移。

        Args:
            df: 已确保 DatetimeIndex 的 DataFrame（内部会copy，不污染调用方数据）
            factor_name, percentile, direction, n_quantiles, shares_per_trade, use_tradable_mask, fc: 同上
            skip_copy: 如果调用方已做过copy，设为True避免重复复制

        Returns:
            Dict 包含:
                df (DataFrame): 添加了 returns 列后的数据
                entries (Series): 入场信号
                exits (Series): 出场信号
                tradable_mask (Series|None): 可交易掩码
                quantile_returns (dict): 各层收益 {Q1: Series, ...}
                pf (vbt.Portfolio): VectorBT Portfolio 对象
                equity (Series): 净值曲线
                returns (Series): 收益率序列
        """
        # 规则5：禁止就地修改传入的DataFrame，必须先copy
        if not skip_copy:
            df = df.copy()

        # Mask-First
        tradable_mask = None
        if use_tradable_mask and "tradable_mask" in df.columns:
            tradable_mask = df["tradable_mask"]

        # 计算前向收益率（t→t+1），避免前视偏差
        df["forward_return"] = calculate_future_return(df)

        # 因子分位数排名
        factor_raw = df[factor_name]
        if tradable_mask is not None:
            factor_clean = factor_raw.where(tradable_mask)
            factor_rank = factor_clean.rolling(fc["rolling_window"], min_periods=fc["rolling_min_periods"]).rank(
                pct=True
            )
        else:
            factor_rank = factor_raw.rolling(fc["rolling_window"], min_periods=1).rank(pct=True)

        # 信号生成
        percentile_threshold = percentile / 100.0
        if direction == "long":
            entries = factor_rank >= percentile_threshold
            exits = factor_rank < percentile_threshold
        else:
            entries = factor_rank <= percentile_threshold
            exits = factor_rank > percentile_threshold

        if tradable_mask is not None:
            entries = entries & tradable_mask.astype(bool)
            exits = exits | (~tradable_mask.astype(bool))

        # 分层收益
        quantile_returns = {}
        for q in range(n_quantiles):
            q_min = q / n_quantiles
            q_max = (q + 1) / n_quantiles
            # 最后一层使用<=，避免rank=1.0的样本被排除
            if q == n_quantiles - 1:
                layer_mask = (factor_rank >= q_min) & (factor_rank <= q_max)
            else:
                layer_mask = (factor_rank >= q_min) & (factor_rank < q_max)
            layer_returns = df.loc[layer_mask, "forward_return"]
            quantile_returns[f"Q{q + 1}"] = layer_returns

        # VectorBT 回测
        # 使用 size_type='percent' + size=1.0 投入全部可用资金
        # 旧版 size=shares_per_trade(100) 导致仅投入极小比例资金，所有绩效指标失真
        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            freq=fc["vbt_freq"],
            cash_sharing=False,
            fees=self.commission_rate,
            slippage=self.slippage,
            size=1.0,
            size_type="percent",
        )

        equity = pf.value()
        returns = pf.returns()

        return {
            "df": df,
            "entries": entries,
            "exits": exits,
            "tradable_mask": tradable_mask,
            "quantile_returns": quantile_returns,
            "pf": pf,
            "equity": equity,
            "returns": returns,
        }

    def _run_single_chunk(
        self,
        df_chunk: pd.DataFrame,
        factor_name: str,
        percentile: int,
        direction: str,
        n_quantiles: int,
        shares_per_trade: int,
        use_tradable_mask: bool,
        fc: Dict,
        chunk_start: int,
        chunk_end: int,
    ) -> Optional[Dict]:
        """
        对单个分块执行回测（委托给 _run_vbt_core，返回简化结果）

        Returns:
            Dict with equity_curve, returns, trades, quantile_returns (all only for the reliable region)
        """
        if not isinstance(df_chunk.index, pd.DatetimeIndex):
            df_chunk = df_chunk.copy()
            if "date" in df_chunk.columns:
                df_chunk = df_chunk.set_index("date")
            df_chunk.index = pd.to_datetime(df_chunk.index)

        # Mask-First 快速检查
        if use_tradable_mask and "tradable_mask" in df_chunk.columns:
            if df_chunk["tradable_mask"].sum() == 0:
                return None

        # 委托给共享核心逻辑
        core = self._run_vbt_core(
            df=df_chunk,
            factor_name=factor_name,
            percentile=percentile,
            direction=direction,
            n_quantiles=n_quantiles,
            shares_per_trade=shares_per_trade,
            use_tradable_mask=use_tradable_mask,
            fc=fc,
        )

        return {
            "equity_curve": core["equity"],
            "returns": core["returns"],
            "quantile_returns": core["quantile_returns"],
            "pf": core["pf"],
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        }

    @staticmethod
    def _stitch_equity_curves(
        chunk_results: List[Dict],
        overlap_size: int,
    ) -> pd.Series:
        """
        拼接各分块的净值曲线

        每个 chunk 的前 overlap_size 个 bar 是 warmup 区（rolling 不可靠），丢弃。

        修复：拼接收益率序列而非净值曲线，避免在 chunk 边界丢失一天的收益。
        原方法对净值曲线做归一化（equity / equity.iloc[0]）会丢弃 chunk 首日
        相对于上一 chunk 末日的收益率。改为：在每个 chunk 内计算 pct_change，
        丢弃 warmup 区的收益率，拼接所有收益率后从收益率重建净值曲线。
        """
        returns_parts = []

        for i, cr in enumerate(chunk_results):
            equity = cr["equity_curve"].copy()
            if len(equity) == 0:
                continue

            # 计算收益率序列
            returns = equity.pct_change()

            # 丢弃 warmup 区的收益率（与原逻辑一致：warmup 区 rolling 不可靠）
            if len(returns) > overlap_size:
                returns = returns.iloc[overlap_size:]
            elif i == 0:
                # 第一个 chunk 数据太少，保留所有收益率
                pass
            else:
                # 重叠区比数据还长，跳过
                continue

            if len(returns) == 0:
                continue

            returns_parts.append(returns)

        if not returns_parts:
            return pd.Series(dtype=float)

        # 拼接所有收益率
        all_returns = pd.concat(returns_parts)
        # 去重：同名时间戳只保留第一个
        all_returns = all_returns[~all_returns.index.duplicated(keep="first")]

        # 从收益率重建净值曲线
        # 第一条收益率（NaN，来自 pct_change 的首行）替换为 0，确保起始净值为 1.0
        all_returns.iloc[0] = 0.0
        equity_curve = (1 + all_returns).cumprod()

        return equity_curve

    def chunked_single_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        percentile: int = 50,
        direction: str = "long",
        n_quantiles: int = 5,
        shares_per_trade: int = 100,
        use_tradable_mask: bool = True,
        freq: str = "D",
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        risk_free_rate: float = 0.03,
    ) -> Dict:
        """
        分块回测：将大数据集切分为重叠块，逐块计算后拼接结果，大幅降低内存峰值。

        Args:
            df: 完整数据
            factor_name: 因子名称
            percentile: 分位数阈值
            direction: 交易方向
            n_quantiles: 分层数量
            shares_per_trade: 每次交易手数
            use_tradable_mask: 是否使用可交易性掩码
            freq: 数据频率
            chunk_size: 每块 bar 数（None=自动）
            overlap_size: 重叠 bar 数（None=自动=滚动窗口大小）

        Returns:
            Dict: 与 single_factor_backtest 相同格式的结果
        """
        fc = _get_freq_config(freq)
        auto_chunk, auto_overlap = self._auto_chunk_config(freq, len(df))

        if chunk_size is None:
            chunk_size = auto_chunk
        if overlap_size is None:
            overlap_size = auto_overlap

        logger.info(
            f"🧩 分块回测: {len(df)} bars → {chunk_size} bar/chunk, "
            f"overlap={overlap_size} bars, 预计 {(len(df) + chunk_size - 1) // chunk_size} 块"
        )

        # 内存优化：下转型
        df = self._to_memory_efficient(df)

        # 切分
        chunks = self._split_chunks(df, chunk_size, overlap_size)

        # 逐块计算
        chunk_results = []
        for i, (chunk_df, chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"  分块 {i + 1}/{len(chunks)}: bars [{chunk_start}:{chunk_end}] ({len(chunk_df)} rows)")
            try:
                cr = self._run_single_chunk(
                    df_chunk=chunk_df,
                    factor_name=factor_name,
                    percentile=percentile,
                    direction=direction,
                    n_quantiles=n_quantiles,
                    shares_per_trade=shares_per_trade,
                    use_tradable_mask=use_tradable_mask,
                    fc=fc,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                if cr is not None:
                    chunk_results.append(cr)
            except Exception as e:
                logger.warning(f"  分块 {i + 1} 回测失败: {e}")
            finally:
                del chunk_df
                gc.collect()

        if not chunk_results:
            raise ValueError("所有分块回测均失败，无有效结果")

        # 拼接净值曲线
        equity_curve = self._stitch_equity_curves(chunk_results, overlap_size)
        if len(equity_curve) == 0:
            raise ValueError("净值曲线拼接结果为空")

        # 从拼接净值计算收益率
        returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        # 拼接分层收益（按时间索引裁剪warmup区，而非按位置）
        quantile_returns = {}
        for q in range(n_quantiles):
            q_parts = []
            for i, cr in enumerate(chunk_results):
                q_key = f"Q{q + 1}"
                if q_key in cr["quantile_returns"]:
                    q_series = cr["quantile_returns"][q_key]
                    if len(cr.get("equity_curve", [])) > overlap_size:
                        # 所有分块（包括i=0）都过滤warmup区，确保分层收益与净值曲线一致
                        warmup_end_time = cr["equity_curve"].index[overlap_size]
                        q_series = q_series[q_series.index >= warmup_end_time]
                    q_parts.append(q_series)
            if q_parts:
                quantile_returns[q_key] = pd.concat(q_parts)

        # 拼接交易记录
        trades_dfs = []
        for cr in chunk_results:
            try:
                pf = cr["pf"]
                if hasattr(pf, "trades") and hasattr(pf.trades, "records_readable"):
                    trades_readable = pf.trades.records_readable
                    if trades_readable is not None and len(trades_readable) > 0:
                        # 过滤掉 warmup 区的交易
                        if "Entry Timestamp" in trades_readable.columns:
                            warmup_cutoff = (
                                df.index[cr["chunk_start"] + overlap_size]
                                if cr["chunk_start"] + overlap_size < len(df.index)
                                else None
                            )
                            if warmup_cutoff is not None:
                                trades_readable = trades_readable[trades_readable["Entry Timestamp"] >= warmup_cutoff]
                        trades_dfs.append(trades_readable)
            except Exception as e:
                logger.debug(f"提取分块交易记录失败: {e}")

        trades_df = None
        if trades_dfs:
            trades_df = pd.concat(trades_dfs, ignore_index=True)
            # 去重
            if "Entry Timestamp" in trades_df.columns and "Exit Timestamp" in trades_df.columns:
                trades_df = trades_df.drop_duplicates(subset=["Entry Timestamp", "Exit Timestamp"])

        # 释放分块中的Portfolio对象，避免内存积累
        for cr in chunk_results:
            cr.pop("pf", None)
        gc.collect()

        # 指标计算（委托risk_metrics统一入口）
        n_bars = len(returns)
        returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns

        if n_bars > 0:
            metrics = calculate_risk_metrics(
                returns_series, risk_free_rate=risk_free_rate, annual_trading_days=fc["annual_bars"]
            )
            total_return = metrics["total_return"]
            annual_return = metrics["annual_return"]
            volatility = metrics["volatility"]
            sharpe_ratio = metrics["sharpe_ratio"]
            sortino_ratio = metrics["sortino_ratio"]
            max_drawdown = metrics["max_drawdown"]
            calmar_ratio = metrics["calmar_ratio"]
            win_rate = metrics["win_rate"]
            var_95 = metrics["var_95"]
            cvar_95 = metrics["cvar_95"]
        else:
            total_return = None
            annual_return = None
            volatility = None
            sharpe_ratio = None
            sortino_ratio = None
            max_drawdown = None
            calmar_ratio = None
            win_rate = None
            var_95 = None
            cvar_95 = None

        trades_count = int(len(trades_df)) if trades_df is not None else 0

        logger.info(
            f"✅ 分块回测完成: {len(chunk_results)} 块 → "
            + (
                f"净值曲线 {len(equity_curve)} bars, 年化收益 {annual_return:.2%}"
                if annual_return is not None
                else f"净值曲线 {len(equity_curve)} bars, 年化收益 N/A"
            )
        )

        return {
            "quantile_returns": quantile_returns,
            "portfolio_returns": returns,
            "equity_curve": equity_curve,
            "trades_count": trades_count,
            "trades": trades_df,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "slippage_info": self.get_slippage_info(),
            "mask_statistics": {
                "total_days": len(df),
                "tradable_days": int(df["tradable_mask"].sum()) if "tradable_mask" in df.columns else len(df),
                "tradable_ratio": float(df["tradable_mask"].mean()) if "tradable_mask" in df.columns else 1.0,
                "limit_up_days": (
                    int((df.get("limit_up_mask", pd.Series(dtype=bool))).sum()) if "limit_up_mask" in df.columns else 0
                ),
                "limit_down_days": (
                    int((df.get("limit_down_mask", pd.Series(dtype=bool))).sum())
                    if "limit_down_mask" in df.columns
                    else 0
                ),
                "suspended_days": int((~df["tradable_mask"]).sum()) if "tradable_mask" in df.columns else 0,
                "mask_applied": "tradable_mask" in df.columns,
            },
            "chunking_info": {
                "chunked": True,
                "num_chunks": len(chunk_results),
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
            },
        }

    def single_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        percentile: int = 50,
        direction: str = "long",
        n_quantiles: int = 5,
        shares_per_trade: int = None,
        use_tradable_mask: bool = True,
        freq: str = "D",
        use_chunking: str = "auto",
    ) -> Dict:
        """
        单因子分层回测（使用vectorbt，支持Mask-First设计）

        Args:
            df: 包含价格和因子数据的DataFrame，必须有 close 列和因子列
            factor_name: 因子名称
            percentile: 分位数阈值（0-100），用于做多/做空判断
            direction: 交易方向，"long"做多或"short"做空
            n_quantiles: 分层数量，默认5层
            shares_per_trade: 已弃用，保留参数仅为向后兼容，实际使用percent模式投入全部资金
            use_tradable_mask: 是否使用Mask-First可交易性掩码（默认True）
            freq: 数据频率，支持 "D"(日频)/"5min"/"15min"/"30min"/"60min"
            use_chunking: 分块模式，"auto"(自动)/"force"(强制)/"off"(禁用)

        Returns:
            Dict: 包含各层收益、整体收益、净值曲线等数据的字典
        """
        # 自动检测：大数据集自动启用分块
        if use_chunking in ("auto", "force"):
            chunk_size, _ = self._auto_chunk_config(freq, len(df))
            if use_chunking == "force" or len(df) > chunk_size * 1.5:
                logger.info(f"📊 数据量 {len(df)} bars > {int(chunk_size * 1.5)} 阈值，自动启用分块回测")
                return self.chunked_single_factor_backtest(
                    df=df,
                    factor_name=factor_name,
                    percentile=percentile,
                    direction=direction,
                    n_quantiles=n_quantiles,
                    shares_per_trade=shares_per_trade,
                    use_tradable_mask=use_tradable_mask,
                    freq=freq,
                )

        # 获取频率配置
        fc = _get_freq_config(freq)
        logger.info(f"回测频率: {fc['description']} (vbt_freq={fc['vbt_freq']}, 年化bar数={fc['annual_bars']})")

        # 确保索引是 DatetimeIndex
        already_copied = False
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            already_copied = True
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)

        # Mask-First: 提取并应用可交易性掩码
        tradable_mask = None
        if use_tradable_mask and "tradable_mask" in df.columns:
            tradable_mask = df["tradable_mask"]
            logger.info(f"✅ VectorBT Backtest: 使用Mask-First设计，可交易比例 {tradable_mask.mean():.1%}")
            if tradable_mask.sum() == 0:
                logger.error("tradable_mask全为False！所有日期都不可交易")
                raise ValueError("tradable_mask全为False！所有日期都不可交易")
        elif use_tradable_mask and "tradable_mask" not in df.columns:
            logger.warning("⚠️ VectorBT Backtest: 未找到tradable_mask列！")

        # 委托给共享核心逻辑（因子排名 → 信号 → VBT回测）
        # skip_copy: 仅在上方已对df做过copy时跳过，避免双重复制
        core = self._run_vbt_core(
            df=df,
            factor_name=factor_name,
            percentile=percentile,
            direction=direction,
            n_quantiles=n_quantiles,
            shares_per_trade=shares_per_trade,
            use_tradable_mask=use_tradable_mask,
            fc=fc,
            skip_copy=already_copied,
        )

        pf = core["pf"]
        equity = core["equity"]
        returns = core["returns"]
        quantile_returns = core["quantile_returns"]
        tradable_mask = core["tradable_mask"]
        returns_clean = returns.dropna()

        # 使用 VectorBT 的 stats() 方法获取所有指标
        stats = pf.stats()

        # 从 stats Series 中提取指标
        # VectorBT 返回的百分比值需要除以100转换为小数
        # 注意: 使用 _safe_stat 辅助函数避免 `or 0` 将 None/0.0 混淆（规则6）
        def _safe_stat(key: str, default=0, divide_by_100: bool = False):
            """安全提取 VectorBT 统计值，None 保持为 None（规则6）"""
            val = stats.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            if divide_by_100:
                return val / 100.0
            return val

        total_return = _safe_stat("Total Return [%]", divide_by_100=True)

        # 年化收益率：如果VectorBT返回None/NaN，则手动计算
        annual_return = _safe_stat("Annual Return [%]", divide_by_100=True)
        if annual_return is None:
            # 手动计算年化收益率 = (1 + 总收益率)^(年化bar数/交易bar数) - 1
            n_days = len(returns_clean)
            if n_days > 0 and total_return is not None and total_return > -1:
                annual_return = (1 + total_return) ** (fc["annual_bars"] / n_days) - 1
            else:
                annual_return = None

        volatility = self._calculate_volatility(returns_clean, fc["annual_bars"])

        # 使用统一的calculate_metrics计算Sharpe/Sortino（扣除无风险利率3%）
        # VectorBT默认rf=0，此处统一为rf=3%以保持与分块回测一致
        metrics = self.calculate_metrics(
            returns_clean, equity_curve=(1 + returns_clean).cumprod(), annual_trading_days=fc["annual_bars"]
        )
        sharpe_ratio = metrics["sharpe_ratio"]
        sortino_ratio = metrics["sortino_ratio"]
        max_drawdown = _safe_stat("Max Drawdown [%]", divide_by_100=True)
        calmar_ratio = _safe_stat("Calmar Ratio")
        win_rate = _safe_stat("Win Rate [%]", divide_by_100=True)

        # VaR 和 CVaR 需要自己计算
        var_95, cvar_95 = self._calculate_var_cvar(returns_clean)

        # 计算交易次数
        trades_count = stats.get("Total Trades", 0) or 0

        # 提取交易记录
        trades_df = self._format_trades_df(pf)

        return {
            "quantile_returns": quantile_returns,
            "portfolio_returns": returns,
            "equity_curve": equity,
            "trades_count": int(trades_count),
            "trades": trades_df,
            # 手动计算的指标（None 保持为 None，符合规则6）
            "total_return": float(total_return) if total_return is not None else None,
            "annual_return": float(annual_return) if annual_return is not None else None,
            "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio is not None else None,
            "sortino_ratio": float(sortino_ratio) if sortino_ratio is not None else None,
            "max_drawdown": float(max_drawdown) if max_drawdown is not None else None,
            "calmar_ratio": float(calmar_ratio) if calmar_ratio is not None else None,
            "win_rate": float(win_rate) if win_rate is not None else None,
            "volatility": float(volatility) if volatility is not None else None,
            "var_95": float(var_95) if var_95 is not None else None,
            "cvar_95": float(cvar_95) if cvar_95 is not None else None,
            # 滑点信息
            "slippage_info": self.get_slippage_info(),
            # Mask-First统计
            "mask_statistics": {
                "total_days": len(df),
                "tradable_days": int(tradable_mask.sum()) if tradable_mask is not None else len(df),
                "tradable_ratio": float(tradable_mask.mean()) if tradable_mask is not None else 1.0,
                "limit_up_days": int(df["is_limit_up"].sum()) if "is_limit_up" in df.columns else 0,
                "limit_down_days": int(df["is_limit_down"].sum()) if "is_limit_down" in df.columns else 0,
                "suspended_days": int(df["is_suspended"].sum()) if "is_suspended" in df.columns else 0,
                "mask_applied": tradable_mask is not None,
            },
        }

    def multi_factor_backtest(
        self,
        df: pd.DataFrame,
        factor_names: List[str],
        weights: Optional[List[float]] = None,
        method: str = "equal_weight",
        percentile: int = 50,
        direction: str = "long",
        n_quantiles: int = 5,
        shares_per_trade: int = 100,
        freq: str = "D",
        use_chunking: str = "auto",
    ) -> Dict:
        """
        多因子组合回测（使用vectorbt）

        Args:
            df: 包含价格和因子数据的DataFrame
            factor_names: 因子名称列表
            weights: 因子权重列表（可选）
            method: 权重分配方法，"equal_weight"等权重, "ic_weight" IC加权, "risk_parity"风险平价
            percentile: 分位数阈值（0-100）
            direction: 交易方向，"long"做多或"short"做空
            n_quantiles: 分层数量
            freq: 数据频率，支持 "D"/"5min"/"15min"/"30min"/"60min"
            use_chunking: 分块模式，"auto"/"force"/"off"

        Returns:
            Dict: 回测结果
        """
        # 确保索引是 DatetimeIndex，且无条件copy避免副作用
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)

        # 1. 标准化因子值（z-score，使用滚动窗口避免前视偏差）
        std_window = min(252, max(20, len(df) // 2))
        for factor_name in factor_names:
            if factor_name in df.columns:
                rolling_mean = df[factor_name].rolling(std_window, min_periods=20).mean()
                rolling_std = df[factor_name].rolling(std_window, min_periods=20).std()
                df[f"{factor_name}_normalized"] = safe_series_divide(
                    df[factor_name] - rolling_mean, rolling_std, fill_value=np.nan
                ).fillna(0)

        # 2. 计算因子组合得分
        normalized_factors = [f"{name}_normalized" for name in factor_names]

        if method == "equal_weight":
            # 等权重
            if weights is None:
                weights = [1.0 / len(normalized_factors)] * len(normalized_factors)
            df["composite_score"] = sum(df[nf] * w for nf, w in zip(normalized_factors, weights))

        elif method == "ic_weight":
            # IC加权：使用滚动窗口历史IC，避免前视偏差
            # 对每个因子，用过去数据计算因子值与未来收益的滚动IC
            df["forward_return"] = calculate_future_return(df)
            ic_window = min(60, max(10, len(df) // 4))

            # 使用滚动Spearman IC作为时变权重，每个时间点仅使用截至前一天的信息
            ic_weight_frames = []
            forward_return_shifted = df["forward_return"].shift(1)
            for factor_name in factor_names:
                norm_factor = f"{factor_name}_normalized"
                # 滚动Spearman IC：因子值（滞后1期，避免前视偏差）与未来收益的相关系数
                # 使用历史窗口内的数据，确保t日的IC权重只用到t-1日及之前的信息
                # forward_return 也需 shift(1)，确保 t 日窗口只用到 t-1→t 的已知收益
                shifted_factor = df[norm_factor].shift(1)
                rolling_ic = calculate_rolling_ic(
                    shifted_factor, forward_return_shifted, window=ic_window, method="spearman", min_periods=10
                )
                ic_abs = rolling_ic.abs()
                ic_weight_frames.append(ic_abs)

            # 逐行计算归一化权重和复合得分
            ic_weight_sum = sum(ic_weight_frames)
            # 除零+NaN保护：权重和为0或NaN时回退到等权
            equal_w = 1.0 / len(normalized_factors)
            # 标记权重和无效的行（0或NaN）
            invalid_mask = (ic_weight_sum.abs() < 1e-10) | ic_weight_sum.isna()
            # 有效行：用 ic_wf / ic_weight_sum 归一化；无效行：直接用等权
            safe_ic_weight_sum = ic_weight_sum.copy()
            safe_ic_weight_sum[invalid_mask] = np.nan  # 无效行标记为NaN，后续fillna处理
            composite_parts = []
            for nf, ic_wf in zip(normalized_factors, ic_weight_frames):
                safe_weight = ic_wf / safe_ic_weight_sum
                # 无效行回退到等权
                safe_weight[invalid_mask] = equal_w
                composite_parts.append(df[nf] * safe_weight)
            df["composite_score"] = sum(composite_parts)
            # 清理临时列，避免污染DataFrame
            df.drop(columns=["forward_return"], inplace=True)

        elif method == "risk_parity":
            # 风险平价：使用滚动波动率，避免前视偏差
            vol_window = min(252, max(60, len(df) // 2))
            vol_weight_frames = []
            for factor_name in factor_names:
                norm_factor = f"{factor_name}_normalized"
                # 滚动波动率，shift(1)避免前视
                rolling_vol = df[norm_factor].rolling(vol_window, min_periods=20).std().shift(1)
                inv_vol = safe_divide(1.0, rolling_vol, default=np.nan)
                vol_weight_frames.append(inv_vol)

            # 逐行计算归一化权重和复合得分
            vol_weight_sum = sum(vol_weight_frames)
            composite_parts = []
            for nf, vw_f in zip(normalized_factors, vol_weight_frames):
                safe_weight = safe_divide(vw_f, vol_weight_sum, default=1.0 / len(normalized_factors))
                composite_parts.append(df[nf] * safe_weight.fillna(1.0 / len(normalized_factors)))
            df["composite_score"] = sum(composite_parts)

        else:
            # 默认等权重
            df["composite_score"] = df[normalized_factors].mean(axis=1)

        # 3. 使用组合得分进行回测
        return self.single_factor_backtest(
            df=df,
            factor_name="composite_score",
            percentile=percentile,
            direction=direction,
            n_quantiles=n_quantiles,
            shares_per_trade=shares_per_trade,
            freq=freq,
            use_chunking=use_chunking,
        )

    def cross_sectional_backtest(
        self,
        df: pd.DataFrame,
        factor_name: str,
        top_percentile: float = 0.2,
        direction: str = "long",
        freq: str = "D",
        use_tradable_mask: bool = True,
    ) -> Dict:
        """
        股票池横截面回测（使用vectorbt）

        Args:
            df: 包含多只股票数据的DataFrame
            factor_name: 因子名称
            top_percentile: 选择股票的百分比（0.2表示前20%）
            direction: "long"做多或"short"做空
            freq: 数据频率，支持 "D"/"5min"/"15min"/"30min"/"60min"
            use_tradable_mask: 是否使用Mask-First可交易性掩码（默认True）

        Returns:
            Dict: 回测结果
        """
        # 规则5：禁止就地修改传入的DataFrame，必须先copy
        df = df.copy()

        # 获取频率配置
        fc = _get_freq_config(freq)
        logger.info(f"横截面回测频率: {fc['description']} (vbt_freq={fc['vbt_freq']}, 年化bar数={fc['annual_bars']})")
        # 确保索引正确
        if "date" not in df.columns:
            df = df.reset_index()

        # Mask-First: 提取可交易性掩码
        tradable_mask_df = None
        if use_tradable_mask and "tradable_mask" in df.columns:
            tradable_mask_df = df.pivot(index="date", columns="stock_code", values="tradable_mask")
            logger.info(f"✅ 横截面回测: 使用Mask-First设计，可交易比例 {tradable_mask_df.mean().mean():.1%}")
        elif use_tradable_mask and "tradable_mask" not in df.columns:
            logger.warning("⚠️ 横截面回测: 未找到tradable_mask列！")

        # 透视数据：将股票代码转为列
        price_df = df.pivot(index="date", columns="stock_code", values="close")
        factor_df = df.pivot(index="date", columns="stock_code", values=factor_name)

        # 确保索引是 DatetimeIndex
        price_df.index = pd.to_datetime(price_df.index)

        # 1. 计算收益率
        _returns_df = price_df.pct_change()  # noqa: F841

        # 2. 向量化选择股票（替代逐日循环，性能提升数十倍）
        # Mask-First: 因子排名前先应用tradable_mask
        if tradable_mask_df is not None:
            factor_clean = factor_df.where(tradable_mask_df.astype(bool))
            ranks = factor_clean.rank(pct=True, axis=1)
        else:
            ranks = factor_df.rank(pct=True, axis=1)

        if direction == "long":
            selected_mask = ranks >= (1 - top_percentile)
        else:
            selected_mask = ranks <= top_percentile
        # NaN位置标记为未选中
        selected_mask = selected_mask & factor_df.notna()

        # Mask-First: 不可交易的股票不允许入场
        if tradable_mask_df is not None:
            selected_mask = selected_mask & tradable_mask_df.astype(bool)

        # 3. 变化驱动信号矩阵（向量化计算入场/出场）
        entries = pd.DataFrame(False, index=price_df.index, columns=price_df.columns, dtype=bool)
        exits = pd.DataFrame(False, index=price_df.index, columns=price_df.columns, dtype=bool)

        if len(selected_mask) > 0:
            # 首日建仓
            first_date = selected_mask.index[0]
            entries.loc[first_date] = entries.loc[first_date] | selected_mask.loc[first_date]

            # 后续日期：新增→入场，被剔除→出场（向量化diff）
            if len(selected_mask) > 1:
                newly_selected = selected_mask & ~selected_mask.shift(1).fillna(False)
                newly_deselected = ~selected_mask & selected_mask.shift(1).fillna(False)
                # 排除首日（已单独处理）
                entries.iloc[1:] = entries.iloc[1:] | newly_selected.iloc[1:]
                exits.iloc[1:] = exits.iloc[1:] | newly_deselected.iloc[1:]

            # 最后一个交易日：平掉所有剩余持仓
            last_date = selected_mask.index[-1]
            exits.loc[last_date] = exits.loc[last_date] | selected_mask.loc[last_date]

        n_dates = len(selected_mask)
        n_entries = int(entries.sum().sum())
        n_exits = int(exits.sum().sum())
        logger.info(
            f"横截面信号: {n_dates}个交易日, "
            f"入场{n_entries}次, 出场{n_exits}次 "
            f"(旧逻辑每日全换手={n_dates * len(price_df.columns)}次)"
        )

        # 4. 使用vectorbt进行回测
        pf = vbt.Portfolio.from_signals(
            close=price_df,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            freq=fc["vbt_freq"],
            cash_sharing=True,
            fees=self.commission_rate,
            slippage=self.slippage,
        )

        # 5. 获取结果
        equity = pf.value()
        returns = pf.returns()
        returns_clean = returns.dropna()

        # 使用 VectorBT 的 stats() 方法获取所有指标
        stats = pf.stats()

        # 从 stats Series 中提取指标（使用 _safe_stat 避免 `or 0` 混淆 None/0.0，规则6）
        def _safe_stat(key: str, default=0, divide_by_100: bool = False):
            """安全提取 VectorBT 统计值，None 保持为 None（规则6）"""
            val = stats.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            if divide_by_100:
                return val / 100.0
            return val

        total_return = _safe_stat("Total Return [%]", divide_by_100=True)

        # 年化收益率：如果VectorBT返回None/NaN，则手动计算
        annual_return = _safe_stat("Annual Return [%]", divide_by_100=True)
        if annual_return is None:
            # 手动计算年化收益率 = (1 + 总收益率)^(年化bar数/交易bar数) - 1
            n_days = len(returns_clean)
            if n_days > 0 and total_return is not None and total_return > -1:
                annual_return = (1 + total_return) ** (fc["annual_bars"] / n_days) - 1
            else:
                annual_return = None

        volatility = self._calculate_volatility(returns_clean, fc["annual_bars"])

        # 使用统一的calculate_metrics计算Sharpe/Sortino（扣除无风险利率3%）
        # VectorBT默认rf=0，此处统一为rf=3%以保持与分块回测一致
        metrics = self.calculate_metrics(
            returns_clean, equity_curve=(1 + returns_clean).cumprod(), annual_trading_days=fc["annual_bars"]
        )
        sharpe_ratio = metrics["sharpe_ratio"]
        sortino_ratio = metrics["sortino_ratio"]
        max_drawdown = _safe_stat("Max Drawdown [%]", divide_by_100=True)
        calmar_ratio = _safe_stat("Calmar Ratio")
        win_rate = _safe_stat("Win Rate [%]", divide_by_100=True)

        # VaR 和 CVaR 需要自己计算
        var_95, cvar_95 = self._calculate_var_cvar(returns_clean)

        # 计算交易次数（每日调仓次数）
        trades_count = stats.get("Total Trades", len(price_df.columns)) or len(price_df.columns)

        # 提取交易记录
        trades_df = self._format_trades_df(pf)

        return {
            "portfolio_returns": returns,
            "equity_curve": equity,
            "trades_count": trades_count,
            "trades": trades_df,
            "daily_selected_count": trades_count,
            # 手动计算的指标（None保持None，符合规则6）
            "total_return": float(total_return) if total_return is not None else None,
            "annual_return": float(annual_return) if annual_return is not None else None,
            "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio is not None else None,
            "sortino_ratio": float(sortino_ratio) if sortino_ratio is not None else None,
            "max_drawdown": float(max_drawdown) if max_drawdown is not None else None,
            "calmar_ratio": float(calmar_ratio) if calmar_ratio is not None else None,
            "win_rate": float(win_rate) if win_rate is not None else None,
            "volatility": float(volatility) if volatility is not None else None,
            "var_95": float(var_95) if var_95 is not None else None,
            "cvar_95": float(cvar_95) if cvar_95 is not None else None,
            # 滑点信息
            "slippage_info": self.get_slippage_info(),
        }

    def run_vectorbt_backtest_from_weights(
        self,
        weights_df: pd.DataFrame,
        price_data: pd.DataFrame,
        rebalance_freq: str = "M",
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> Dict:
        """
        基于权重DataFrame执行回测（用于StockRanker等场景）

        Args:
            weights_df: 包含date/ticker/weight列的DataFrame
            price_data: 包含date/ticker/close列的DataFrame
            rebalance_freq: 再平衡频率
            date_col: 日期列名
            ticker_col: 股票代码列名

        Returns:
            Dict: 回测结果
        """
        try:
            # 构建价格透视表
            price_pivot = price_data.pivot(index=date_col, columns=ticker_col, values="close")
            price_pivot.index = pd.to_datetime(price_pivot.index)

            # 构建权重透视表
            weight_pivot = weights_df.pivot(index=date_col, columns=ticker_col, values="weight")
            weight_pivot.index = pd.to_datetime(weight_pivot.index)
            weight_pivot = weight_pivot.reindex(price_pivot.index).ffill().fillna(0)

            # 归一化权重
            weight_pivot = weight_pivot.div(weight_pivot.sum(axis=1).replace(0, 1), axis=0)

            # 使用VectorBT的从权重构建组合
            pf = vbt.Portfolio.from_pypf(
                vbt.PYPortfolio(
                    close=price_pivot,
                    weight=weight_pivot,
                    init_cash=self.initial_capital,
                    fee=self.commission_rate,
                    freq="1D",
                )
            )

            # 计算指标
            returns = pf.returns()
            equity_curve = pf.value()

            metrics = self.calculate_metrics(returns, equity_curve)

            return {
                "success": True,
                "metrics": metrics,
                "equity_curve": equity_curve,
                "total_return": metrics.get("total_return") if metrics.get("total_return") is not None else 0,
                "sharpe_ratio": metrics.get("sharpe_ratio") if metrics.get("sharpe_ratio") is not None else 0,
                "max_drawdown": metrics.get("max_drawdown") if metrics.get("max_drawdown") is not None else 0,
            }

        except Exception as e:
            logger.error(f"权重回测失败: {e}", exc_info=True)
            # fallback: 简单计算等权组合收益
            try:
                price_pivot = price_data.pivot(index=date_col, columns=ticker_col, values="close")
                price_pivot.index = pd.to_datetime(price_pivot.index)
                returns = price_pivot.pct_change().mean(axis=1).dropna()
                if len(returns) == 0:
                    return {"error": "权重回测fallback失败：无法计算收益率", "success": False}
                equity_curve = (1 + returns).cumprod()
                metrics = self.calculate_metrics(returns, equity_curve)
                return {
                    "success": True,
                    "metrics": metrics,
                    "equity_curve": equity_curve,
                    "fallback": True,
                    "fallback_reason": str(e),
                }
            except Exception as e2:
                return {"error": f"权重回测失败: {e2}", "success": False}

    def calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series = None,
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.03,
    ) -> Dict:
        """
        计算性能指标（委托risk_metrics + empyrical）

        Args:
            returns: 收益率序列
            equity_curve: 净值曲线（可选，当前保留参数以兼容调用方）
            annual_trading_days: 年化交易日数，默认252
            risk_free_rate: 无风险利率，默认3%

        Returns:
            Dict: 包含各种性能指标的字典
        """
        returns_clean = returns.dropna()

        if len(returns_clean) == 0:
            return self._empty_metrics()

        return calculate_risk_metrics(returns_clean, risk_free_rate, annual_trading_days)

    def _calculate_volatility(self, returns: pd.Series, annual_trading_days: int = 252) -> Optional[float]:
        """计算年化波动率（委托risk_metrics统一入口，符合规则2）"""
        return _calc_volatility(returns, annual_trading_days=annual_trading_days)

    def _format_trades_df(self, pf) -> Optional[pd.DataFrame]:
        """从VectorBT Portfolio提取并格式化交易记录为中文可读格式

        Args:
            pf: VectorBT Portfolio对象

        Returns:
            格式化后的交易记录DataFrame，无交易记录时返回None
        """
        trades_df = None
        try:
            if hasattr(pf, "trades") and hasattr(pf.trades, "records_readable"):
                trades_readable = pf.trades.records_readable

                if trades_readable is not None and len(trades_readable) > 0:
                    trades_df = trades_readable.copy()

                    # 创建列名映射（英文 -> 中文）
                    column_mapping = {
                        "Trade Id": "交易ID",
                        "Column": "股票代码",
                        "Size": "数量",
                        "Entry Timestamp": "入场时间",
                        "Avg Entry Price": "入场价格",
                        "Entry Fees": "入场手续费",
                        "Exit Timestamp": "出场时间",
                        "Avg Exit Price": "出场价格",
                        "Exit Fees": "出场手续费",
                        "PnL": "收益",
                        "Return": "收益率",
                        "Direction": "方向",
                        "Status": "状态",
                        "Parent Id": "父ID",
                    }

                    # 重命名列
                    trades_df.rename(columns=column_mapping, inplace=True)

                    # 转换方向和状态为中文
                    if "方向" in trades_df.columns:
                        trades_df["方向"] = trades_df["方向"].map({"Long": "做多", "Short": "做空"}).fillna("未知")

                    if "状态" in trades_df.columns:
                        trades_df["状态"] = trades_df["状态"].map({"Open": "持仓中", "Closed": "已平仓"}).fillna("未知")

                    # 将入场时间设为索引
                    if "入场时间" in trades_df.columns:
                        try:
                            trades_df["入场时间"] = pd.to_datetime(trades_df["入场时间"], errors="coerce")
                        except Exception as e:
                            logger.debug(f"入场时间转换失败: {e}")
                        trades_df.set_index("入场时间", inplace=True)

                    # 转换出场时间为可读格式
                    if "出场时间" in trades_df.columns:
                        try:
                            exit_time_series = trades_df["出场时间"].copy()
                            mask = exit_time_series.notna()
                            if mask.any():
                                formatted = exit_time_series[mask].dt.strftime("%Y-%m-%d")
                                trades_df["出场时间"] = exit_time_series.astype(object)
                                trades_df.loc[mask, "出场时间"] = formatted.values
                        except Exception as e:
                            logger.warning(f"出场时间格式化失败: {e}")

                    # 对于单股票回测，如果股票代码列全是0，则删除该列
                    if "股票代码" in trades_df.columns:
                        unique_codes = trades_df["股票代码"].unique()
                        if len(unique_codes) == 1 and (unique_codes[0] == 0 or unique_codes[0] == "0"):
                            trades_df = trades_df.drop(columns=["股票代码"])

                    # 删除不需要的列
                    for col in ["Exit Trade Id", "Position Id"]:
                        if col in trades_df.columns:
                            trades_df = trades_df.drop(columns=[col])

                    # 计算交易价值（入场价格 * 数量）
                    if "入场价格" in trades_df.columns and "数量" in trades_df.columns:
                        trades_df["价值"] = trades_df["入场价格"] * trades_df["数量"].abs()
        except Exception as e:
            logger.warning(f"提取VectorBT交易记录失败: {e}")
            logger.debug(traceback.format_exc())
            trades_df = None

        return trades_df

    def _calculate_var_cvar(self, returns: pd.Series, cutoff: float = 0.05) -> tuple:
        """计算VaR和CVaR（委托risk_metrics统一入口，符合规则2）"""
        returns_clean = returns.dropna()
        if len(returns_clean) < 2:
            return None, None
        metrics = calculate_risk_metrics(returns_clean)
        return metrics.get("var_95"), metrics.get("cvar_95")

    def _empty_metrics(self) -> Dict:
        """返回空的性能指标字典"""
        return _empty_metrics()


def check_vectorbt_available() -> bool:
    """检查vectorbt是否可用"""
    return True
