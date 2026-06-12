"""
策略基类 - 定义策略接口
"""

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

from backend.services.risk_metrics import (
    calculate_risk_metrics as _calculate_risk_metrics,
    _empty_metrics as _risk_empty_metrics,
)
from backend.utils.return_calculator import calculate_future_return


class BaseStrategy(ABC):
    """策略抽象基类"""

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        **kwargs,
    ):
        """
        初始化策略

        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            **kwargs: 策略特定参数
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.params = kwargs

        # 回测结果存储
        self.equity_curve = None
        self.positions = None
        self.trades = None

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Args:
            df: 包含价格和因子数据的DataFrame

        Returns:
            pd.Series: 信号序列，1表示买入，-1表示卖出，0表示持有
        """
        pass

    @abstractmethod
    def calculate_weights(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        计算持仓权重

        Args:
            df: 数据
            signals: 交易信号

        Returns:
            pd.Series: 权重序列，范围[-1, 1]
        """
        pass

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        执行回测

        Args:
            df: 包含价格和因子数据的DataFrame

        Returns:
            Dict: 回测结果
        """
        df = df.copy()

        # 确保数据按日期排序，同时保留 MultiIndex 结构
        if isinstance(df.index, pd.MultiIndex):
            # MultiIndex (date, asset) 保持不变，这是多股票回测的标准格式
            pass
        elif "date" in df.columns:
            df = df.sort_values("date")
        elif df.index.name != "date":
            original_index_name = df.index.name
            df = df.reset_index()
            # Try to sort by date column if available
            sort_col = None
            for col in ["date", "index", original_index_name]:
                if col in df.columns:
                    sort_col = col
                    break
            if sort_col:
                df = df.sort_values(sort_col)

        # 1. 生成交易信号
        signals = self.generate_signals(df)

        # 2. 计算权重
        weights = self.calculate_weights(df, signals)

        # 3. 计算下一期收益率
        # t日决策（权重），t+1日获得收益，无前视偏差
        df["next_return"] = calculate_future_return(df)

        # 4. 计算组合收益（权重 * 下一期收益率）
        portfolio_returns = weights * df["next_return"]

        # MultiIndex (date, asset) 下需按日期聚合得到组合层面收益
        is_multiindex = isinstance(df.index, pd.MultiIndex)
        if is_multiindex:
            portfolio_returns = portfolio_returns.groupby(level=0).sum()

        # 5. 扣除手续费（简化版：假设每次调仓产生手续费）
        # portfolio_returns 是比例收益率，手续费也必须保持比例
        # weight_change 是权重变化比例，commission_rate 是费率
        # 比例手续费 = 权重变化 * 费率
        # 首期视作从0建仓，diff首行NaN用初始权重填充
        if is_multiindex:
            # MultiIndex 下按资产分组计算权重变化，再按日期聚合手续费
            weight_change = weights.groupby(level=1).diff().abs().fillna(weights.abs())
            commission = weight_change * self.commission_rate
            commission = commission.groupby(level=0).sum()
        else:
            weight_change = weights.diff().abs().fillna(weights.abs())
            commission = weight_change * self.commission_rate
        portfolio_returns = portfolio_returns - commission

        # 6. 计算净值曲线
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital

        # 7. 计算交易次数
        if is_multiindex:
            # MultiIndex 下按资产分组统计交易次数
            trades_count = int(
                weights.groupby(level=1)
                .diff()
                .fillna(0)
                .ne(0)
                .groupby(level=0)
                .any()
                .sum()
            )
        else:
            trades_count = (weights.diff().fillna(0) != 0).sum()

        # 8. 计算持仓历史
        positions = weights.copy()
        positions.name = "position"

        # 存储结果
        self.equity_curve = equity
        self.positions = positions
        self.trades = trades_count

        return {
            "portfolio_returns": portfolio_returns,
            "equity_curve": equity,
            "positions": positions,
            "trades_count": int(trades_count),
            "weights": weights,
            "signals": signals,
        }

    def calculate_metrics(
        self,
        returns: pd.Series,
        annual_trading_days: int = 252,
        risk_free_rate: float = 0.03,
    ) -> Dict:
        """
        计算性能指标（委托risk_metrics统一入口，符合规则0和代码复用原则）
        """
        return _calculate_risk_metrics(
            returns,
            risk_free_rate=risk_free_rate,
            annual_trading_days=annual_trading_days,
        )

    def _empty_metrics(self) -> Dict:
        """返回空的性能指标（委托risk_metrics统一入口）"""
        return _risk_empty_metrics()

    def get_name(self) -> str:
        """获取策略名称"""
        return self.__class__.__name__

    def get_description(self) -> str:
        """获取策略描述"""
        return self.__doc__ or "无描述"
