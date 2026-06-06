"""
策略基类 - 定义策略接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
import numpy as np

try:
    import empyrical
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False


class BaseStrategy(ABC):
    """策略抽象基类"""

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        **kwargs
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
    def calculate_weights(
        self,
        df: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
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

        # 确保数据按日期排序
        if "date" in df.columns:
            df = df.sort_values("date")
        elif df.index.name != "date":
            df = df.reset_index()

        # 1. 生成交易信号
        signals = self.generate_signals(df)

        # 2. 计算权重
        weights = self.calculate_weights(df, signals)

        # 3. 计算下一期收益率
        # pct_change(1) = (close[t] - close[t-1]) / close[t-1]，即当日收益率
        # shift(-1) 取下一日的收益率，用于与当日权重配对
        # 这样在t日决策（权重），t+1日获得收益，无前视偏差
        df["next_return"] = df["close"].pct_change(1).shift(-1)

        # 4. 计算组合收益（权重 * 下一期收益率）
        portfolio_returns = weights * df["next_return"]

        # 5. 扣除手续费（简化版：假设每次调仓产生手续费）
        # 权重变化时产生手续费，基于初始资金简化计算
        weight_change = weights.diff().abs()
        commission = weight_change * self.initial_capital * self.commission_rate
        portfolio_returns = portfolio_returns - commission

        # 6. 计算净值曲线
        equity = (1 + portfolio_returns.fillna(0)).cumprod() * self.initial_capital

        # 7. 计算交易次数
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
        risk_free_rate: float = 0.03
    ) -> Dict:
        """
        计算性能指标

        Args:
            returns: 收益率序列
            annual_trading_days: 年化交易日数
            risk_free_rate: 无风险利率

        Returns:
            Dict: 性能指标
        """
        returns_clean = returns.dropna()

        if len(returns_clean) == 0:
            return self._empty_metrics()

        returns_arr = returns_clean.values

        if EMPYRICAL_AVAILABLE:
            # 使用empyrical计算标准风险指标
            total_return = float(empyrical.cum_returns_final(returns_arr))
            annual_return = float(empyrical.annual_return(returns_arr, period='daily', annualization=annual_trading_days))
            volatility = float(empyrical.annual_volatility(returns_arr, period='daily', annualization=annual_trading_days))
            sharpe_ratio = float(empyrical.sharpe_ratio(returns_arr, risk_free=risk_free_rate, period='daily', annualization=annual_trading_days))
            sortino_ratio = float(empyrical.sortino_ratio(returns_arr, required_return=risk_free_rate, period='daily', annualization=annual_trading_days))
            max_drawdown = float(empyrical.max_drawdown(returns_arr))
            calmar_ratio = float(empyrical.calmar_ratio(returns_arr, period='daily', annualization=annual_trading_days))
        else:
            # 总收益率
            total_return = (1 + returns_clean).prod() - 1

            # 年化收益率
            n_days = len(returns_clean)
            annual_return = (1 + total_return) ** (annual_trading_days / n_days) - 1

            # 波动率
            volatility = returns_clean.std() * np.sqrt(annual_trading_days)

            # 夏普比率
            daily_rf = risk_free_rate / annual_trading_days
            excess_returns = returns_clean - daily_rf
            sharpe_ratio = (
                excess_returns.mean() * annual_trading_days / volatility
                if volatility > 0
                else 0.0
            )

            # 最大回撤
            equity = (1 + returns_clean).cumprod()
            peak = equity.cummax()
            drawdown = (peak - equity) / peak
            max_drawdown = drawdown.max()

            # 卡玛比率
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0

            # 索提诺比率（标准下行偏差公式）
            daily_rf = risk_free_rate / annual_trading_days
            excess_returns = returns_clean - daily_rf
            downside_diff = np.minimum(excess_returns, 0)
            downside_std = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(annual_trading_days)
            if downside_std > 0:
                sortino_ratio = (excess_returns.mean() * annual_trading_days) / downside_std
            else:
                sortino_ratio = 0.0

        # 胜率
        win_rate = (returns_clean > 0).mean()

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "win_rate": float(win_rate),
            "sortino_ratio": float(sortino_ratio),
        }

    def _empty_metrics(self) -> Dict:
        """返回空的性能指标"""
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "sortino_ratio": 0.0,
        }

    def get_name(self) -> str:
        """获取策略名称"""
        return self.__class__.__name__

    def get_description(self) -> str:
        """获取策略描述"""
        return self.__doc__ or "无描述"
