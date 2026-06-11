"""
策略注册表 - 管理所有策略
"""

from typing import Dict, Type, List
from backend.strategies.base_strategy import BaseStrategy
from backend.strategies import (
    EqualWeightStrategy,
    MarketCapStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
)


class StrategyRegistry:
    """策略注册表"""

    # 注册的策略
    _strategies: Dict[str, Type[BaseStrategy]] = {
        "equal_weight": EqualWeightStrategy,
        "market_cap": MarketCapStrategy,
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseStrategy]):
        """
        注册新策略

        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def get_strategy(cls, name: str, **kwargs) -> BaseStrategy:
        """
        获取策略实例

        Args:
            name: 策略名称
            **kwargs: 策略参数

        Returns:
            策略实例

        Raises:
            ValueError: 策略不存在
        """
        if name not in cls._strategies:
            raise ValueError(f"策略 '{name}' 不存在。可用策略: {list(cls._strategies.keys())}")

        strategy_class = cls._strategies[name]
        return strategy_class(**kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        列出所有已注册的策略

        Returns:
            策略名称列表
        """
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, name: str) -> Dict:
        """
        获取策略信息

        Args:
            name: 策略名称

        Returns:
            策略信息字典
        """
        if name not in cls._strategies:
            raise ValueError(f"策略 '{name}' 不存在")

        strategy_class = cls._strategies[name]
        instance = strategy_class()

        return {
            "name": name,
            "class_name": strategy_class.__name__,
            "description": instance.get_description(),
            "params": instance.params,
        }

    @classmethod
    def get_all_strategies_info(cls) -> List[Dict]:
        """
        获取所有策略的信息

        Returns:
            策略信息列表
        """
        return [cls.get_strategy_info(name) for name in cls._strategies.keys()]


# 全局策略注册表实例
strategy_registry = StrategyRegistry()
