"""
演示：滑点在因子分析 vs 策略回测中的作用差异

通过实际代码对比，清晰展示两者的区别
"""
import numpy as np
import pandas as pd
from typing import Dict

print("=" * 80)
print("📊 滑点作用范围分析报告")
print("=" * 80)

# ============================================================
# 1️⃣ 因子分析阶段（Factor Analysis）- 不使用滑点
# ============================================================
print("\n" + "=" * 80)
print("🔬 第一阶段：因子分析（IC/IR/分层收益）")
print("=" * 80)

def calculate_factor_ic(factor_values: pd.Series, returns: pd.Series) -> Dict:
    """
    计算因子的信息系数（IC）

    这是纯统计指标，衡量因子预测能力
    ❌ 不涉及任何交易成本
    ❌ 不模拟买卖过程
    ✅ 只评估因子信号质量
    """
    # 清理数据
    valid_mask = factor_values.notna() & returns.notna()
    factor_clean = factor_values[valid_mask]
    returns_clean = returns[valid_mask]

    # 计算IC（Pearson相关系数）
    ic = factor_clean.corr(returns_clean)

    # 计算Rank IC（Spearman相关系数）
    rank_ic = factor_clean.corr(returns_clean, method='spearman')

    # 模拟数据示例
    print(f"\n📈 IC计算过程（纯统计分析）：")
    print(f"   输入：因子值序列（{len(factor_clean)}个观测点）")
    print(f"   输入：未来收益率序列（{len(returns_clean)}个观测点）")
    print(f"   计算：Pearson相关系数 → IC = {ic:.4f}")
    print(f"   计算：Spearman相关系数 → Rank IC = {rank_ic:.4f}")
    print(f"\n   ⚠️ 注意：整个过程没有使用滑点参数！")

    return {
        'IC': ic,
        'Rank_IC': rank_ic,
        'IC_STD': ic.std() if isinstance(ic, (pd.Series, np.ndarray)) else 0,
        'IR': ic / 0.05 if abs(ic) > 0 else 0,  # 假设IC标准差为5%
    }

def calculate_quantile_returns(factor_values: pd.Series, returns: pd.Series, n_quantiles: int = 5) -> Dict:
    """
    计算分层收益率（Quantile Returns）

    根据因子值将股票分成N层，计算每层的平均收益
    ❌ 不考虑交易成本
    ❌ 假设可以无成本地完美调仓
    ✅ 展示因子的单调性和区分度
    """
    # 分层
    quantiles = pd.qcut(factor_values.rank(pct=True), q=n_quantiles, labels=False, duplicates='drop')

    results = {}
    for q in range(n_quantiles):
        mask = quantiles == q
        if mask.sum() > 0:
            avg_return = returns[mask].mean()
            results[f'Q{q+1}'] = avg_return

    print(f"\n📊 分层收益计算（理论收益）：")
    for q, ret in results.items():
        print(f"   {q}层: {ret*100:.2f}%/期")
    print(f"\n   ⚠️ 注意：这是\"理想状态\"下的收益，未扣除任何成本！")

    return results


# ============================================================
# 2️⃣ 策略回测阶段（Backtesting）- 使用滑点
# ============================================================
print("\n" + "=" * 80)
print("💰 第二阶段：策略回测（模拟真实交易）")
print("=" * 80)

class SimpleBacktest:
    """简化版回测引擎"""

    def __init__(self, initial_capital: float, commission_rate: float, slippage: float):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.cash = initial_capital
        self.position = 0
        self.trades_log = []

    def execute_trade(self, signal: int, price: float) -> float:
        """
        执行交易（包含滑点和手续费）

        ✅ 这里才真正使用滑点！
        """
        if signal == 0 or self.position == signal:
            return 0.0

        trade_value = 0.0

        if signal == 1 and self.position == 0:  # 买入
            # 滑点：买入价格 = 市价 × (1 + slippage)
            execution_price = price * (1 + self.slippage)
            commission = execution_price * self.commission_rate

            total_cost = execution_price + commission
            trade_value = -total_cost  # 支出现金

            self.position = 1
            self.trades_log.append({
                'type': 'BUY',
                'market_price': price,
                'execution_price': execution_price,
                'slippage_cost': (execution_price - price),
                'commission': commission,
                'total_cost': total_cost,
            })

        elif signal == -1 and self.position == 1:  # 卖出
            # 滑点：卖出价格 = 市价 × (1 - slippage)
            execution_price = price * (1 - self.slippage)
            commission = execution_price * self.commission_rate

            total_revenue = execution_price - commission
            trade_value = total_revenue  # 收入现金

            self.position = 0
            self.trades_log.append({
                'type': 'SELL',
                'market_price': price,
                'execution_price': execution_price,
                'slippage_cost': (price - execution_price),
                'commission': commission,
                'total_revenue': total_revenue,
            })

        return trade_value


def run_backtest_with_different_slippages(
    signals: pd.Series,
    prices: pd.Series,
    slippage_values: list = [0.0, 0.001, 0.002, 0.003]
) -> Dict:
    """
    对比不同滑点下的回测结果
    """
    print("\n💸 不同滑点设置对策略收益的影响：\n")

    results = {}
    for slip in slippage_values:
        bt = SimpleBacktest(
            initial_capital=1000000,
            commission_rate=0.0003,  # 万三手续费
            slippage=slip
        )

        total_pnl = 0.0
        for i in range(1, len(signals)):
            pnl = bt.execute_trade(signals.iloc[i], prices.iloc[i])
            total_pnl += pnl

        final_return = (total_pnl / 1000000) * 100
        results[f'{slip*100:.1f}%'] = {
            'final_pnl': total_pnl,
            'return_pct': final_return,
            'n_trades': len(bt.trades_log),
        }

        print(f"   滑点={slip*100:.1f}%: 收益={final_return:+.2f}% | 交易次数={len(bt.trades_log)}")

    return results


# ============================================================
# 🎯 实际演示
# ============================================================
print("\n" + "=" * 80)
print("🎮 实际对比演示")
print("=" * 80)

np.random.seed(42)
n_days = 252  # 一年交易日

# 生成模拟数据
factor_values = pd.Series(np.random.randn(n_days), name='momentum_factor')
returns = pd.Series(np.random.randn(n_days) * 0.02, name='daily_return')  # 日均波动2%
prices = (1 + returns).cumprod() * 100  # 从100元开始的价格序列

# 生成简单信号（因子值>0时持有）
signals = (factor_values > 0).astype(int).diff().fillna(0)
signals[signals == -1] = -1  # 平仓信号

print("\n📌 第一步：因子分析（不使用滑点）")
print("-" * 60)
ic_result = calculate_factor_ic(factor_values, returns.shift(-1))
quantile_result = calculate_quantile_returns(factor_values, returns.shift(-1))

print(f"\n✅ 因子分析结论：")
print(f"   IC = {ic_result['IC']:.4f} ({'有效' if abs(ic_result['IC']) > 0.03 else '无效'})")
print(f"   Rank_IC = {ic_result['Rank_IC']:.4f}")
print(f"   IR = {ic_result['IR']:.2f}")

print("\n\n📌 第二步：策略回测（使用滑点）")
print("-" * 60)
backtest_results = run_backtest_with_different_slippages(signals, prices)


# ============================================================
# 📊 总结对比
# ============================================================
print("\n" + "=" * 80)
print("🎯 核心结论")
print("=" * 80)

print("""
┌─────────────────┬────────────────────┬────────────────────┐
│     阶段        │      是否用滑点     │       目的         │
├─────────────────┼────────────────────┼────────────────────┤
│  因子分析        │       ❌ 不使用     │ 评估因子预测能力    │
│  (IC/IR/分层)   │                    │  (信号质量)        │
├─────────────────┼────────────────────┼────────────────────┤
│  策略回测        │       ✅ 使用      │ 模拟真实交易盈亏    │
│  (净值/收益)    │                    │  (执行效果)        │
└─────────────────┴────────────────────┴────────────────────┘

💡 关键洞察：

1️⃣ 因子分析回答："这个因子能预测收益吗？"
   • 衡量指标：IC、IR、分层收益单调性
   • 理论假设：可以无成本、即时、完整地执行交易
   • 类比：实验室环境下的药物有效性测试

2️⃣ 策略回测回答："用这个因子交易能赚钱吗？"
   • 衡量指标：净收益率、夏普比率、最大回撤
   • 现实约束：滑点、手续费、流动性冲击、延迟
   • 类比：临床试验中的实际疗效

3️⃣ 两者的关系：
   • 好的因子分析 ≠ 一定能赚钱（可能被交易成本侵蚀）
   • 差的因子分析 = 一定赚不到钱
   • 最优策略：先筛选好因子（高IC），再优化执行（低滑点）
""")


# ============================================================
# ⚠️ 特殊情况：什么时候因子分析也要考虑成本？
# ============================================================
print("\n" + "=" * 80)
print("⚠️ 进阶：何时需要在因子分析中考虑成本？")
print("=" * 80)

print("""
虽然标准的因子分析不使用滑点，但在以下场景需要特殊处理：

🔴 场景1：高频因子（换手率 > 50倍/年）
   问题：即使IC很高，频繁调仓的成本会吃掉所有利润
   解决：在因子评分中加入"成本调整后的IC"
   公式：Net_IC ≈ Gross_IC × (1 - turnover × 2 × slippage)

🔴 场景2：小市值/流动性差的股票池
   问题：因子可能偏好小市值股，但实际无法以理论价格成交
   解决：使用"可投资 universe"过滤，或在IC计算中加权

🔴 场景3：因子拥挤度（Factor Crowding）
   问题：大家都用同一个因子 → 交易拥堵 → 实际滑点飙升
   解决：监控因子拥挤度指标，动态调整预期收益

🟡 FactorHub 的解决方案：
   ✓ 第一阶段：标准因子分析（纯IC/IR，不考虑成本）
   ✓ 第二阶段：综合评分服务（ComprehensiveScoringService）
     - 包含换手率惩罚
     - 包含滑点敏感性分析（我们刚实现的！）
     - 给出"成本调整后"的综合得分

📚 推荐阅读：
   1. Grinold & Kahn《Active Portfolio Management》第4章
   2. 《因子投资：方法与实践》 - 交易成本章节
   3. AQR论文 "Trading Costs and Factor Investing"
""")
