# 阶段 4：组合成策略 —— 将因子落地为可回测的策略

## 目标
将筛选出的有效因子组合成完整的选股策略，实现从信号生成到组合构建、回测验证的全流程落地。

## 预计时长
1-2 周

## 输出物
1 个完整的双因子选股策略（含回测代码、绩效报告、参数说明）

---

## 步骤详解

### Step 1：策略设计（2天）

#### 1.1 确定策略框架

```
策略名称：[自定义名称]
策略类型：双因子选股策略
调仓频率：月度（每月最后一个交易日）
股票池：全A股（剔除ST、停牌、上市不足60日）
基准指数：中证500 / 沪深300
```

#### 1.2 因子选择与权重

**因子组合原则**：
- 因子间相关性低（|ρ| < 0.5）
- 因子风格互补（如价值 + 动量，质量 + 低波）
- 避免同类型因子简单叠加

**推荐组合示例**：

| 组合方案 | 因子A | 因子B | 组合逻辑 |
|---------|------|------|---------|
| 方案1 | EP（估值） | ROE（质量） | 便宜的好公司 |
| 方案2 | 20日动量 | 20日波动率 | 趋势强且波动低 |
| 方案3 | BP（估值） | 营收增长率（成长） | 价值成长平衡 |

**因子合成方法**：

1. **等权法**
   ```python
   combined_factor = (factor_A_rank + factor_B_rank) / 2
   ```

2. **IC加权法**
   ```python
   # 按历史IC均值加权
   weight_A = IC_mean_A / (IC_mean_A + IC_mean_B)
   weight_B = IC_mean_B / (IC_mean_A + IC_mean_B)
   combined_factor = weight_A * factor_A_rank + weight_B * factor_B_rank
   ```

3. **ICIR加权法**（推荐）
   ```python
   # 按ICIR加权，更稳定
   weight_A = ICIR_A / (ICIR_A + ICIR_B)
   weight_B = ICIR_B / (ICIR_A + ICIR_B)
   combined_factor = weight_A * factor_A_rank + weight_B * factor_B_rank
   ```

#### 1.3 选股规则设计

```python
def select_stocks(combined_factor, date, top_n=50):
    """
    每月选股逻辑
    """
    # 1. 获取当月截面数据
   截面 = combined_factor[combined_factor['trade_date'] == date].copy()
    
    # 2. 基础筛选
    截面 = 截面[截面['is_st'] == False]
    截面 = 截面[截面['list_days'] >= 60]
    截面 = 截面[截面['is_suspend'] == False]
    
    # 3. 可选：行业中性约束
    # 每个行业最多选N只
    
    # 4. 可选：市值约束
    # 排除市值最小的10%股票（流动性考虑）
    
    # 5. 按合成因子排序，取前N只
    截面 = 截面.sort_values('combined_factor', ascending=False)
    selected = 截面.head(top_n)
    
    return selected[['code', 'name', 'combined_factor', 'weight']]
```

#### 1.4 权重分配方案

1. **等权配置**
   - 每只入选股票权重 = 1 / N
   - 简单，但可能受小市值股票影响

2. **市值加权**
   - 按流通市值加权
   - 更接近基准，跟踪误差小

3. **因子加权**
   - 按因子值大小加权（因子值越高，权重越大）
   - 更充分利用因子信息

4. **风险平价（进阶）**
   - 按波动率倒数加权
   - 降低组合波动

---

### Step 2：回测框架搭建（3天）

#### 2.1 回测参数设置

```python
backtest_config = {
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'initial_capital': 10_000_000,  # 初始资金1000万
    'benchmark': '000905.SH',       # 中证500
    'rebalance_freq': 'monthly',    # 月度调仓
    'rebalance_day': -1,            # 每月最后一个交易日
    'max_holding': 50,              # 最多持有50只
    'commission_rate': 0.0003,      # 手续费率 0.03%
    'slippage': 0.001,              # 滑点 0.1%
    'tax_rate': 0.001,              # 印花税 0.1%（卖出）
}
```

#### 2.2 回测主循环

```python
def backtest(strategy, config):
    """
    回测主函数
    """
    # 初始化
    portfolio = Portfolio(config['initial_capital'])
    trade_dates = get_trade_dates(config['start_date'], config['end_date'])
    rebalance_dates = [d for d in trade_dates if is_rebalance_day(d)]
    
    records = []
    
    for i, date in enumerate(trade_dates):
        # 1. 获取当日行情
        daily_data = get_daily_data(date)
        
        # 2. 调仓日执行调仓
        if date in rebalance_dates:
            # 生成交易信号
            signals = strategy.generate_signals(date)
            
            # 计算目标持仓
            target_positions = strategy.calculate_weights(signals)
            
            # 执行交易（考虑手续费、滑点）
            trades = portfolio.rebalance(target_positions, daily_data)
            
            # 记录交易
            records.extend(trades)
        
        # 3. 每日更新净值
        portfolio.update_nav(daily_data)
        
        # 4. 记录每日状态
        records.append({
            'date': date,
            'nav': portfolio.nav,
            'holdings': portfolio.holdings.copy()
        })
    
    return records
```

#### 2.3 交易成本模型

```python
def calculate_trade_cost(trade_value, is_buy, config):
    """
    计算交易成本
    """
    commission = trade_value * config['commission_rate']
    commission = max(commission, 5)  # 最低5元
    
    slippage = trade_value * config['slippage']
    
    tax = 0
    if not is_buy:  # 卖出收印花税
        tax = trade_value * config['tax_rate']
    
    total_cost = commission * 2 + slippage + tax  # 买卖双边佣金
    return total_cost
```

---

### Step 3：绩效分析（2天）

#### 3.1 收益指标

| 指标 | 计算方式 | 说明 |
|-----|---------|------|
| 累计收益 | $\prod(1+r_t) - 1$ | 策略总收益 |
| 年化收益 | $(1 + \text{累计收益})^{\frac{252}{n}} - 1$ | 按交易日年化 |
| 超额收益 | 策略收益 - 基准收益 | 相对基准的表现 |
| 年化超额 | 超额收益的年化值 | 核心评价指标 |

#### 3.2 风险指标

| 指标 | 计算方式 | 说明 |
|-----|---------|------|
| 年化波动率 | $\sigma(r_t) \times \sqrt{252}$ | 收益波动程度 |
| 最大回撤 | $\max_{t} \frac{峰值 - 当前}{峰值}$ | 最大亏损幅度 |
| 下行波动率 | 仅负收益的波动率 | 下行风险 |
| 最大回撤恢复时间 | 从最大回撤恢复到新高的天数 | 恢复能力 |

#### 3.3 风险调整收益指标

| 指标 | 计算方式 | 说明 |
|-----|---------|------|
| 夏普比率 | $\frac{R_p - R_f}{\sigma_p}$ | 单位总风险的超额收益 |
| 信息比率 | $\frac{R_p - R_b}{\sigma(R_p - R_b)}$ | 单位跟踪误差的超额收益 |
| 卡玛比率 | $\frac{R_p}{\text{最大回撤}}$ | 单位回撤的收益 |
| 索提诺比率 | $\frac{R_p - R_f}{\sigma_{下行}}$ | 单位下行风险的收益 |

#### 3.4 绩效分析代码

```python
class PerformanceAnalyzer:
    def __init__(self, strategy_returns, benchmark_returns):
        self.strategy = strategy_returns
        self.benchmark = benchmark_returns
        self.excess = strategy_returns - benchmark_returns
    
    def annual_return(self):
        return (1 + self.strategy).prod() ** (252 / len(self.strategy)) - 1
    
    def annual_volatility(self):
        return self.strategy.std() * np.sqrt(252)
    
    def sharpe_ratio(self, risk_free=0.03):
        return (self.annual_return() - risk_free) / self.annual_volatility()
    
    def information_ratio(self):
        return self.excess.mean() / self.excess.std() * np.sqrt(252)
    
    def max_drawdown(self):
        cumulative = (1 + self.strategy).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def calmar_ratio(self):
        return self.annual_return() / abs(self.max_drawdown())
    
    def generate_report(self):
        return {
            '年化收益': f"{self.annual_return():.2%}",
            '年化波动': f"{self.annual_volatility():.2%}",
            '最大回撤': f"{self.max_drawdown():.2%}",
            '夏普比率': f"{self.sharpe_ratio():.2f}",
            '信息比率': f"{self.information_ratio():.2f}",
            '卡玛比率': f"{self.calmar_ratio():.2f}",
        }
```

#### 3.5 可视化输出

必须生成的图表：
1. **策略 vs 基准净值曲线**
2. **超额收益累计曲线**
3. **月度收益热力图**
4. **回撤曲线**
5. **年度收益对比柱状图**
6. **行业分布饼图**
7. **市值分布箱线图**

---

### Step 4：参数敏感性分析（2天）

#### 4.1 关键参数

| 参数 | 默认值 | 测试范围 |
|-----|-------|---------|
| 持股数量 | 50 | [20, 30, 50, 80, 100] |
| 因子权重 | ICIR加权 | [等权, IC加权, ICIR加权] |
| 调仓频率 | 月度 | [周度, 双周, 月度, 季度] |
| 因子去极值分位 | 1% / 99% | [0.5%, 1%, 2%, 5%] |
| 换手率惩罚 | 无 | [有/无] |

#### 4.2 参数遍历脚本

```python
def parameter_sensitivity_analysis(param_grid, strategy_class, data):
    """
    参数敏感性分析
    """
    results = []
    
    for params in ParameterGrid(param_grid):
        # 初始化策略
        strategy = strategy_class(**params)
        
        # 运行回测
        backtest_result = backtest(strategy, data)
        
        # 计算绩效
        perf = PerformanceAnalyzer(backtest_result.returns, benchmark_returns)
        
        results.append({
            **params,
            'annual_return': perf.annual_return(),
            'sharpe': perf.sharpe_ratio(),
            'max_drawdown': perf.max_drawdown(),
            'info_ratio': perf.information_ratio()
        })
    
    return pd.DataFrame(results)
```

---

### Step 5：策略文档输出（1天）

#### 策略文档模板

```markdown
# 策略报告：[策略名称]

## 一、策略概述
- 策略名称：
- 策略类型：双因子选股策略
- 调仓频率：
- 股票池：
- 基准指数：

## 二、因子说明
### 因子A
- 名称：
- 逻辑：
- 权重：

### 因子B
- 名称：
- 逻辑：
- 权重：

### 因子合成
- 合成方法：
- 预处理：

## 三、回测设置
- 回测区间：
- 初始资金：
- 手续费率：
- 滑点：
- 印花税：

## 四、回测结果
### 收益指标
- 年化收益：
- 年化超额：
- 累计收益：

### 风险指标
- 年化波动：
- 最大回撤：
- 最大回撤发生时间：

### 风险调整收益
- 夏普比率：
- 信息比率：
- 卡玛比率：

## 五、持仓分析
- 平均持股数量：
- 平均换手率：
- 行业分布：
- 市值分布：

## 六、参数敏感性
- 最优参数组合：
- 参数稳定性评价：

## 七、风险提示
- 策略失效场景：
- 最大风险点：
- 改进方向：

## 八、附录
- 回测代码
- 完整绩效数据
- 分年度表现
```

---

## 检查清单

- [ ] 双因子组合方案已确定
- [ ] 因子合成方法已选定
- [ ] 选股规则已明确
- [ ] 回测框架搭建完成
- [ ] 交易成本模型已纳入
- [ ] 回测结果已生成
- [ ] 绩效指标已计算（收益、风险、风险调整收益）
- [ ] 7张核心图表已生成
- [ ] 参数敏感性分析已完成
- [ ] 策略文档撰写完成

---

## 常见问题

**Q1：回测收益很高，但担心过拟合？**
> 1. 检查参数是否过多，是否经过大量优化
> 2. 做样本外测试（用最近1-2年数据验证）
> 3. 简化策略，减少参数
> 4. 检查是否在特定市场阶段表现异常好

**Q2：策略换手率太高怎么办？**
> 1. 加入换手率惩罚项
> 2. 延长调仓周期（如从周度改为月度）
> 3. 加入持仓惯性（如上期持仓给予一定权重）
> 4. 提高选股阈值，减少边际股票更换

**Q3：策略在小盘股中表现好，但实盘容量有限？**
> 1. 加入市值约束（如只选市值前80%）
> 2. 测试策略在大/中盘股中的表现
> 3. 考虑策略的资金容量限制
