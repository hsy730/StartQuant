"""
导出服务 - 导出回测结果到Excel多Sheet
"""
from typing import Dict
import pandas as pd
from datetime import datetime


class ExportService:
    """导出服务"""

    def __init__(self):
        pass

    def export_backtest_to_excel(
        self,
        backtest_result: Dict,
        output_path: str,
        metrics: Dict = None,
        strategy_name: str = "策略"
    ):
        """
        导出回测结果到Excel（多Sheet）

        Args:
            backtest_result: 回测结果字典
            output_path: 输出文件路径
            metrics: 性能指标（可选）
            strategy_name: 策略名称
        """
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: 摘要
            self._write_summary_sheet(writer, backtest_result, metrics, strategy_name)

            # Sheet 2: 净值曲线
            self._write_equity_sheet(writer, backtest_result)

            # Sheet 3: 持仓历史
            self._write_positions_sheet(writer, backtest_result)

            # Sheet 4: 交易记录
            self._write_trades_sheet(writer, backtest_result)

            # Sheet 5: 收益率分析
            self._write_returns_sheet(writer, backtest_result)

    def _write_summary_sheet(
        self,
        writer: pd.ExcelWriter,
        backtest_result: Dict,
        metrics: Dict,
        strategy_name: str
    ):
        """写入摘要Sheet"""
        summary_data = []

        # 策略名称
        summary_data.append(["策略名称", strategy_name])
        summary_data.append(["导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        summary_data.append([])

        # 回测参数
        summary_data.append(["回测参数"])
        summary_data.append(["初始资金", f"{backtest_result.get('initial_capital', 1000000):,.0f} 元"])
        summary_data.append(["交易次数", f"{backtest_result.get('trades_count', 0)} 次"])
        summary_data.append([])

        # 性能指标
        if metrics:
            def _fmt(val, fmt, default=0):
                return f"{val:{fmt}}" if val is not None else f"{default:{fmt}}"

            summary_data.append(["性能指标"])
            summary_data.append(["总收益率", _fmt(metrics.get('total_return'), '.2%', 0)])
            summary_data.append(["年化收益率", _fmt(metrics.get('annual_return'), '.2%', 0)])
            summary_data.append(["波动率", _fmt(metrics.get('volatility'), '.2%', 0)])
            summary_data.append(["夏普比率", _fmt(metrics.get('sharpe_ratio'), '.2f', 0)])
            summary_data.append(["最大回撤", _fmt(metrics.get('max_drawdown'), '.2%', 0)])
            summary_data.append(["卡玛比率", _fmt(metrics.get('calmar_ratio'), '.2f', 0)])
            summary_data.append(["胜率", _fmt(metrics.get('win_rate'), '.2%', 0)])
            summary_data.append(["索提诺比率", _fmt(metrics.get('sortino_ratio'), '.2f', 0)])

        # 创建DataFrame
        df_summary = pd.DataFrame(summary_data, columns=["项目", "值"])

        # 写入Excel
        df_summary.to_excel(writer, sheet_name="摘要", index=False)

    def _write_equity_sheet(
        self,
        writer: pd.ExcelWriter,
        backtest_result: Dict
    ):
        """写入净值曲线Sheet"""
        equity = backtest_result.get("equity_curve")

        if equity is not None:
            df_equity = equity.to_frame(name="净值")
            df_equity.index.name = "日期"

            # 计算收益率
            df_equity["收益率"] = df_equity["净值"].pct_change()
            df_equity["累计收益率"] = (df_equity["净值"] / df_equity["净值"].iloc[0] - 1)

            df_equity.to_excel(writer, sheet_name="净值曲线")

    def _write_positions_sheet(
        self,
        writer: pd.ExcelWriter,
        backtest_result: Dict
    ):
        """写入持仓历史Sheet"""
        positions = backtest_result.get("positions")
        weights = backtest_result.get("weights")
        signals = backtest_result.get("signals")

        if positions is not None:
            df_positions = pd.DataFrame(index=positions.index)
            df_positions["持仓"] = positions.values

            if weights is not None:
                df_positions["权重"] = weights.values

            if signals is not None:
                df_positions["信号"] = signals.values

            df_positions.index.name = "日期"
            df_positions.to_excel(writer, sheet_name="持仓历史")

    def _write_trades_sheet(
        self,
        writer: pd.ExcelWriter,
        backtest_result: Dict
    ):
        """写入交易记录Sheet"""
        weights = backtest_result.get("weights")

        if weights is not None:
            # 找出权重变化的点
            weight_changes = weights.diff().fillna(0)
            trade_dates = weight_changes[weight_changes != 0].index

            if len(trade_dates) > 0:
                trades_list = []

                for i, date in enumerate(trade_dates):
                    prev_weight = weights.loc[weight_changes.index.get_loc(date) - 1] if i > 0 else 0
                    curr_weight = weights.loc[date]

                    trades_list.append({
                        "日期": date,
                        "操作": "买入" if curr_weight > prev_weight else "卖出",
                        "前权重": f"{prev_weight:.2%}",
                        "后权重": f"{curr_weight:.2%}",
                        "变化量": f"{(curr_weight - prev_weight):.2%}",
                    })

                df_trades = pd.DataFrame(trades_list)
                df_trades.to_excel(writer, sheet_name="交易记录", index=False)
            else:
                # 无交易记录
                pd.DataFrame({"消息": ["无交易记录"]}).to_excel(
                    writer, sheet_name="交易记录", index=False
                )

    def _write_returns_sheet(
        self,
        writer: pd.ExcelWriter,
        backtest_result: Dict
    ):
        """写入收益率分析Sheet"""
        portfolio_returns = backtest_result.get("portfolio_returns")

        if portfolio_returns is not None:
            # 日收益率
            df_returns = portfolio_returns.to_frame(name="日收益率")

            # 月度收益率
            monthly_returns = (1 + portfolio_returns).resample("ME").prod() - 1
            df_returns["月度收益率"] = monthly_returns

            # 年度收益率
            annual_returns = (1 + portfolio_returns).resample("YE").prod() - 1
            df_returns["年度收益率"] = annual_returns

            # 统计信息
            stats_data = [
                ["总收益率", f"{(1 + portfolio_returns).prod() - 1:.2%}"],
                ["平均日收益率", f"{portfolio_returns.mean():.4%}"],
                ["收益率标准差", f"{portfolio_returns.std():.4%}"],
                ["最大日收益", f"{portfolio_returns.max():.4%}"],
                ["最大日亏损", f"{portfolio_returns.min():.4%}"],
                ["正收益天数", f"{(portfolio_returns > 0).sum()} 天"],
                ["负收益天数", f"{(portfolio_returns < 0).sum()} 天"],
                ["零收益天数", f"{(portfolio_returns == 0).sum()} 天"],
            ]

            df_stats = pd.DataFrame(stats_data, columns=["统计项目", "值"])

            df_returns.index.name = "日期"
            df_returns.to_excel(writer, sheet_name="收益率分析")

            # 将统计信息写入同一个Sheet
            df_stats.to_excel(
                writer, sheet_name="收益率分析", startrow=len(df_returns) + 3
            )

    def export_comparison_to_excel(
        self,
        comparison_result: Dict,
        output_path: str
    ):
        """
        导出策略对比结果到Excel

        Args:
            comparison_result: 策略对比结果
            output_path: 输出文件路径
        """
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: 指标对比
            metrics_df = comparison_result["metrics_comparison"]
            metrics_df.to_excel(writer, sheet_name="指标对比")

            # Sheet 2: 排名
            ranking = comparison_result["ranking"]
            df_ranking = pd.DataFrame(ranking)
            df_ranking.to_excel(writer, sheet_name="排名")

            # Sheet 3: 净值曲线对比
            equity_curves = comparison_result["equity_curves"]
            df_equities = pd.DataFrame(equity_curves)
            df_equities.index.name = "日期"
            df_equities.to_excel(writer, sheet_name="净值曲线")

            # Sheet 4: 统计检验
            statistical_tests = comparison_result["statistical_tests"]
            if statistical_tests:
                tests_list = []

                for test_key, test_result in statistical_tests.items():
                    tests_list.append({
                        "对比": test_key,
                        "独立T检验p值": f"{test_result['independent_t_test']['p_value']:.4f}",
                        "显著性": "是" if test_result['independent_t_test']['significant'] else "否",
                        "配对T检验p值": f"{test_result['paired_t_test']['p_value']:.4f}",
                        "显著性(配对)": "是" if test_result['paired_t_test']['significant'] else "否",
                        "相关系数": f"{test_result['correlation']:.4f}",
                    })

                df_tests = pd.DataFrame(tests_list)
                df_tests.to_excel(writer, sheet_name="统计检验", index=False)


# 全局导出服务实例
export_service = ExportService()
