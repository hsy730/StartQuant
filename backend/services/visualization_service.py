"""
可视化增强服务 - 生成专业的分析图表
"""

from typing import Dict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class VisualizationService:
    """可视化服务"""

    def __init__(self):
        pass

    def plot_factor_decay(
        self,
        factor_returns: pd.Series,
        factor_name: str = "因子",
        n_periods: int = 20,
    ) -> go.Figure:
        """
        绘制因子衰减曲线

        Args:
            factor_returns: 因子收益率序列（多日持仓的累计收益率）
            factor_name: 因子名称
            n_periods: 显示期数

        Returns:
            Plotly图表对象
        """
        # 计算累计收益
        cumulative_returns = (1 + factor_returns).cumprod()

        # 创建图表
        fig = go.Figure()

        # 添加累计收益曲线
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index[:n_periods],
                y=cumulative_returns.values[:n_periods],
                mode="lines+markers",
                name="累计收益",
                line=dict(color="#2E86DE", width=2),
                marker=dict(size=8),
            )
        )

        # 添加零线
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="基准线")

        # 更新布局
        fig.update_layout(
            title=f"{factor_name} - 因子衰减曲线（前{n_periods}期）",
            xaxis_title="持仓期数",
            yaxis_title="累计收益",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )

        return fig

    def plot_factor_correlation_network(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5,
        width: int = 800,
        height: int = 600,
    ) -> go.Figure:
        """
        绘制因子相关性网络图

        Args:
            correlation_matrix: 因子相关性矩阵
            threshold: 相关性阈值（只显示大于此阈值的边）
            width: 图表宽度
            height: 图表高度

        Returns:
            Plotly图表对象
        """
        if not NETWORKX_AVAILABLE:
            # 如果networkx不可用，返回热图
            return self._plot_correlation_heatmap(correlation_matrix)

        # 创建网络图
        G = nx.Graph()

        # 添加节点
        for factor in correlation_matrix.columns:
            G.add_node(factor)

        # 添加边（只保留相关性绝对值大于阈值的边）
        for i, factor1 in enumerate(correlation_matrix.columns):
            for j, factor2 in enumerate(correlation_matrix.columns):
                if i < j:  # 避免重复
                    corr = correlation_matrix.loc[factor1, factor2]
                    if abs(corr) >= threshold:
                        G.add_edge(factor1, factor2, weight=abs(corr))

        # 计算布局
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"spring_layout失败，使用circular_layout: {e}")
            pos = nx.circular_layout(G)

        # 创建边轨迹
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = edge[2].get("weight", 0)
            edge_text.append(f"{edge[0]}-{edge[1]}: {weight:.3f}")

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # 创建节点轨迹
        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            # 节点大小基于连接数
            node_size.append(10 + len(list(G.neighbors(node))) * 3)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="bottom center",
            marker=dict(
                size=node_size,
                color="#2E86DE",
                line=dict(width=2, color="#FFF"),
            ),
        )

        # 创建图表
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=f"因子相关性网络图（阈值 ≥ {threshold}）", font=dict(size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="节点大小表示连接数",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white",
                height=height,
                width=width,
            ),
        )

        return fig

    def _plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> go.Figure:
        """
        绘制相关性热图（当networkx不可用时的备选方案）

        Args:
            correlation_matrix: 相关性矩阵

        Returns:
            Plotly图表对象
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="相关性"),
            )
        )

        fig.update_layout(
            title="因子相关性热图",
            template="plotly_white",
            height=600,
        )

        return fig

    def plot_factor_radar(
        self,
        factors_data: Dict[str, Dict[str, float]],
        max_score: float = 1.0,
    ) -> go.Figure:
        """
        绘制因子表现雷达图

        Args:
            factors_data: 因子数据字典
                格式: {
                    "因子名": {
                        "ic": 0.05,
                        "ir": 1.2,
                        "return": 0.15,
                        ...
                    }
                }
            max_score: 各指标的最大值（用于归一化）

        Returns:
            Plotly图表对象
        """
        # 获取所有指标
        all_metrics = set()
        for factor_data in factors_data.values():
            all_metrics.update(factor_data.keys())

        all_metrics = sorted(list(all_metrics))

        # 为每个因子创建雷达图
        fig = go.Figure()

        colors = ["#2E86DE", "#EE5253", "#10AC84", "#F39C12", "#9B59B6"]

        for i, (factor_name, factor_data) in enumerate(factors_data.items()):
            # 归一化数据
            values = []
            for metric in all_metrics:
                value = factor_data.get(metric, 0)
                normalized = min(abs(value) / max_score, 1.0) if max_score > 0 else 0
                values.append(normalized)

            # 添加轨迹
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # 闭合雷达图
                    theta=all_metrics + [all_metrics[0]],
                    fill="toself",
                    name=factor_name,
                    line_color=colors[i % len(colors)],
                )
            )

        # 更新布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                ),
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1,
            ),
            title="因子表现雷达图",
            template="plotly_white",
            height=600,
        )

        return fig

    def plot_ic_distribution(
        self,
        ic_series: pd.Series,
        factor_name: str = "因子",
    ) -> go.Figure:
        """
        绘制IC分布直方图

        Args:
            ic_series: IC序列
            factor_name: 因子名称

        Returns:
            Plotly图表对象
        """
        # 移除NaN值
        ic_clean = ic_series.dropna()

        # 创建图表
        fig = go.Figure()

        # 添加直方图
        fig.add_trace(
            go.Histogram(
                x=ic_clean,
                nbinsx=30,
                name="IC分布",
                marker_color="#2E86DE",
                opacity=0.7,
            )
        )

        # 添加均值线
        mean_ic = ic_clean.mean()
        fig.add_vline(x=mean_ic, line_dash="dash", line_color="red", annotation_text=f"均值: {mean_ic:.4f}")

        # 添加零线
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="零线")

        # 更新布局
        fig.update_layout(
            title=f"{factor_name} - IC分布直方图",
            xaxis_title="IC值",
            yaxis_title="频数",
            template="plotly_white",
            height=500,
            bargap=0.1,
        )

        return fig

    def plot_ic_time_series(
        self,
        ic_series: pd.Series,
        factor_name: str = "因子",
        window: int = 20,
    ) -> go.Figure:
        """
        绘制IC时间序列图（含滚动均值）

        Args:
            ic_series: IC时间序列
            factor_name: 因子名称
            window: 滚动窗口大小

        Returns:
            Plotly图表对象
        """
        # 计算滚动均值
        rolling_mean = ic_series.rolling(window=window, min_periods=1).mean()

        # 创建图表
        fig = go.Figure()

        # 添加IC序列
        fig.add_trace(
            go.Scatter(
                x=ic_series.index,
                y=ic_series.values,
                mode="lines",
                name="IC值",
                line=dict(color="#2E86DE", width=1),
                opacity=0.6,
            )
        )

        # 添加滚动均值
        fig.add_trace(
            go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean.values,
                mode="lines",
                name=f"滚动均值({window}期)",
                line=dict(color="#EE5253", width=2),
            )
        )

        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="零线")

        # 更新布局
        fig.update_layout(
            title=f"{factor_name} - IC时间序列",
            xaxis_title="日期",
            yaxis_title="IC值",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig

    def plot_layered_returns(
        self,
        layered_returns: Dict[str, pd.Series],
        factor_name: str = "因子",
    ) -> go.Figure:
        """
        绘制分层收益对比图

        Args:
            layered_returns: 分层收益字典
                格式: {"Q1": returns1, "Q2": returns2, ...}
            factor_name: 因子名称

        Returns:
            Plotly图表对象
        """
        # 计算累计收益
        cumulative_returns = {}
        for layer, returns in layered_returns.items():
            cumulative_returns[layer] = (1 + returns).cumprod()

        # 创建图表
        fig = go.Figure()

        colors = ["#EE5253", "#F39C12", "#3498DB", "#2ECC71", "#9B59B6"]

        # 添加各层收益曲线
        for i, (layer, returns) in enumerate(cumulative_returns.items()):
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns.values,
                    mode="lines",
                    name=layer,
                    line=dict(color=colors[i % len(colors)], width=2),
                )
            )

        # 更新布局
        fig.update_layout(
            title=f"{factor_name} - 分层累计收益对比",
            xaxis_title="日期",
            yaxis_title="累计收益",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig

    def plot_turnover_analysis(
        self,
        turnover_series: pd.Series,
        factor_name: str = "因子",
    ) -> go.Figure:
        """
        绘制换手率分析图

        Args:
            turnover_series: 换手率时间序列
            factor_name: 因子名称

        Returns:
            Plotly图表对象
        """
        # 创建子图
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("换手率时间序列", "换手率分布"),
            vertical_spacing=0.15,
        )

        # 子图1: 时间序列
        fig.add_trace(
            go.Scatter(
                x=turnover_series.index,
                y=turnover_series.values,
                mode="lines",
                name="换手率",
                line=dict(color="#2E86DE", width=1.5),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # 子图2: 分布直方图
        fig.add_trace(
            go.Histogram(
                x=turnover_series.values,
                nbinsx=30,
                name="分布",
                marker_color="#10AC84",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 添加平均线
        avg_turnover = turnover_series.mean()
        fig.add_hline(
            y=avg_turnover,
            line_dash="dash",
            line_color="red",
            row=1,
            col=1,
            annotation_text=f"均值: {avg_turnover:.3f}",
        )

        # 更新布局
        fig.update_layout(
            title=f"{factor_name} - 换手率分析",
            template="plotly_white",
            height=700,
            showlegend=False,
        )

        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="换手率", row=1, col=1)
        fig.update_xaxes(title_text="换手率", row=2, col=1)
        fig.update_yaxes(title_text="频数", row=2, col=1)

        return fig

    def plot_multi_factor_comparison(
        self,
        factors_metrics: Dict[str, Dict[str, float]],
    ) -> go.Figure:
        """
        绘制多因子对比条形图

        Args:
            factors_metrics: 因子指标字典
                格式: {
                    "因子名": {
                        "ic_mean": 0.05,
                        "ir": 1.2,
                        "sharpe": 1.5,
                        ...
                    }
                }

        Returns:
            Plotly图表对象
        """
        # 准备数据
        factors = []
        metrics = []
        values = []

        for factor_name, metrics_dict in factors_metrics.items():
            for metric_name, metric_value in metrics_dict.items():
                factors.append(factor_name)
                metrics.append(metric_name.upper())
                values.append(metric_value)

        # 创建图表
        fig = go.Figure()

        for i, metric in enumerate(set(metrics)):
            metric_data = [(f, m, v) for f, m, v in zip(factors, metrics, values) if m == metric]

            metric_factors = [x[0] for x in metric_data]
            metric_values = [x[2] for x in metric_data]

            fig.add_trace(
                go.Bar(
                    x=metric_factors,
                    y=metric_values,
                    name=metric,
                )
            )

        # 更新布局
        fig.update_layout(
            title="多因子指标对比",
            xaxis_title="因子",
            yaxis_title="指标值",
            barmode="group",
            template="plotly_white",
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig


# 全局可视化服务实例
visualization_service = VisualizationService()
