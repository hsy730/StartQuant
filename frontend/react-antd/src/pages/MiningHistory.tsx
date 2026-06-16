import { useState, useEffect, useRef } from "react";
import {
  Card,
  Table,
  Tag,
  Button,
  Space,
  message,
  Drawer,
  Descriptions,
  Progress,
  Row,
  Col,
  Popconfirm,
} from "antd";
import {
  HistoryOutlined,
  DeleteOutlined,
  EyeOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  ExperimentOutlined,
  StopOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import * as echarts from "echarts";
import { api } from "@/services/api";
import dayjs from "dayjs";

interface MiningHistoryItem {
  id: number;
  task_id: string;
  status: string;
  algorithm: string;
  stock_codes: string[];
  start_date: string;
  end_date: string;
  progress: number;
  best_fitness: number;
  factor_count: number;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

interface MiningHistoryDetail {
  id: number;
  task_id: string;
  status: string;
  algorithm: string;
  stock_codes: string[];
  base_factors: string[];
  start_date: string;
  end_date: string;
  freq: string;
  progress: number;
  current_generation: number;
  total_generations: number;
  best_fitness: number;
  avg_fitness: number;
  fitness_history: { best: number[]; average: number[] } | null;
  result: any;
  process_info: Record<string, any> | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

const algoLabels: Record<string, string> = {
  genetic: "遗传规划",
  pysr: "PySR符号回归",
  tree_prescreen: "树模型预筛选",
  gflownet: "GFlowNet增强GP",
  deep_implicit: "深度隐式因子",
};

const algoColors: Record<string, string> = {
  genetic: "blue",
  pysr: "purple",
  tree_prescreen: "green",
  gflownet: "orange",
  deep_implicit: "magenta",
};

const statusConfig: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  pending: { color: "default", icon: <ClockCircleOutlined />, label: "等待中" },
  running: { color: "processing", icon: <SyncOutlined spin />, label: "挖掘中" },
  completed: { color: "success", icon: <CheckCircleOutlined />, label: "已完成" },
  failed: { color: "error", icon: <CloseCircleOutlined />, label: "失败" },
  cancelled: { color: "warning", icon: <StopOutlined />, label: "已取消" },
  aborted: { color: "error", icon: <WarningOutlined />, label: "已中止" },
};

// 挖掘过程信息展示组件（适配不同算法）
const ProcessInfoSection: React.FC<{ info: Record<string, any> }> = ({ info }) => {
  const algorithm = info.algorithm || "genetic";
  const items: { label: string; value: string }[] = [];

  // 通用
  items.push({ label: "发现因子数", value: String(info.factors_found ?? "-") });
  if (info.cancelled) items.push({ label: "状态", value: "已取消" });

  // 算法特定
  if (algorithm === "genetic") {
    if (info.population_size) items.push({ label: "种群大小", value: String(info.population_size) });
    if (info.n_generations) items.push({ label: "总代数", value: String(info.n_generations) });
    if (info.actual_generations) items.push({ label: "实际代数", value: String(info.actual_generations) });
    if (info.elite_size) items.push({ label: "精英数量", value: String(info.elite_size) });
    if (info.cx_prob != null) items.push({ label: "交叉概率", value: String(info.cx_prob) });
    if (info.mut_prob != null) items.push({ label: "变异概率", value: String(info.mut_prob) });
    if (info.fitness_objective) items.push({ label: "适应度目标", value: String(info.fitness_objective) });
    if (info.use_nsga2 != null) items.push({ label: "NSGA-II", value: info.use_nsga2 ? "启用" : "禁用" });
    if (info.use_extended_primitives != null) items.push({ label: "扩展原语", value: info.use_extended_primitives ? "启用" : "禁用" });
    if (info.cv_folds > 0) items.push({ label: "交叉验证", value: `${info.cv_folds}折` });
    if (info.parsimony_coeff != null) items.push({ label: "简约性系数", value: String(info.parsimony_coeff) });
  } else if (algorithm === "pysr") {
    if (info.niterations) items.push({ label: "迭代次数", value: String(info.niterations) });
    if (info.populations) items.push({ label: "种群数", value: String(info.populations) });
    if (info.population_size) items.push({ label: "种群大小", value: String(info.population_size) });
    if (info.maxsize) items.push({ label: "最大复杂度", value: String(info.maxsize) });
    if (info.maxdepth) items.push({ label: "最大深度", value: String(info.maxdepth) });
    if (info.parsimony != null) items.push({ label: "简约性", value: String(info.parsimony) });
    if (info.procs) items.push({ label: "并行进程", value: String(info.procs) });
    if (info.equations_found != null) items.push({ label: "发现方程数", value: String(info.equations_found) });
  } else if (algorithm === "tree_prescreen") {
    if (info.tree_model_type) items.push({ label: "树模型", value: String(info.tree_model_type) });
    if (info.top_k) items.push({ label: "Top-K", value: String(info.top_k) });
    if (info.importance_threshold != null) items.push({ label: "重要性阈值", value: String(info.importance_threshold) });
    if (info.tree_n_estimators) items.push({ label: "树数量", value: String(info.tree_n_estimators) });
    if (info.tree_max_depth) items.push({ label: "树深度", value: String(info.tree_max_depth) });
    if (info.downstream_algorithm) items.push({ label: "下游算法", value: String(info.downstream_algorithm) });
    if (info.n_selected != null) items.push({ label: "筛选特征数", value: String(info.n_selected) });
  } else if (algorithm === "gflownet") {
    if (info.n_trajectories) items.push({ label: "轨迹数", value: String(info.n_trajectories) });
    if (info.n_iterations) items.push({ label: "总迭代", value: String(info.n_iterations) });
    if (info.actual_iterations) items.push({ label: "实际迭代", value: String(info.actual_iterations) });
    if (info.hidden_dim) items.push({ label: "隐藏维度", value: String(info.hidden_dim) });
    if (info.learning_rate) items.push({ label: "学习率", value: String(info.learning_rate) });
    if (info.max_expression_depth) items.push({ label: "最大深度", value: String(info.max_expression_depth) });
    if (info.temperature) items.push({ label: "温度", value: String(info.temperature) });
    if (info.reward_scale) items.push({ label: "奖励缩放", value: String(info.reward_scale) });
  } else if (algorithm === "deep_implicit") {
    if (info.d_model) items.push({ label: "模型维度", value: String(info.d_model) });
    if (info.n_heads) items.push({ label: "注意力头数", value: String(info.n_heads) });
    if (info.n_layers) items.push({ label: "层数", value: String(info.n_layers) });
    if (info.n_latent_factors) items.push({ label: "隐因子数", value: String(info.n_latent_factors) });
    if (info.seq_length) items.push({ label: "序列长度", value: String(info.seq_length) });
    if (info.n_epochs) items.push({ label: "总Epoch", value: String(info.n_epochs) });
    if (info.actual_epochs) items.push({ label: "实际Epoch", value: String(info.actual_epochs) });
    if (info.early_stopping_patience) items.push({ label: "早停耐心", value: String(info.early_stopping_patience) });
    if (info.learning_rate) items.push({ label: "学习率", value: String(info.learning_rate) });
    if (info.batch_size) items.push({ label: "批次大小", value: String(info.batch_size) });
    if (info.dropout != null) items.push({ label: "Dropout", value: String(info.dropout) });
  }

  return (
    <Row gutter={[16, 8]}>
      {items.map((item, idx) => (
        <Col span={8} key={idx}>
          <span style={{ color: "#64748b" }}>{item.label}: </span>
          <b>{item.value}</b>
        </Col>
      ))}
    </Row>
  );
};

const MiningHistory: React.FC = () => {
  const [history, setHistory] = useState<MiningHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [detailVisible, setDetailVisible] = useState(false);
  const [detail, setDetail] = useState<MiningHistoryDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const savedFactorIds = useRef<Set<number>>(new Set()); // 已保存到因子池的 generated_factor_id
  const savingIndex = useRef<number | null>(null); // 正在保存的因子索引
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);

  const loadHistory = async (p: number = page, ps: number = pageSize) => {
    setLoading(true);
    try {
      const offset = (p - 1) * ps;
      const response = (await api.getMiningHistory({
        limit: ps,
        offset,
      })) as any;
      if (response.success) {
        setHistory(response.data.items);
        setTotal(response.data.total);
      }
    } catch (error) {
      console.error("加载挖掘历史失败:", error);
      message.error("加载挖掘历史失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
    };
  }, []);

  const viewDetail = async (taskId: string) => {
    setDetailVisible(true);
    setModalOpen(false);
    setDetailLoading(true);
    setDetail(null);
    savedFactorIds.current.clear(); // 重置已保存状态
    try {
      const response = (await api.getMiningHistoryDetail(taskId)) as any;
      if (response.success) {
        setDetail(response.data);
      }
    } catch (error) {
      console.error("获取挖掘详情失败:", error);
      message.error("获取挖掘详情失败");
    } finally {
      setDetailLoading(false);
    }
  };

  // 当 Modal 动画完成且 detail 数据就绪时，渲染进化曲线
  useEffect(() => {
    if (!modalOpen || !detail?.fitness_history || !detail.fitness_history.best?.length) return;

    // 使用 requestAnimationFrame 确保 DOM 已完成布局
    const timer = requestAnimationFrame(() => {
      renderChart(detail.fitness_history!);
    });
    return () => cancelAnimationFrame(timer);
  }, [modalOpen, detail]);

  const deleteRecord = async (taskId: string) => {
    try {
      const response = (await api.deleteMiningHistory(taskId)) as any;
      if (response.success) {
        message.success("删除成功");
        loadHistory();
      }
    } catch (error) {
      console.error("删除失败:", error);
      message.error("删除失败");
    }
  };

  // 将发现的因子保存到因子池
  const saveFactorToPool = async (factor: any, index: number, retryCount = 0) => {
    // 验证门控
    if (factor.overall_passed === false) {
      message.warning(
        `该因子未通过验证（得分: ${factor.validation_score?.toFixed(1)}），不能保存到因子池`
      );
      return;
    }

    // 已保存过则跳过
    if (factor.generated_factor_id && savedFactorIds.current.has(factor.generated_factor_id)) {
      message.info("该因子已在因子池中");
      return;
    }

    savingIndex.current = index;

    try {
      const today = new Date();
      const dateStr = [
        today.getFullYear(),
        String(today.getMonth() + 1).padStart(2, "0"),
        String(today.getDate()).padStart(2, "0"),
        String(today.getHours()).padStart(2, "0"),
        String(today.getMinutes()).padStart(2, "0"),
      ].join("");

      const factorName =
        retryCount === 0
          ? `Mined_${index + 1}_${dateStr}`
          : `Mined_${index + 1}_${dateStr}_${retryCount}`;

      // 表达式包装为完整函数
      const processedExpr = factor.expression
        .replace(/\bopen\b/g, "df['open']")
        .replace(/\bclose\b/g, "df['close']")
        .replace(/\bhigh\b/g, "df['high']")
        .replace(/\blow\b/g, "df['low']")
        .replace(/\bvolume\b/g, "df['volume']");

      const sourceLabel =
        factor.source === "pysr"
          ? "PySR符号回归"
          : factor.source === "genetic"
            ? "遗传规划"
            : "因子挖掘";

      const code = `def calculate_factor(df):
    """
    ${sourceLabel}挖掘因子
    表达式: ${factor.expression}
    IC: ${factor.ic?.toFixed(4)}
    IR: ${factor.ir?.toFixed(4)}
    """
    import pandas as pd
    import numpy as np

    try:
        result = ${processedExpr}
        return result
    except Exception as e:
        return pd.Series(0, index=df.index)
`;

      const response = (await api.createFactor({
        name: factorName,
        code,
        category: sourceLabel,
        description: `${sourceLabel} | 表达式: ${factor.expression} | IC: ${factor.ic?.toFixed(4)} | IR: ${factor.ir?.toFixed(4)}`,
        formula_type: "function",
        generated_factor_id: factor.generated_factor_id || null,
      })) as any;

      if (response.success) {
        message.success(`因子 "${factorName}" 已添加到因子池`);
        if (factor.generated_factor_id) {
          savedFactorIds.current.add(factor.generated_factor_id);
        }
        // 强制刷新卡片以更新按钮状态
        setDetail((prev) => prev ? { ...prev } : prev);
      } else {
        message.error("保存失败: " + (response.data?.detail || response.message || "未知错误"));
      }
    } catch (error: any) {
      const errorMsg =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        "未知错误";

      if (errorMsg.includes("未通过验证")) {
        message.error("保存失败: " + errorMsg);
      } else if (errorMsg.includes("已存在") && retryCount < 5) {
        return saveFactorToPool(factor, index, retryCount + 1);
      } else {
        message.error("保存失败: " + errorMsg);
      }
    } finally {
      savingIndex.current = null;
    }
  };

  const renderChart = (fitnessHistory: { best: number[]; average: number[] }) => {
    if (!chartRef.current) {
      console.warn("[进化曲线] chartRef 尚未挂载，跳过渲染");
      return;
    }

    let chart = chartInstanceRef.current;
    if (!chart) {
      chart = echarts.init(chartRef.current);
      chartInstanceRef.current = chart;
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1);

    chart.setOption(
      {
        title: { text: "进化曲线", left: "center", textStyle: { fontSize: 14 } },
        tooltip: { trigger: "axis" },
        legend: { data: ["最优适应度", "平均适应度"], bottom: 0 },
        grid: { left: "3%", right: "4%", bottom: "12%", containLabel: true },
        xAxis: { type: "category", name: "代数", data: generations },
        yAxis: { type: "value", name: "适应度", scale: true },
        series: [
          {
            name: "最优适应度",
            type: "line",
            data: fitnessHistory.best,
            smooth: true,
            itemStyle: { color: "#3b82f6" },
          },
          {
            name: "平均适应度",
            type: "line",
            data: fitnessHistory.average,
            smooth: true,
            itemStyle: { color: "#22c55e" },
          },
        ],
      },
      true
    );
  };

  const columns = [
    {
      title: "时间",
      dataIndex: "created_at",
      key: "created_at",
      width: 170,
      render: (val: string) => (val ? dayjs(val).format("YYYY-MM-DD HH:mm:ss") : "-"),
    },
    {
      title: "算法",
      dataIndex: "algorithm",
      key: "algorithm",
      width: 130,
      render: (val: string) => (
        <Tag color={algoColors[val] || "default"}>
          {algoLabels[val] || val}
        </Tag>
      ),
    },
    {
      title: "股票",
      dataIndex: "stock_codes",
      key: "stock_codes",
      width: 160,
      render: (val: string[]) => {
        if (!val || val.length === 0) return "-";
        const display = val.length > 2 ? `${val.slice(0, 2).join(", ")}...` : val.join(", ");
        return <span style={{ fontSize: 12 }}>{display}</span>;
      },
    },
    {
      title: "日期范围",
      key: "date_range",
      width: 180,
      render: (_: any, record: MiningHistoryItem) =>
        record.start_date && record.end_date
          ? `${record.start_date} ~ ${record.end_date}`
          : "-",
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (val: string) => {
        const cfg = statusConfig[val] || { color: "default", icon: null, label: val };
        return (
          <Tag color={cfg.color} icon={cfg.icon}>
            {cfg.label}
          </Tag>
        );
      },
    },
    {
      title: "进度",
      dataIndex: "progress",
      key: "progress",
      width: 120,
      render: (val: number, record: MiningHistoryItem) =>
        record.status === "completed" ? (
          <Progress percent={100} size="small" status="success" />
        ) : record.status === "failed" || record.status === "aborted" ? (
          <Progress percent={val} size="small" status="exception" />
        ) : (
          <Progress percent={val} size="small" status="active" />
        ),
    },
    {
      title: "最优适应度",
      dataIndex: "best_fitness",
      key: "best_fitness",
      width: 110,
      render: (val: number) => (val != null ? val.toFixed(4) : "-"),
    },
    {
      title: "因子数",
      dataIndex: "factor_count",
      key: "factor_count",
      width: 80,
      render: (val: number) => val || 0,
    },
    {
      title: "操作",
      key: "actions",
      width: 120,
      render: (_: any, record: MiningHistoryItem) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => viewDetail(record.task_id)}
          >
            详情
          </Button>
          <Popconfirm
            title="确定删除此记录？"
            onConfirm={() => deleteRecord(record.task_id)}
            okText="确定"
            cancelText="取消"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: 16,
        position: "relative",
      }}
    >
      <div
        style={{
          position: "fixed",
          inset: 0,
          background:
            "radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.08) 0%, transparent 50%), radial-gradient(circle at 80% 70%, rgba(6, 182, 212, 0.06) 0%, transparent 50%)",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      <div style={{ position: "relative", zIndex: 10 }}>
        {/* 页面头部 */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 24,
            padding: "20px 24px",
            background: "rgba(255, 255, 255, 0.85)",
            border: "1px solid rgba(59, 130, 246, 0.12)",
            borderRadius: 16,
            backdropFilter: "blur(20px)",
            boxShadow: "0 2px 12px rgba(59, 130, 246, 0.06)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <HistoryOutlined
              style={{ fontSize: 28, color: "#3b82f6" }}
            />
            <div>
              <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
                挖掘历史记录
              </h1>
              <p style={{ margin: 0, color: "#64748b", fontSize: 13 }}>
                查看因子挖掘任务的历史记录和详细结果
              </p>
            </div>
          </div>
          <Button onClick={() => loadHistory()}>刷新</Button>
        </div>

        {/* 历史列表 */}
        <Card
          style={{
            borderRadius: 12,
            border: "1px solid rgba(59, 130, 246, 0.12)",
            boxShadow: "0 2px 12px rgba(59, 130, 246, 0.06)",
          }}
        >
          <Table
            columns={columns}
            dataSource={history}
            rowKey="task_id"
            loading={loading}
            pagination={{
              current: page,
              pageSize: pageSize,
              total: total,
              showSizeChanger: true,
              showTotal: (t) => `共 ${t} 条记录`,
              onChange: (p, ps) => {
                setPage(p);
                setPageSize(ps);
                loadHistory(p, ps);
              },
            }}
            size="middle"
          />
        </Card>
      </div>

      {/* 详情侧滑框 */}
      <Drawer
        title={
          <Space>
            <ExperimentOutlined />
            <span>挖掘任务详情</span>
          </Space>
        }
        open={detailVisible}
        afterOpenChange={(open) => setModalOpen(open)}
        onClose={() => {
          setDetailVisible(false);
          setModalOpen(false);
          setDetail(null);
          // 销毁 ECharts 实例，防止内存泄漏
          if (chartInstanceRef.current) {
            chartInstanceRef.current.dispose();
            chartInstanceRef.current = null;
          }
        }}
        width={960}
        destroyOnClose
      >
        {detailLoading ? (
          <div style={{ textAlign: "center", padding: "40px 0" }}>
            <SyncOutlined spin style={{ fontSize: 24 }} />
            <p style={{ marginTop: 12, color: "#64748b" }}>加载中...</p>
          </div>
        ) : detail ? (
          <div>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="任务ID">
                <span style={{ fontFamily: "monospace", fontSize: 12 }}>
                  {detail.task_id}
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {(() => {
                  const cfg = statusConfig[detail.status] || {
                    color: "default",
                    icon: null,
                    label: detail.status,
                  };
                  return (
                    <Tag color={cfg.color} icon={cfg.icon}>
                      {cfg.label}
                    </Tag>
                  );
                })()}
              </Descriptions.Item>
              <Descriptions.Item label="算法">
                <Tag color={algoColors[detail.algorithm] || "default"}>
                  {algoLabels[detail.algorithm] || detail.algorithm}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="数据频率">
                {detail.freq === "D" ? "日线" : detail.freq}
              </Descriptions.Item>
              <Descriptions.Item label="日期范围" span={2}>
                {detail.start_date} ~ {detail.end_date}
              </Descriptions.Item>
              <Descriptions.Item label="股票代码" span={2}>
                {detail.stock_codes?.join(", ") || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="基础因子" span={2}>
                <div style={{ maxHeight: 80, overflow: "auto" }}>
                  {detail.base_factors?.map((f, i) => (
                    <Tag key={i} style={{ marginBottom: 4 }}>
                      {f}
                    </Tag>
                  )) || "-"}
                </div>
              </Descriptions.Item>
              <Descriptions.Item label="最优适应度">
                {detail.best_fitness != null ? detail.best_fitness.toFixed(4) : "-"}
              </Descriptions.Item>
              <Descriptions.Item label="平均适应度">
                {detail.avg_fitness != null ? detail.avg_fitness.toFixed(4) : "-"}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {detail.created_at
                  ? dayjs(detail.created_at).format("YYYY-MM-DD HH:mm:ss")
                  : "-"}
              </Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {detail.completed_at
                  ? dayjs(detail.completed_at).format("YYYY-MM-DD HH:mm:ss")
                  : "-"}
              </Descriptions.Item>
              {detail.error && (
                <Descriptions.Item label="错误信息" span={2}>
                  <span style={{ color: "#ef4444" }}>{detail.error}</span>
                </Descriptions.Item>
              )}
            </Descriptions>

            {/* 挖掘过程信息 */}
            {detail.process_info && (
              <Card
                size="small"
                title={
                  <Space>
                    <InfoCircleOutlined />
                    <span>挖掘过程详情</span>
                    <Tag color={algoColors[detail.algorithm] || "default"}>
                      {algoLabels[detail.algorithm] || detail.algorithm}
                    </Tag>
                  </Space>
                }
                style={{ marginTop: 16, borderRadius: 8 }}
              >
                <ProcessInfoSection info={detail.process_info} />
              </Card>
            )}

            {/* 进化曲线 */}
            {detail.fitness_history &&
              detail.fitness_history.best?.length > 0 && (
                <div style={{ marginTop: 24 }}>
                  <h4 style={{ marginBottom: 12 }}>进化曲线</h4>
                  <div
                    ref={chartRef}
                    style={{ height: 400, border: "1px solid #f0f0f0", borderRadius: 8 }}
                  />
                </div>
              )}

            {/* 挖掘结果因子列表 */}
            {detail.result?.factors && detail.result.factors.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <h4 style={{ marginBottom: 12 }}>
                  发现的因子 ({detail.result.factors.length})
                </h4>
                <div style={{ maxHeight: 500, overflow: "auto" }}>
                  {detail.result.factors.map(
                    (factor: any, index: number) => {
                      const isSaved =
                        factor.generated_factor_id != null &&
                        savedFactorIds.current.has(factor.generated_factor_id);
                      const isSaving = savingIndex.current === index;
                      const canSave =
                        factor.overall_passed !== false && !isSaved;

                      return (
                        <Card
                          key={index}
                          size="small"
                          style={{
                            marginBottom: 8,
                            border: isSaved
                              ? "1px solid #22c55e"
                              : "1px solid #f0f0f0",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                            }}
                          >
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <Space wrap>
                                <Tag color="blue">Top {index + 1}</Tag>
                                <Tag
                                  color={
                                    algoColors[factor.source] || "default"
                                  }
                                >
                                  {algoLabels[factor.source] ||
                                    factor.source}
                                </Tag>
                                {isSaved && (
                                  <Tag color="success">已入库</Tag>
                                )}
                              </Space>
                              <div
                                style={{
                                  fontFamily: "monospace",
                                  fontSize: 12,
                                  marginTop: 4,
                                  color: "#475569",
                                  wordBreak: "break-all",
                                }}
                              >
                                {factor.expression}
                              </div>
                            </div>
                            <div
                              style={{
                                textAlign: "right",
                                minWidth: 140,
                                marginLeft: 16,
                              }}
                            >
                              <div>
                                IC:{" "}
                                <span
                                  style={{
                                    color:
                                      factor.ic > 0 ? "#22c55e" : "#ef4444",
                                    fontWeight: 600,
                                  }}
                                >
                                  {factor.ic?.toFixed(4)}
                                </span>
                              </div>
                              <div>
                                IR:{" "}
                                <span
                                  style={{
                                    color:
                                      factor.ir > 0 ? "#22c55e" : "#ef4444",
                                    fontWeight: 600,
                                  }}
                                >
                                  {factor.ir?.toFixed(4)}
                                </span>
                              </div>
                              <div>
                                验证:{" "}
                                <Tag
                                  color={
                                    factor.overall_passed
                                      ? "success"
                                      : "error"
                                  }
                                  style={{ fontSize: 11 }}
                                >
                                  {factor.overall_passed
                                    ? "通过"
                                    : "未通过"}
                                </Tag>
                              </div>
                              <div style={{ marginTop: 6 }}>
                                {isSaved ? (
                                  <Button
                                    size="small"
                                    type="link"
                                    disabled
                                    icon={<CheckCircleOutlined />}
                                    style={{ color: "#22c55e", padding: 0 }}
                                  >
                                    已在因子池
                                  </Button>
                                ) : (
                                  <Button
                                    size="small"
                                    type="primary"
                                    icon={<SaveOutlined />}
                                    loading={isSaving}
                                    disabled={!canSave}
                                    onClick={() =>
                                      saveFactorToPool(factor, index)
                                    }
                                  >
                                    {isSaving
                                      ? "保存中..."
                                      : factor.overall_passed === false
                                        ? "未通过验证"
                                        : "添加到因子池"}
                                  </Button>
                                )}
                              </div>
                            </div>
                          </div>
                        </Card>
                      );
                    }
                  )}
                </div>
              </div>
            )}
          </div>
        ) : null}
      </Drawer>
    </div>
  );
};

export default MiningHistory;
