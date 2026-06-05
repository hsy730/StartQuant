import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  Card,
  Form,
  Input,
  DatePicker,
  Button,
  Select,
  InputNumber,
  Progress,
  Row,
  Col,
  message,
  Space,
  Divider,
  Tag,
  Spin,
  Alert,
} from "antd";
import {
  PlayCircleOutlined,
  SaveOutlined,
  BarChartOutlined,
  RocketOutlined,
  SyncOutlined,
  LoadingOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  AimOutlined,
  BulbOutlined,
  ClockCircleOutlined,
  StopOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import * as echarts from "echarts";
import { api } from "@/services/api";
import dayjs from "dayjs";
import "./FactorMining.css";

const { Option } = Select;
const { RangePicker } = DatePicker;

interface Factor {
  id: number;
  name: string;
  code: string;
  category: string;
  source: "preset" | "user";
  description?: string;
}

interface MinedFactor {
  name: string;
  expression: string;
  ic: number;
  ir: number;
  fitness: number;
  source?: string;
  overall_passed?: boolean;  // 是否通过验证
  validation_score?: number;  // 验证得分
  generated_factor_id?: number | null;  // generated_factors 表记录ID
}

interface MiningStatus {
  task_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  current_generation: number;
  total_generations: number;
  best_fitness: number;
  avg_fitness: number;
  fitness_history?: {
    best: number[];
    average: number[];
  };
  started_at?: string | null;
  error?: string;
}

interface MiningResult {
  factors: MinedFactor[];
  best_fitness: number;
  avg_fitness: number;
  generations: number;
  fitness_history?: {
    best: number[];
    average: number[];
  };
}

const FactorMining: React.FC = () => {

// 挖掘过程信息展示组件（适配不同算法）
const ProcessInfoDisplay: React.FC<{ info: Record<string, any> }> = ({ info }) => {
  const algorithm = info.algorithm || "genetic";

  // 通用信息行
  const renderCommonInfo = () => (
    <Row gutter={[16, 8]} style={{ marginBottom: 8 }}>
      <Col span={8}>
        <span style={{ color: "#64748b" }}>发现因子数: </span>
        <span style={{ fontWeight: 600 }}>{info.factors_found ?? "-"}</span>
      </Col>
      {info.cancelled && (
        <Col span={8}>
          <Tag color="warning">已取消</Tag>
        </Col>
      )}
    </Row>
  );

  // 遗传规划信息
  if (algorithm === "genetic") {
    return (
      <div>
        {renderCommonInfo()}
        <Row gutter={[16, 8]}>
          <Col span={8}><span style={{ color: "#64748b" }}>种群大小: </span><b>{info.population_size}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>总代数: </span><b>{info.n_generations}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>实际代数: </span><b>{info.actual_generations ?? info.n_generations}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>精英数量: </span><b>{info.elite_size}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>交叉概率: </span><b>{info.cx_prob}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>变异概率: </span><b>{info.mut_prob}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>适应度目标: </span><b>{info.fitness_objective}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>NSGA-II: </span><b>{info.use_nsga2 ? "启用" : "禁用"}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>扩展原语: </span><b>{info.use_extended_primitives ? "启用" : "禁用"}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>交叉验证: </span><b>{info.cv_folds > 0 ? `${info.cv_folds}折` : "禁用"}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>简约性系数: </span><b>{info.parsimony_coeff}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>多样性惩罚: </span><b>{info.diversity_penalty_coeff}</b></Col>
        </Row>
      </div>
    );
  }

  // PySR信息
  if (algorithm === "pysr") {
    return (
      <div>
        {renderCommonInfo()}
        <Row gutter={[16, 8]}>
          <Col span={8}><span style={{ color: "#64748b" }}>迭代次数: </span><b>{info.niterations}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>种群数: </span><b>{info.populations}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>种群大小: </span><b>{info.population_size}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>最大复杂度: </span><b>{info.maxsize}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>最大深度: </span><b>{info.maxdepth}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>简约性: </span><b>{info.parsimony}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>并行进程: </span><b>{info.procs}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>发现方程数: </span><b>{info.equations_found ?? "-"}</b></Col>
        </Row>
      </div>
    );
  }

  // 树模型预筛选信息
  if (algorithm === "tree_prescreen") {
    return (
      <div>
        {renderCommonInfo()}
        <Row gutter={[16, 8]}>
          <Col span={8}><span style={{ color: "#64748b" }}>树模型: </span><b>{info.tree_model_type}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>Top-K: </span><b>{info.top_k}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>重要性阈值: </span><b>{info.importance_threshold}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>树数量: </span><b>{info.tree_n_estimators}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>树深度: </span><b>{info.tree_max_depth}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>下游算法: </span><b>{info.downstream_algorithm}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>筛选特征数: </span><b>{info.n_selected ?? "-"}</b></Col>
        </Row>
        {info.feature_importance && (
          <div style={{ marginTop: 8 }}>
            <span style={{ color: "#64748b" }}>特征重要性: </span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
              {Object.entries(info.feature_importance)
                .sort(([, a]: any, [, b]: any) => b - a)
                .slice(0, 10)
                .map(([name, val]: any) => (
                  <Tag key={name} style={{ fontSize: 11 }}>{name}: {typeof val === "number" ? val.toFixed(4) : val}</Tag>
                ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // GFlowNet信息
  if (algorithm === "gflownet") {
    return (
      <div>
        {renderCommonInfo()}
        <Row gutter={[16, 8]}>
          <Col span={8}><span style={{ color: "#64748b" }}>轨迹数: </span><b>{info.n_trajectories}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>总迭代: </span><b>{info.n_iterations}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>实际迭代: </span><b>{info.actual_iterations ?? info.n_iterations}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>隐藏维度: </span><b>{info.hidden_dim}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>学习率: </span><b>{info.learning_rate}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>最大深度: </span><b>{info.max_expression_depth}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>温度: </span><b>{info.temperature}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>奖励缩放: </span><b>{info.reward_scale}</b></Col>
        </Row>
      </div>
    );
  }

  // 深度隐式因子信息
  if (algorithm === "deep_implicit") {
    return (
      <div>
        {renderCommonInfo()}
        <Row gutter={[16, 8]}>
          <Col span={8}><span style={{ color: "#64748b" }}>模型维度: </span><b>{info.d_model}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>注意力头数: </span><b>{info.n_heads}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>层数: </span><b>{info.n_layers}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>隐因子数: </span><b>{info.n_latent_factors}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>序列长度: </span><b>{info.seq_length}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>总Epoch: </span><b>{info.n_epochs}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>实际Epoch: </span><b>{info.actual_epochs ?? info.n_epochs}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>早停耐心: </span><b>{info.early_stopping_patience}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>学习率: </span><b>{info.learning_rate}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>批次大小: </span><b>{info.batch_size}</b></Col>
          <Col span={8}><span style={{ color: "#64748b" }}>Dropout: </span><b>{info.dropout}</b></Col>
        </Row>
        {info.model_info && (
          <div style={{ marginTop: 8 }}>
            <span style={{ color: "#64748b" }}>模型信息: </span>
            <span style={{ fontSize: 12 }}>{JSON.stringify(info.model_info)}</span>
          </div>
        )}
      </div>
    );
  }

  // 通用回退
  return (
    <div>
      {renderCommonInfo()}
      <pre style={{ fontSize: 11, maxHeight: 200, overflow: "auto" }}>
        {JSON.stringify(info, null, 2)}
      </pre>
    </div>
  );
};
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const evolutionChartRef = useRef<HTMLDivElement>(null);
  const resultChartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);
  const resultChartInstanceRef = useRef<echarts.ECharts | null>(null);

  const [factors, setFactors] = useState<Factor[]>([]);
  const [loading, setLoading] = useState(false);
  const [mining, setMining] = useState(false);
  const [currentStockCode, setCurrentStockCode] = useState<string>("");
  const [stockPools, setStockPools] = useState<Array<{ id: string; name: string; description: string }>>([]);
  const [loadingPoolStocks, setLoadingPoolStocks] = useState(false);

  const [miningStatus, setMiningStatus] = useState<MiningStatus | null>(null);
  const [miningResult, setMiningResult] = useState<MiningResult | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0); // 挖掘已用时间（秒）
  const miningStartTimeRef = useRef<number | null>(null); // 挖掘开始时间戳
  const elapsedTimeIntervalRef = useRef<NodeJS.Timeout | null>(null); // 计时器ID
  const [savedFactorNames, setSavedFactorNames] = useState<Set<string>>(
    new Set(),
  ); // 已保存的因子名称
  const [savedFactorIds, setSavedFactorIds] = useState<Map<number, number>>(
    new Map(),
  ); // 已保存的因子ID映射 (index -> factor_id)
  const [renameModalVisible, setRenameModalVisible] = useState(false);
  const [renameTarget, setRenameTarget] = useState<{
    factor: MinedFactor;
    index: number;
  } | null>(null);
  const [customFactorName, setCustomFactorName] = useState("");

  // 加载因子列表
  const loadFactors = async () => {
    try {
      const response = (await api.getFactors()) as any;
      if (response.success) {
        setFactors(response.data);
      }
    } catch (error) {
      console.error("加载因子列表失败:", error);
    }
  };

  // 加载预设股票池列表
  const loadStockPools = async () => {
    try {
      const res = (await api.getStockPools()) as any;
      if (res.success && res.data) {
        setStockPools(res.data);
      }
    } catch (error) {
      console.error("加载股票池列表失败:", error);
    }
  };

  useEffect(() => {
    loadFactors();
    loadStockPools();

    // 检查是否有正在进行的挖掘任务（页面刷新后恢复状态）
    recoverActiveMiningTask();

    // 设置默认日期范围
    const endDate = dayjs();
    const startDate = dayjs().subtract(1, "year");
    form.setFieldsValue({
      dateRange: [startDate, endDate],
      freq: 'D',
      population_size: 50,
      n_generations: 10,
      mutation_rate: 0.2,
      crossover_rate: 0.7,
      elite_size: 5,
      fitness_objective: "ic_mean",
      ic_threshold: 0.03,
      // Phase 2-7: quality-boosting defaults
      parsimony_coeff: 0.001,
      diversity_penalty_coeff: 0.1,
      cv_folds: 0,
      use_extended_primitives: true,
      max_tree_depth: 17,
      use_nsga2: true,
      algorithm: "genetic",
      pysr_niterations: 40,
      pysr_populations: 30,
      pysr_maxsize: 30,
      pysr_maxdepth: 5,
      pysr_parsimony: 0.0032,
      pysr_procs: 8,
      // GFlowNet默认参数
      gflownet_n_trajectories: 200,
      gflownet_n_iterations: 50,
      gflownet_hidden_dim: 128,
      gflownet_learning_rate: 1e-3,
      gflownet_max_expression_depth: 5,
      gflownet_temperature: 1.0,
      gflownet_reward_scale: 10.0,
      gflownet_buffer_size: 1000,
      // 深度隐式因子默认参数
      deep_d_model: 64,
      deep_n_heads: 4,
      deep_n_layers: 3,
      deep_d_ff: 256,
      deep_n_latent_factors: 5,
      deep_dropout: 0.1,
      deep_seq_length: 20,
      deep_learning_rate: 1e-4,
      deep_n_epochs: 50,
      deep_batch_size: 32,
      deep_weight_decay: 1e-5,
      deep_early_stopping_patience: 5,
    });

    return () => {
      // 清理定时器
      if (window.miningInterval) {
        clearInterval(window.miningInterval);
      }
      // 清理计时器
      if (elapsedTimeIntervalRef.current) {
        clearInterval(elapsedTimeIntervalRef.current);
        elapsedTimeIntervalRef.current = null;
      }
      // 清理图表
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
      if (resultChartInstanceRef.current) {
        resultChartInstanceRef.current.dispose();
        resultChartInstanceRef.current = null;
      }
    };
  }, []);

  // 恢复活跃的挖掘任务（页面刷新后）
  const recoverActiveMiningTask = async () => {
    try {
      const response = (await api.getActiveMiningTasks()) as any;
      if (response.success && response.data && response.data.length > 0) {
        // 取最新的活跃任务
        const activeTask = response.data[0];
        const taskId = activeTask.task_id;

        console.log("Recovering active mining task:", taskId, activeTask.status);

        // 恢复挖掘状态
        setMining(true);
        setMiningStatus({
          task_id: taskId,
          status: activeTask.status,
          current_generation: activeTask.current_generation || 0,
          total_generations: activeTask.total_generations || 10,
          best_fitness: activeTask.best_fitness || 0,
          avg_fitness: activeTask.avg_fitness || 0,
          fitness_history: activeTask.fitness_history || { best: [], average: [] },
          started_at: activeTask.started_at || null,
        });

        // 恢复计时器（使用数据库的started_at计算已用时间）
        if (activeTask.started_at) {
          miningStartTimeRef.current = new Date(activeTask.started_at).getTime();
        } else {
          miningStartTimeRef.current = Date.now();
        }
        elapsedTimeIntervalRef.current = setInterval(() => {
          if (miningStartTimeRef.current) {
            const elapsed = Math.floor(
              (Date.now() - miningStartTimeRef.current) / 1000,
            );
            setElapsedTime(elapsed);
          }
        }, 1000);

        // 恢复轮询
        window.miningInterval = setInterval(() => {
          checkMiningProgress(taskId);
        }, 2000);

        message.info("检测到正在进行的挖掘任务，已恢复进度");
      }
    } catch (error) {
      console.debug("检查活跃挖掘任务失败（可能没有进行中的任务）:", error);
    }
  };

  // 开始挖掘
  const startMining = async (values: any) => {
    const selectedFactors = values.base_factors || [];

    // 根据股票池模式获取股票代码列表
    let stockCodesList: string[] = [];
    const stockMode = values.stock_mode || 'preset';

    if (stockMode === 'preset' && values.stock_pool_id) {
      // 预设股票池模式 - 从后端获取成分股
      setLoadingPoolStocks(true);
      try {
        const res = (await api.getStockPoolStocks(values.stock_pool_id)) as any;
        if (res.success && res.data) {
          stockCodesList = res.data.map((s: any) => s.code);
        }
      } catch (error) {
        message.error("获取股票池成分股失败");
        setLoadingPoolStocks(false);
        return;
      }
      setLoadingPoolStocks(false);
      setCurrentStockCode(values.stock_pool_id);
    } else {
      // 自定义股票代码模式
      const rawInput = values.stock_codes_input || '000001';
      stockCodesList = rawInput
        .split(/[,\n，、\s]+/)
        .map((s: string) => s.trim())
        .filter((s: string) => s.length > 0)
        .map((code: string) => {
          const cleanCode = code.replace(/\.(SH|SZ)$/i, '');
          if (cleanCode.startsWith('6')) return cleanCode + '.SH';
          if (cleanCode.startsWith('0') || cleanCode.startsWith('3')) return cleanCode + '.SZ';
          return cleanCode;
        })
        .filter((s: string) => s.length > 0);
      setCurrentStockCode(rawInput.replace(/[,\n，、\s]+/g, '_').substring(0, 20));
    }

    if (stockCodesList.length === 0) {
      message.warning("请至少选择一只股票");
      return;
    }

    const [startDate, endDate] = values.dateRange;
    const requestData = {
      stock_code: stockCodesList.length === 1 ? stockCodesList[0] : stockCodesList[0],
      stock_codes: stockCodesList,
      base_factors: selectedFactors,
      start_date: startDate.format("YYYY-MM-DD"),
      end_date: endDate.format("YYYY-MM-DD"),
      population_size: values.population_size,
      n_generations: values.n_generations,
      cx_prob: values.crossover_rate,
      mut_prob: values.mutation_rate,
      elite_size: values.elite_size,
      fitness_objective: values.fitness_objective,
      ic_threshold: values.ic_threshold,
      parsimony_coeff: values.parsimony_coeff ?? 0.001,
      diversity_penalty_coeff: values.diversity_penalty_coeff ?? 0.1,
      cv_folds: values.cv_folds ?? 0,
      use_extended_primitives: values.use_extended_primitives ?? true,
      max_tree_depth: values.max_tree_depth ?? 17,
      use_nsga2: values.use_nsga2 ?? true,
      algorithm: values.algorithm ?? "genetic",
      // PySR参数
      pysr_niterations: values.pysr_niterations ?? 40,
      pysr_populations: values.pysr_populations ?? 30,
      pysr_maxsize: values.pysr_maxsize ?? 30,
      pysr_maxdepth: values.pysr_maxdepth ?? 5,
      pysr_parsimony: values.pysr_parsimony ?? 0.0032,
      pysr_procs: values.pysr_procs ?? 8,
      // GFlowNet参数
      gflownet_n_trajectories: values.gflownet_n_trajectories ?? 200,
      gflownet_n_iterations: values.gflownet_n_iterations ?? 50,
      gflownet_hidden_dim: values.gflownet_hidden_dim ?? 128,
      gflownet_learning_rate: values.gflownet_learning_rate ?? 1e-3,
      gflownet_max_expression_depth: values.gflownet_max_expression_depth ?? 5,
      gflownet_temperature: values.gflownet_temperature ?? 1.0,
      gflownet_reward_scale: values.gflownet_reward_scale ?? 10.0,
      gflownet_buffer_size: values.gflownet_buffer_size ?? 1000,
      // 深度隐式因子参数
      deep_d_model: values.deep_d_model ?? 64,
      deep_n_heads: values.deep_n_heads ?? 4,
      deep_n_layers: values.deep_n_layers ?? 3,
      deep_d_ff: values.deep_d_ff ?? 256,
      deep_n_latent_factors: values.deep_n_latent_factors ?? 5,
      deep_dropout: values.deep_dropout ?? 0.1,
      deep_seq_length: values.deep_seq_length ?? 20,
      deep_learning_rate: values.deep_learning_rate ?? 1e-4,
      deep_n_epochs: values.deep_n_epochs ?? 50,
      deep_batch_size: values.deep_batch_size ?? 32,
      deep_weight_decay: values.deep_weight_decay ?? 1e-5,
      deep_early_stopping_patience: values.deep_early_stopping_patience ?? 5,
      freq: values.freq || 'D',
      period: values.freq && values.freq !== 'D' ? values.freq.replace('min', '') : undefined,
    };

    try {
      setLoading(true);
      setMining(true);
      setMiningResult(null);
      setElapsedTime(0); // 重置计时器
      setSavedFactorNames(new Set()); // 新挖掘时清除记录

      const response = (await api.startGeneticMining(requestData)) as any;

      if (response.success) {
        const newTaskId = response.data.task_id;

        // 记录挖掘开始时间
        miningStartTimeRef.current = Date.now();

        // 启动计时器，每秒更新一次已用时间
        elapsedTimeIntervalRef.current = setInterval(() => {
          if (miningStartTimeRef.current) {
            const elapsed = Math.floor(
              (Date.now() - miningStartTimeRef.current) / 1000,
            );
            setElapsedTime(elapsed);
          }
        }, 1000);

        // 轮询获取进度
        window.miningInterval = setInterval(() => {
          checkMiningProgress(newTaskId);
        }, 2000);

        message.success("挖掘任务已启动");
      }
    } catch (error) {
      console.error("启动挖掘失败:", error);
      message.error("启动挖掘失败");
      setMining(false);
    } finally {
      setLoading(false);
    }
  };

  // 检查挖掘进度
  const checkMiningProgress = async (currentTaskId: string) => {
    try {
      const response = (await api.getMiningStatus(currentTaskId)) as any;

      if (response.success) {
        const statusData = response.data as MiningStatus;
        setMiningStatus(statusData);

        console.log(
          "Mining status:",
          statusData.status,
          "Generation:",
          statusData.current_generation,
          "/",
          statusData.total_generations,
        );

        // 更新进化曲线 - 使用完整的历史数据
        if (
          statusData.fitness_history &&
          statusData.fitness_history.best.length > 0
        ) {
          console.log(
            "Updating evolution chart with history:",
            statusData.fitness_history,
          );
          updateEvolutionChart(statusData.fitness_history);
        } else if (statusData.current_generation > 0) {
          // 降级方案：如果没有历史数据，用当前值生成
          console.log("Using fallback for evolution chart");
          updateEvolutionChart({
            best: Array(statusData.current_generation).fill(
              statusData.best_fitness,
            ),
            average: Array(statusData.current_generation).fill(
              statusData.avg_fitness,
            ),
          });
        }

        // 检查是否完成
        if (statusData.status === "completed") {
          console.log(
            "Mining completed, clearing interval and getting results",
          );
          if (window.miningInterval) {
            clearInterval(window.miningInterval);
          }
          // 清除计时器
          if (elapsedTimeIntervalRef.current) {
            clearInterval(elapsedTimeIntervalRef.current);
            elapsedTimeIntervalRef.current = null;
          }
          miningStartTimeRef.current = null;
          setMining(false);
          await getMiningResults(currentTaskId);
        } else if (statusData.status === "failed") {
          console.log("Mining failed:", statusData.error);
          if (window.miningInterval) {
            clearInterval(window.miningInterval);
          }
          // 清除计时器
          if (elapsedTimeIntervalRef.current) {
            clearInterval(elapsedTimeIntervalRef.current);
            elapsedTimeIntervalRef.current = null;
          }
          miningStartTimeRef.current = null;
          setMining(false);
          message.error(`挖掘失败: ${statusData.error || "未知错误"}`);
        } else if (statusData.status === "cancelled") {
          console.log("Mining cancelled by user");
          if (window.miningInterval) {
            clearInterval(window.miningInterval);
          }
          if (elapsedTimeIntervalRef.current) {
            clearInterval(elapsedTimeIntervalRef.current);
            elapsedTimeIntervalRef.current = null;
          }
          miningStartTimeRef.current = null;
          setMining(false);
          message.warning("挖掘任务已取消");
        }
      }
    } catch (error) {
      console.error("获取进度失败:", error);
      // 不中断轮询，继续尝试获取进度
      // 只有在任务不存在时才停止
      if (error instanceof Error && error.message.includes("任务不存在")) {
        if (window.miningInterval) {
          clearInterval(window.miningInterval);
        }
        // 清除计时器
        if (elapsedTimeIntervalRef.current) {
          clearInterval(elapsedTimeIntervalRef.current);
          elapsedTimeIntervalRef.current = null;
        }
        miningStartTimeRef.current = null;
        setMining(false);
        message.error("任务不存在或已过期");
      }
    }
  };

  // 取消挖掘任务
  const cancelMining = async () => {
    if (!miningStatus?.task_id) return;
    try {
      await api.cancelMiningTask(miningStatus.task_id);
      message.info("正在取消挖掘任务...");
    } catch (error) {
      console.error("取消挖掘任务失败:", error);
      message.error("取消挖掘任务失败");
    }
  };

  // 获取挖掘结果
  const getMiningResults = async (currentTaskId: string) => {
    try {
      console.log("Fetching mining results for task:", currentTaskId);
      const response = (await api.getMiningResults(currentTaskId)) as any;

      if (response.success) {
        console.log("Mining results received:", response.data);
        setMiningResult(response.data);

        // 绘制最终进化曲线（使用新的图表实例）
        if (response.data.fitness_history) {
          console.log(
            "Updating result chart with history:",
            response.data.fitness_history,
          );
          // 延迟一下确保DOM已渲染
          setTimeout(() => {
            updateResultChart(response.data.fitness_history);
          }, 200);
        }
      } else {
        message.error("获取结果失败: " + (response.message || "未知错误"));
      }
    } catch (error) {
      console.error("获取结果失败:", error);
      message.error("获取结果失败");
    }
  };

  // 更新进化曲线（进度中）
  const updateEvolutionChart = (fitnessHistory: {
    best: number[];
    average: number[];
  }) => {
    console.log(
      "updateEvolutionChart called, DOM element:",
      evolutionChartRef.current,
    );

    if (!evolutionChartRef.current) {
      console.error("Progress chart DOM element is null");
      return;
    }

    let chart = chartInstanceRef.current;
    if (!chart) {
      console.log("Initializing new progress chart instance");
      chart = echarts.init(evolutionChartRef.current);
      chartInstanceRef.current = chart;
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1);

    const option = {
      title: {
        text: "进化曲线",
        left: "center",
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: "axis",
      },
      legend: {
        data: ["最优适应度", "平均适应度"],
        bottom: 0,
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "10%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        name: "代数",
        data: generations,
      },
      yAxis: {
        type: "value",
        name: "适应度",
        scale: true,
      },
      series: [
        {
          name: "最优适应度",
          type: "line",
          data: fitnessHistory.best,
          smooth: true,
          itemStyle: { color: "#3b82f6" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
                { offset: 1, color: "rgba(59, 130, 246, 0.05)" },
              ],
            },
          },
        },
        {
          name: "平均适应度",
          type: "line",
          data: fitnessHistory.average,
          smooth: true,
          itemStyle: { color: "#22c55e" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(34, 197, 94, 0.3)" },
                { offset: 1, color: "rgba(34, 197, 94, 0.05)" },
              ],
            },
          },
        },
      ],
    };

    try {
      chart.setOption(option, true);
      console.log("Progress chart updated successfully");
    } catch (error) {
      console.error("Error updating progress chart:", error);
    }
  };

  // 更新最终结果图表
  const updateResultChart = (fitnessHistory: {
    best: number[];
    average: number[];
  }) => {
    console.log(
      "updateResultChart called, DOM element:",
      resultChartRef.current,
    );

    if (!resultChartRef.current) {
      console.error("Result chart DOM element is null");
      return;
    }

    let chart = resultChartInstanceRef.current;
    if (!chart) {
      console.log("Initializing new result chart instance");
      chart = echarts.init(resultChartRef.current);
      resultChartInstanceRef.current = chart;
    }

    const generations = fitnessHistory.best.map((_, i) => i + 1);

    const option = {
      title: {
        text: "完整进化曲线",
        left: "center",
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: "axis",
      },
      legend: {
        data: ["最优适应度", "平均适应度"],
        bottom: 0,
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "10%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        name: "代数",
        data: generations,
      },
      yAxis: {
        type: "value",
        name: "适应度",
        scale: true,
      },
      series: [
        {
          name: "最优适应度",
          type: "line",
          data: fitnessHistory.best,
          smooth: true,
          itemStyle: { color: "#3b82f6" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
                { offset: 1, color: "rgba(59, 130, 246, 0.05)" },
              ],
            },
          },
        },
        {
          name: "平均适应度",
          type: "line",
          data: fitnessHistory.average,
          smooth: true,
          itemStyle: { color: "#22c55e" },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(34, 197, 94, 0.3)" },
                { offset: 1, color: "rgba(34, 197, 94, 0.05)" },
              ],
            },
          },
        },
      ],
    };

    try {
      chart.setOption(option, true);
      console.log("Result chart updated successfully");
    } catch (error) {
      console.error("Error updating result chart:", error);
    }
  };

  // 保存单个因子（带重试机制）
  const saveFactor = async (
    factor: MinedFactor,
    index: number,
    retryCount: number = 0,
    customName?: string,
  ) => {
    // 验证门控：未通过验证的因子不允许保存
    if (factor.overall_passed === false) {
      message.warning(
        `因子 "${factor.name}" 未通过验证（得分: ${factor.validation_score?.toFixed(1)}），不能保存到因子库。请先通过因子验证。`
      );
      return null;
    }

    // 生成因子名称：Mined_Factor_序号_年月日时分秒_股票代码
    const today = new Date();
    const dateStr = [
      today.getFullYear(),
      String(today.getMonth() + 1).padStart(2, "0"),
      String(today.getDate()).padStart(2, "0"),
      String(today.getHours()).padStart(2, "0"),
      String(today.getMinutes()).padStart(2, "0"),
      String(today.getSeconds()).padStart(2, "0"),
    ].join("");

    // 确保使用有效的股票代码
    const stockCode = currentStockCode || "Unknown";
    const baseFactorName = customName || `Mined_Factor_${index + 1}_${dateStr}_${stockCode}`;

    // 根据重试次数生成名称
    let factorName: string;
    if (retryCount === 0) {
      factorName = baseFactorName;
    } else {
      factorName = `${baseFactorName}_${retryCount}`;
    }

    try {
      // 将表达式包装成完整的函数
      const generateFactorFunction = (expr: string) => {
        // 为表达式添加 df 前缀
        const processedExpr = expr
          .replace(/\bopen\b/g, "df['open']")
          .replace(/\bclose\b/g, "df['close']")
          .replace(/\bhigh\b/g, "df['high']")
          .replace(/\blow\b/g, "df['low']")
          .replace(/\bvolume\b/g, "df['volume']");

        const sourceLabel = factor.source === "pysr" ? "PySR符号回归" : factor.source === "genetic" ? "遗传规划" : "因子挖掘";
        return `def calculate_factor(df):
    """
    ${sourceLabel}挖掘因子
    表达式: ${expr}
    IC: ${factor.ic?.toFixed(4)}
    IR: ${factor.ir?.toFixed(4)}
    """
    import pandas as pd
    import numpy as np

    try:
        result = ${processedExpr}
        return result
    except Exception as e:
        # 如果计算失败，返回全0序列
        return pd.Series(0, index=df.index)
`;
      };

      const sourceLabel2 = factor.source === "pysr" ? "PySR符号回归" : factor.source === "genetic" ? "遗传规划" : "因子挖掘";
      const factorData: any = {
        name: factorName,
        code: generateFactorFunction(factor.expression),
        category: "遗传挖掘",
        description: `通过${sourceLabel2}挖掘的因子 | 表达式: ${factor.expression} | IC: ${factor.ic?.toFixed(4)} | IR: ${factor.ir?.toFixed(4)} | 适应度: ${factor.fitness?.toFixed(4)}`,
        formula_type: "function",
        generated_factor_id: factor.generated_factor_id || null,
      };

      console.log("Saving factor:", factorData);
      console.log("Factor code length:", factorData.code.length);

      const response = (await api.createFactor(factorData)) as any;

      if (response.success) {
        message.success(`因子 "${factorName}" 已保存到自定义因子库`);
        // 记录已保存的因子
        setSavedFactorNames((prev) => new Set(prev).add(factorName));
        // 记录因子ID，用于后续"跳转分析"
        const savedId = response.data?.id;
        if (savedId != null) {
          setSavedFactorIds((prev) => new Map(prev).set(index, savedId));
        }
        // 刷新因子列表
        await loadFactors();
        return savedId;
      } else {
        message.error(
          "保存失败: " +
            (response.data?.detail || response.message || "未知错误"),
        );
        return null;
      }
    } catch (error: any) {
      console.error("保存因子失败:", error);
      const errorMsg =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        "未知错误";

      // 如果是验证门控拒绝，直接提示
      if (errorMsg.includes("未通过验证")) {
        message.error("保存因子失败: " + errorMsg);
        return null;
      }

      // 如果是"已存在"错误且重试次数少于5次，使用新名称重试
      if (errorMsg.includes("已存在") && retryCount < 5) {
        console.log(
          `因子名称 ${factorName} 已存在，尝试使用新名称 (重试 ${retryCount + 1}/5)`,
        );
        return await saveFactor(factor, index, retryCount + 1, customName);
      } else {
        message.error("保存因子失败: " + errorMsg);
        return null;
      }
    }
  };

  // 跳转到因子分析页面
  const handleAnalyzeFactor = (index: number) => {
    const factorId = savedFactorIds.get(index);
    if (factorId != null) {
      navigate(`/factor-detail?id=${factorId}`);
    } else {
      message.warning("请先保存因子到因子库");
    }
  };

  // 打开重命名弹窗
  const handleOpenRename = (factor: MinedFactor, index: number) => {
    setRenameTarget({ factor, index });
    setCustomFactorName("");
    setRenameModalVisible(true);
  };

  // 确认重命名并保存
  const handleRenameSave = async () => {
    if (!renameTarget) return;
    const { factor, index } = renameTarget;
    const name = customFactorName.trim();
    if (!name) {
      message.warning("请输入因子名称");
      return;
    }
    await saveFactor(factor, index, 0, name);
    setRenameModalVisible(false);
    setRenameTarget(null);
  };

  // 保存单个因子到后端（带重试机制）
  const saveSingleFactorWithRetry = async (
    factor: MinedFactor,
    index: number,
    dateStr: string,
    stockCode: string,
  ): Promise<{ success: boolean; name?: string; renamed?: boolean }> => {
    // 验证门控
    if (factor.overall_passed === false) {
      return { success: false };
    }

    const baseFactorName = `Mined_Factor_${index + 1}_${dateStr}_${stockCode}`;

    for (let retry = 0; retry <= 5; retry++) {
      const factorName =
        retry === 0 ? baseFactorName : `${baseFactorName}_${retry}`;

      try {
        // 生成完整的因子函数代码
        const processedExpr = factor.expression
          .replace(/\bopen\b/g, "df['open']")
          .replace(/\bclose\b/g, "df['close']")
          .replace(/\bhigh\b/g, "df['high']")
          .replace(/\blow\b/g, "df['low']")
          .replace(/\bvolume\b/g, "df['volume']");

        const srcLbl = factor.source === "pysr" ? "PySR符号回归" : factor.source === "genetic" ? "遗传规划" : "因子挖掘";
        const factorCode = `def calculate_factor(df):
    """
    ${srcLbl}挖掘因子
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

        const factorData: any = {
          name: factorName,
          code: factorCode,
          category: "遗传挖掘",
          description: `通过${srcLbl}挖掘的因子 | 表达式: ${factor.expression} | IC: ${factor.ic?.toFixed(4)} | IR: ${factor.ir?.toFixed(4)} | 适应度: ${factor.fitness?.toFixed(4)}`,
          formula_type: "function",
          generated_factor_id: factor.generated_factor_id || null,
        };

        const response = (await api.createFactor(factorData)) as any;

        if (response.success) {
          return {
            success: true,
            name: factorName,
            renamed: retry > 0,
          };
        }
      } catch (error: any) {
        const errorMsg =
          error.response?.data?.detail ||
          error.response?.data?.message ||
          error.message ||
          "未知错误";

        // 验证门控拒绝，不再重试
        if (errorMsg.includes("未通过验证")) {
          return { success: false };
        }

        // 如果是"已存在"错误且还可以重试，继续循环
        if (errorMsg.includes("已存在") && retry < 5) {
          console.log(
            `因子 ${factorName} 已存在，下一次尝试使用: ${baseFactorName}_${retry + 1}`,
          );
          continue;
        } else {
          return { success: false };
        }
      }
    }

    return { success: false };
  };

  // 保存全部因子
  const saveAllFactors = async () => {
    if (
      !miningResult ||
      !miningResult.factors ||
      miningResult.factors.length === 0
    ) {
      message.warning("没有可保存的因子");
      return;
    }

    // 筛选出通过验证的因子
    const validFactors = miningResult.factors.filter(
      (f) => f.overall_passed !== false
    );
    const skippedCount = miningResult.factors.length - validFactors.length;

    if (validFactors.length === 0) {
      message.warning("所有因子均未通过验证，无法保存到因子库");
      return;
    }

    if (skippedCount > 0) {
      message.info(`已跳过 ${skippedCount} 个未通过验证的因子`);
    }

    // 生成日期字符串（包含时分秒）
    const today = new Date();
    const dateStr = [
      today.getFullYear(),
      String(today.getMonth() + 1).padStart(2, "0"),
      String(today.getDate()).padStart(2, "0"),
      String(today.getHours()).padStart(2, "0"),
      String(today.getMinutes()).padStart(2, "0"),
      String(today.getSeconds()).padStart(2, "0"),
    ].join("");

    // 确保使用有效的股票代码
    const stockCode = currentStockCode || "Unknown";

    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < validFactors.length; i++) {
      const factor = validFactors[i];
      const originalIndex = miningResult.factors.indexOf(factor);

      // 直接生成唯一的因子名称（包含序号、日期时间、股票代码）
      const factorName = `Mined_Factor_${originalIndex + 1}_${dateStr}_${stockCode}`;

      try {
        // 生成完整的因子函数代码
        const processedExpr = factor.expression
          .replace(/\bopen\b/g, "df['open']")
          .replace(/\bclose\b/g, "df['close']")
          .replace(/\bhigh\b/g, "df['high']")
          .replace(/\blow\b/g, "df['low']")
          .replace(/\bvolume\b/g, "df['volume']");

        const srcLbl3 = factor.source === "pysr" ? "PySR符号回归" : factor.source === "genetic" ? "遗传规划" : "因子挖掘";
        const factorCode = `def calculate_factor(df):
    """
    ${srcLbl3}挖掘因子
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

        const factorData: any = {
          name: factorName,
          code: factorCode,
          category: "遗传挖掘",
          description: `通过${srcLbl3}挖掘的因子 | 表达式: ${factor.expression} | IC: ${factor.ic?.toFixed(4)} | IR: ${factor.ir?.toFixed(4)} | 适应度: ${factor.fitness?.toFixed(4)}`,
          formula_type: "function",
          generated_factor_id: factor.generated_factor_id || null,
        };

        const response = (await api.createFactor(factorData)) as any;

        if (response.success) {
          successCount++;
          message.success(`因子 "${factorName}" 已保存到自定义因子库`);
          setSavedFactorNames((prev) => new Set(prev).add(factorName));
          await loadFactors();
        } else {
          failCount++;
          message.error(
            "保存失败: " +
              (response.data?.detail || response.message || "未知错误"),
          );
        }
      } catch (error: any) {
        failCount++;
        console.error("保存因子失败:", error);
        const errorMsg =
          error.response?.data?.detail ||
          error.response?.data?.message ||
          error.message ||
          "未知错误";
        message.error(`保存因子失败: ${errorMsg}`);
      }
    }

    // 显示结果消息
    if (failCount === 0) {
      message.success(`成功保存 ${successCount} 个因子到自定义因子库`);
    } else {
      message.warning(
        `保存完成: 成功 ${successCount} 个, 失败 ${failCount} 个`,
      );
    }
  };

  // 计算进度百分比
  const getProgressPercent = () => {
    if (!miningStatus || miningStatus.total_generations === 0) return 0;
    return Math.round(
      (miningStatus.current_generation / miningStatus.total_generations) * 100,
    );
  };

  // 格式化挖掘时长
  const formatElapsedTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}小时${minutes}分${secs}秒`;
    } else if (minutes > 0) {
      return `${minutes}分${secs}秒`;
    } else {
      return `${secs}秒`;
    }
  };

  return (
    <div className="factor-mining-container">
      {/* 背景 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      <div className="factor-mining-content">
        <div className="page-header">
          <div className="header-content">
            <RocketOutlined className="header-icon" />
            <div>
              <h1 className="page-title">因子挖掘</h1>
              <p className="page-subtitle">
                使用遗传算法与符号回归自动发现最优因子表达式
              </p>
            </div>
          </div>
        </div>

        <Row gutter={[24, 24]}>
          {/* 左侧配置面板 */}
          <Col xs={24} lg={8}>
            <Card title="因子挖掘配置" className="config-card">
              <Form form={form} layout="vertical" onFinish={startMining}>
                {/* 基础配置 */}
                <Divider
                  styles={{ content: { margin: 0 } }}
                  titlePlacement="left"
                >
                  基础配置
                </Divider>

                <Form.Item
                  label="股票池"
                  name="stock_mode"
                  initialValue="preset"
                  rules={[{ required: true, message: "请选择股票池模式" }]}
                >
                  <Select placeholder="选择股票池模式">
                    <Option value="preset">预设股票池</Option>
                    <Option value="custom">自定义股票</Option>
                  </Select>
                </Form.Item>

                <Form.Item noStyle shouldUpdate={(prev, cur) => prev?.stock_mode !== cur?.stock_mode}>
                  {({ getFieldValue }) => {
                    const mode = getFieldValue('stock_mode') || 'preset'
                    return mode === 'preset' ? (
                      <Form.Item
                        label="选择预设股票池"
                        name="stock_pool_id"
                        rules={[{ required: true, message: "请选择预设股票池" }]}
                      >
                        <Select
                          placeholder={stockPools.length > 0 ? "选择股票池" : "加载中..."}
                          loading={loadingPoolStocks || stockPools.length === 0}
                          key={`pool-select-${stockPools.length}`}
                        >
                          {stockPools.map((pool: any) => (
                            <Option key={pool.id} value={pool.id}>{pool.name}</Option>
                          ))}
                        </Select>
                      </Form.Item>
                    ) : (
                      <Form.Item
                        label="股票代码（逗号或换行分隔）"
                        name="stock_codes_input"
                        initialValue="000001"
                        rules={[{ required: true, message: "请输入股票代码" }]}
                      >
                        <Input.TextArea
                          placeholder="例如: 000001,600519,000858"
                          autoSize={{ minRows: 2, maxRows: 4 }}
                        />
                      </Form.Item>
                    )
                  }}
                </Form.Item>

                <Form.Item
                  label="日期范围"
                  name="dateRange"
                  rules={[{ required: true, message: "请选择日期范围" }]}
                >
                  <RangePicker style={{ width: "100%" }} />
                </Form.Item>

                <Form.Item label="数据频率" name="freq" tooltip="数据采样频率。日线适合中长周期因子，分钟级适合日内高频因子。注意：分钟级数据量远大于日线，挖掘耗时显著增加">
                  <Select>
                    <Option value="D">日线</Option>
                    <Option value="5min">5分钟</Option>
                    <Option value="15min">15分钟</Option>
                    <Option value="30min">30分钟</Option>
                    <Option value="60min">60分钟</Option>
                  </Select>
                </Form.Item>

                {/* 基础因子选择 */}
                <Divider
                  styles={{ content: { margin: 0 } }}
                  titlePlacement="left"
                >
                  基础因子选择
                </Divider>
                <p className="text-hint">
                  选择作为遗传算法输入的基础因子（可搜索因子名称）
                </p>

                <Form.Item
                  name="base_factors"
                  rules={[
                    { required: true, message: "请至少选择一个基础因子" },
                  ]}
                >
                  <Select
                    mode="multiple"
                    placeholder="输入因子名称搜索，如：RSI、MACD、SMA"
                    style={{ width: "100%" }}
                    showSearch
                    filterOption={(input, option) => {
                      const label = String(option?.label ?? "");
                      const value = String(option?.value ?? "");
                      return (
                        label.toLowerCase().includes(input.toLowerCase()) ||
                        value.toLowerCase().includes(input.toLowerCase())
                      );
                    }}
                    optionLabelProp="label"
                    maxTagCount="responsive"
                    size="large"
                    classNames={{ popup: "factor-select-dropdown" }}
                    listHeight={400}
                  >
                    {factors.map((factor) => (
                      <Option
                        key={factor.id}
                        value={factor.name}
                        label={factor.name}
                      >
                        <div
                          style={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 4,
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                            }}
                          >
                            <span style={{ fontWeight: 500 }}>
                              {factor.name}
                            </span>
                            <Tag
                              color={
                                factor.source === "preset"
                                  ? "success"
                                  : "warning"
                              }
                            >
                              {factor.source === "preset" ? "预置" : "自定义"}
                            </Tag>
                            <Tag color="blue">{factor.category}</Tag>
                          </div>
                          <div
                            style={{
                              fontSize: 12,
                              color: "#64748b",
                              fontFamily: "monospace",
                            }}
                          >
                            {factor.code}
                          </div>
                          {factor.description && (
                            <div style={{ fontSize: 12, color: "#94a3b8" }}>
                              {factor.description}
                            </div>
                          )}
                        </div>
                      </Option>
                    ))}
                  </Select>
                </Form.Item>

                <Form.Item noStyle shouldUpdate>
                  {() => {
                    const selectedCount =
                      form.getFieldValue("base_factors")?.length || 0;
                    return (
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: 16,
                        }}
                      >
                        <span className="text-hint">
                          已选择{" "}
                          <strong style={{ color: "#3b82f6" }}>
                            {selectedCount}
                          </strong>{" "}
                          个因子
                        </span>
                        <Space size="small">
                          <Button
                            type="link"
                            size="small"
                            onClick={() => {
                              form.setFieldsValue({
                                base_factors: factors.map((f) => f.name),
                              });
                            }}
                          >
                            全选
                          </Button>
                          <Button
                            type="link"
                            size="small"
                            onClick={() => {
                              form.setFieldsValue({ base_factors: [] });
                            }}
                          >
                            清空
                          </Button>
                        </Space>
                      </div>
                    );
                  }}
                </Form.Item>

                {/* 算法参数 */}
                <Divider
                  styles={{ content: { margin: 0 } }}
                  titlePlacement="left"
                >
                  算法参数
                </Divider>

                <Form.Item
                  label="挖掘算法"
                  name="algorithm"
                  tooltip="选择因子挖掘的核心算法。遗传规划(GP)通过进化搜索公式，可解释性强；PySR基于物理启发搜索简洁方程；树模型预筛选先用LightGBM筛选重要特征再回归，适合高维场景；GFlowNet用策略网络构建公式，探索更高效；深度隐式因子用Transformer学习时变隐因子，不追求可解释性"
                >
                  <Select>
                    <Option value="genetic">遗传规划 (DEAP)</Option>
                    <Option value="pysr">符号回归 (PySR)</Option>
                    <Option value="tree_prescreen">树模型预筛选</Option>
                    <Option value="gflownet">GFlowNet增强GP</Option>
                    <Option value="deep_implicit">深度隐式因子(Transformer)</Option>
                  </Select>
                </Form.Item>

                <Form.Item noStyle shouldUpdate>
                  {() => {
                    const algo = form.getFieldValue("algorithm") || "genetic";
                    // 各算法对应的参数面板
                    const showGP = algo === "genetic" || algo === "tree_prescreen" || algo === "gflownet";
                    const showPySR = algo === "pysr" || algo === "tree_prescreen";
                    const showGFlowNet = algo === "gflownet";
                    const showDeepImplicit = algo === "deep_implicit";
                    return (
                      <>
                        {showGP && (
                          <div style={{ marginBottom: 16, padding: "12px 16px", background: "rgba(59,130,246,0.05)", borderRadius: 8, border: "1px solid rgba(59,130,246,0.15)" }}>
                            <div style={{ fontWeight: 600, marginBottom: 8, color: "#3b82f6", fontSize: 13 }}>
                              遗传规划参数 (DEAP)
                            </div>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="种群大小"
                                  name="population_size"
                                  tooltip="每代进化中的候选公式数量。越大搜索越充分但越慢，推荐50-100。小种群(30)易早熟，大种群(150+)耗时长"
                                >
                                  <InputNumber
                                    min={10}
                                    max={200}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="迭代次数"
                                  name="n_generations"
                                  tooltip="进化迭代的总轮数。越多越可能找到优质因子，但收益递减。快速验证可用10-20代，正式挖掘建议30-50代"
                                >
                                  <InputNumber
                                    min={1}
                                    max={100}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>

                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item label="变异率" name="mutation_rate" tooltip="子代公式随机变异的概率。变异产生新算子/操作数，是探索新公式的主要手段。推荐0.2-0.4，过高导致搜索随机化，过低则探索不足">
                                  <InputNumber
                                    min={0}
                                    max={1}
                                    step={0.05}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item label="交叉率" name="crossover_rate" tooltip="两个父代公式交换子树生成子代的概率。交叉继承优秀子结构，推荐0.5-0.8。与变异率互补：交叉利用已有知识，变异探索新方向">
                                  <InputNumber
                                    min={0}
                                    max={1}
                                    step={0.05}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>

                            <Form.Item
                              label="精英保留数量"
                              name="elite_size"
                              tooltip="每代直接进入下一代的顶级公式数量。防止进化过程中丢失最优解，推荐3-5。过大(>10)会降低种群多样性"
                            >
                              <InputNumber min={0} max={20} style={{ width: "100%" }} />
                            </Form.Item>
                          </div>
                        )}

                        {showPySR && (
                          <div style={{ marginBottom: 16, padding: "12px 16px", background: "rgba(168,85,247,0.05)", borderRadius: 8, border: "1px solid rgba(168,85,247,0.15)" }}>
                            <div style={{ fontWeight: 600, marginBottom: 8, color: "#a855f7", fontSize: 13 }}>
                              符号回归参数 (PySR)
                            </div>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="迭代次数"
                                  name="pysr_niterations"
                                  tooltip="PySR符号回归的搜索轮数。每轮在当前最优方程基础上尝试简化或组合，越多越可能发现简洁高IC方程。快速验证用20-40，正式挖掘用60-100"
                                >
                                  <InputNumber
                                    min={10}
                                    max={200}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="种群数"
                                  name="pysr_populations"
                                  tooltip="PySR同时进化的独立种群数。不同种群独立搜索，最终合并结果。越多搜索越全面但越慢，推荐15-40"
                                >
                                  <InputNumber
                                    min={5}
                                    max={100}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>

                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="最大表达式大小"
                                  name="pysr_maxsize"
                                  tooltip="生成公式的最大运算节点数。控制公式复杂度上限，节点越多公式越灵活但越容易过拟合。推荐20-35，超过40通常无额外收益"
                                >
                                  <InputNumber
                                    min={5}
                                    max={50}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="最大表达式深度"
                                  name="pysr_maxdepth"
                                  tooltip="公式嵌套层数上限。如 sin(log(x)) 深度为2。深度越大公式表达能力越强但越难解释。推荐3-6，超过7易过拟合"
                                >
                                  <InputNumber
                                    min={2}
                                    max={10}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>

                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="简约系数"
                                  name="pysr_parsimony"
                                  tooltip="简约性惩罚系数。值越大越偏好简洁公式，防止过拟合。0=不惩罚，0.001-0.005=轻度惩罚(推荐)，0.01+=强惩罚(只保留极简公式)"
                                >
                                  <InputNumber
                                    min={0}
                                    max={0.1}
                                    step={0.0005}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="并行进程数"
                                  name="pysr_procs"
                                  tooltip="PySR底层Julia引擎的并行进程数。设为CPU核心数可最大化速度，但会占用更多内存。推荐4-8，内存不足时降低"
                                >
                                  <InputNumber
                                    min={1}
                                    max={32}
                                    style={{ width: "100%" }}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>
                          </div>
                        )}

                        {showGFlowNet && (
                          <div style={{ marginBottom: 16, padding: "12px 16px", background: "rgba(245,158,11,0.05)", borderRadius: 8, border: "1px solid rgba(245,158,11,0.15)" }}>
                            <div style={{ fontWeight: 600, marginBottom: 8, color: "#f59e0b", fontSize: 13 }}>
                              GFlowNet参数
                            </div>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="轨迹数量"
                                  name="gflownet_n_trajectories"
                                  tooltip="每轮训练中采样构建的公式轨迹数。越多训练越稳定但越慢，推荐100-300。低于50训练不稳定，高于500收益递减"
                                >
                                  <InputNumber min={50} max={500} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="迭代次数"
                                  name="gflownet_n_iterations"
                                  tooltip="策略网络训练的总轮数。越多网络越能学到高质量公式的构建模式，推荐30-80。过少(10)策略不成熟，过多(200)可能过拟合"
                                >
                                  <InputNumber min={10} max={200} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="隐藏层维度"
                                  name="gflownet_hidden_dim"
                                  tooltip="策略网络的隐藏层维度。决定网络对公式空间的建模能力，推荐64-256。小维度(32)表达能力弱，大维度(512)训练慢且易过拟合"
                                >
                                  <InputNumber min={32} max={512} step={32} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="学习率"
                                  name="gflownet_learning_rate"
                                  tooltip="策略网络参数更新步长。过大(>0.01)训练不稳定，过小(<0.0001)收敛极慢。推荐0.001-0.005"
                                >
                                  <InputNumber min={1e-5} max={1e-1} step={1e-4} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="最大表达式深度"
                                  name="gflownet_max_expression_depth"
                                  tooltip="GFlowNet生成公式的最大嵌套层数。与GP的max_tree_depth类似，控制公式复杂度。推荐3-6，过深易过拟合"
                                >
                                  <InputNumber min={3} max={10} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="温度参数"
                                  name="gflownet_temperature"
                                  tooltip="采样时的温度参数。低温度(0.1-0.5)倾向选择高概率动作，结果稳定但探索少；高温度(2.0-5.0)增加随机性，探索更多样但效率低。推荐0.5-1.5"
                                >
                                  <InputNumber min={0.1} max={5.0} step={0.1} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                            </Row>
                          </div>
                        )}

                        {showDeepImplicit && (
                          <div style={{ marginBottom: 16, padding: "12px 16px", background: "rgba(236,72,153,0.05)", borderRadius: 8, border: "1px solid rgba(236,72,153,0.15)" }}>
                            <div style={{ fontWeight: 600, marginBottom: 8, color: "#ec4899", fontSize: 13 }}>
                              深度隐式因子参数 (Transformer)
                            </div>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="模型维度"
                                  name="deep_d_model"
                                  tooltip="Transformer模型的特征嵌入维度。决定模型对时序模式的建模能力，推荐32-128。小维度欠拟合，大维度训练慢且需更多数据"
                                >
                                  <InputNumber min={32} max={256} step={32} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="注意力头数"
                                  name="deep_n_heads"
                                  tooltip="多头自注意力的头数。每个头关注不同的时序模式，头数越多捕获的模式越丰富。推荐2-8，需能整除嵌入维度"
                                >
                                  <InputNumber min={2} max={16} step={2} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="层数"
                                  name="deep_n_layers"
                                  tooltip="Transformer编码器堆叠层数。层数越深模型越能捕获复杂时序依赖，但也越容易过拟合。推荐2-4层，数据量大时可增至6"
                                >
                                  <InputNumber min={1} max={8} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="隐式因子数"
                                  name="deep_n_latent_factors"
                                  tooltip="模型输出的隐式因子个数。每个因子是一个独立的时变信号源，推荐3-10。过多因子会稀释信号且难解释"
                                >
                                  <InputNumber min={1} max={20} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="训练轮次"
                                  name="deep_n_epochs"
                                  tooltip="模型训练的epoch数。越多拟合越充分但可能过拟合，配合Early Stopping自动停止。推荐30-100，数据量大时可增加"
                                >
                                  <InputNumber min={10} max={200} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="批大小"
                                  name="deep_batch_size"
                                  tooltip="每次梯度更新的样本数。小batch(16-32)正则化效果好但训练慢，大batch(64-128)训练快但可能泛化差。推荐32-64"
                                >
                                  <InputNumber min={8} max={128} step={8} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  label="Dropout"
                                  name="deep_dropout"
                                  tooltip="训练时随机丢弃神经元的比例。防止过拟合，0=不丢弃，0.1-0.3=轻度正则化(推荐)，0.5=强正则化(数据少时用)"
                                >
                                  <InputNumber min={0} max={0.5} step={0.05} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  label="学习率"
                                  name="deep_learning_rate"
                                  tooltip="Adam优化器的学习率。控制参数更新步长，过大训练震荡，过小收敛慢。推荐1e-4到5e-4，通常不需要调整"
                                >
                                  <InputNumber min={1e-6} max={1e-2} step={1e-5} style={{ width: "100%" }} />
                                </Form.Item>
                              </Col>
                            </Row>
                          </div>
                        )}
                      </>
                    );
                  }}
                </Form.Item>

                {/* 适应度函数 */}
                <Divider
                  styles={{ content: { margin: 0 } }}
                  titlePlacement="left"
                >
                  适应度函数
                </Divider>

                <Form.Item label="优化目标" name="fitness_objective" tooltip="适应度函数决定进化方向，不同目标侧重不同因子特性">
                  <Select>
                    <Option value="ic_mean">IC均值</Option>
                    <Option value="ir_ratio">IR比率</Option>
                    <Option value="sharpe">夏普比率</Option>
                    <Option value="combined">综合得分</Option>
                  </Select>
                </Form.Item>

                <Form.Item noStyle shouldUpdate>
                  {() => {
                    const objective =
                      form.getFieldValue("fitness_objective") || "ic_mean";

                    let thresholdLabel = "阈值";
                    let thresholdPlaceholder = "0.03";
                    let thresholdTooltip = "筛选因子的阈值";

                    if (objective === "ic_mean") {
                      thresholdLabel = "IC阈值";
                      thresholdPlaceholder = "例如：0.03";
                      thresholdTooltip = "因子Rank IC的绝对均值下限。推荐：0.03（宽松，挖掘更多因子）~ 0.05（严格，只要强因子）。IC>0.03为弱因子，>0.05为中等，>0.1为强因子";
                      if (form.getFieldValue("ic_threshold") >= 1.0) form.setFieldValue("ic_threshold", 0.03);
                    } else if (objective === "ir_ratio") {
                      thresholdLabel = "IR阈值";
                      thresholdPlaceholder = "例如：0.5";
                      thresholdTooltip = "信息比率(IR=IC均值/IC标准差)下限，衡量因子预测稳定性。推荐：0.3（宽松）~ 1.0（严格）。IR>0.5说明因子信号较稳定，>1.0非常稳定";
                      if (form.getFieldValue("ic_threshold") < 0.1) form.setFieldValue("ic_threshold", 0.5);
                    } else if (objective === "sharpe") {
                      thresholdLabel = "夏普阈值";
                      thresholdPlaceholder = "例如：1.0";
                      thresholdTooltip = "基于多空组合的夏普比率下限。实际以IR近似代理（IR≈年化夏普/√周期数）。推荐：0.5（宽松）~ 2.0（严格）。夏普>1.0为可接受，>2.0为优秀";
                      if (form.getFieldValue("ic_threshold") < 0.1) form.setFieldValue("ic_threshold", 1.0);
                    } else if (objective === "combined") {
                      thresholdLabel = "综合阈值";
                      thresholdPlaceholder = "例如：0.3";
                      thresholdTooltip = "综合得分采用代际Z-Score归一化加权：60%×Norm(IC) + 40%×Norm(IR)。IC和IR先通过前一代（GA/GFlowNet）或全部方程（PySR）的统计量做Z-Score归一化：z=clip((x-μ)/σ,-3,3)，再映射到[0,1]：Norm=(z+3)/6，最后按权重求和。σ有下界保护max(σ,max(0.01×μ,0.005))防止收敛时Z-Score爆炸。第一代使用先验值(IC_μ=0.03,IC_σ=0.02,IR_μ=0.5,IR_σ=0.3)冷启动。得分范围[0,1]，推荐：0.3（宽松）~ 0.6（严格）";
                      if (form.getFieldValue("ic_threshold") < 0.1) form.setFieldValue("ic_threshold", 0.3);
                    }

                    return (
                      <Form.Item
                        label={thresholdLabel}
                        name="ic_threshold"
                        tooltip={thresholdTooltip}
                      >
                        <InputNumber
                          min={0}
                          step={0.01}
                          style={{ width: "100%" }}
                          placeholder={thresholdPlaceholder}
                        />
                      </Form.Item>
                    );
                  }}
                </Form.Item>

                {/* Phase 2-7: 高级优化参数 */}
                <Divider
                  styles={{ content: { margin: 0 } }}
                  titlePlacement="left"
                >
                  高级优化参数
                </Divider>

                <Row gutter={12}>
                  <Col span={12}>
                    <Form.Item
                      label="简约系数"
                      name="parsimony_coeff"
                      tooltip="简约性压力系数。对复杂公式施加适应度惩罚，防止表达式膨胀(过拟合)。0=关闭，0.0005-0.002=轻度(推荐)，0.005+=强压力(只保留极简公式)。与NSGA-II互斥时建议二选一"
                    >
                      <InputNumber
                        min={0}
                        max={0.1}
                        step={0.0005}
                        style={{ width: "100%" }}
                      />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="多样性惩罚"
                      name="diversity_penalty_coeff"
                      tooltip="多样性保护系数。基于Jaccard相似度惩罚与种群中已有公式雷同的个体，鼓励探索不同方向。0=关闭，0.05-0.2=推荐范围，0.5+=强压力(可能牺牲质量换多样性)"
                    >
                      <InputNumber
                        min={0}
                        max={1}
                        step={0.05}
                        style={{ width: "100%" }}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={12}>
                  <Col span={12}>
                    <Form.Item
                      label="交叉验证折数"
                      name="cv_folds"
                      tooltip="时间序列CV折数。将数据按时间顺序分段，在训练集上进化、验证集上筛选，有效控制过拟合。0=关闭(快但易过拟合)，3-5折=推荐(稍慢但更可靠)。开启后适应度取验证集表现"
                    >
                      <Select>
                        <Option value={0}>关闭</Option>
                        <Option value={2}>2折</Option>
                        <Option value={3}>3折</Option>
                        <Option value={5}>5折</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="最大树深度"
                      name="max_tree_depth"
                      tooltip="表达式树的最大深度。深度越大公式越复杂(如嵌套函数调用)，17=DEAP默认(允许复杂公式)，10以下=限制为简单公式。过深易过拟合，过浅表达力不足"
                    >
                      <InputNumber
                        min={3}
                        max={25}
                        step={1}
                        style={{ width: "100%" }}
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Row gutter={12}>
                  <Col span={12}>
                    <Form.Item
                      label="扩展基元集"
                      name="use_extended_primitives"
                      tooltip="扩展基元集开关。开启后增加时序窗口算子(如TS_MEAN、TS_STD、TS_RANK等约25个)，可挖掘动量/反转类因子；关闭仅保留基础算术运算(9个)，公式更简洁但表达力受限。强烈建议开启"
                      valuePropName="checked"
                    >
                      <Select>
                        <Option value={true}>启用（~25个基元）</Option>
                        <Option value={false}>仅基础（9个基元）</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="NSGA-II多目标"
                      name="use_nsga2"
                      tooltip="NSGA-II多目标进化。同时优化IC均值(越高越好)和表达式复杂度(越低越好)，产出Pareto前沿上的非支配解。开启后不再需要手动设简约系数。推荐开启，适合不确定简约系数时使用"
                      valuePropName="checked"
                    >
                      <Select>
                        <Option value={true}>启用</Option>
                        <Option value={false}>关闭（锦标赛选择）</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                </Row>

                <Form.Item>
                  <Button
                    type="primary"
                    htmlType="submit"
                    icon={<PlayCircleOutlined />}
                    loading={loading}
                    block
                    size="large"
                    disabled={mining}
                  >
                    {mining ? "挖掘中..." : "开始挖掘"}
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </Col>

          {/* 右侧结果展示 */}
          <Col xs={24} lg={16}>
            <Card title="挖掘结果" className="result-card">
              {/* 等待提示 */}
              {!mining && !miningStatus && !miningResult && (
                <div className="placeholder-content">
                  <BarChartOutlined className="placeholder-icon" />
                  <p className="placeholder-text">
                    配置参数后点击"开始挖掘"按钮
                  </p>
                  <p className="placeholder-hint">
                    遗传规划与符号回归将自动搜索最优因子表达式
                  </p>
                </div>
              )}

              {/* 挖掘进度和完成状态 */}
              {(mining || miningStatus) && !miningResult && (
                <div className="mining-progress">
                  {/* 挖掘状态提示 */}
                  {mining && (
                    <Alert
                      message={
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <Space>
                            <SyncOutlined spin />
                            <span>挖掘进行中...</span>
                            <ClockCircleOutlined />
                            <span style={{ color: "#64748b" }}>
                              已用时: {formatElapsedTime(elapsedTime)}
                            </span>
                          </Space>
                          <Button
                            size="small"
                            danger
                            icon={<StopOutlined />}
                            onClick={cancelMining}
                          >
                            取消挖掘
                          </Button>
                        </div>
                      }
                      type="info"
                      showIcon={false}
                      style={{
                        marginBottom: 16,
                        background: "rgba(59, 130, 246, 0.1)",
                        border: "1px solid rgba(59, 130, 246, 0.2)",
                      }}
                    />
                  )}

                  {/* 进度条和统计信息 - 只有当有 miningStatus 时才显示 */}
                  {miningStatus && (
                    <>
                      <div className="progress-section">
                        <div className="progress-header">
                          <span className="progress-label">挖掘进度</span>
                          <span className="progress-value">
                            {getProgressPercent()}%
                          </span>
                        </div>
                        <Progress
                          percent={getProgressPercent()}
                          status={mining ? "active" : "success"}
                          strokeColor={{
                            "0%": "#3b82f6",
                            "100%": "#22c55e",
                          }}
                        />
                      </div>

                      <Row gutter={16} style={{ marginTop: 24 }}>
                        <Col span={8}>
                          <div className="stat-item">
                            <p className="stat-label">当前代数</p>
                            <p className="stat-value">
                              {miningStatus.current_generation}/
                              {miningStatus.total_generations}
                            </p>
                          </div>
                        </Col>
                        <Col span={8}>
                          <div className="stat-item">
                            <p className="stat-label">最优适应度</p>
                            <p className="stat-value stat-primary">
                              {miningStatus.best_fitness?.toFixed(4) || "-"}
                            </p>
                          </div>
                        </Col>
                        <Col span={8}>
                          <div className="stat-item">
                            <p className="stat-label">平均适应度</p>
                            <p className="stat-value">
                              {miningStatus.avg_fitness?.toFixed(4) || "-"}
                            </p>
                          </div>
                        </Col>
                      </Row>
                    </>
                  )}

                  {/* 如果正在挖掘但还没有状态数据，显示加载提示 */}
                  {mining && !miningStatus && (
                    <div
                      style={{
                        textAlign: "center",
                        padding: "40px 0",
                        color: "#64748b",
                      }}
                    >
                      <Spin size="large" />
                      <p style={{ marginTop: 16 }}>正在执行挖掘任务...</p>
                    </div>
                  )}

                  {/* 进化曲线图表 - 只有当有数据时才显示 */}
                  {miningStatus && (
                    <div className="chart-section" style={{ marginTop: 24 }}>
                      <h4 className="chart-title">进化曲线（实时）</h4>
                      <div
                        ref={evolutionChartRef}
                        className="chart-container"
                        style={{ height: "300px" }}
                      ></div>
                    </div>
                  )}
                </div>
              )}

              {/* 挖掘完成提示 */}
              {miningStatus &&
                miningStatus.status === "completed" &&
                !miningResult && (
                  <div style={{ textAlign: "center", padding: "24px" }}>
                    <Spin size="large" tip="正在加载挖掘结果..." />
                  </div>
                )}

              {/* 最终结果 */}
              {miningResult && (
                <div className="mining-result">
                  {/* 挖掘摘要 */}
                  <div className="result-summary" style={{ marginBottom: 24 }}>
                    <Row gutter={16}>
                      <Col span={6}>
                        <div className="stat-item">
                          <p className="stat-label">总代数</p>
                          <p className="stat-value">
                            {miningResult.generations}
                          </p>
                        </div>
                      </Col>
                      <Col span={6}>
                        <div className="stat-item">
                          <p className="stat-label">最优适应度</p>
                          <p className="stat-value stat-primary">
                            {miningResult.best_fitness?.toFixed(4)}
                          </p>
                        </div>
                      </Col>
                      <Col span={6}>
                        <div className="stat-item">
                          <p className="stat-label">发现因子数</p>
                          <p className="stat-value">
                            {miningResult.factors?.length || 0}
                          </p>
                        </div>
                      </Col>
                      <Col span={6}>
                        <div className="stat-item">
                          <p className="stat-label">挖掘耗时</p>
                          <p className="stat-value">
                            {formatElapsedTime(elapsedTime)}
                          </p>
                        </div>
                      </Col>
                    </Row>
                  </div>

                  {/* 挖掘过程信息 */}
                  {miningResult.process_info && (
                    <Card
                      size="small"
                      title={
                        <Space>
                          <InfoCircleOutlined />
                          <span>挖掘过程详情</span>
                          <Tag color="blue">
                            {miningResult.process_info.algorithm_label || miningResult.process_info.algorithm}
                          </Tag>
                        </Space>
                      }
                      style={{ marginBottom: 24, borderRadius: 8 }}
                    >
                      <ProcessInfoDisplay info={miningResult.process_info} />
                    </Card>
                  )}

                  {/* 最终进化曲线 */}
                  <div className="chart-section" style={{ marginBottom: 24 }}>
                    <h4 className="chart-title">完整进化曲线</h4>
                    <div
                      ref={resultChartRef}
                      className="chart-container"
                      style={{ height: "300px" }}
                    ></div>
                  </div>

                  <Divider />

                  <h3 className="result-title">发现的因子</h3>

                  {!miningResult.factors ||
                  miningResult.factors.length === 0 ? (
                    <Alert
                      message="未发现符合条件的因子"
                      type="info"
                      showIcon
                      style={{ marginTop: 16 }}
                    />
                  ) : (
                    <div className="factors-list">
                      {miningResult.factors.map((factor, index) => (
                        <Card key={index} className="factor-card" size="small">
                          <div className="factor-header">
                            <div className="factor-info">
                              <Space>
                                <Tag color="blue">Top {index + 1}</Tag>
                                <Tag color={factor.source === "pysr" ? "purple" : factor.source === "genetic" ? "blue" : "default"}>
                                  {factor.source === "pysr" ? "PySR" : factor.source === "genetic" ? "GP" : factor.source === "simulated" ? "模拟" : "混合"}
                                </Tag>
                                <span className="factor-name">
                                  {factor.name || `Factor_${index + 1}`}
                                </span>
                              </Space>
                              <div className="factor-expression">
                                {factor.expression}
                              </div>
                            </div>
                            <div className="factor-stats">
                              <div className="stat-row">
                                <span className="stat-label">IC:</span>
                                <span
                                  className={`stat-value ${factor.ic > 0 ? "positive" : "negative"}`}
                                >
                                  {factor.ic?.toFixed(4)}
                                </span>
                              </div>
                              <div className="stat-row">
                                <span className="stat-label">IR:</span>
                                <span
                                  className={`stat-value ${factor.ir > 0 ? "positive" : "negative"}`}
                                >
                                  {factor.ir?.toFixed(4)}
                                </span>
                              </div>
                              <div className="stat-row">
                                <span className="stat-label">验证:</span>
                                <span
                                  className={`stat-value ${factor.overall_passed ? "positive" : "negative"}`}
                                >
                                  {factor.overall_passed ? "通过" : "未通过"}
                                  {factor.validation_score != null ? ` (${factor.validation_score.toFixed(1)})` : ""}
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="factor-actions">
                            <Button
                              type="primary"
                              size="small"
                              icon={<SaveOutlined />}
                              onClick={() => saveFactor(factor, index)}
                              disabled={factor.overall_passed === false}
                            >
                              {factor.overall_passed === false ? "未通过验证" : "保存到因子库"}
                            </Button>
                            <Button
                              size="small"
                              icon={<EditOutlined />}
                              onClick={() => handleOpenRename(factor, index)}
                              disabled={factor.overall_passed === false}
                            >
                              重命名保存
                            </Button>
                            <Button
                              size="small"
                              icon={<SearchOutlined />}
                              onClick={() => handleAnalyzeFactor(index)}
                              disabled={!savedFactorIds.has(index)}
                            >
                              分析
                            </Button>
                          </div>
                        </Card>
                      ))}
                    </div>
                  )}

                  <div className="result-actions" style={{ marginTop: 24 }}>
                    <Space>
                      <Button
                        type="primary"
                        icon={<SaveOutlined />}
                        onClick={saveAllFactors}
                      >
                        全部保存到因子库
                      </Button>
                    </Space>
                  </div>
                </div>
              )}
            </Card>
          </Col>
        </Row>
      </div>

      {/* 重命名保存弹窗 */}
      <Modal
        title="自定义因子名称"
        open={renameModalVisible}
        onOk={handleRenameSave}
        onCancel={() => {
          setRenameModalVisible(false);
          setRenameTarget(null);
        }}
        okText="保存"
        cancelText="取消"
      >
        <div style={{ marginBottom: 12 }}>
          <p style={{ color: "#64748b", fontSize: 13 }}>
            为因子设置一个有意义的名称，方便后续查找和管理。
          </p>
          {renameTarget && (
            <div style={{ marginBottom: 8, fontSize: 12, color: "#888" }}>
              表达式: <code style={{ fontSize: 11 }}>{renameTarget.factor.expression}</code>
            </div>
          )}
        </div>
        <Input
          placeholder="请输入因子名称"
          value={customFactorName}
          onChange={(e) => setCustomFactorName(e.target.value)}
          onPressEnter={handleRenameSave}
          autoFocus
        />
      </Modal>
    </div>
  );
};

export default FactorMining;
