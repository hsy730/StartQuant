"""
GFlowNet增强遗传规划因子挖掘服务

核心思想：传统GP使用随机变异/交叉来探索公式空间，效率较低。
GFlowNet（Generative Flow Network）学习一个策略网络，逐步构建公式树，
实现结构化、高效的公式空间探索。

公式构建过程:
    空树 → 选择算子节点 → 添加操作数节点 → 完整公式树 → 评估适应度

策略网络学习为更容易产生高适应度因子的公式结构分配更高概率。

训练方法: Trajectory Balance Loss
    L_TB = (log Z_θ - log R(x) + Σ log P_F(a|s))^2

其中 Z_θ 是可学习的基础流，R(x) 是奖励，P_F 是前向策略。
"""
import logging
import random
import math
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch 可用性检测
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    GFLOWNET_AVAILABLE = True
except ImportError:
    GFLOWNET_AVAILABLE = False
    logger.warning("PyTorch库未安装，GFlowNet因子挖掘功能将不可用。请运行: pip install torch")

from backend.services.factor_generator_service import factor_generator_service
from backend.services.factor_validation_service import factor_validation_service
from backend.services.alphalens_analysis_service import alphalens_analysis_service, ALPHALENS_AVAILABLE
from backend.services.data_service import data_service
from backend.services.factor_primitives import (
    create_pset,
    tree_to_expression,
    tree_to_placeholder_expr,
    compile_tree,
    expression_similarity,
)

MAX_EVAL_STOCKS = 50

# ---------------------------------------------------------------------------
# 算子定义 - GFlowNet 动作空间
# ---------------------------------------------------------------------------

# 一元算子（arity=1）
UNARY_OPS = ["neg", "abs", "log", "sqrt", "rank"]
# 二元算子（arity=2）
BINARY_OPS = ["add", "sub", "mul", "div"]
# 所有算子
ALL_OPS = UNARY_OPS + BINARY_OPS

# 算子元数映射
OP_ARITY = {op: 1 for op in UNARY_OPS}
OP_ARITY.update({op: 2 for op in BINARY_OPS})

# 扩展算子（可选，用于更丰富的表达式）
EXTENDED_UNARY_OPS = [
    "ts_mean_5", "ts_mean_10", "ts_mean_20",
    "ts_std_5", "ts_std_10", "ts_std_20",
    "ts_delay_1", "ts_delay_5",
    "ts_delta_1", "ts_delta_5",
    "sigmoid", "tanh",
]
EXTENDED_BINARY_OPS = [
    "ts_corr_5", "ts_corr_10", "ts_corr_20",
    "max", "min",
]

# ---------------------------------------------------------------------------
# 表达式树节点
# ---------------------------------------------------------------------------

class ExprNode:
    """表达式树节点"""

    def __init__(self, op: Optional[str] = None, factor_idx: Optional[int] = None):
        self.op = op            # 算子名称（如 "add", "neg"）或 None
        self.factor_idx = factor_idx  # 基础因子索引（叶节点）或 None
        self.children: List["ExprNode"] = []

    @property
    def is_leaf(self) -> bool:
        return self.op is None and self.factor_idx is not None

    @property
    def is_empty(self) -> bool:
        return self.op is None and self.factor_idx is None

    @property
    def arity(self) -> int:
        if self.is_leaf:
            return 0
        return OP_ARITY.get(self.op, 0)

    @property
    def is_complete(self) -> bool:
        """节点是否完整（有足够子节点）"""
        if self.is_empty:
            return False
        if self.is_leaf:
            return True
        return len(self.children) == self.arity

    def depth(self) -> int:
        if self.is_leaf or self.is_empty:
            return 0
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def to_string(self) -> str:
        """转换为DEAP兼容的Lisp风格表达式字符串"""
        if self.is_leaf:
            return f"factor_{self.factor_idx}"
        if self.is_empty:
            return "EMPTY"
        child_strs = [c.to_string() for c in self.children]
        return f"{self.op}({', '.join(child_strs)})"

    def node_count(self) -> int:
        if self.is_leaf:
            return 1
        if self.is_empty:
            return 0
        return 1 + sum(c.node_count() for c in self.children)

    def copy(self) -> "ExprNode":
        node = ExprNode(op=self.op, factor_idx=self.factor_idx)
        node.children = [c.copy() for c in self.children]
        return node


# ---------------------------------------------------------------------------
# 表达式状态编码
# ---------------------------------------------------------------------------

class ExpressionState:
    """表达式构建状态，用于策略网络输入编码"""

    def __init__(self, n_factors: int, max_depth: int, use_extended: bool = True):
        self.n_factors = n_factors
        self.max_depth = max_depth
        self.use_extended = use_extended
        self.root: Optional[ExprNode] = None
        self._action_history: List[int] = []

        # 动作空间定义
        self._build_action_space()

    def _build_action_space(self):
        """构建动作空间"""
        # 动作类型:
        #   0..N-1: 选择基础因子 factor_0..factor_{N-1}
        #   N..N+M-1: 选择算子
        #   N+M: STOP（完成构建）

        ops = ALL_OPS[:]
        if self.use_extended:
            ops += EXTENDED_UNARY_OPS + EXTENDED_BINARY_OPS

        self.factor_actions = list(range(self.n_factors))
        self.op_actions = {op: self.n_factors + i for i, op in enumerate(ops)}
        self.stop_action = self.n_factors + len(ops)
        self.all_ops = ops

        # 反向映射: action_id -> (type, value)
        self.action_map: Dict[int, Tuple[str, object]] = {}
        for i in range(self.n_factors):
            self.action_map[i] = ("factor", i)
        for op, action_id in self.op_actions.items():
            self.action_map[action_id] = ("op", op)
        self.action_map[self.stop_action] = ("stop", None)

        self.n_actions = self.stop_action + 1

    def get_incomplete_node(self) -> Optional[Tuple[ExprNode, List[int]]]:
        """找到第一个不完整的节点（深度优先，左优先）

        Returns:
            (node, path) 或 None（树已完整）
        """
        if self.root is None:
            return None  # 空树，需要在根节点放置算子

        def _dfs(node: ExprNode, path: List[int]) -> Optional[Tuple[ExprNode, List[int]]]:
            if not node.is_complete and not node.is_empty:
                return (node, path)
            for i, child in enumerate(node.children):
                result = _dfs(child, path + [i])
                if result is not None:
                    return result
            return None

        return _dfs(self.root, [])

    def current_depth(self) -> int:
        if self.root is None:
            return 0
        return self.root.depth()

    def is_complete(self) -> bool:
        """表达式是否构建完成"""
        if self.root is None:
            return False
        if not self.root.is_complete:
            return False
        return True

    def can_add_op(self) -> bool:
        """是否可以添加算子（深度未超限）"""
        return self.current_depth() < self.max_depth

    def get_valid_actions(self) -> List[int]:
        """获取当前状态下的合法动作"""
        if self.is_complete():
            # 树已完整，只能STOP
            return [self.stop_action]

        if self.root is None:
            # 空树：只能选算子（不能直接放叶节点作为根，这样太简单）
            valid = [self.op_actions[op] for op in self.all_ops if self.can_add_op()]
            # 也允许直接选因子作为单因子表达式
            valid.extend(self.factor_actions)
            return valid

        incomplete = self.get_incomplete_node()
        if incomplete is None:
            # 所有节点都完整
            return [self.stop_action]

        node, path = incomplete
        # 需要填充子节点：可以选算子（如果深度允许）或因子
        valid = list(self.factor_actions)  # 总是可以选因子
        if self.can_add_op():
            valid.extend([self.op_actions[op] for op in self.all_ops])
        return valid

    def apply_action(self, action: int) -> bool:
        """应用动作，更新表达式状态

        Returns:
            True 如果动作成功应用
        """
        if action not in self.action_map:
            return False

        action_type, value = self.action_map[action]

        if action_type == "stop":
            return True

        if action_type == "factor":
            leaf = ExprNode(factor_idx=value)
            if self.root is None:
                # 单因子表达式
                self.root = leaf
            else:
                incomplete = self.get_incomplete_node()
                if incomplete is None:
                    return False
                node, path = incomplete
                node.children.append(leaf)
            self._action_history.append(action)
            return True

        if action_type == "op":
            op_node = ExprNode(op=value)
            if self.root is None:
                self.root = op_node
            else:
                incomplete = self.get_incomplete_node()
                if incomplete is None:
                    return False
                node, path = incomplete
                node.children.append(op_node)
            self._action_history.append(action)
            return True

        return False

    def encode(self) -> np.ndarray:
        """将当前状态编码为固定大小的向量，用于策略网络输入

        编码方式:
        - 动作历史: one-hot序列，padding到max_depth*3长度
        - 当前深度: 标量
        - 未完成节点数: 标量
        - 节点总数: 标量
        """
        max_len = self.max_depth * 3  # 最大动作序列长度
        action_dim = self.n_actions

        # 动作历史编码
        history_encoded = np.zeros(max_len * action_dim, dtype=np.float32)
        for i, action in enumerate(self._action_history[:max_len]):
            offset = i * action_dim
            history_encoded[offset + action] = 1.0

        # 状态特征
        depth = self.current_depth() / max(self.max_depth, 1)
        incomplete_count = 0
        if self.root is not None:
            incomplete_count = self._count_incomplete(self.root)
        incomplete_norm = incomplete_count / 10.0

        # 节点数
        node_count = 0
        if self.root is not None:
            node_count = self.root.node_count()
        node_norm = node_count / 20.0

        state_features = np.array([depth, incomplete_norm, node_norm], dtype=np.float32)

        return np.concatenate([history_encoded, state_features])

    def _count_incomplete(self, node: ExprNode) -> int:
        count = 0
        if not node.is_leaf and not node.is_complete:
            count += (node.arity - len(node.children))
        for child in node.children:
            count += self._count_incomplete(child)
        return count

    def get_expression_string(self) -> str:
        if self.root is None:
            return ""
        return self.root.to_string()


# ===========================================================================
# 以下类需要 PyTorch，仅在 GFLOWNET_AVAILABLE 时定义
# ===========================================================================

if GFLOWNET_AVAILABLE:

    # -------------------------------------------------------------------
    # GFlowNet 策略网络
    # -------------------------------------------------------------------

    class PolicyNetwork(nn.Module):
        """GFlowNet前向策略网络 - MLP架构

        输入: 当前表达式状态编码
        输出: 各动作的概率分布
        """

        def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128, n_layers: int = 3):
            super().__init__()
            self.n_actions = n_actions

            layers = []
            in_dim = state_dim
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, n_actions))

            self.network = nn.Sequential(*layers)

            # 可学习的基础流 Z
            self.log_z = nn.Parameter(torch.zeros(1))

        def forward(self, state: torch.Tensor, valid_actions_mask: torch.Tensor) -> torch.Tensor:
            """
            Args:
                state: (batch, state_dim) 状态编码
                valid_actions_mask: (batch, n_actions) 合法动作掩码

            Returns:
                (batch, n_actions) 动作对数概率（仅合法动作有值）
            """
            logits = self.network(state)

            # 将非法动作的logit设为负无穷
            logits = logits.masked_fill(~valid_actions_mask, float('-inf'))

            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs

    # -------------------------------------------------------------------
    # 经验回放缓冲区
    # -------------------------------------------------------------------

    class ReplayBuffer:
        """经验回放缓冲区，存储轨迹用于训练"""

        def __init__(self, max_size: int = 1000):
            self.max_size = max_size
            self.buffer: List[Dict] = []

        def add(self, trajectory: Dict):
            self.buffer.append(trajectory)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

        def sample(self, n: int) -> List[Dict]:
            if len(self.buffer) == 0:
                return []
            n = min(n, len(self.buffer))
            return random.sample(self.buffer, n)

        def __len__(self):
            return len(self.buffer)

    # -------------------------------------------------------------------
    # GFlowNet 因子挖掘服务
    # -------------------------------------------------------------------

    class GFlowNetMiningService:
        """GFlowNet增强因子挖掘服务

        使用GFlowNet（Generative Flow Network）学习公式构建策略，
        相比传统GP的随机变异/交叉，能够更高效地探索公式空间。

        核心流程:
        1. 策略网络采样多条公式构建轨迹
        2. 评估每条轨迹生成的因子表达式
        3. 使用Trajectory Balance Loss更新策略网络
        4. 维护Hall-of-Fame保存最优表达式
        """

        def __init__(
            self,
            base_factors: List[str],
            data: pd.DataFrame,
            return_column: str = "return",
            factor_calculator=None,
            max_eval_stocks: int = MAX_EVAL_STOCKS,
            # ---- GFlowNet 核心参数 ----
            n_trajectories: int = 200,
            n_iterations: int = 50,
            hidden_dim: int = 128,
            learning_rate: float = 1e-3,
            max_expression_depth: int = 5,
            temperature: float = 1.0,
            reward_scale: float = 10.0,
            buffer_size: int = 1000,
            # ---- 扩展算子 ----
            use_extended_primitives: bool = True,
            # ---- 适应度 ----
            fitness_objective: str = "ic_mean",
            parsimony_coeff: float = 0.001,
            diversity_penalty_coeff: float = 0.1,
            # ---- 交叉验证 ----
            cv_folds: int = 0,
        ):
            if not GFLOWNET_AVAILABLE:
                raise ImportError("PyTorch库未安装，请运行: pip install torch")

            self.base_factor_codes = base_factors
            self.data = data
            self.return_column = return_column
            self.factor_calculator = factor_calculator
            self.max_eval_stocks = max_eval_stocks

            # GFlowNet参数
            self.n_trajectories = n_trajectories
            self.n_iterations = n_iterations
            self.hidden_dim = hidden_dim
            self.learning_rate = learning_rate
            self.max_expression_depth = max_expression_depth
            self.temperature = temperature
            self.reward_scale = reward_scale
            self.buffer_size = buffer_size

            # 表达式参数
            self.use_extended_primitives = use_extended_primitives

            # 适应度参数
            self.fitness_objective = fitness_objective
            self.parsimony_coeff = parsimony_coeff
            self.diversity_penalty_coeff = diversity_penalty_coeff
            self.cv_folds = cv_folds

            self.return_values = data[return_column] if return_column in data.columns else None

            # 股票池
            self.stock_codes: List[str] = []
            self.stock_pool_data: Dict[str, pd.DataFrame] = {}
            self.stock_pool_return_values: Dict[str, pd.Series] = {}
            self.stock_pool_base_factor_values: Dict[str, dict] = {}
            self._sampled_stock_codes: List[str] = []

            # 预计算基础因子
            self.base_factor_values: Dict[str, dict] = {}
            self._precompute_base_factors()

            # GP原语集（用于编译表达式）
            self.pset = create_pset(
                max(len(self.base_factor_values), 1),
                extended=self.use_extended_primitives,
            )

            # Hall-of-Fame
            self._halloffame: List[Dict] = []

            # 进度回调
            self.progress_callback = None
            self._cancel_flag = False

            # Iterative Z-Score normalization for combined score
            # Collect raw IC/IR per iteration → compute Z-Score stats → apply next iteration
            self._gen_ic_values: List[float] = []   # raw IC values collected in current iteration
            self._gen_ir_values: List[float] = []   # raw IR values collected in current iteration
            # Prior cold-start values based on domain knowledge of quantitative factors
            _PRIOR_IC_MEAN = 0.03
            _PRIOR_IC_STD = 0.02
            _PRIOR_IR_MEAN = 0.5
            _PRIOR_IR_STD = 0.3
            self._zscore_ic_mean: float = _PRIOR_IC_MEAN  # μ of IC from previous iteration
            self._zscore_ic_std: float = _PRIOR_IC_STD   # σ of IC from previous iteration
            self._zscore_ir_mean: float = _PRIOR_IR_MEAN  # μ of IR from previous iteration
            self._zscore_ir_std: float = _PRIOR_IR_STD   # σ of IR from previous iteration
            self._has_zscore_stats: bool = True  # Prior values are valid from the start

            # 经验回放
            self.replay_buffer = ReplayBuffer(max_size=buffer_size)

            # 初始化策略网络
            self._init_policy_network()

        # ---------------------------------------------------------------
        # 基础因子预计算
        # ---------------------------------------------------------------

        def _precompute_base_factors(self):
            """预计算基础因子值"""
            if self.factor_calculator is None:
                from backend.services.factor_service import factor_service
                self.factor_calculator = factor_service.calculator

            logger.info(f"GFlowNet预计算 {len(self.base_factor_codes)} 个基础因子...")

            for i, factor_code in enumerate(self.base_factor_codes):
                try:
                    fv = self.factor_calculator.calculate(self.data, factor_code)
                    if fv is not None and len(fv.dropna()) > 0:
                        var_name = f"factor_{i}"
                        self.base_factor_values[var_name] = {
                            "code": factor_code,
                            "values": fv,
                        }
                        logger.info(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: {len(fv.dropna())} 个有效值")
                    else:
                        logger.warning(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算失败或无有效值")
                except Exception as e:
                    logger.warning(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算出错 - {e}")

            logger.info(f"GFlowNet成功预计算 {len(self.base_factor_values)} 个基础因子")

        def _init_policy_network(self):
            """初始化策略网络和优化器"""
            n_factors = max(len(self.base_factor_values), 1)

            # 创建临时状态来获取编码维度
            tmp_state = ExpressionState(
                n_factors=n_factors,
                max_depth=self.max_expression_depth,
                use_extended=self.use_extended_primitives,
            )
            state_dim = len(tmp_state.encode())
            n_actions = tmp_state.n_actions

            self._state_template = tmp_state
            self._state_dim = state_dim
            self._n_actions = n_actions

            self.policy_net = PolicyNetwork(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dim=self.hidden_dim,
            )
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

            logger.info(
                f"策略网络初始化: state_dim={state_dim}, n_actions={n_actions}, "
                f"hidden_dim={self.hidden_dim}"
            )

        # ---------------------------------------------------------------
        # 股票池
        # ---------------------------------------------------------------

        def set_stock_pool(self, stock_codes: List[str], start_date: str, end_date: str):
            """设置股票池用于截面IC评估"""
            self.stock_codes = stock_codes
            self.stock_pool_data = data_service.get_multiple_stocks_data(stock_codes, start_date, end_date)

            for code, df in self.stock_pool_data.items():
                if "close" in df.columns:
                    df["return"] = df["close"].pct_change()
                self.stock_pool_return_values[code] = (
                    df[self.return_column] if self.return_column in df.columns else None
                )

                if self.factor_calculator is None:
                    from backend.services.factor_service import factor_service
                    self.factor_calculator = factor_service.calculator

                stock_base_factors = {}
                for i, factor_code in enumerate(self.base_factor_codes):
                    try:
                        fv = self.factor_calculator.calculate(df, factor_code)
                        if fv is not None and len(fv.dropna()) > 0:
                            var_name = f"factor_{i}"
                            stock_base_factors[var_name] = {
                                "code": factor_code,
                                "values": fv,
                            }
                    except Exception as e:
                        logger.warning(f"Stock {code} factor {factor_code} compute error: {e}")
                self.stock_pool_base_factor_values[code] = stock_base_factors

            self._refresh_stock_sample()
            logger.info(
                f"GFlowNet股票池已设置: {len(self.stock_pool_data)} 只股票, "
                f"评估样本={len(self._sampled_stock_codes)}"
            )

        def _refresh_stock_sample(self):
            available = list(self.stock_pool_base_factor_values.keys())
            if len(available) <= self.max_eval_stocks:
                self._sampled_stock_codes = available
            else:
                self._sampled_stock_codes = random.sample(available, self.max_eval_stocks)

        def set_progress_callback(self, callback):
            """设置进度回调函数

            Args:
                callback: 签名为 callback(iteration, total_iterations, best_fitness, avg_fitness)
            """
            self.progress_callback = callback

        def request_cancel(self):
            """请求取消挖掘任务"""
            self._cancel_flag = True
            logger.info("收到取消请求，将在当前迭代结束后停止")

        # ---------------------------------------------------------------
        # 表达式采样
        # ---------------------------------------------------------------

        def _sample_trajectory(self) -> Tuple[ExpressionState, List[Tuple[np.ndarray, int, List[int]]], float]:
            """使用当前策略采样一条公式构建轨迹

            Returns:
                (final_state, trajectory_info, log_prob)
                trajectory_info: [(state_encoded, action, valid_actions), ...]
            """
            n_factors = max(len(self.base_factor_values), 1)
            state = ExpressionState(
                n_factors=n_factors,
                max_depth=self.max_expression_depth,
                use_extended=self.use_extended_primitives,
            )

            trajectory_info = []
            total_log_prob = 0.0

            max_steps = self.max_expression_depth * 4  # 安全上限
            for _ in range(max_steps):
                if state.is_complete():
                    break

                valid_actions = state.get_valid_actions()
                if not valid_actions:
                    break

                # 编码当前状态
                state_encoded = state.encode()
                state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0)

                # 构建合法动作掩码
                mask = torch.zeros(self._n_actions, dtype=torch.bool)
                for a in valid_actions:
                    mask[a] = True
                mask_tensor = mask.unsqueeze(0)

                # 获取策略输出
                with torch.no_grad():
                    log_probs = self.policy_net(state_tensor, mask_tensor).squeeze(0)

                # 温度缩放采样
                probs = torch.exp(log_probs / self.temperature)
                probs = probs / probs.sum()

                # 采样动作
                action_tensor = torch.multinomial(probs, 1)
                action = action_tensor.item()

                # 记录轨迹
                trajectory_info.append((state_encoded, action, valid_actions))
                total_log_prob += log_probs[action].item()

                # 应用动作
                if not state.apply_action(action):
                    break

            return state, trajectory_info, total_log_prob

        def _sample_batch_trajectories(self, n: int) -> List[Tuple[ExpressionState, List, float]]:
            """批量采样多条轨迹"""
            trajectories = []
            for _ in range(n):
                state, info, log_prob = self._sample_trajectory()
                trajectories.append((state, info, log_prob))
            return trajectories

        # ---------------------------------------------------------------
        # 表达式评估
        # ---------------------------------------------------------------

        def _evaluate_expression(self, expr_state: ExpressionState) -> float:
            """评估表达式状态对应的因子适应度

            尝试将表达式编译为可执行函数，计算因子值，然后用IC评估。
            """
            expr_str = expr_state.get_expression_string()
            if not expr_str or not expr_state.is_complete():
                return 0.0

            try:
                # 尝试使用DEAP编译表达式
                fv = self._compute_factor_from_string(expr_str)
                if fv is None:
                    return 0.0
            except Exception:
                return 0.0

            # 根据是否有股票池选择评估方式
            if len(self.stock_pool_data) >= 2:
                fitness = self._evaluate_cross_sectional_ic_from_values(fv, expr_str)
            elif len(self.stock_pool_data) == 1:
                fitness = self._evaluate_single_stock_ic_from_values(fv)
            else:
                fitness = self._evaluate_single_stock_ic_from_values(fv)

            # 简约性惩罚
            if expr_state.root is not None:
                parsimony_penalty = self.parsimony_coeff * expr_state.root.node_count()
            else:
                parsimony_penalty = 0.0

            # 多样性惩罚
            diversity_penalty = 0.0
            if self.diversity_penalty_coeff > 0 and self._halloffame:
                for hof_entry in self._halloffame:
                    sim = expression_similarity(expr_str, hof_entry.get("placeholder_expression", ""))
                    if sim > 0.7:
                        diversity_penalty += self.diversity_penalty_coeff * sim

            return max(fitness - parsimony_penalty - diversity_penalty, 0.0)

        def _compute_factor_from_string(self, expr_str: str) -> Optional[pd.Series]:
            """从Lisp风格表达式字符串计算因子值

            尝试通过DEAP的compile机制执行表达式。
            如果失败，尝试直接用Python eval计算。
            """
            # 方法1: 使用DEAP编译
            try:
                from deap import gp as deap_gp

                # 将表达式字符串解析为PrimitiveTree
                tree = deap_gp.PrimitiveTree.from_string(expr_str, self.pset)
                func = deap_gp.compile(tree, self.pset)

                ordered = []
                for i in range(len(self.base_factor_values)):
                    info = self.base_factor_values.get(f"factor_{i}")
                    if info is None:
                        return None
                    ordered.append(info["values"])

                result = func(*ordered)

                if isinstance(result, (int, float, np.number)):
                    idx = ordered[0].index if ordered else None
                    if idx is None:
                        return None
                    result = pd.Series(float(result), index=idx)

                if not isinstance(result, pd.Series):
                    return None

                result = result.replace([np.inf, -np.inf], np.nan)
                valid = result.notna().sum()
                if valid == 0 or valid < len(result) * 0.1:
                    return None

                return result
            except Exception:
                pass

            # 方法2: 直接用Python eval（安全受限，仅用于简单表达式）
            try:
                return self._eval_expression_direct(expr_str)
            except Exception:
                return None

        def _eval_expression_direct(self, expr_str: str) -> Optional[pd.Series]:
            """直接用Python eval计算简单表达式（fallback）"""
            # 构建命名空间
            namespace = {}
            for var_name, info in self.base_factor_values.items():
                namespace[var_name] = info["values"]

            # 添加安全算子
            from backend.services.factor_primitives import (
                safe_div, safe_log, safe_sqrt, pct_rank,
                ts_mean, ts_std, ts_delay, ts_delta, ts_corr,
                _pair_max, _pair_min, _sigmoid, _tanh,
            )
            namespace["add"] = np.add
            namespace["sub"] = np.subtract
            namespace["mul"] = np.multiply
            namespace["div"] = safe_div
            namespace["neg"] = np.negative
            namespace["abs"] = np.abs
            namespace["log"] = safe_log
            namespace["sqrt"] = safe_sqrt
            namespace["rank"] = pct_rank
            namespace["ts_mean_5"] = lambda a: ts_mean(a, 5)
            namespace["ts_mean_10"] = lambda a: ts_mean(a, 10)
            namespace["ts_mean_20"] = lambda a: ts_mean(a, 20)
            namespace["ts_std_5"] = lambda a: ts_std(a, 5)
            namespace["ts_std_10"] = lambda a: ts_std(a, 10)
            namespace["ts_std_20"] = lambda a: ts_std(a, 20)
            namespace["ts_delay_1"] = lambda a: ts_delay(a, 1)
            namespace["ts_delay_5"] = lambda a: ts_delay(a, 5)
            namespace["ts_delta_1"] = lambda a: ts_delta(a, 1)
            namespace["ts_delta_5"] = lambda a: ts_delta(a, 5)
            namespace["ts_corr_5"] = lambda a, b: ts_corr(a, b, 5)
            namespace["ts_corr_10"] = lambda a, b: ts_corr(a, b, 10)
            namespace["ts_corr_20"] = lambda a, b: ts_corr(a, b, 20)
            namespace["max"] = _pair_max
            namespace["min"] = _pair_min
            namespace["sigmoid"] = _sigmoid
            namespace["tanh"] = _tanh

            result = eval(expr_str, {"__builtins__": {}}, namespace)

            if isinstance(result, (int, float, np.number)):
                first_factor = list(self.base_factor_values.values())[0]
                result = pd.Series(float(result), index=first_factor["values"].index)

            if not isinstance(result, pd.Series):
                return None

            result = result.replace([np.inf, -np.inf], np.nan)
            valid = result.notna().sum()
            if valid == 0 or valid < len(result) * 0.1:
                return None

            return result

        def _evaluate_cross_sectional_ic_from_values(self, fv: pd.Series, expr_str: str) -> float:
            """截面IC评估（多股票）"""
            factor_values_dict: Dict[str, pd.Series] = {}

            for code in self._sampled_stock_codes:
                try:
                    base_factors = self.stock_pool_base_factor_values.get(code, {})
                    if not base_factors:
                        continue
                    stock_fv = self._eval_expression_on_stock(expr_str, base_factors)
                    if stock_fv is not None and len(stock_fv.dropna()) >= 10:
                        factor_values_dict[code] = stock_fv.dropna()
                except Exception:
                    continue

            if len(factor_values_dict) < 2:
                return self._evaluate_single_stock_ic_from_values(fv)

            if not ALPHALENS_AVAILABLE:
                return self._evaluate_single_stock_ic_from_values(fv)

            try:
                all_dates = set()
                for s in factor_values_dict.values():
                    all_dates.update(s.index)
                all_dates = sorted(all_dates)

                pricing_df = pd.DataFrame(index=all_dates)
                for stock_code in factor_values_dict:
                    df = self.stock_pool_data.get(stock_code)
                    if df is not None and "close" in df.columns:
                        pricing_df[stock_code] = df["close"]
                pricing_df = pricing_df.dropna(how="all")

                factor_data = alphalens_analysis_service.prepare_factor_data(
                    factor_values_dict=factor_values_dict,
                    pricing_df=pricing_df,
                )

                if factor_data is None or factor_data.empty:
                    return 0.0

                ic_results = alphalens_analysis_service.analyze_ic(factor_data)

                if "error" in ic_results:
                    return 0.0

                raw_fitness = self._route_fitness(ic_results)

                # 交叉验证惩罚
                cv_penalty = self._cv_penalty(factor_values_dict)
                raw_fitness = raw_fitness * (1.0 - cv_penalty)

                return raw_fitness

            except Exception as e:
                logger.warning(f"GFlowNet截面IC评估失败: {e}")
                return self._evaluate_single_stock_ic_from_values(fv)

        def _eval_expression_on_stock(self, expr_str: str, stock_base_factors: dict) -> Optional[pd.Series]:
            """在单只股票上计算表达式"""
            try:
                from deap import gp as deap_gp
                tree = deap_gp.PrimitiveTree.from_string(expr_str, self.pset)
                func = deap_gp.compile(tree, self.pset)

                ordered = []
                for i in range(len(self.base_factor_values)):
                    info = stock_base_factors.get(f"factor_{i}")
                    if info is None:
                        return None
                    ordered.append(info["values"])

                result = func(*ordered)

                if isinstance(result, (int, float, np.number)):
                    idx = ordered[0].index if ordered else None
                    if idx is None:
                        return None
                    result = pd.Series(float(result), index=idx)

                if not isinstance(result, pd.Series):
                    return None

                result = result.replace([np.inf, -np.inf], np.nan)
                return result
            except Exception:
                return None

        def _evaluate_single_stock_ic_from_values(self, fv: pd.Series) -> float:
            """单股票IC评估"""
            if fv is None or len(fv.dropna()) < 10:
                return 0.0

            if self.return_values is not None:
                validation = factor_validation_service.validate_factor(
                    factor_values=fv,
                    return_values=self.return_values,
                    existing_factors=None,
                )
                fitness = validation["score"] / 100.0
            else:
                fitness = fv.std() / (fv.mean() + 1e-8)
                # CV代理不反映预测能力，返回0.0避免误导
                logger.warning("无收益率数据时无法评估因子预测能力，适应度设为0")
                fitness = 0.0

            return fitness

        # ---------------------------------------------------------------
        # 适应度路由
        # ---------------------------------------------------------------

        def _route_fitness(self, ic_results: dict) -> float:
            """根据fitness_objective选择适应度值

            对于 combined 模式，IC 和 IR 先通过代际 Z-Score 归一化，
            使得 60/40 权重在 IC（典型0.01-0.10）和 IR（典型0.3-2.0）
            不同量纲下仍然有效。

            公式:
                z_ic = clip((IC - μ_ic) / (σ_ic + ε), -3, 3)
                z_ir = clip((IR - μ_ir) / (σ_ir + ε), -3, 3)
                Norm(IC) = (z_ic + 3) / 6   → maps [-3σ, +3σ] to [0, 1]
                Norm(IR) = (z_ir + 3) / 6
                combined  = 0.6 * Norm(IC) + 0.4 * Norm(IR)

            统计量从前一轮迭代收集，通过 _update_zscore_stats() 更新。
            第一轮迭代使用先验冷启动值。
            """
            best_ic = 0.0
            best_ir = 0.0

            for ic_type in ["spearman_ic", "pearson_ic"]:
                ic_type_data = ic_results.get(ic_type, {})
                for period_key, period_stats in ic_type_data.items():
                    if not isinstance(period_stats, dict) or "error" in period_stats:
                        continue
                    mean_ic = period_stats.get("mean_ic")
                    std_ic = period_stats.get("std_ic")
                    if mean_ic is None or std_ic is None:
                        continue
                    mean_ic = abs(float(mean_ic))
                    std_ic = float(std_ic)
                    ir = abs(mean_ic / std_ic) if std_ic > 1e-10 else 0.0
                    if mean_ic > best_ic:
                        best_ic = mean_ic
                    if ir > best_ir:
                        best_ir = ir

            # Collect raw IC/IR for iterative Z-Score computation
            self._gen_ic_values.append(best_ic)
            self._gen_ir_values.append(best_ir)

            if self.fitness_objective == "ir_ratio":
                return best_ir
            elif self.fitness_objective == "sharpe":
                return best_ir
            elif self.fitness_objective == "combined":
                # Z-Score normalize using previous iteration's statistics (with prior cold-start)
                z_ic = max(-3.0, min((best_ic - self._zscore_ic_mean) / (self._zscore_ic_std + 1e-8), 3.0))
                z_ir = max(-3.0, min((best_ir - self._zscore_ir_mean) / (self._zscore_ir_std + 1e-8), 3.0))
                # Map from [-3, 3] to [0, 1]
                norm_ic = (z_ic + 3.0) / 6.0
                norm_ir = (z_ir + 3.0) / 6.0
                return 0.6 * norm_ic + 0.4 * norm_ir
            else:
                return best_ic

        def _update_zscore_stats(self):
            """Compute Z-Score normalization stats from the current iteration's
            collected IC/IR values.  Called at each iteration boundary.

            Requirements: at least 5 valid values to compute stable statistics.
            Applies σ lower-bound protection: max(σ, max(0.01*μ, 0.005)) to
            prevent Z-Score explosion when trajectories converge.
            After computing, clears the collection lists for the next iteration.
            """
            valid_ic = [v for v in self._gen_ic_values if v > 1e-10]
            valid_ir = [v for v in self._gen_ir_values if v > 1e-10]

            if len(valid_ic) >= 5 and len(valid_ir) >= 5:
                ic_mean = float(np.mean(valid_ic))
                ic_std = float(np.std(valid_ic))
                ir_mean = float(np.mean(valid_ir))
                ir_std = float(np.std(valid_ir))

                # σ lower-bound protection: prevent Z-Score explosion on convergence
                ic_std = max(ic_std, max(0.01 * ic_mean, 0.005))
                ir_std = max(ir_std, max(0.01 * ir_mean, 0.005))

                if ic_std < self._zscore_ic_std * 0.1 or ir_std < self._zscore_ir_std * 0.1:
                    logger.warning(
                        f"Z-Score σ very small (IC σ={ic_std:.6f}, IR σ={ir_std:.6f}), "
                        f"search may be stagnating"
                    )

                self._zscore_ic_mean = ic_mean
                self._zscore_ic_std = ic_std
                self._zscore_ir_mean = ir_mean
                self._zscore_ir_std = ir_std
                self._has_zscore_stats = True
                logger.debug(
                    f"Z-Score stats updated: IC μ={self._zscore_ic_mean:.4f} σ={self._zscore_ic_std:.4f}, "
                    f"IR μ={self._zscore_ir_mean:.4f} σ={self._zscore_ir_std:.4f}"
                )

            # Clear for next iteration
            self._gen_ic_values = []
            self._gen_ir_values = []

        def _cv_penalty(self, factor_values_dict: Dict[str, pd.Series]) -> float:
            """交叉验证过拟合惩罚"""
            if self.cv_folds < 2:
                return 0.0

            fold_ics: List[float] = []
            for stock_code, fv in factor_values_dict.items():
                ret = self.stock_pool_return_values.get(stock_code)
                if ret is None:
                    continue
                aligned = pd.DataFrame({"factor": fv, "return": ret}).dropna()
                if len(aligned) < self.cv_folds * 20:
                    continue

                n = len(aligned)
                fold_size = n // self.cv_folds
                for k in range(self.cv_folds):
                    start = k * fold_size
                    end = start + fold_size if k < self.cv_folds - 1 else n
                    segment = aligned.iloc[start:end]
                    if len(segment) >= 10:
                        ic = segment["factor"].corr(segment["return"])
                        if not np.isnan(ic):
                            fold_ics.append(abs(ic))

            if len(fold_ics) < self.cv_folds:
                return 0.0

            min_ic = min(fold_ics)
            max_ic = max(fold_ics)
            if max_ic < 1e-10:
                return 1.0
            penalty = 1.0 - (min_ic / max_ic)
            return max(0.0, min(penalty, 1.0))

        # ---------------------------------------------------------------
        # Trajectory Balance Loss
        # ---------------------------------------------------------------

        def _compute_tb_loss(self, trajectories: List[Dict]) -> torch.Tensor:
            """计算Trajectory Balance Loss

            L_TB = (log Z_θ - log R(x) + Σ log P_F(a|s))^2

            Args:
                trajectories: 轨迹列表，每个包含:
                    - log_prob: 前向策略的累积对数概率
                    - reward: 奖励值
                    - state_actions: [(state_encoded, action, valid_actions), ...]

            Returns:
                标量损失
            """
            if not trajectories:
                return torch.tensor(0.0, requires_grad=True)

            losses = []
            for traj in trajectories:
                reward = traj["reward"]
                if reward < 1e-10:
                    # 零奖励轨迹，跳过或给小奖励
                    reward = 1e-10

                log_reward = math.log(reward)

                # 重新计算前向策略的对数概率（需要梯度）
                log_pf = 0.0
                for state_encoded, action, valid_actions in traj["state_actions"]:
                    state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0)
                    mask = torch.zeros(self._n_actions, dtype=torch.bool)
                    for a in valid_actions:
                        mask[a] = True
                    mask_tensor = mask.unsqueeze(0)

                    log_probs = self.policy_net(state_tensor, mask_tensor).squeeze(0)
                    log_pf = log_pf + log_probs[action]

                # TB loss: (log Z - log R + log P_F)^2
                log_z = self.policy_net.log_z
                loss = (log_z - log_reward + log_pf) ** 2
                losses.append(loss)

            return torch.stack(losses).mean()

        # ---------------------------------------------------------------
        # Hall-of-Fame 管理
        # ---------------------------------------------------------------

        def _update_halloffame(self, candidates: List[Dict], max_size: int = 20):
            """更新Hall-of-Fame

            Args:
                candidates: 候选因子列表，每个包含 expression, fitness, placeholder_expression
                max_size: HoF最大容量
            """
            for candidate in candidates:
                if candidate["fitness"] < 1e-10:
                    continue

                # 去重检查
                expr = candidate.get("placeholder_expression", "")
                is_duplicate = False
                for hof_entry in self._halloffame:
                    sim = expression_similarity(expr, hof_entry.get("placeholder_expression", ""))
                    if sim > 0.9:
                        # 如果新表达式更好，替换旧的
                        if candidate["fitness"] > hof_entry["fitness"]:
                            self._halloffame.remove(hof_entry)
                        else:
                            is_duplicate = True
                        break

                if not is_duplicate:
                    self._halloffame.append(candidate)

            # 按适应度排序，保留top
            self._halloffame.sort(key=lambda x: x["fitness"], reverse=True)
            if len(self._halloffame) > max_size:
                self._halloffame = self._halloffame[:max_size]

        # ---------------------------------------------------------------
        # 训练循环
        # ---------------------------------------------------------------

        def mine_factors(self) -> Dict:
            """执行GFlowNet因子挖掘

            训练循环:
            1. 采样n_trajectories条公式构建轨迹
            2. 评估每条轨迹的适应度
            3. 计算Trajectory Balance Loss
            4. 更新策略网络
            5. 更新Hall-of-Fame
            6. 报告进度

            Returns:
                dict with keys: success, best_factors, fitness_history, policy_loss_history
            """
            if not GFLOWNET_AVAILABLE:
                return {"success": False, "message": "PyTorch库未安装", "best_factors": []}

            logger.info("开始GFlowNet因子挖掘...")
            logger.info(
                f"参数: n_trajectories={self.n_trajectories}, n_iterations={self.n_iterations}, "
                f"hidden_dim={self.hidden_dim}, lr={self.learning_rate}, "
                f"max_depth={self.max_expression_depth}, temperature={self.temperature}, "
                f"reward_scale={self.reward_scale}, fitness_objective={self.fitness_objective}"
            )

            fitness_history_best = []
            fitness_history_avg = []
            policy_loss_history = []

            for iteration in range(1, self.n_iterations + 1):
                # 取消检查
                if self._cancel_flag:
                    logger.info(f"GFlowNet挖掘任务在第 {iteration} 次迭代被用户取消")
                    break

                # ---- Step 1: 采样轨迹 ----
                raw_trajectories = self._sample_batch_trajectories(self.n_trajectories)

                # ---- Step 2: 评估适应度 ----
                trajectory_data = []
                best_fitness_iter = 0.0
                fitness_sum = 0.0
                n_valid = 0

                for expr_state, traj_info, log_prob in raw_trajectories:
                    if not expr_state.is_complete():
                        # 未完成的表达式，给零奖励
                        trajectory_data.append({
                            "reward": 1e-10,
                            "log_prob": log_prob,
                            "state_actions": traj_info,
                            "expression": "",
                            "placeholder_expression": "",
                            "fitness": 0.0,
                        })
                        continue

                    expr_str = expr_state.get_expression_string()
                    fitness = self._evaluate_expression(expr_state)

                    # 奖励 = 缩放后的适应度
                    reward = fitness * self.reward_scale + 1e-10

                    trajectory_data.append({
                        "reward": reward,
                        "log_prob": log_prob,
                        "state_actions": traj_info,
                        "expression": self._convert_expression_to_code(expr_str),
                        "placeholder_expression": expr_str,
                        "fitness": fitness,
                    })

                    if fitness > best_fitness_iter:
                        best_fitness_iter = fitness
                    fitness_sum += fitness
                    n_valid += 1

                # ---- Step 3: 更新Hall-of-Fame ----
                hof_candidates = [
                    {
                        "expression": t["expression"],
                        "placeholder_expression": t["placeholder_expression"],
                        "fitness": t["fitness"],
                    }
                    for t in trajectory_data
                    if t["fitness"] > 0
                ]
                self._update_halloffame(hof_candidates)

                # ---- Step 4: 经验回放 ----
                for t in trajectory_data:
                    self.replay_buffer.add(t)

                replay_samples = self.replay_buffer.sample(min(32, len(self.replay_buffer)))
                training_data = trajectory_data + replay_samples

                # ---- Step 5: 计算TB Loss并更新策略 ----
                self.optimizer.zero_grad()
                loss = self._compute_tb_loss(training_data)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()

                loss_val = loss.item()
                policy_loss_history.append(loss_val)

                avg_fitness = fitness_sum / max(n_valid, 1)
                fitness_history_best.append(best_fitness_iter)
                fitness_history_avg.append(avg_fitness)

                # ---- Step 6: 进度报告 ----
                if self.progress_callback:
                    self.progress_callback(iteration, self.n_iterations, best_fitness_iter, avg_fitness)

                logger.info(
                    f"Iteration {iteration}/{self.n_iterations} - "
                    f"Best: {best_fitness_iter:.4f}, Avg: {avg_fitness:.4f}, "
                    f"Loss: {loss_val:.4f}, HoF: {len(self._halloffame)}, "
                    f"Valid: {n_valid}/{self.n_trajectories}"
                )

                # 刷新股票评估样本
                if len(self.stock_pool_data) >= 2:
                    self._refresh_stock_sample()

                # Update Z-Score normalization stats from this iteration's evaluations
                self._update_zscore_stats()

            # ---- 构建返回结果 ----
            best_factors = []
            for i, hof_entry in enumerate(self._halloffame):
                factor_info = {
                    "rank": i + 1,
                    "expression": hof_entry["expression"],
                    "placeholder_expression": hof_entry.get("placeholder_expression", ""),
                    "fitness": hof_entry["fitness"],
                    "complexity": self._estimate_complexity(hof_entry.get("placeholder_expression", "")),
                    "source": "gflownet",
                }

                # 尝试获取详细验证结果
                try:
                    fv = self._compute_factor_from_string(hof_entry.get("placeholder_expression", ""))
                    if fv is not None and self.return_values is not None:
                        validation = factor_validation_service.validate_factor(
                            factor_values=fv,
                            return_values=self.return_values,
                        )
                        factor_info["validation"] = validation
                except Exception:
                    pass

                best_factors.append(factor_info)

            # 按验证分数排序
            def _sort_key(f):
                v = f.get("validation", {})
                if v and isinstance(v, dict):
                    return v.get("score", f.get("fitness", 0))
                return f.get("fitness", 0)

            best_factors.sort(key=_sort_key, reverse=True)
            for idx, fi in enumerate(best_factors):
                fi["rank"] = idx + 1

            logger.info(f"GFlowNet挖掘完成: 发现 {len(best_factors)} 个候选因子")

            return {
                "success": True,
                "best_factors": best_factors,
                "fitness_history": {
                    "best": fitness_history_best,
                    "average": fitness_history_avg,
                },
                "policy_loss_history": policy_loss_history,
            }

        # ---------------------------------------------------------------
        # 辅助方法
        # ---------------------------------------------------------------

        def _convert_expression_to_code(self, expr_str: str) -> str:
            """将占位符表达式转换为真实因子代码"""
            mapping = {}
            for var_name, info in self.base_factor_values.items():
                mapping[var_name] = info["code"]
            result = expr_str
            for var_name in sorted(mapping, key=len, reverse=True):
                code = mapping[var_name]
                result = result.replace(var_name, f"({code})")
            return result

        def _estimate_complexity(self, expr_str: str) -> float:
            """估算表达式复杂度（节点数）"""
            if not expr_str:
                return 0.0
            # 简单启发式：统计算子和因子数量
            op_count = sum(1 for op in ALL_OPS + EXTENDED_UNARY_OPS + EXTENDED_BINARY_OPS if op in expr_str)
            factor_count = expr_str.count("factor_")
            return float(op_count + factor_count)

    # -------------------------------------------------------------------
    # 工厂函数
    # -------------------------------------------------------------------

    def create_gflownet_mining_service(
        base_factors: List[str],
        data: pd.DataFrame,
        factor_calculator=None,
        **kwargs
    ) -> GFlowNetMiningService:
        """创建配置好的GFlowNetMiningService实例

        接受的关键字参数（转发到构造函数）:

        * ``n_trajectories`` – 每次迭代采样的轨迹数 (默认 200)
        * ``n_iterations`` – 策略训练迭代次数 (默认 50)
        * ``hidden_dim`` – MLP隐藏层维度 (默认 128)
        * ``learning_rate`` – 策略学习率 (默认 1e-3)
        * ``max_expression_depth`` – 最大表达式树深度 (默认 5)
        * ``temperature`` – 采样温度 (默认 1.0)
        * ``reward_scale`` – 奖励缩放因子 (默认 10.0)
        * ``buffer_size`` – 经验回放缓冲区大小 (默认 1000)
        * ``use_extended_primitives`` – 启用扩展算子 (默认 True)
        * ``fitness_objective`` – ic_mean / ir_ratio / sharpe / combined (默认 ic_mean)
        * ``parsimony_coeff`` – 复杂度惩罚系数 (默认 0.001)
        * ``diversity_penalty_coeff`` – 多样性惩罚系数 (默认 0.1)
        * ``cv_folds`` – 交叉验证折数 (默认 0)
        """
        return GFlowNetMiningService(
            base_factors=base_factors,
            data=data,
            factor_calculator=factor_calculator,
            **kwargs
        )
