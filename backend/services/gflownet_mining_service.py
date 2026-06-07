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

from backend.services.base_mining_service import BaseMiningService
from backend.services.factor_generator_service import factor_generator_service
from backend.services.factor_validation_service import factor_validation_service
from backend.services.alphalens_analysis_service import alphalens_analysis_service
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

    class GFlowNetMiningService(BaseMiningService):
        """GFlowNet增强因子挖掘服务

        使用GFlowNet（Generative Flow Network）学习公式构建策略，
        相比传统GP的随机变异/交叉，能够更高效地探索公式空间。

        核心流程:
        1. 策略网络采样多条公式构建轨迹
        2. 评估每条轨迹生成的因子表达式
        3. 使用Trajectory Balance Loss更新策略网络
        4. 维护Hall-of-Fame保存最优表达式
        """

        _service_name = "GFlowNet"

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

            super().__init__(
                base_factors=base_factors,
                data=data,
                return_column=return_column,
                factor_calculator=factor_calculator,
                max_eval_stocks=max_eval_stocks,
                fitness_objective=fitness_objective,
                cv_folds=cv_folds,
            )

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
            self.parsimony_coeff = parsimony_coeff
            self.diversity_penalty_coeff = diversity_penalty_coeff

            # GP原语集（用于编译表达式）
            self.pset = create_pset(
                max(len(self.base_factor_values), 1),
                extended=self.use_extended_primitives,
            )

            # Hall-of-Fame
            self._halloffame: List[Dict] = []

            # 经验回放
            self.replay_buffer = ReplayBuffer(max_size=buffer_size)

            # 初始化策略网络
            self._init_policy_network()

        # ---------------------------------------------------------------
        # 策略网络初始化
        # ---------------------------------------------------------------

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
                # 无收益率数据时，使用变异系数(CV)作为代理适应度
                cv_value = fv.std() / (fv.mean() + 1e-8)
                if np.isfinite(cv_value) and abs(fv.mean()) > 1e-8:
                    fitness = cv_value
                else:
                    # 均值接近0或CV无效时，无法评估，设为0避免误导
                    logger.warning("无收益率数据且CV代理无效，适应度设为0")
                    fitness = 0.0

            return fitness

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
