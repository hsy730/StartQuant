"""
因子生成器服务 - 基于预置因子生成新因子
"""

from typing import List, Dict
import pandas as pd
import numpy as np
import random
from itertools import combinations
from scipy.stats import spearmanr

from backend.utils.safe_math import safe_ir


class FactorGeneratorService:
    """因子生成器服务"""

    def __init__(self):
        # 可用的运算符（扩展）
        self.operators = {
            "+": "加法",
            "-": "减法",
            "*": "乘法",
            "/": "除法",
            "**": "幂运算",
            "%": "取模",
        }

        # 可用的统计函数（扩展）
        self.statistics = {
            "rank": "排名",
            "zscore": "Z-score标准化",
            "mean": "均值",
            "std": "标准差",
            "max": "最大值",
            "min": "最小值",
            "median": "中位数",
            "skew": "偏度",
            "kurtosis": "峰度",
            "quantile": "分位数",
            "diff": "差分",
            "pct_change": "百分比变化",
            "log": "对数变换",
            "abs": "绝对值",
            "sqrt": "平方根",
            "exp": "指数",
        }

        # 可用的技术指标（扩展）
        self.indicators = {
            "SMA": "简单移动平均",
            "EMA": "指数移动平均",
            "RSI": "相对强弱指标",
            "MACD": "MACD",
            "BBANDS": "布林带",
            "STOCH": "随机指标",
            "ADX": "平均趋向指数",
            "CCI": "顺势指标",
            "ATR": "真实波幅",
            "VOLATILITY": "波动率",
        }

    def generate_binary_combinations(
        self, base_factors: List[str], max_depth: int = 3, max_combinations: int = 100
    ) -> List[str]:
        """
        生成二元运算组合因子

        Args:
            base_factors: 基础因子列表
            max_depth: 最大深度（嵌套层数）
            max_combinations: 最大组合数

        Returns:
            因子表达式列表
        """
        expressions = []

        if len(base_factors) < 2:
            return expressions

        # 生成深度1的组合
        for factor1, factor2 in combinations(base_factors, 2):
            for op in self.operators.keys():
                if op == "/":
                    expr = f"safe_divide({factor1}, {factor2}, default=np.nan)"
                else:
                    expr = f"({factor1} {op} {factor2})"
                expressions.append(expr)

        # 生成深度2的组合（如果需要）
        if max_depth >= 2 and len(base_factors) >= 3:
            for _ in range(min(max_combinations // 2, 50)):  # 限制数量
                # 随机选择3个因子
                selected = random.sample(base_factors, min(3, len(base_factors)))
                ops = random.sample(list(self.operators.keys()), 2)

                # 生成嵌套表达式
                expr = f"(({selected[0]} {ops[0]} {selected[1]}) {ops[1]} {selected[2]})"
                expressions.append(expr)

        # 生成深度3的组合（如果需要）
        if max_depth >= 3 and len(base_factors) >= 4:
            for _ in range(min(max_combinations // 4, 25)):  # 限制数量
                selected = random.sample(base_factors, min(4, len(base_factors)))
                ops = random.sample(list(self.operators.keys()), 3)

                expr = f"(({selected[0]} {ops[0]} {selected[1]}) {ops[1]} ({selected[2]} {ops[2]} {selected[3]}))"
                expressions.append(expr)

        return expressions[:max_combinations]

    def generate_statistical_combinations(
        self, base_factors: List[str], window_sizes: List[int] = [5, 10, 20, 60], max_combinations: int = 50
    ) -> List[str]:
        """
        生成统计函数组合因子

        Args:
            base_factors: 基础因子列表
            window_sizes: 窗口大小列表
            max_combinations: 最大组合数

        Returns:
            因子表达式列表（使用pandas链式调用语法）
        """
        expressions = []

        # 需要窗口参数的函数（使用rolling）
        window_functions = {
            "mean": "mean()",
            "std": "std()",
            "max": "max()",
            "min": "min()",
            "median": "median()",
            "skew": "skew()",
            "kurtosis": "kurtosis()",
        }

        # 不需要窗口参数的函数
        no_window_functions = {
            "diff": "diff()",
            "pct_change": "pct_change()",
            "abs": "abs()",
        }

        # 特殊处理的函数
        special_functions = {
            "rank": "rank(pct=True)",
            "log": "np.log",
            "sqrt": "np.sqrt",
            "exp": "np.exp",
            "zscore": None,  # 需要特殊处理
        }

        for factor in base_factors:
            for stat_func in self.statistics.keys():
                if stat_func in window_functions:
                    # 这些函数使用rolling
                    for window in window_sizes:
                        # 生成pandas链式调用语法: factor.rolling(window).mean()
                        expr = f"({factor}.rolling({window}, min_periods=1).{window_functions[stat_func]})"
                        expressions.append(expr)
                elif stat_func in special_functions:
                    if special_functions[stat_func] is not None:
                        # 生成pandas/numpy函数调用语法
                        expr = f"{special_functions[stat_func]}({factor})"
                        expressions.append(expr)
                    elif stat_func == "zscore":
                        # zscore需要特殊处理: (x - mean) / std，使用safe_divide防止除零
                        expr = (
                            f"safe_divide({factor} - {factor}.rolling(252, min_periods=1).mean(), "
                            f"{factor}.rolling(252, min_periods=1).std(), default=np.nan)"
                        )
                        expressions.append(expr)
                elif stat_func in no_window_functions:
                    # 直接调用方法
                    expr = f"({factor}.{no_window_functions[stat_func]})"
                    expressions.append(expr)
                elif stat_func == "quantile":
                    # 分位数函数
                    for q in [0.25, 0.5, 0.75]:
                        expr = f"({factor}.rolling(252, min_periods=1).quantile({q}))"
                        expressions.append(expr)

        return expressions[:max_combinations]

    def generate_indicator_combinations(
        self, base_factors: List[str], price_column: str = "close", max_combinations: int = 30
    ) -> List[str]:
        """
        生成技术指标组合因子

        Args:
            base_factors: 基础因子列表
            price_column: 价格列名
            max_combinations: 最大组合数

        Returns:
            因子表达式列表
        """
        expressions = []

        # 为每个基础因子生成与技术指标的组合
        for factor in base_factors:
            for indicator in self.indicators.keys():
                if indicator == "SMA":
                    for window in [5, 10, 20, 60]:
                        expr = f"safe_divide({factor}, SMA({price_column}, {window}), default=np.nan)"
                        expressions.append(expr)
                        expr = f"({factor} - SMA({price_column}, {window}))"
                        expressions.append(expr)
                elif indicator == "EMA":
                    for window in [5, 10, 20, 60]:
                        expr = f"safe_divide({factor}, EMA({price_column}, {window}), default=np.nan)"
                        expressions.append(expr)
                elif indicator == "RSI":
                    expr = f"({factor} * RSI({price_column}, 14))"
                    expressions.append(expr)
                elif indicator == "MACD":
                    expr = f"({factor} * MACD({price_column}))"
                    expressions.append(expr)

        return expressions[:max_combinations]

    def generate_hybrid_factors(self, base_factors: List[str], n_factors: int = 100) -> List[Dict]:
        """
        生成混合因子（结合多种方法）

        Args:
            base_factors: 基础因子列表
            n_factors: 生成因子数量

        Returns:
            因子字典列表，包含表达式和元数据
        """
        factors = []

        # 1. 二元运算组合（40%）
        n_binary = int(n_factors * 0.4)
        binary_exprs = self.generate_binary_combinations(base_factors, max_combinations=n_binary)

        for expr in binary_exprs:
            factors.append(
                {
                    "expression": expr,
                    "type": "binary_operation",
                    "complexity": "medium",
                }
            )

        # 2. 统计函数组合（30%）
        n_statistical = int(n_factors * 0.3)
        stat_exprs = self.generate_statistical_combinations(base_factors, max_combinations=n_statistical)

        for expr in stat_exprs:
            factors.append(
                {
                    "expression": expr,
                    "type": "statistical",
                    "complexity": "low",
                }
            )

        # 3. 技术指标组合（20%）
        n_indicator = int(n_factors * 0.2)
        indicator_exprs = self.generate_indicator_combinations(base_factors, max_combinations=n_indicator)

        for expr in indicator_exprs:
            factors.append(
                {
                    "expression": expr,
                    "type": "indicator_based",
                    "complexity": "high",
                }
            )

        # 4. 随机组合（10%）
        n_random = n_factors - len(factors)

        for _ in range(n_random):
            if len(base_factors) >= 2:
                factor1, factor2 = random.sample(base_factors, 2)
                op = random.choice(list(self.operators.keys()))

                # 随机添加统计函数
                if random.random() < 0.3:
                    stat_func = random.choice(list(self.statistics.keys()))
                    if stat_func in ["mean", "std", "max", "min"]:
                        window = random.choice([5, 10, 20])
                        expr = f"{stat_func}({factor1} {op} {factor2}, {window})"
                    else:
                        expr = f"{stat_func}({factor1} {op} {factor2})"
                else:
                    expr = f"({factor1} {op} {factor2})"

                factors.append(
                    {
                        "expression": expr,
                        "type": "random_hybrid",
                        "complexity": random.choice(["low", "medium", "high"]),
                    }
                )

        # 打乱顺序
        random.shuffle(factors)

        return factors[:n_factors]

    def compile_expression_to_code(self, expression: str, data_column: str = "close") -> str:
        """
        将因子表达式编译为可执行代码

        Args:
            expression: 因子表达式，如 mean(close, 20) / std(close, 20)
            data_column: 数据列名

        Returns:
            可执行的Python代码
        """

        # 使用正则进行函数替换，避免简单字符串替换的互相干扰
        # 定义函数映射表（按函数名长度降序排列，避免短名匹配长名）
        # 格式: (函数名, 替换模板, 是否需要特殊处理)
        func_map = [
            # 技术指标 → talib
            ("BBANDS", "talib.BBANDS({args})"),
            ("STOCH", "talib.STOCH({args})"),
            ("MACD", "talib.MACD({args})"),
            ("SMA", "talib.SMA({args})"),
            ("EMA", "talib.EMA({args})"),
            ("RSI", "talib.RSI({args})"),
            ("ADX", "talib.ADX({args})"),
            ("CCI", "talib.CCI({args})"),
            ("ATR", "talib.ATR({args})"),
            # 滚动统计函数 → df.rolling().func()
            ("kurtosis", "df['{col}'].rolling(window=252, min_periods=1).kurtosis()"),
            ("quantile", "df['{col}'].rolling(window=252, min_periods=1).quantile({args})"),
            ("median", "df['{col}'].rolling(window=252, min_periods=1).median()"),
            ("skew", "df['{col}'].rolling(window=252, min_periods=1).skew()"),
            ("std", "df['{col}'].rolling(window=252, min_periods=1).std()"),
            ("mean", "df['{col}'].rolling(window=252, min_periods=1).mean()"),
            ("max", "df['{col}'].rolling(window=252, min_periods=1).max()"),
            ("min", "df['{col}'].rolling(window=252, min_periods=1).min()"),
            # 排名函数
            ("rank", "df['{col}'].rolling(252, min_periods=1).rank(pct=True)"),
            # 差分/变化率
            ("diff", "df['{col}'].diff({args})"),
            ("pct_change", "df['{col}'].pct_change({args})"),
            # 数学函数
            ("log", "np.log({args})"),
            ("abs", "np.abs({args})"),
            ("sqrt", "np.sqrt({args})"),
            ("exp", "np.exp({args})"),
            # zscore 特殊处理
            ("zscore", None),  # 手动处理
        ]

        # 按函数名长度降序排列，确保最长匹配优先（避免短名误匹配长名前缀）
        func_map.sort(key=lambda x: len(x[0]), reverse=True)

        # 用栈解析表达式，逐层替换函数调用
        def split_args_by_comma(args_str: str) -> list:
            """按逗号分割参数列表，尊重括号嵌套"""
            parts = []
            depth = 0
            current = []
            for ch in args_str:
                if ch == "(":
                    depth += 1
                    current.append(ch)
                elif ch == ")":
                    depth -= 1
                    current.append(ch)
                elif ch == "," and depth == 0:
                    parts.append("".join(current).strip())
                    current = []
                else:
                    current.append(ch)
            if current:
                parts.append("".join(current).strip())
            return parts

        # 已知的非列名标识符（Python 关键字、内置函数、模块名等）
        _non_col_identifiers = {
            "True",
            "False",
            "None",
            "and",
            "or",
            "not",
            "in",
            "is",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "with",
            "as",
            "def",
            "class",
            "return",
            "import",
            "from",
            "try",
            "except",
            "finally",
            "raise",
            "pass",
            "break",
            "continue",
            "lambda",
            "yield",
            "del",
            "global",
            "nonlocal",
            "assert",
        }

        def parse_and_replace(expr: str) -> str:
            """递归解析表达式并替换函数调用"""
            expr = expr.strip()

            # 匹配函数调用: func_name(args)
            # 从最外层函数开始解析
            i = 0
            while i < len(expr):
                if expr[i].isalpha() or expr[i] == "_":
                    # 找到函数名
                    j = i
                    while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                        j += 1
                    func_name = expr[i:j]
                    # 跳过空白
                    k = j
                    while k < len(expr) and expr[k] == " ":
                        k += 1
                    if k < len(expr) and expr[k] == "(":
                        # 找到匹配的右括号
                        paren_depth = 1
                        m = k + 1
                        while m < len(expr) and paren_depth > 0:
                            if expr[m] == "(":
                                paren_depth += 1
                            elif expr[m] == ")":
                                paren_depth -= 1
                            m += 1
                        args = expr[k + 1 : m - 1]  # noqa: E203
                        # 递归解析参数
                        parsed_args = parse_and_replace(args)
                        # 替换函数
                        prefix = expr[:i]
                        suffix = expr[m:]
                        # 查找函数映射
                        replaced = False
                        for fname, template in func_map:
                            if fname == func_name and template is not None:
                                if "{col}" in template:
                                    # 含 {col} 的模板：第一个参数是数据源，其余参数是函数参数
                                    arg_list = split_args_by_comma(parsed_args)
                                    # 确定数据源：第一个参数
                                    if arg_list:
                                        first_arg = arg_list[0].strip()
                                        # 判断是否为简单列名（纯标识符）
                                        is_simple_col = (
                                            first_arg.isidentifier()
                                            and not first_arg.startswith("np.")
                                            and not first_arg.startswith("df[")
                                        )
                                        if is_simple_col:
                                            data_source = f"df['{first_arg}']"
                                        else:
                                            data_source = f"({first_arg})"
                                    else:
                                        data_source = f"df['{data_column}']"
                                    # 剩余参数（除第一个外的所有参数）
                                    remaining_args = ", ".join(arg_list[1:]) if len(arg_list) > 1 else ""
                                    # 确定窗口大小：第二个参数（如果存在且为数字），用于 rolling 函数
                                    window = 252
                                    if len(arg_list) >= 2:
                                        try:
                                            window = int(arg_list[1].strip())
                                        except (ValueError, TypeError):
                                            window = 252
                                    # 构建替换：先替换 df['{col}'] 整体为 data_source，再替换其余占位符
                                    # 必须先替换 df['{col}'] 再替换 {col}，避免展开后无法匹配多引用场景
                                    replacement = template.replace("df['{col}']", data_source)
                                    replacement = replacement.replace("{col}", data_column)
                                    replacement = replacement.replace("{args}", remaining_args)
                                    if "window=252" in replacement and window != 252:
                                        replacement = replacement.replace("window=252", f"window={window}", 1)
                                else:
                                    replacement = template.replace("{args}", parsed_args)
                                replaced = True
                                break
                        if not replaced:
                            # 未映射的函数，保持原样
                            replacement = f"{func_name}({parsed_args})"
                        return prefix + replacement + parse_and_replace(suffix)
                    else:
                        # 标识符后面没有 '('，不是函数调用
                        # 如果是裸列名（非关键字、非已替换表达式），转为 df['...'] 引用
                        if (
                            func_name not in _non_col_identifiers
                            and not func_name.startswith("np.")
                            and not func_name.startswith("df[")
                            and not func_name.startswith("talib.")
                        ):
                            col_ref = f"df['{func_name}']"
                            return expr[:i] + col_ref + parse_and_replace(expr[j:])
                        i = j
                else:
                    i += 1
            return expr

        # 如果表达式是 zscore(expr)，特殊处理
        if expression.strip().startswith("zscore("):
            inner = expression.strip()[len("zscore(") : -1]  # noqa: E203
            parsed_inner = parse_and_replace(inner)
            # zscore: (x - mean) / std，使用safe_divide防止除零
            code = (
                f"safe_divide({parsed_inner} - ({parsed_inner}).rolling(252, min_periods=1).mean(), "
                f"({parsed_inner}).rolling(252, min_periods=1).std(), "
                f"default=np.nan)"
            )
        else:
            code = parse_and_replace(expression)

        # 包装成完整的代码
        full_code = f"""
import talib
import pandas as pd
import numpy as np
from backend.utils.safe_math import safe_divide

def calculate_factor(df):
    '''计算因子: {expression}'''

    # 确保有必要的列
    if '{data_column}' not in df.columns:
        raise ValueError("数据中缺少 '{data_column}' 列")

    # 计算因子
    try:
        factor = {code}
        return factor
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"计算因子时出错: {{e}}")
        return pd.Series(index=df.index, dtype=float)
"""

        return full_code

    def validate_expression(self, expression: str) -> tuple[bool, str]:
        """
        验证因子表达式是否有效

        Args:
            expression: 因子表达式

        Returns:
            (是否有效, 错误信息)
        """
        # 基本语法检查
        if not expression or expression.strip() == "":
            return False, "表达式为空"

        # 检查括号匹配
        if expression.count("(") != expression.count(")"):
            return False, "括号不匹配"

        # 检查是否有非法字符
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/()., _")
        for char in expression:
            if char not in allowed_chars:
                return False, f"包含非法字符: {char}"

        # 检查是否有运算符
        has_operator = any(op in expression for op in ["+", "-", "*", "/"])
        has_function = any(func in expression for func in self.statistics.keys())

        if not (has_operator or has_function):
            return False, "表达式缺少运算符或函数"

        return True, ""

    def parse_expression(self, expression: str) -> Dict:
        """
        解析因子表达式，提取结构

        Args:
            expression: 因子表达式

        Returns:
            解析后的结构信息
        """
        structure = {
            "expression": expression,
            "components": [],
            "operators": [],
            "functions": [],
            "depth": 0,
        }

        # 提取运算符
        for op in self.operators.keys():
            if op in expression:
                structure["operators"].append(op)

        # 提取函数
        for func in self.statistics.keys():
            if f"{func}(" in expression:
                structure["functions"].append(func)

        for func in self.indicators.keys():
            if f"{func}(" in expression:
                structure["functions"].append(func)

        # 计算深度（括号嵌套层数）
        max_depth = 0
        current_depth = 0
        for char in expression:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth -= 1

        structure["depth"] = max_depth

        return structure

    def preselect_factors(
        self,
        factors: List[Dict],
        factor_data_map: Dict[str, pd.Series],
        return_data: pd.Series,
        ic_threshold: float = 0.03,
        ir_threshold: float = 0.5,
        min_valid_ratio: float = 0.7,
    ) -> List[Dict]:
        """
        预筛选因子 - 根据IC、IR等指标筛选有潜力的因子

        Args:
            factors: 因子字典列表
            factor_data_map: 因子数据字典 {factor_name: factor_series}
            return_data: 收益率数据
            ic_threshold: IC阈值（绝对值）
            ir_threshold: IR阈值
            min_valid_ratio: 最小有效数据比例

        Returns:
            筛选后的因子列表
        """
        selected_factors = []

        for factor_info in factors:
            expression = factor_info["expression"]
            _factor_name = f"factor_{len(selected_factors)}"  # noqa: F841

            if expression in factor_data_map:
                factor_values = factor_data_map[expression]

                # 对齐数据
                aligned_data = pd.DataFrame({"factor": factor_values, "return": return_data}).dropna()

                # 检查数据比例
                valid_ratio = len(aligned_data) / len(factor_values) if len(factor_values) > 0 else 0.0
                if valid_ratio < min_valid_ratio:
                    continue

                # 计算IC（Spearman秩相关，与业界标准一致）
                valid = aligned_data["factor"].notna() & aligned_data["return"].notna()
                if valid.sum() < 5:
                    continue
                ic, _ = spearmanr(aligned_data["factor"][valid], aligned_data["return"][valid])
                if pd.isna(ic):
                    ic = 0.0

                if pd.isna(ic) or abs(ic) < ic_threshold:
                    continue

                # 计算IR（IC均值/IC标准差）- 使用滚动Spearman IC
                window = 20
                min_periods = 10
                factor_vals = aligned_data["factor"]
                return_vals = aligned_data["return"]

                def _rolling_spearman(x):
                    y_aligned = return_vals.loc[x.index]
                    valid = x.notna() & y_aligned.notna()
                    if valid.sum() < min_periods:
                        return np.nan
                    r, _ = spearmanr(x[valid], y_aligned[valid])
                    return r

                rolling_ic_series = factor_vals.rolling(window=window, min_periods=min_periods).apply(
                    _rolling_spearman, raw=False
                )
                rolling_ic_values = rolling_ic_series.tolist()

                if rolling_ic_values:
                    ic_mean = np.nanmean(rolling_ic_values)
                    ic_std = np.nanstd(rolling_ic_values)

                    ir = safe_ir(float(ic_mean), float(ic_std), default=None)
                    if ir is None:
                        # Rule 7.10: IC_std≈0 且 IC_mean≠0 → IR→∞，因子极其稳定
                        if abs(float(ic_mean)) > 1e-10:
                            ir = float("inf")
                        else:
                            ir = 0
                else:
                    ir = 0

                if ir < ir_threshold:
                    continue

                # 通过筛选
                factor_info_copy = factor_info.copy()
                factor_info_copy["ic"] = float(ic)
                factor_info_copy["ir"] = float(ir) if ir is not None and np.isfinite(ir) else None
                factor_info_copy["valid_ratio"] = float(valid_ratio)
                selected_factors.append(factor_info_copy)

        return selected_factors

    def calculate_factor_metrics(self, factor_values: pd.Series, return_values: pd.Series) -> Dict:
        """
        计算因子的质量指标

        Args:
            factor_values: 因子值序列
            return_values: 收益率序列

        Returns:
            质量指标字典
        """
        # 对齐数据
        aligned_data = pd.DataFrame({"factor": factor_values, "return": return_values}).dropna()

        if len(aligned_data) < 10:
            return {"valid": False, "message": "数据不足"}

        # 计算IC（Spearman秩相关，与业界标准一致）
        valid = aligned_data["factor"].notna() & aligned_data["return"].notna()
        if valid.sum() < 5:
            return {"valid": False, "message": "有效数据不足"}
        ic, _ = spearmanr(aligned_data["factor"][valid], aligned_data["return"][valid])
        if pd.isna(ic):
            ic = 0.0

        # 计算滚动IR - 使用滚动Spearman IC
        window = 20
        min_periods = 10
        factor_vals = aligned_data["factor"]
        return_vals = aligned_data["return"]

        def _rolling_spearman(x):
            y_aligned = return_vals.loc[x.index]
            v = x.notna() & y_aligned.notna()
            if v.sum() < min_periods:
                return np.nan
            r, _ = spearmanr(x[v], y_aligned[v])
            return r

        rolling_ic_series = factor_vals.rolling(window=window, min_periods=min_periods).apply(
            _rolling_spearman, raw=False
        )

        rolling_ic = rolling_ic_series

        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()
        ir = safe_ir(float(ic_mean), float(ic_std), default=None)

        # 计算胜率
        ic_win_rate = (rolling_ic > 0).sum() / rolling_ic.count() if rolling_ic.count() > 0 else 0

        # 计算因子分布特征
        factor_stats = {
            "mean": float(aligned_data["factor"].mean()),
            "std": float(aligned_data["factor"].std()),
            "skew": float(aligned_data["factor"].skew()),
            "kurtosis": float(aligned_data["factor"].kurtosis()),
        }

        return {
            "valid": True,
            "ic": float(ic),
            "ir": float(ir) if ir is not None else None,
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "ic_win_rate": float(ic_win_rate),
            "n_obs": len(aligned_data),
            "factor_stats": factor_stats,
        }


# 全局因子生成器服务实例
factor_generator_service = FactorGeneratorService()
