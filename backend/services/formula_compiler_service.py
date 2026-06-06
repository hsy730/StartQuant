"""
公式编译器服务 - 可视化因子构建
"""
import ast
import re
from typing import Dict, List, Optional, Any, Tuple


class FormulaCompilerService:
    """公式编译器服务类 - 将可视化公式树编译为可执行代码"""

    # 支持的因子元素
    AVAILABLE_ELEMENTS = {
        # 价格数据
        "price": {
            "open": "开盘价",
            "high": "最高价",
            "low": "最低价",
            "close": "收盘价",
            "volume": "成交量",
            "amount": "成交额",
        },
        # 技术指标
        "indicators": {
            "SMA": "简单移动平均",
            "EMA": "指数移动平均",
            "RSI": "相对强弱指标",
            "MACD": "MACD",
            "ADX": "平均趋向指标",
            "CCI": "顺势指标",
            "ATR": "平均真实波幅",
            "BBANDS": "布林带",
            "OBV": "能量潮",
        },
        # 运算符
        "operators": {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            ">": "大于",
            "<": "小于",
            ">=": "大于等于",
            "<=": "小于等于",
            "==": "等于",
        },
        # 统计函数
        "statistics": {
            "mean": "均值",
            "std": "标准差",
            "max": "最大值",
            "min": "最小值",
            "median": "中位数",
            "rank": "排名",
            "zscore": "Z-score标准化",
        },
    }

    def __init__(self):
        pass

    def compile_formula(self, formula_tree: Dict) -> str:
        """
        将公式树编译为可执行代码

        Args:
            formula_tree: 公式树结构

        Returns:
            可执行的Python代码

        公式树示例:
        {
            "type": "operation",
            "operator": "/",
            "left": {
                "type": "column",
                "value": "close"
            },
            "right": {
                "type": "function",
                "name": "SMA",
                "args": [
                    {
                        "type": "column",
                        "value": "close"
                    },
                    {
                        "type": "literal",
                        "value": 20
                    }
                ]
            }
        }
        """
        try:
            code = self._compile_node(formula_tree)

            # 包装为完整的表达式
            return code

        except Exception as e:
            raise ValueError(f"公式编译失败: {e}")

    def _compile_node(self, node: Dict) -> str:
        """递归编译节点"""
        node_type = node.get("type")

        if node_type == "column":
            # 数据列
            return f'df["{node["value"]}"]'

        elif node_type == "literal":
            # 字面量
            value = node["value"]
            if isinstance(value, str):
                return f'"{value}"'
            return str(value)

        elif node_type == "function":
            # 函数调用
            func_name = node["name"]
            args = node.get("args", [])

            # 编译参数
            compiled_args = [self._compile_node(arg) for arg in args]

            # 特殊处理技术指标
            if func_name in ["SMA", "EMA", "RSI", "MACD", "ADX", "CCI", "ATR", "BBANDS", "OBV"]:
                if func_name == "SMA":
                    return f"SMA({compiled_args[0]}, timeperiod={compiled_args[1]})"
                elif func_name == "EMA":
                    return f"EMA({compiled_args[0]}, timeperiod={compiled_args[1]})"
                elif func_name == "RSI":
                    return f"RSI({compiled_args[0]}, timeperiod={compiled_args[1]})"
                elif func_name == "MACD":
                    return f"MACD({compiled_args[0]}, fastperiod=12, slowperiod=26, signalperiod=9)[0]"
                elif func_name == "BBANDS":
                    return f"BBANDS({compiled_args[0]}, timeperiod=20)[1]"  # 返回中轨(middleband)
                elif func_name == "ATR":
                    return f"ATR({compiled_args[0]}, timeperiod={compiled_args[1]})"
                elif func_name == "OBV":
                    return f"OBV({compiled_args[0]}, {compiled_args[1]})"
                else:
                    return f"{func_name}({', '.join(compiled_args)})"
            elif func_name in ["mean", "std", "max", "min"]:
                return f'df["{compiled_args[0]}"].{func_name}()'
            elif func_name == "rank":
                return f'df["{compiled_args[0]}"].rank()'
            elif func_name == "zscore":
                # 使用滚动窗口zscore避免前视偏差，默认窗口20
                window = compiled_args[1] if len(compiled_args) > 1 else 20
                return f'(df["{compiled_args[0]}"] - df["{compiled_args[0]}"].rolling({window}).mean()) / df["{compiled_args[0]}"].rolling({window}).std().replace(0, np.nan)'
            else:
                return f"{func_name}({', '.join(compiled_args)})"

        elif node_type == "operation":
            # 运算符
            operator = node["operator"]
            left = self._compile_node(node["left"])
            right = self._compile_node(node["right"])

            return f"({left} {operator} {right})"

        else:
            raise ValueError(f"未知的节点类型: {node_type}")

    def validate_formula(self, formula_code: str) -> Tuple[bool, str]:
        """
        验证公式代码

        Args:
            formula_code: 公式代码

        Returns:
            (是否有效, 错误消息)
        """
        try:
            # 尝试编译代码
            if formula_code.strip().startswith("def "):
                # 函数形式
                compile(formula_code, '<string>', 'exec')
            else:
                # 表达式形式
                # 提供测试上下文
                test_context = {
                    "df": None,
                    "SMA": lambda x, **kwargs: x,
                    "EMA": lambda x, **kwargs: x,
                    "RSI": lambda x, **kwargs: x,
                }
                # 只验证语法，不执行
                ast.parse(formula_code, mode='eval')

            return True, "公式验证通过"

        except SyntaxError as e:
            return False, f"语法错误: {e}"
        except Exception as e:
            return False, f"验证失败: {e}"

    def parse_expression(self, expression: str) -> Dict:
        """
        解析表达式为公式树

        Args:
            expression: 表达式字符串

        Returns:
            公式树
        """
        try:
            # 使用AST解析表达式
            tree = ast.parse(expression, mode='eval')
            return self._ast_to_formula_tree(tree.body)

        except Exception as e:
            raise ValueError(f"表达式解析失败: {e}")

    def _ast_to_formula_tree(self, node: ast.AST) -> Dict:
        """将AST节点转换为公式树节点"""
        if isinstance(node, ast.Name):
            # 变量名（如 close, open）
            return {"type": "column", "value": node.id}

        elif isinstance(node, ast.Constant):
            # 常量
            return {"type": "literal", "value": node.value}

        elif isinstance(node, ast.BinOp):
            # 二元运算
            return {
                "type": "operation",
                "operator": self._ast_op_to_str(node.op),
                "left": self._ast_to_formula_tree(node.left),
                "right": self._ast_to_formula_tree(node.right),
            }

        elif isinstance(node, ast.Call):
            # 函数调用
            func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
            args = [self._ast_to_formula_tree(arg) for arg in node.args]

            return {
                "type": "function",
                "name": func_name,
                "args": args,
            }

        else:
            raise ValueError(f"不支持的AST节点类型: {type(node)}")

    def _ast_op_to_str(self, op: ast.operator) -> str:
        """将AST运算符转换为字符串"""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Gt: ">",
            ast.Lt: "<",
            ast.GtE: ">=",
            ast.LtE: "<=",
            ast.Eq: "==",
        }
        return op_map.get(type(op), str(op))

    def get_available_elements(self) -> Dict:
        """
        获取可用的因子元素

        Returns:
            可用元素字典
        """
        return self.AVAILABLE_ELEMENTS

    def simplify_formula(self, formula_code: str) -> str:
        """
        简化公式代码

        Args:
            formula_code: 原始公式代码

        Returns:
            简化后的代码
        """
        # 移除多余的空行
        lines = [line.strip() for line in formula_code.split("\n") if line.strip()]
        return "\n".join(lines)


# 全局公式编译器服务实例
formula_compiler_service = FormulaCompilerService()
