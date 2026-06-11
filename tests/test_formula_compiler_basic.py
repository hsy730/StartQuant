"""
公式编译器基础功能测试

测试 FormulaCompilerService 的核心编译和验证功能：
- 节点编译（column, literal, operation, function）
- 技术指标编译（SMA, EMA, RSI, MACD, BBANDS）
- 统计函数编译（mean, std, rank, zscore）
- 公式验证（allowed, disallowed, syntax error）
- 复杂公式编译与执行验证
"""

import numpy as np
import pandas as pd

from backend.services.formula_compiler_service import FormulaCompilerService


class TestCompileColumn:
    """列节点编译"""

    def test_compile_column(self):
        """column 节点编译为 df["close"]"""
        compiler = FormulaCompilerService()
        tree = {"type": "column", "value": "close"}
        code = compiler.compile_formula(tree)
        assert code == 'df["close"]'


class TestCompileLiteral:
    """字面量节点编译"""

    def test_compile_literal_number(self):
        """数字字面量编译为数字字符串"""
        compiler = FormulaCompilerService()
        tree = {"type": "literal", "value": 20}
        code = compiler.compile_formula(tree)
        assert code == "20"

    def test_compile_literal_string(self):
        """字符串字面量编译为带引号的字符串"""
        compiler = FormulaCompilerService()
        tree = {"type": "literal", "value": "hello"}
        code = compiler.compile_formula(tree)
        assert code == '"hello"'


class TestCompileOperation:
    """运算符节点编译"""

    def test_compile_operation(self):
        """运算符节点编译为 (left op right)"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "operation",
            "operator": "/",
            "left": {"type": "column", "value": "close"},
            "right": {"type": "literal", "value": 2},
        }
        code = compiler.compile_formula(tree)
        assert code == 'safe_series_divide(df["close"], 2)'


class TestCompileSMA:
    """SMA 编译"""

    def test_compile_sma(self):
        """SMA(close, 20) 编译正确"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "SMA",
            "args": [
                {"type": "column", "value": "close"},
                {"type": "literal", "value": 20},
            ],
        }
        code = compiler.compile_formula(tree)
        assert code == 'SMA(df["close"], timeperiod=20)'


class TestCompileEMA:
    """EMA 编译"""

    def test_compile_ema(self):
        """EMA(close, 12) 编译正确"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "EMA",
            "args": [
                {"type": "column", "value": "close"},
                {"type": "literal", "value": 12},
            ],
        }
        code = compiler.compile_formula(tree)
        assert code == 'EMA(df["close"], timeperiod=12)'


class TestCompileRSI:
    """RSI 编译"""

    def test_compile_rsi(self):
        """RSI(close, 14) 编译正确"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "RSI",
            "args": [
                {"type": "column", "value": "close"},
                {"type": "literal", "value": 14},
            ],
        }
        code = compiler.compile_formula(tree)
        assert code == 'RSI(df["close"], timeperiod=14)'


class TestCompileMACD:
    """MACD 编译"""

    def test_compile_macd_default(self):
        """MACD(close) 编译为 MACD 线 (index 0)"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "MACD",
            "args": [
                {"type": "column", "value": "close"},
            ],
        }
        code = compiler.compile_formula(tree)
        assert code == 'MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]'

    def test_compile_macd_signal(self):
        """MACD(close, 1) 编译为 signal 线"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "MACD",
            "args": [
                {"type": "column", "value": "close"},
                {"type": "literal", "value": 1},
            ],
        }
        code = compiler.compile_formula(tree)
        assert code == 'MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)[1]'


class TestCompileBBANDS:
    """BBANDS 编译"""

    def test_compile_bbands(self):
        """BBANDS(close) 编译为中轨 (middleband)"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "BBANDS",
            "args": [
                {"type": "column", "value": "close"},
            ],
        }
        code = compiler.compile_formula(tree)
        # 返回值顺序注释已标注（规则4）：TA-Lib返回(upper,middle,lower)
        assert 'BBANDS(df["close"], timeperiod=20)[1]' in code


class TestCompileStatFunctions:
    """统计函数编译"""

    def test_compile_mean(self):
        """mean(close) 编译为 df["close"].mean()"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "mean",
            "args": [{"type": "column", "value": "close"}],
        }
        code = compiler.compile_formula(tree)
        assert code == 'df["close"].mean()'

    def test_compile_std(self):
        """std(close) 编译为 df["close"].std()"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "std",
            "args": [{"type": "column", "value": "close"}],
        }
        code = compiler.compile_formula(tree)
        assert code == 'df["close"].std()'

    def test_compile_rank(self):
        """rank(close) 编译为 df["close"].rank()"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "rank",
            "args": [{"type": "column", "value": "close"}],
        }
        code = compiler.compile_formula(tree)
        assert code == 'df["close"].rank()'

    def test_compile_zscore_default(self):
        """zscore(close) 使用默认窗口 20 编译"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "zscore",
            "args": [{"type": "column", "value": "close"}],
        }
        code = compiler.compile_formula(tree)
        # 默认窗口为20
        assert "rolling(20)" in code
        assert 'df["close"]' in code

    def test_compile_zscore_custom_window(self):
        """zscore(close, 10) 使用自定义窗口 10 编译"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "function",
            "name": "zscore",
            "args": [
                {"type": "column", "value": "close"},
                {"type": "literal", "value": 10},
            ],
        }
        code = compiler.compile_formula(tree)
        assert "rolling(10)" in code
        assert 'df["close"]' in code


class TestValidateFormula:
    """公式验证"""

    def test_validate_formula_allowed(self):
        """允许的函数名通过验证"""
        compiler = FormulaCompilerService()
        # SMA 是白名单中的函数
        is_valid, msg = compiler.validate_formula('SMA(df["close"], timeperiod=20)')
        assert is_valid is True

    def test_validate_formula_disallowed(self):
        """不允许的函数名验证失败"""
        compiler = FormulaCompilerService()
        # eval 是危险函数，不在白名单中
        is_valid, msg = compiler.validate_formula('eval("print(1)")')
        assert is_valid is False
        assert "不允许的函数" in msg

    def test_validate_formula_syntax_error(self):
        """语法错误验证失败"""
        compiler = FormulaCompilerService()
        is_valid, msg = compiler.validate_formula('df["close" + ')
        assert is_valid is False
        assert "语法错误" in msg


class TestCompileComplexFormula:
    """复杂公式编译与执行验证"""

    def test_compile_complex_formula(self):
        """编译 close / SMA(close, 20) 并验证执行结果正确"""
        compiler = FormulaCompilerService()
        tree = {
            "type": "operation",
            "operator": "/",
            "left": {"type": "column", "value": "close"},
            "right": {
                "type": "function",
                "name": "SMA",
                "args": [
                    {"type": "column", "value": "close"},
                    {"type": "literal", "value": 20},
                ],
            },
        }
        code = compiler.compile_formula(tree)
        assert code == 'safe_series_divide(df["close"], SMA(df["close"], timeperiod=20))'

        # 创建测试数据并执行
        df = pd.DataFrame(
            {
                "close": np.random.seed(42) or np.cumsum(np.random.randn(50)) + 100,
            }
        )

        # 模拟 SMA 函数（使用 pandas rolling mean）
        def SMA(series, timeperiod=20):
            return series.rolling(window=timeperiod).mean()

        from backend.utils.safe_math import safe_series_divide

        local_vars = {"df": df, "SMA": SMA, "np": np, "safe_series_divide": safe_series_divide}
        result = eval(code, {"__builtins__": {}}, local_vars)

        # 手动计算期望结果（使用 safe_series_divide 的行为：零分母→NaN）
        expected = safe_series_divide(df["close"], df["close"].rolling(window=20).mean())

        pd.testing.assert_series_equal(result, expected)

        # 验证非 NaN 部分有意义
        valid_result = result.dropna()
        assert len(valid_result) > 0
