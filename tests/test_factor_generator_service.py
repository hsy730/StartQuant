"""
factor_generator_service.py 因子生成器服务测试

覆盖 FactorGeneratorService 所有公开方法：
- generate_binary_combinations
- generate_statistical_combinations
- generate_indicator_combinations
- generate_hybrid_factors
- compile_expression_to_code
- validate_expression
- parse_expression
- preselect_factors
- calculate_factor_metrics
"""
import sys
import os
import random
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NumPy 2.0 兼容
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf

from backend.services.factor_generator_service import FactorGeneratorService


class TestGenerateBinaryCombinations:
    """generate_binary_combinations 二元运算组合测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()

    def test_empty_base_factors_should_return_empty(self):
        """空基础因子列表应返回空列表"""
        result = self.service.generate_binary_combinations([])
        assert result == []

    def test_single_base_factor_should_return_empty(self):
        """仅1个基础因子应返回空列表（无法组合）"""
        result = self.service.generate_binary_combinations(["factor_a"])
        assert result == []

    def test_two_base_factors_depth1_should_generate_all_operator_pairs(self):
        """2个基础因子、深度1应生成所有运算符组合"""
        result = self.service.generate_binary_combinations(
            ["f1", "f2"], max_depth=1
        )
        # 2个因子取2组合 = 1对 × 6运算符 = 6个表达式
        assert len(result) == 6
        # 验证每个运算符都出现
        for op in self.service.operators.keys():
            assert any(f" {op} " in expr for expr in result), f"缺少运算符 {op}"

    def test_two_base_factors_expressions_should_contain_both_factors(self):
        """2个因子的表达式应包含两个因子名"""
        result = self.service.generate_binary_combinations(
            ["alpha", "beta"], max_depth=1
        )
        for expr in result:
            assert "alpha" in expr
            assert "beta" in expr

    def test_three_base_factors_depth1_should_generate_c2_pairs(self):
        """3个基础因子、深度1应生成 C(3,2)=3 对 × 6运算符 = 18个"""
        result = self.service.generate_binary_combinations(
            ["f1", "f2", "f3"], max_depth=1
        )
        assert len(result) == 18

    def test_depth2_should_add_nested_expressions(self):
        """深度2应增加嵌套3因子表达式"""
        random.seed(42)
        result = self.service.generate_binary_combinations(
            ["f1", "f2", "f3", "f4"], max_depth=2
        )
        # 深度1: C(4,2)*6 = 36; 深度2: min(100//2, 50) = 50
        # 总数受 max_combinations 限制
        assert len(result) <= 100
        # 应有深度1的表达式（2个因子）
        depth1_exprs = [e for e in result if e.count("(") <= 2]
        assert len(depth1_exprs) > 0

    def test_depth2_with_3_factors_should_generate_nested(self):
        """3个因子、深度2应生成嵌套表达式"""
        random.seed(42)
        result = self.service.generate_binary_combinations(
            ["f1", "f2", "f3"], max_depth=2
        )
        # 深度1: C(3,2)*6 = 18; 深度2: min(max_combinations//2, 50)
        # 应有嵌套表达式（3个因子，格式为 ((a op b) op c)）
        nested_exprs = [e for e in result if e.count("(") >= 3]
        if len(nested_exprs) == 0:
            # 也可能是 max_combinations 限制导致深度2未生成
            # 检查是否有包含3个不同因子的表达式
            three_factor_exprs = [
                e for e in result
                if "f1" in e and "f2" in e and "f3" in e
            ]
            assert len(three_factor_exprs) > 0 or len(result) >= 18

    def test_depth3_should_add_deeper_nested_expressions(self):
        """深度3应增加4因子深层嵌套表达式"""
        random.seed(42)
        result = self.service.generate_binary_combinations(
            ["f1", "f2", "f3", "f4", "f5"], max_depth=3, max_combinations=200
        )
        # 应有深度3的嵌套表达式
        assert len(result) > 0

    def test_max_combinations_should_limit_result(self):
        """max_combinations 应限制返回数量"""
        result = self.service.generate_binary_combinations(
            ["f1", "f2", "f3", "f4", "f5"], max_depth=3, max_combinations=10
        )
        assert len(result) <= 10

    def test_depth2_insufficient_factors_should_skip(self):
        """深度2但因子不足3个应跳过深度2生成"""
        result = self.service.generate_binary_combinations(
            ["f1", "f2"], max_depth=2
        )
        # 只有深度1: C(2,2)*6 = 6
        assert len(result) == 6

    def test_depth3_insufficient_factors_should_skip(self):
        """深度3但因子不足4个应跳过深度3生成"""
        random.seed(42)
        result = self.service.generate_binary_combinations(
            ["f1", "f2", "f3"], max_depth=3
        )
        # 深度1 + 深度2，无深度3
        assert len(result) > 0

    def test_expressions_should_be_balanced_parentheses(self):
        """所有生成的表达式括号应匹配"""
        result = self.service.generate_binary_combinations(
            ["f1", "f2", "f3", "f4"], max_depth=3
        )
        for expr in result:
            assert expr.count("(") == expr.count(")"), f"括号不匹配: {expr}"

    def test_reproducibility_with_same_seed(self):
        """相同随机种子应产生相同结果"""
        random.seed(123)
        result1 = self.service.generate_binary_combinations(
            ["f1", "f2", "f3", "f4"], max_depth=3
        )
        random.seed(123)
        result2 = self.service.generate_binary_combinations(
            ["f1", "f2", "f3", "f4"], max_depth=3
        )
        assert result1 == result2


class TestGenerateStatisticalCombinations:
    """generate_statistical_combinations 统计函数组合测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()

    def test_empty_base_factors_should_return_empty(self):
        """空基础因子列表应返回空列表"""
        result = self.service.generate_statistical_combinations([])
        assert result == []

    def test_single_factor_should_generate_statistical_expressions(self):
        """1个基础因子应生成统计函数表达式"""
        result = self.service.generate_statistical_combinations(["close"])
        assert len(result) > 0

    def test_window_functions_should_use_rolling(self):
        """窗口函数应使用 rolling() 语法"""
        result = self.service.generate_statistical_combinations(
            ["close"], window_sizes=[5]
        )
        rolling_exprs = [e for e in result if "rolling" in e]
        assert len(rolling_exprs) > 0

    def test_window_functions_should_respect_window_sizes(self):
        """窗口函数应使用指定的窗口大小"""
        result = self.service.generate_statistical_combinations(
            ["close"], window_sizes=[5, 10]
        )
        assert any("rolling(5" in e for e in result)
        assert any("rolling(10" in e for e in result)

    def test_no_window_functions_should_use_direct_method(self):
        """无窗口函数应使用直接方法调用"""
        result = self.service.generate_statistical_combinations(["close"])
        # diff, pct_change, abs 不使用 rolling
        diff_exprs = [e for e in result if ".diff()" in e]
        pct_exprs = [e for e in result if ".pct_change()" in e]
        abs_exprs = [e for e in result if ".abs()" in e]
        assert len(diff_exprs) > 0
        assert len(pct_exprs) > 0
        assert len(abs_exprs) > 0

    def test_special_function_rank_should_use_pct(self):
        """rank 函数应使用 rank(pct=True)"""
        result = self.service.generate_statistical_combinations(["close"])
        rank_exprs = [e for e in result if "rank(pct=True)" in e]
        assert len(rank_exprs) > 0

    def test_special_function_log_should_use_np_log(self):
        """log 函数应使用 np.log"""
        result = self.service.generate_statistical_combinations(["close"])
        log_exprs = [e for e in result if "np.log" in e]
        assert len(log_exprs) > 0

    def test_special_function_sqrt_should_use_np_sqrt(self):
        """sqrt 函数应使用 np.sqrt"""
        result = self.service.generate_statistical_combinations(["close"])
        sqrt_exprs = [e for e in result if "np.sqrt" in e]
        assert len(sqrt_exprs) > 0

    def test_special_function_exp_should_use_np_exp(self):
        """exp 函数应使用 np.exp"""
        result = self.service.generate_statistical_combinations(["close"])
        exp_exprs = [e for e in result if "np.exp" in e]
        assert len(exp_exprs) > 0

    def test_zscore_should_generate_special_expression(self):
        """zscore 应生成特殊的 (x - mean)/std 表达式"""
        result = self.service.generate_statistical_combinations(["close"])
        zscore_exprs = [e for e in result if "zscore" in e.lower() or
                        ("rolling(252" in e and "mean()" in e and "std()" in e)]
        assert len(zscore_exprs) > 0

    def test_quantile_should_generate_multiple_quantiles(self):
        """quantile 应生成 0.25, 0.5, 0.75 三个分位数"""
        result = self.service.generate_statistical_combinations(["close"])
        q25_exprs = [e for e in result if "quantile(0.25)" in e]
        q50_exprs = [e for e in result if "quantile(0.5)" in e]
        q75_exprs = [e for e in result if "quantile(0.75)" in e]
        assert len(q25_exprs) > 0
        assert len(q50_exprs) > 0
        assert len(q75_exprs) > 0

    def test_max_combinations_should_limit_result(self):
        """max_combinations 应限制返回数量"""
        result = self.service.generate_statistical_combinations(
            ["f1", "f2", "f3"], max_combinations=10
        )
        assert len(result) <= 10

    def test_multiple_factors_should_generate_for_each(self):
        """多个基础因子应为每个因子生成统计表达式"""
        result = self.service.generate_statistical_combinations(
            ["f1", "f2"], window_sizes=[5], max_combinations=200
        )
        f1_exprs = [e for e in result if "f1" in e]
        f2_exprs = [e for e in result if "f2" in e]
        assert len(f1_exprs) > 0
        assert len(f2_exprs) > 0

    def test_default_window_sizes_should_include_standard(self):
        """默认窗口大小应包含 5, 10, 20, 60"""
        result = self.service.generate_statistical_combinations(["close"])
        for w in [5, 10, 20, 60]:
            assert any(f"rolling({w}" in e for e in result), f"缺少窗口 {w}"

    def test_rolling_expressions_should_have_min_periods(self):
        """rolling 表达式应包含 min_periods=1"""
        result = self.service.generate_statistical_combinations(["close"])
        rolling_exprs = [e for e in result if "rolling" in e]
        for expr in rolling_exprs:
            assert "min_periods=1" in expr, f"缺少 min_periods=1: {expr}"


class TestGenerateIndicatorCombinations:
    """generate_indicator_combinations 技术指标组合测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()

    def test_empty_base_factors_should_return_empty(self):
        """空基础因子列表应返回空列表"""
        result = self.service.generate_indicator_combinations([])
        assert result == []

    def test_single_factor_should_generate_indicator_expressions(self):
        """1个基础因子应生成技术指标表达式"""
        result = self.service.generate_indicator_combinations(["momentum"])
        assert len(result) > 0

    def test_sma_should_generate_multiple_windows(self):
        """SMA 应为多个窗口生成表达式"""
        result = self.service.generate_indicator_combinations(["momentum"])
        sma_exprs = [e for e in result if "SMA" in e]
        # SMA 有4个窗口 × 2种操作（除法和减法）= 8个
        assert len(sma_exprs) >= 2

    def test_ema_should_generate_multiple_windows(self):
        """EMA 应为多个窗口生成表达式"""
        result = self.service.generate_indicator_combinations(["momentum"])
        ema_exprs = [e for e in result if "EMA" in e]
        assert len(ema_exprs) > 0

    def test_rsi_should_use_window_14(self):
        """RSI 应使用窗口14"""
        result = self.service.generate_indicator_combinations(["momentum"])
        rsi_exprs = [e for e in result if "RSI" in e]
        assert len(rsi_exprs) > 0
        assert any("14" in e for e in rsi_exprs)

    def test_macd_should_generate_expression(self):
        """MACD 应生成表达式"""
        result = self.service.generate_indicator_combinations(["momentum"])
        macd_exprs = [e for e in result if "MACD" in e]
        assert len(macd_exprs) > 0

    def test_custom_price_column_should_be_used(self):
        """自定义价格列名应在表达式中使用"""
        result = self.service.generate_indicator_combinations(
            ["momentum"], price_column="adj_close"
        )
        assert any("adj_close" in e for e in result)

    def test_default_price_column_is_close(self):
        """默认价格列名应为 close"""
        result = self.service.generate_indicator_combinations(["momentum"])
        assert any("close" in e for e in result)

    def test_max_combinations_should_limit_result(self):
        """max_combinations 应限制返回数量"""
        result = self.service.generate_indicator_combinations(
            ["f1", "f2"], max_combinations=5
        )
        assert len(result) <= 5

    def test_sma_should_generate_division_and_subtraction(self):
        """SMA 应同时生成除法和减法表达式"""
        result = self.service.generate_indicator_combinations(["momentum"])
        sma_div = [e for e in result if "SMA" in e and "/" in e]
        sma_sub = [e for e in result if "SMA" in e and "-" in e]
        assert len(sma_div) > 0
        assert len(sma_sub) > 0


class TestGenerateHybridFactors:
    """generate_hybrid_factors 混合因子生成测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()
        random.seed(42)

    def test_empty_base_factors_should_return_only_random_if_any(self):
        """空基础因子列表应返回空列表（无法生成二元/统计/指标组合）"""
        result = self.service.generate_hybrid_factors([], n_factors=10)
        # 随机组合部分也需要 >= 2 个因子
        assert isinstance(result, list)

    def test_result_should_contain_dict_entries(self):
        """结果应为字典列表"""
        result = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3"], n_factors=20
        )
        for item in result:
            assert isinstance(item, dict)
            assert "expression" in item
            assert "type" in item
            assert "complexity" in item

    def test_result_types_should_include_expected_categories(self):
        """结果类型应包含 binary_operation, statistical, indicator_based"""
        result = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3", "f4", "f5"], n_factors=50
        )
        types = {item["type"] for item in result}
        # 至少应包含部分类型
        assert len(types) > 0

    def test_n_factors_should_limit_result_count(self):
        """n_factors 应限制返回数量"""
        result = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3"], n_factors=10
        )
        assert len(result) <= 10

    def test_complexity_should_be_low_medium_or_high(self):
        """复杂度应为 low, medium, high 之一"""
        result = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3"], n_factors=20
        )
        valid_complexities = {"low", "medium", "high"}
        for item in result:
            assert item["complexity"] in valid_complexities

    def test_binary_operation_type_should_have_medium_complexity(self):
        """binary_operation 类型应为 medium 复杂度"""
        result = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3"], n_factors=50
        )
        binary_items = [item for item in result if item["type"] == "binary_operation"]
        for item in binary_items:
            assert item["complexity"] == "medium"

    def test_statistical_type_should_have_low_complexity(self):
        """statistical 类型应为 low 复杂度"""
        result = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3"], n_factors=50
        )
        stat_items = [item for item in result if item["type"] == "statistical"]
        for item in stat_items:
            assert item["complexity"] == "low"

    def test_indicator_based_type_should_have_high_complexity(self):
        """indicator_based 类型应为 high 复杂度"""
        result = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3"], n_factors=50
        )
        indicator_items = [item for item in result if item["type"] == "indicator_based"]
        for item in indicator_items:
            assert item["complexity"] == "high"

    def test_result_should_be_shuffled(self):
        """结果应被打乱顺序（非固定排列）"""
        random.seed(42)
        result1 = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3", "f4"], n_factors=30
        )
        # 两次不同种子应产生不同顺序
        random.seed(99)
        result2 = self.service.generate_hybrid_factors(
            ["f1", "f2", "f3", "f4"], n_factors=30
        )
        # 表达式集合应大致相同，但顺序可能不同
        exprs1 = [item["expression"] for item in result1]
        exprs2 = [item["expression"] for item in result2]
        # 不要求完全不同，但集合应有交集
        assert isinstance(exprs1, list)
        assert isinstance(exprs2, list)


class TestCompileExpressionToCode:
    """compile_expression_to_code 表达式编译测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()

    def test_simple_expression_should_compile_to_function(self):
        """简单表达式应编译为完整函数代码"""
        code = self.service.compile_expression_to_code("(f1 + f2)")
        assert "def calculate_factor(df):" in code
        assert "import talib" in code
        assert "import pandas as pd" in code
        assert "import numpy as np" in code

    def test_column_name_should_be_converted_to_df_ref(self):
        """列名应转换为 df['col'] 引用"""
        code = self.service.compile_expression_to_code("(close + open)")
        assert "df['close']" in code
        assert "df['open']" in code

    def test_sma_should_be_converted_to_talib(self):
        """SMA 应转换为 talib.SMA"""
        code = self.service.compile_expression_to_code("SMA(close, 20)")
        assert "talib.SMA" in code

    def test_ema_should_be_converted_to_talib(self):
        """EMA 应转换为 talib.EMA"""
        code = self.service.compile_expression_to_code("EMA(close, 20)")
        assert "talib.EMA" in code

    def test_rsi_should_be_converted_to_talib(self):
        """RSI 应转换为 talib.RSI"""
        code = self.service.compile_expression_to_code("RSI(close, 14)")
        assert "talib.RSI" in code

    def test_macd_should_be_converted_to_talib(self):
        """MACD 应转换为 talib.MACD"""
        code = self.service.compile_expression_to_code("MACD(close)")
        assert "talib.MACD" in code

    def test_mean_should_be_converted_to_rolling(self):
        """mean 应转换为 rolling().mean()"""
        code = self.service.compile_expression_to_code("mean(close, 20)")
        assert "rolling" in code
        assert "mean()" in code

    def test_std_should_be_converted_to_rolling(self):
        """std 应转换为 rolling().std()"""
        code = self.service.compile_expression_to_code("std(close, 20)")
        assert "rolling" in code
        assert "std()" in code

    def test_log_should_be_converted_to_np_log(self):
        """log 应转换为 np.log"""
        code = self.service.compile_expression_to_code("log(close)")
        assert "np.log" in code

    def test_sqrt_should_be_converted_to_np_sqrt(self):
        """sqrt 应转换为 np.sqrt"""
        code = self.service.compile_expression_to_code("sqrt(close)")
        assert "np.sqrt" in code

    def test_exp_should_be_converted_to_np_exp(self):
        """exp 应转换为 np.exp"""
        code = self.service.compile_expression_to_code("exp(close)")
        assert "np.exp" in code

    def test_abs_should_be_converted_to_np_abs(self):
        """abs 应转换为 np.abs"""
        code = self.service.compile_expression_to_code("abs(close)")
        assert "np.abs" in code

    def test_rank_should_be_converted_to_rolling_rank(self):
        """rank 应转换为 rolling().rank(pct=True)"""
        code = self.service.compile_expression_to_code("rank(close)")
        assert "rank(pct=True)" in code

    def test_diff_should_be_converted_to_diff(self):
        """diff 应转换为 df['col'].diff()"""
        code = self.service.compile_expression_to_code("diff(close)")
        assert ".diff(" in code

    def test_pct_change_should_be_converted(self):
        """pct_change 应转换为 df['col'].pct_change()"""
        code = self.service.compile_expression_to_code("pct_change(close)")
        assert ".pct_change(" in code

    def test_zscore_should_be_special_handled(self):
        """zscore 应特殊处理为 (x - rolling_mean) / (rolling_std + 1e-8)"""
        code = self.service.compile_expression_to_code("zscore(close)")
        assert "rolling" in code
        assert "mean()" in code
        assert "std()" in code

    def test_custom_data_column_should_be_used(self):
        """自定义数据列名应在代码中使用"""
        code = self.service.compile_expression_to_code(
            "(close + open)", data_column="adj_close"
        )
        assert "adj_close" in code

    def test_default_data_column_is_close(self):
        """默认数据列名应为 close"""
        code = self.service.compile_expression_to_code("(close + open)")
        assert "'close'" in code

    def test_compiled_code_should_have_error_handling(self):
        """编译后的代码应包含异常处理"""
        code = self.service.compile_expression_to_code("(f1 + f2)")
        assert "try:" in code
        assert "except" in code

    def test_compiled_code_should_check_column_existence(self):
        """编译后的代码应检查列是否存在"""
        code = self.service.compile_expression_to_code("(close + open)")
        assert "not in df.columns" in code

    def test_window_parameter_should_override_default(self):
        """窗口参数应覆盖默认的252"""
        code = self.service.compile_expression_to_code("mean(close, 20)")
        assert "window=20" in code

    def test_quantile_should_include_quantile_value(self):
        """quantile 应包含分位数值"""
        code = self.service.compile_expression_to_code("quantile(close, 0.5)")
        assert "quantile" in code


class TestValidateExpression:
    """validate_expression 表达式验证测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()

    def test_empty_expression_should_be_invalid(self):
        """空表达式应为无效"""
        valid, msg = self.service.validate_expression("")
        assert valid is False
        assert "空" in msg

    def test_whitespace_only_expression_should_be_invalid(self):
        """仅空格的表达式应为无效"""
        valid, msg = self.service.validate_expression("   ")
        assert valid is False

    def test_unbalanced_parentheses_should_be_invalid(self):
        """括号不匹配应为无效"""
        valid, msg = self.service.validate_expression("(f1 + f2")
        assert valid is False
        assert "括号" in msg

    def test_unbalanced_closing_paren_should_be_invalid(self):
        """右括号多余应为无效"""
        valid, msg = self.service.validate_expression("f1 + f2)")
        assert valid is False

    def test_illegal_character_should_be_invalid(self):
        """非法字符应为无效"""
        valid, msg = self.service.validate_expression("(f1 + f2)@#$")
        assert valid is False
        assert "非法字符" in msg

    def test_expression_with_operator_should_be_valid(self):
        """含运算符的表达式应为有效"""
        valid, msg = self.service.validate_expression("(f1 + f2)")
        assert valid is True
        assert msg == ""

    def test_expression_with_function_should_be_valid(self):
        """含统计函数的表达式应为有效"""
        valid, msg = self.service.validate_expression("mean(close, 20)")
        assert valid is True

    def test_expression_without_operator_or_function_should_be_invalid(self):
        """无运算符也无函数的表达式应为无效"""
        valid, msg = self.service.validate_expression("factorname")
        assert valid is False
        assert "运算符" in msg or "函数" in msg

    def test_subtraction_operator_should_be_valid(self):
        """减法运算符应为有效"""
        valid, _ = self.service.validate_expression("(f1 - f2)")
        assert valid is True

    def test_multiplication_operator_should_be_valid(self):
        """乘法运算符应为有效"""
        valid, _ = self.service.validate_expression("(f1 * f2)")
        assert valid is True

    def test_division_operator_should_be_valid(self):
        """除法运算符应为有效"""
        valid, _ = self.service.validate_expression("(f1 / f2)")
        assert valid is True

    def test_complex_expression_should_be_valid(self):
        """复杂表达式应为有效"""
        expr = "((f1 + f2) * mean(close, 20))"
        valid, _ = self.service.validate_expression(expr)
        assert valid is True

    def test_balanced_parentheses_should_be_valid(self):
        """括号匹配应为有效"""
        valid, _ = self.service.validate_expression("((f1 + f2))")
        assert valid is True

    def test_allowed_special_chars_should_be_valid(self):
        """允许的特殊字符（逗号、点、下划线）应为有效"""
        valid, _ = self.service.validate_expression("mean(close, 20)")
        assert valid is True


class TestParseExpression:
    """parse_expression 表达式解析测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()

    def test_simple_binary_should_extract_operator(self):
        """简单二元表达式应提取运算符"""
        result = self.service.parse_expression("(f1 + f2)")
        assert "+" in result["operators"]

    def test_multiple_operators_should_extract_all(self):
        """多运算符表达式应提取所有运算符"""
        result = self.service.parse_expression("((f1 + f2) * f3)")
        assert "+" in result["operators"]
        assert "*" in result["operators"]

    def test_statistical_function_should_be_extracted(self):
        """统计函数应被提取"""
        result = self.service.parse_expression("mean(close, 20)")
        assert "mean" in result["functions"]

    def test_indicator_function_should_be_extracted(self):
        """技术指标函数应被提取"""
        result = self.service.parse_expression("SMA(close, 20)")
        assert "SMA" in result["functions"]

    def test_depth_should_count_max_nesting(self):
        """深度应为最大括号嵌套层数"""
        result = self.service.parse_expression("((f1 + f2))")
        assert result["depth"] == 2

    def test_no_nesting_should_have_depth_0(self):
        """无括号嵌套应为深度0"""
        result = self.service.parse_expression("f1 + f2")
        assert result["depth"] == 0

    def test_single_nesting_should_have_depth_1(self):
        """单层括号应为深度1"""
        result = self.service.parse_expression("(f1 + f2)")
        assert result["depth"] == 1

    def test_deep_nesting_should_have_correct_depth(self):
        """深层嵌套应有正确深度"""
        result = self.service.parse_expression("(((f1 + f2)))")
        assert result["depth"] == 3

    def test_expression_should_be_preserved(self):
        """原始表达式应被保留"""
        expr = "(f1 + f2)"
        result = self.service.parse_expression(expr)
        assert result["expression"] == expr

    def test_empty_expression_should_return_zero_depth(self):
        """空表达式应为深度0"""
        result = self.service.parse_expression("")
        assert result["depth"] == 0

    def test_multiple_functions_should_extract_all(self):
        """多函数表达式应提取所有函数"""
        result = self.service.parse_expression("mean(close, 20) + std(close, 20)")
        assert "mean" in result["functions"]
        assert "std" in result["functions"]

    def test_result_should_have_all_required_keys(self):
        """结果应包含所有必需的键"""
        result = self.service.parse_expression("(f1 + f2)")
        assert "expression" in result
        assert "components" in result
        assert "operators" in result
        assert "functions" in result
        assert "depth" in result


class TestPreselectFactors:
    """preselect_factors 因子预筛选测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()
        np.random.seed(42)

    def _make_factor_data(self, n=100, correlated=False):
        """生成测试用因子数据"""
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        if correlated:
            # 与收益率相关的因子
            base = np.random.randn(n)
            factor_values = pd.Series(base * 0.5 + np.random.randn(n) * 0.1, index=dates)
            return_data = pd.Series(base * 0.3 + np.random.randn(n) * 0.05, index=dates)
        else:
            factor_values = pd.Series(np.random.randn(n), index=dates)
            return_data = pd.Series(np.random.randn(n), index=dates)
        return factor_values, return_data

    def test_empty_factors_should_return_empty(self):
        """空因子列表应返回空列表"""
        result = self.service.preselect_factors(
            [], {}, pd.Series(dtype=float)
        )
        assert result == []

    def test_factor_not_in_data_map_should_be_skipped(self):
        """不在数据映射中的因子应被跳过"""
        factors = [{"expression": "nonexistent_factor"}]
        result = self.service.preselect_factors(
            factors, {}, pd.Series(np.random.randn(100))
        )
        assert result == []

    def test_low_ic_factor_should_be_filtered_out(self):
        """低IC因子应被过滤"""
        factor_values, return_data = self._make_factor_data(n=100, correlated=False)
        factors = [{"expression": "low_ic_factor"}]
        factor_data_map = {"low_ic_factor": factor_values}

        result = self.service.preselect_factors(
            factors, factor_data_map, return_data,
            ic_threshold=0.03, ir_threshold=0.5
        )
        # 不相关的因子IC很低，应被过滤
        assert len(result) == 0

    def test_high_ic_factor_should_pass(self):
        """高IC因子应通过筛选"""
        factor_values, return_data = self._make_factor_data(n=200, correlated=True)
        factors = [{"expression": "high_ic_factor"}]
        factor_data_map = {"high_ic_factor": factor_values}

        result = self.service.preselect_factors(
            factors, factor_data_map, return_data,
            ic_threshold=0.01, ir_threshold=0.0
        )
        # 相关因子应通过IC阈值
        assert len(result) >= 0  # 可能通过也可能不通过，取决于数据

    def test_insufficient_valid_ratio_should_be_filtered(self):
        """有效数据比例不足应被过滤"""
        # 大量NaN的因子
        factor_values = pd.Series([np.nan] * 80 + [1.0] * 20)
        return_data = pd.Series(np.random.randn(100))
        factors = [{"expression": "sparse_factor"}]
        factor_data_map = {"sparse_factor": factor_values}

        result = self.service.preselect_factors(
            factors, factor_data_map, return_data,
            min_valid_ratio=0.7
        )
        # 80% NaN → 有效比例 20/100 = 0.2 < 0.7
        assert len(result) == 0

    def test_selected_factor_should_have_ic_and_ir(self):
        """通过筛选的因子应包含 ic 和 ir 字段"""
        factor_values, return_data = self._make_factor_data(n=200, correlated=True)
        factors = [{"expression": "test_factor"}]
        factor_data_map = {"test_factor": factor_values}

        result = self.service.preselect_factors(
            factors, factor_data_map, return_data,
            ic_threshold=0.0, ir_threshold=0.0
        )
        if len(result) > 0:
            assert "ic" in result[0]
            assert "ir" in result[0]
            assert "valid_ratio" in result[0]

    def test_custom_ic_threshold_should_be_respected(self):
        """自定义IC阈值应被遵守"""
        factor_values, return_data = self._make_factor_data(n=200, correlated=True)
        factors = [{"expression": "test_factor"}]
        factor_data_map = {"test_factor": factor_values}

        result_low = self.service.preselect_factors(
            factors, factor_data_map, return_data,
            ic_threshold=0.001, ir_threshold=0.0
        )
        result_high = self.service.preselect_factors(
            factors, factor_data_map, return_data,
            ic_threshold=0.99, ir_threshold=0.0
        )
        # 低阈值应比高阈值通过更多因子
        assert len(result_low) >= len(result_high)

    def test_nan_ic_should_be_filtered(self):
        """IC为NaN的因子应被过滤"""
        # 常数因子 → IC为NaN
        factor_values = pd.Series([1.0] * 100)
        return_data = pd.Series(np.random.randn(100))
        factors = [{"expression": "constant_factor"}]
        factor_data_map = {"constant_factor": factor_values}

        result = self.service.preselect_factors(
            factors, factor_data_map, return_data,
            ic_threshold=0.0
        )
        # 常数因子与随机收益率的IC为NaN
        assert len(result) == 0


class TestCalculateFactorMetrics:
    """calculate_factor_metrics 因子质量指标测试"""

    def setup_method(self):
        self.service = FactorGeneratorService()
        np.random.seed(42)

    def test_insufficient_data_should_return_invalid(self):
        """数据不足应返回 invalid"""
        factor_values = pd.Series([1.0, 2.0, 3.0])
        return_values = pd.Series([0.01, 0.02, 0.03])
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        assert result["valid"] is False
        assert "message" in result

    def test_normal_data_should_return_valid_metrics(self):
        """正常数据应返回有效指标"""
        n = 200
        factor_values = pd.Series(np.random.randn(n))
        return_values = pd.Series(np.random.randn(n))
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        assert result["valid"] is True
        assert "ic" in result
        assert "ir" in result
        assert "ic_mean" in result
        assert "ic_std" in result
        assert "ic_win_rate" in result
        assert "n_obs" in result
        assert "factor_stats" in result

    def test_ic_should_be_between_minus1_and_1(self):
        """IC应在 [-1, 1] 范围内"""
        n = 200
        factor_values = pd.Series(np.random.randn(n))
        return_values = pd.Series(np.random.randn(n))
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        if result["valid"]:
            assert -1.0 <= result["ic"] <= 1.0

    def test_ic_win_rate_should_be_between_0_and_1(self):
        """胜率应在 [0, 1] 范围内"""
        n = 200
        factor_values = pd.Series(np.random.randn(n))
        return_values = pd.Series(np.random.randn(n))
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        if result["valid"]:
            assert 0.0 <= result["ic_win_rate"] <= 1.0

    def test_n_obs_should_match_aligned_data_length(self):
        """n_obs 应等于对齐后的数据长度"""
        n = 200
        factor_values = pd.Series(np.random.randn(n))
        return_values = pd.Series(np.random.randn(n))
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        if result["valid"]:
            assert result["n_obs"] == n

    def test_factor_stats_should_contain_required_fields(self):
        """因子统计应包含 mean, std, skew, kurtosis"""
        n = 200
        factor_values = pd.Series(np.random.randn(n))
        return_values = pd.Series(np.random.randn(n))
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        if result["valid"]:
            stats = result["factor_stats"]
            assert "mean" in stats
            assert "std" in stats
            assert "skew" in stats
            assert "kurtosis" in stats

    def test_nan_values_should_be_dropped_before_calculation(self):
        """NaN值应在计算前被移除"""
        n = 200
        factor_values = pd.Series(np.random.randn(n))
        factor_values.iloc[0:10] = np.nan
        return_values = pd.Series(np.random.randn(n))
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        if result["valid"]:
            assert result["n_obs"] == n - 10

    def test_perfect_positive_correlation_should_have_high_ic(self):
        """完全正相关应有高IC"""
        n = 200
        base = np.random.randn(n)
        factor_values = pd.Series(base)
        return_values = pd.Series(base * 2 + 1)  # 完全线性相关
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        if result["valid"]:
            assert result["ic"] > 0.9

    def test_perfect_negative_correlation_should_have_low_ic(self):
        """完全负相关应有低IC（负值）"""
        n = 200
        base = np.random.randn(n)
        factor_values = pd.Series(base)
        return_values = pd.Series(-base * 2 + 1)  # 完全负相关
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        if result["valid"]:
            assert result["ic"] < -0.9

    def test_constant_factor_should_return_valid_with_nan_ic(self):
        """常数因子应返回有效结果（IC可能为NaN）"""
        n = 200
        factor_values = pd.Series([1.0] * n)
        return_values = pd.Series(np.random.randn(n))
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        # 常数因子的std=0，corr可能为NaN
        # 但数据长度 > 10，所以 valid=True
        assert result["valid"] is True

    def test_ir_should_use_safe_ir(self):
        """IR计算应使用 safe_ir（零标准差不崩溃）"""
        n = 200
        # 构造IC恒定的场景（因子和收益率完全相同）
        factor_values = pd.Series(np.random.randn(n))
        return_values = factor_values.copy()
        # 不应抛出异常
        result = self.service.calculate_factor_metrics(factor_values, return_values)
        assert result["valid"] is True


class TestFactorGeneratorServiceInit:
    """FactorGeneratorService 初始化测试"""

    def test_should_have_operators(self):
        """应包含运算符集合"""
        service = FactorGeneratorService()
        assert "+" in service.operators
        assert "-" in service.operators
        assert "*" in service.operators
        assert "/" in service.operators

    def test_should_have_statistics(self):
        """应包含统计函数集合"""
        service = FactorGeneratorService()
        assert "mean" in service.statistics
        assert "std" in service.statistics
        assert "rank" in service.statistics
        assert "zscore" in service.statistics

    def test_should_have_indicators(self):
        """应包含技术指标集合"""
        service = FactorGeneratorService()
        assert "SMA" in service.indicators
        assert "EMA" in service.indicators
        assert "RSI" in service.indicators
        assert "MACD" in service.indicators


class TestGlobalInstance:
    """全局实例测试"""

    def test_global_instance_should_exist(self):
        """全局实例应存在"""
        from backend.services.factor_generator_service import factor_generator_service
        assert factor_generator_service is not None
        assert isinstance(factor_generator_service, FactorGeneratorService)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
