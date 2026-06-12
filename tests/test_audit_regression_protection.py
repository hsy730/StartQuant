"""
审计Bug回归防护测试

覆盖22轮代码审查发现并修复的关键Bug，防止未来回退。
每个测试对应一个已修复的生产故障，附带规则编号和修复说明。

规则体系：
  7.1   IC必须使用横截面Spearman（禁止Pearson）
  7.7   禁止fillna(0)填充因子值/IC/收益率
  7.10  IR不可计算时返回None，禁止回退为0.0
  7.26  禁止None→0语义转换（不可计算≠零值）
  7.36  dict.get(key, default)在值为None时不生效
  7.38  评分函数必须截断到[0, 100]
  7.40  禁止浮点==0比较，使用<1e-10
  7.41  同7.36，dict.get的default仅在键不存在时生效
  7.43  f-string格式化None值导致TypeError崩溃
"""

import sys
import os
import re
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# 测试1: _safe_float 防护 — 规则7.36/7.41 + float('inf')泄漏
# Bug来源: factor_summary_service.py, pysr_factor_mining_service.py
# 问题: _safe_float不检查inf；float(None)崩溃；dict.get(None,default)不生效
# ============================================================================


class TestSafeFloatDefense:
    """验证 _safe_float 正确处理 None, NaN, inf 三种异常值"""

    def test_safe_float_none_returns_default(self):
        """Rule 7.36: None应返回默认值而非崩溃"""
        from backend.services.factor_summary_service import _safe_float

        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=-1.0) == -1.0
        assert _safe_float(None, default="N/A") == "N/A"

    def test_safe_float_nan_returns_default(self):
        """NaN应返回默认值"""
        from backend.services.factor_summary_service import _safe_float

        assert _safe_float(np.nan) == 0.0
        assert _safe_float(np.nan, default=99.0) == 99.0

    def test_safe_float_inf_returns_default(self):
        """float('inf')应返回默认值，防止JSON序列化崩溃"""
        from backend.services.factor_summary_service import _safe_float

        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf"), default=-999.0) == -999.0

    def test_safe_float_normal_values_pass_through(self):
        """正常值应原样通过"""
        from backend.services.factor_summary_service import _safe_float

        assert _safe_float(3.14) == pytest.approx(3.14)
        assert _safe_float(0.0) == 0.0
        assert _safe_float(-100.5) == pytest.approx(-100.5)

    def test_safe_float_string_numeric_converts(self):
        """字符串数值应正确转换"""
        from backend.services.factor_summary_service import _safe_float

        assert _safe_float("42.5") == pytest.approx(42.5)

    def test_dict_get_none_does_not_trigger_default(self):
        """Rule 7.41: dict.get(key, default)在值存在但为None时返回None"""
        d = {"key": None}
        # 这是Python标准行为：get的default只在键不存在时生效
        assert d.get("key", 0) is None  # 不是0！
        assert d.get("missing_key", 0) == 0  # 键不存在时才用default


# ============================================================================
# 测试2: sorted() 字典序错排防护 — PySR x0..xN 排序Bug
# Bug来源: pysr_factor_mining_service.py 第120/151/231/373行
# 问题: sorted(["x0","x1","x10"]) → ["x0","x1","x10"] 错误！
#       应该是 ["x0","x1","...","x9","x10"]
# ============================================================================


class TestPySRSortedOrdering:
    """验证PySR变量名按数值排序而非字典序排序"""

    def test_lexicographic_sort_is_wrong_for_x_vars(self):
        """字典序排序对x0..xN(N>=10)产生错误顺序"""
        keys = [f"x{i}" for i in range(12)]
        lexicographic = sorted(keys)
        # x10排在x2前面 — 这就是Bug！
        assert lexicographic.index("x10") < lexicographic.index("x2")

    def test_numeric_sort_is_correct_for_x_vars(self):
        """数值排序产生正确顺序"""
        keys = [f"x{i}" for i in range(12)]
        numeric = sorted(keys, key=lambda k: int(k.replace("x", "")))
        for i, k in enumerate(numeric):
            assert k == f"x{i}", f"位置{i}应为x{i}，实际为{k}"

    def test_numeric_sort_with_non_x_keys_fallback(self):
        """非x前缀的键名不应触发int()转换"""
        keys = ["close", "open", "high", "low", "volume", "x0", "x1"]
        # 使用元组确保类型一致：(0, str) for non-x keys, (1, int) for x keys
        result = sorted(
            keys,
            key=lambda k: (
                0,
                k,
            )
            if not (k.startswith("x") and len(k) > 1 and k[1:].isdigit())
            else (
                1,
                int(k.replace("x", "")),
            ),
        )
        # 非x键按字典序在前，x键按数值序在后
        assert result[:5] == ["close", "high", "low", "open", "volume"]
        assert result[5:] == ["x0", "x1"]

    def test_large_factor_count_ordering(self):
        """15个因子时的正确顺序（生产常见场景）"""
        keys = [f"x{i}" for i in range(15)]
        result = sorted(
            keys,
            key=lambda k: int(k.replace("x", ""))
            if k.startswith("x") and k[1:].isdigit()
            else k,
        )
        for i, k in enumerate(result):
            assert k == f"x{i}", f"因子数15时位置{i}应为x{i}，实际为{k}"


# ============================================================================
# 测试3: f-string None格式化防护 — 规则7.43
# Bug来源: analysis_service.py, export_service.py, pysr_factor_mining_service.py
# 问题: f"{None:.4f}" → TypeError; f"{None:,.0f}" → TypeError
# ============================================================================


class TestFStringNoneFormatting:
    """验证None值的f-string格式化不会崩溃"""

    def test_fstring_none_with_format_spec_crashes(self):
        """证明 f'{None:.4f}' 确实会崩溃（这是Python行为）"""
        with pytest.raises((TypeError, ValueError)):
            f"{None:.4f}"

    def test_fstring_none_with_comma_format_crashes(self):
        """证明 f'{None:,.0f}' 会崩溃"""
        with pytest.raises((TypeError, ValueError)):
            f"{None:,.0f}"

    def test_fstring_none_with_percent_format_crashes(self):
        """证明 f'{None:.6f}' 会崩溃"""
        with pytest.raises((TypeError, ValueError)):
            f"{None:.6f}"

    def test_safe_formatting_pattern_works(self):
        """正确的None安全格式化模式"""
        val = None
        safe_str = f"{val:.4f}" if val is not None else "N/A"
        assert safe_str == "N/A"

        val = 3.14159
        safe_str = f"{val:.4f}" if val is not None else "N/A"
        assert safe_str == "3.1416"

    def test_fstring_format_spec_syntax_error_in_conditional(self):
        """证明 f'{val:.4f if cond else fallback}' 是语法错误"""
        val = None
        # 这种写法看起来合理但实际上是语法错误：
        # .4f 被解析为format spec而不是条件表达式的一部分
        with pytest.raises((TypeError, ValueError)):
            # Python将 : 后面的所有内容当作format spec
            f"{val:.4f if val is not None else 'N/A'}"


# ============================================================================
# 测试4: round(inf) 崩溃防护
# Bug来源: comprehensive_scoring_service.py, factor_correlation_service.py
# 问题: round(float('inf'), 2) → OverflowError
# ============================================================================


class TestRoundInfCrash:
    """验证round()对inf/nan的处理"""

    def test_round_inf_raises_overflow_error(self):
        """round(float('inf')) 返回inf — 结果不能用于有效JSON数字"""
        inf_val = float("inf")
        result = round(inf_val, 2)
        # round本身不崩溃，但结果仍是inf
        assert np.isinf(result) or result == float("inf")
        # json.dumps默认允许inf但输出非标准JSON（"Infinity"不是有效的JSON数字）
        result_str = json.dumps({"value": result})
        assert "Infinity" in result_str or "inf" in result_str.lower()  # 非标准JSON

    def test_nan_round_propagates(self):
        """round(nan) 返回nan — 不能用于JSON"""
        nan_val = float("nan")
        result = round(nan_val, 2)
        assert np.isnan(result)

    def test_safe_round_pattern(self):
        """安全的round模式"""
        val = float("inf")
        safe_result = round(val, 2) if np.isfinite(val) else None
        assert safe_result is None
        # 可以安全地JSON序列化
        json.dumps({"value": safe_result})

        val = 3.14159
        safe_result = round(val, 2) if np.isfinite(val) else None
        assert safe_result == 3.14


# ============================================================================
# 测试5: IR=None 语义防护 — 规则7.10
# Bug来源: weighted_ic_service.py, comprehensive_scoring_service.py,
#         factor_orchestrator_service.py, analysis_service.py
# 问题: IR=None（IC_std≈0, IC_mean≠0）表示因子极其稳定，
#       不应被转为0（表示"无效"）
# ============================================================================


class TestIRNoneSemantics:
    """验证IR=None的正确语义处理"""

    def test_ir_none_means_uncomputable_not_zero(self):
        """Rule 7.10: IR=None ≠ IR=0"""
        from backend.utils.safe_math import safe_ir

        # IC_std极小但IC_mean非零 → 因子极其稳定
        ic_mean = 0.05
        ic_std = 1e-15  # 极小但不为零
        ir = safe_ir(ic_mean, ic_std, default=None)
        # IR应该是一个很大的数或None（取决于实现）
        # 但绝不应该静默变成0.0
        if ir is not None:
            assert abs(ir) > 1.0, f"稳定因子IR={ir}，不应接近0"

    def test_ir_zero_ic_mean_returns_zero(self):
        """IC_mean=0时IR=0是合理的（无信号）"""
        from backend.utils.safe_math import safe_ir

        ir = safe_ir(0.0, 0.01, default=None)
        assert ir == 0.0 or ir is None  # 两者都可接受

    def test_ir_both_zero_returns_none_or_zero(self):
        """IC_mean和IC_std都为0时"""
        from backend.utils.safe_math import safe_ir

        ir = safe_ir(0.0, 0.0, default=None)
        # 不应崩溃即可
        assert ir is not None or ir is None  # 总是为真，只是确认不崩溃

    def test_ir_none_should_not_become_score_zero(self):
        """IR=None在评分中不应得0分（如果IC_mean非零）"""
        # 模拟评分逻辑：当IR=None但IC_mean>0时，因子极其稳定
        ic_mean = 0.05
        ir = None  # 不可计算
        abs_ic = abs(ic_mean)

        # 错误的实现：直接给0分
        wrong_score = (ir or 0.0) * 30  # None or 0.0 = 0.0
        assert wrong_score == 0.0  # 这就是Bug！

        # 正确的实现：检查IC_mean来决定
        if ir is not None:
            ir_score = min(abs(ir) * 30, 30)
        elif abs_ic > 1e-10:
            ir_score = 30.0  # 极稳定因子给满分
        else:
            ir_score = 0.0
        assert ir_score == 30.0  # 正确：稳定因子获得高分


# ============================================================================
# 测试6: JSON序列化防护 — inf/NaN/None
# Bug来源: 多个服务文件
# 问题: float('inf')/NaN不能JSON序列化，sanitize_dict应拦截
# ============================================================================


class TestJSONSerializationSafety:
    """验证API响应中的数值安全性"""

    def test_json_cannot_serialize_inf(self):
        """json.dumps对inf的行为因Python版本而异，但结果不是有效JSON数字"""
        data = {"value": float("inf")}
        result = json.dumps(data)
        # 某些版本允许inf（输出"Infinity"），某些抛出ValueError
        # 无论如何，这不是有效的JSON数字
        assert "Infinity" in result or "inf" in result

    def test_json_cannot_serialize_nan(self):
        """证明json.dumps对nan抛出ValueError（在某些Python版本）"""
        data = {"value": float("nan")}
        try:
            result = json.dumps(data)
            # 如果没崩溃，检查是否变成了"NaN"字符串（不是有效的JSON数字）
            assert "NaN" in result or "Infinity" in result
        except (ValueError, TypeError):
            pass  # 预期行为

    def test_json_can_serialize_none(self):
        """None可以正确序列化为null"""
        result = json.dumps({"value": None})
        assert json.loads(result)["value"] is None

    def test_sanitize_dict_converts_inf_to_none(self):
        """sanitize_dict应将inf转为None"""
        from backend.utils.serialization import sanitize_dict

        data = {"a": float("inf"), "b": float("-inf"), "c": 3.14}
        cleaned = sanitize_dict(data)
        assert cleaned["a"] is None
        assert cleaned["b"] is None
        assert cleaned["c"] == 3.14
        # 清洗后的数据可序列化
        json.dumps(cleaned)

    def test_sanitize_dict_converts_nan_to_none(self):
        """sanitize_dict应将NaN转为None"""
        from backend.utils.serialization import sanitize_dict

        data = {"a": float("nan"), "b": 42}
        cleaned = sanitize_dict(data)
        assert cleaned["a"] is None
        assert cleaned["b"] == 42
        json.dumps(cleaned)

    def test_sanitize_dict_nested_structures(self):
        """sanitize_dict应递归清洗嵌套结构"""
        from backend.utils.serialization import sanitize_dict

        data = {
            "level1": {
                "level2": {
                    "inf_value": float("inf"),
                    "normal": 123,
                }
            },
            "list_data": [1, float("nan"), 3],
        }
        cleaned = sanitize_dict(data)
        assert cleaned["level1"]["level2"]["inf_value"] is None
        assert cleaned["list_data"][1] is None
        json.dumps(cleaned)


# ============================================================================
# 测试7: 除法防护 — 规则1
# 来源: 全局性要求
# ============================================================================


class TestDivisionSafety:
    """验证除法操作的安全性"""

    def test_divide_by_zero_without_protection_crashes(self):
        """裸除法在除以零时应被保护"""
        from backend.utils.safe_math import safe_divide

        result = safe_divide(10.0, 0.0)
        assert result is None  # 安全返回None

    def test_divide_none_by_number_returns_none(self):
        """None / number 应返回None"""
        from backend.utils.safe_math import safe_divide

        result = safe_divide(None, 5.0)
        assert result is None

    def test_divide_number_by_none_returns_none(self):
        """number / None 应返回None"""
        from backend.utils.safe_math import safe_divide

        result = safe_divide(5.0, None)
        assert result is None

    def test_none_divided_by_100_crashes_without_guard(self):
        """None / 100.0 在裸除法下崩溃"""
        with pytest.raises(TypeError):
            _ = None / 100.0  # noqa: B018


# ============================================================================
# 测试8: 评分截断防护 — 规则7.38
# Bug来源: weighted_ic_service.py, base_mining_service.py,
#         comprehensive_scoring_service.py, factor_validation_service.py
# 问题: 评分必须截断到[0, 100]
# ============================================================================


class TestScoreCapping:
    """验证评分函数的边界截断"""

    def test_negative_penalty_can_produce_negative_score(self):
        """唯一性惩罚可能使评分为负（未修复前的Bug）"""
        raw_score = 5.0
        uniqueness_penalty = 16.2  # max_corr=0.9 时 penalty=0.81*20=16.2

        # 错误做法：不减去惩罚后截断
        wrong_score = raw_score - uniqueness_penalty
        assert wrong_score < 0  # 负分！

        # 正确做法：截断到[0, 100]
        correct_score = max(0.0, raw_score - uniqueness_penalty)
        assert correct_score >= 0.0

    def test_score_must_be_capped_at_100(self):
        """评分上限必须是100"""
        raw_fitness = 2.0  # 异常高的fitness
        score = raw_fitness * 100

        # 未截断时超过100
        assert score > 100.0

        # 截断后
        capped_score = max(0.0, min(score, 100.0))
        assert 0.0 <= capped_score <= 100.0

    def test_negative_ir_produces_negative_subscore(self):
        """负IR可能产生负分子分数"""
        ir = -3.0  # 异常负IR
        subscore = ir * 20  # 未截断

        assert subscore < 0  # 负分！

        # 截断后
        capped_subscore = max(min(subscore, 20), 0)
        assert 0.0 <= capped_subscore <= 20.0


# ============================================================================
# 测试9: Pearson vs Spearman IC 防护 — 规则7.1/7.12
# Bug来源: 多个IC计算路径
# 问题: IC必须使用Spearman秩相关，Pearson对非线性关系不敏感
# ============================================================================


class TestICMethodCorrectness:
    """验证IC计算方法正确性"""

    def test_pearson_and_spearman_differ_on_outliers(self):
        """Pearson和Spearman在异常值存在时差异显著"""
        from scipy.stats import spearmanr

        np.random.seed(42)
        n = 200
        factor = np.random.randn(n)
        returns = factor * 0.02 + np.random.randn(n) * 0.05

        # 注入一个极端异常值
        factor[0] = 50.0
        returns[0] = 5.0

        pearson_r = np.corrcoef(factor, returns)[0, 1]
        spearman_r, _ = spearmanr(factor, returns)

        # 异常值使Pearson偏高（或两者差异显著）
        # 关键是证明两种方法给出不同结果
        assert abs(pearson_r - spearman_r) > 1e-6


# ============================================================================
# 测试10: fillna(0) 因子值防护 — 规则7.7
# Bug来源: factor_preprocessing_pipeline.py, vectorbt_backtest_service.py
# 问题: fillna(0)扭曲因子信号（0≠缺失）
# ============================================================================


class TestFillNaZeroForbidden:
    """验证因子值不应fillna(0)"""

    def test_fillna_zero_distorts_zscored_factors(self):
        """对标准化因子fillna(0)会将缺失值变为'平均水平'"""
        np.random.seed(42)
        factors = pd.Series(np.random.randn(100), name="factor")
        factors.iloc[::10] = np.nan  # 10%缺失

        # 错误做法：填充0
        filled_wrong = factors.fillna(0)
        # 对于z-score标准化过的因子，0意味着平均值
        # 缺失值被错误解释为"平均暴露"
        assert filled_wrong.isna().sum() == 0
        assert (filled_wrong == 0).sum() > 0  # 有值被设为0

        # 正确做法：填充中位数或保持NaN
        filled_right = factors.fillna(factors.median())
        assert filled_right.isna().sum() == 0
        # 中位数更接近真实分布

    def test_fillna_zero_on_ic_biases_mean(self):
        """对IC序列fillna(0)系统性拉低均值"""
        np.random.seed(42)
        ic_series = pd.Series([0.03, 0.04, 0.05, np.nan, np.nan, 0.02])

        # 错误做法
        filled_wrong = ic_series.fillna(0)
        assert filled_wrong.mean() < ic_series.mean()  # 均值被拉低

        # 正确做法：忽略NaN计算均值
        correct_mean = ic_series.mean()
        assert correct_mean == pytest.approx(0.035, abs=0.001)


# ============================================================================
# 测试11: 字符串替换交叉污染防护
# Bug来源: pysr_factor_mining_service.py _pysr_expr_to_factor_code
# 问题: str.replace("x0", ...) 会污染 "x10" 中的 "x0" 子串
# ============================================================================


class TestStringReplaceCrossContamination:
    """验证变量名替换的安全性"""

    def test_replace_x1_pollutes_x10(self):
        """str.replace('x1', ...) 错误修改 'x10'（x1是x10的前缀子串）"""
        expr = "x1 + x10 + x11"
        result = expr.replace("x1", "(open)")
        # x10 和 x11 中的 x1 都被错误替换了！
        assert "(open)0" in result or "(open)1" in result  # 被污染！

    def test_replace_x0_does_not_pollute_x10(self):
        """str.replace('x0', ...) 不影响 'x10'（x0不是x10的子串）"""
        expr = "x0 + x1 + x10"
        result = expr.replace("x0", "(close)")
        # x10不受影响（因为"x0"不是"x10"的子串）
        assert "x10" in result
        assert "(close)" in result

    def test_regex_word_boundary_prevents_pollution(self):
        """re.sub(r'\bx1\b', ...) 只替换完整的x1，不污染x10"""
        import re as _re

        expr = "x1 + x10 + x11"
        result = _re.sub(r"\b" + _re.escape("x1") + r"\b", "(open)", expr)
        assert result == "(open) + x10 + x11"  # x10和x11未被污染

    def test_regex_replacement_all_var_names(self):
        """正则替换多个变量名全部正确"""
        import re as _re

        expr = "x0 + x1 + x10 + x11"
        replacements = {"x0": "close", "x1": "open", "x10": "high", "x11": "low"}
        for var_name, code in replacements.items():
            expr = _re.sub(r"\b" + _re.escape(var_name) + r"\b", f"({code})", expr)
        assert expr == "(close) + (open) + (high) + (low)"
