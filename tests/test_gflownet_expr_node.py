"""
GFlowNet表达式节点完整性回归测试

防护Bug:
1. OP_ARITY未包含扩展算子，导致is_complete()误判无参数节点为完整
2. ExprNode.is_complete未递归检查子节点，导致不完整表达式被返回
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 标记torch是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from backend.services.gflownet_mining_service import (
        ExprNode,
        OP_ARITY,
        UNARY_OPS,
        BINARY_OPS,
        EXTENDED_UNARY_OPS,
        EXTENDED_BINARY_OPS,
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch未安装")
class TestGFlowNetExprNode:
    """GFlowNet表达式节点测试"""

    def test_all_extended_unary_ops_in_op_arity(self):
        """所有扩展一元算子必须在OP_ARITY中有定义"""
        for op in EXTENDED_UNARY_OPS:
            assert op in OP_ARITY, f"扩展一元算子 '{op}' 未在OP_ARITY中定义"
            assert OP_ARITY[op] == 1, f"一元算子 '{op}' 的arity应为1，但得到 {OP_ARITY[op]}"

    def test_all_extended_binary_ops_in_op_arity(self):
        """所有扩展二元算子必须在OP_ARITY中有定义"""
        for op in EXTENDED_BINARY_OPS:
            assert op in OP_ARITY, f"扩展二元算子 '{op}' 未在OP_ARITY中定义"
            assert OP_ARITY[op] == 2, f"二元算子 '{op}' 的arity应为2，但得到 {OP_ARITY[op]}"

    def test_basic_ops_arity_unchanged(self):
        """基础算子的arity应保持不变"""
        for op in UNARY_OPS:
            assert OP_ARITY[op] == 1, f"基础一元算子 '{op}' arity被修改"
        for op in BINARY_OPS:
            assert OP_ARITY[op] == 2, f"基础二元算子 '{op}' arity被修改"

    def test_leaf_node_is_complete(self):
        """叶子节点应被认为是完整的"""
        leaf = ExprNode(factor_idx=0)
        assert leaf.is_complete is True

    def test_unary_op_with_child_is_complete(self):
        """一元操作符有1个子节点时应完整"""
        root = ExprNode(op="neg")
        root.children.append(ExprNode(factor_idx=0))
        assert root.is_complete is True

    def test_unary_op_without_child_is_not_complete(self):
        """一元操作符无子节点时应不完整"""
        root = ExprNode(op="neg")
        assert root.is_complete is False

    def test_extended_unary_op_with_child_is_complete(self):
        """扩展一元操作符有1个子节点时应完整"""
        for op in EXTENDED_UNARY_OPS:
            root = ExprNode(op=op)
            root.children.append(ExprNode(factor_idx=0))
            assert root.is_complete is True, f"扩展一元算子 '{op}' 有子节点时应完整"

    def test_extended_unary_op_without_child_is_not_complete(self):
        """扩展一元操作符无子节点时应不完整"""
        for op in EXTENDED_UNARY_OPS:
            root = ExprNode(op=op)
            assert root.is_complete is False, f"扩展一元算子 '{op}' 无子节点时应不完整"

    def test_binary_op_with_two_children_is_complete(self):
        """二元操作符有2个子节点时应完整"""
        root = ExprNode(op="add")
        root.children.append(ExprNode(factor_idx=0))
        root.children.append(ExprNode(factor_idx=1))
        assert root.is_complete is True

    def test_binary_op_with_one_child_is_not_complete(self):
        """二元操作符只有1个子节点时应不完整"""
        root = ExprNode(op="add")
        root.children.append(ExprNode(factor_idx=0))
        assert root.is_complete is False

    def test_nested_expression_is_complete(self):
        """嵌套表达式在全部子节点完整时应完整"""
        # add(neg(factor_0), factor_1)
        root = ExprNode(op="add")
        neg_node = ExprNode(op="neg")
        neg_node.children.append(ExprNode(factor_idx=0))
        root.children.append(neg_node)
        root.children.append(ExprNode(factor_idx=1))
        assert root.is_complete is True

    def test_nested_expression_with_incomplete_child_is_not_complete(self):
        """嵌套表达式中任一子节点不完整时，根节点应不完整"""
        # add(neg(), factor_1) — neg缺少子节点
        root = ExprNode(op="add")
        neg_node = ExprNode(op="neg")  # 无子节点，不完整
        root.children.append(neg_node)
        root.children.append(ExprNode(factor_idx=1))
        assert root.is_complete is False, "子节点不完整时根节点应不完整"

    def test_deep_nested_incomplete_expression(self):
        """深层嵌套的不完整表达式应被正确识别"""
        # ts_std_5(ts_mean_10()) — ts_mean_10缺少子节点
        root = ExprNode(op="ts_std_5")
        inner = ExprNode(op="ts_mean_10")  # 无子节点，不完整
        root.children.append(inner)
        assert root.is_complete is False, "深层嵌套不完整节点应被识别"

    def test_extended_binary_op_with_two_children_is_complete(self):
        """扩展二元操作符有2个子节点时应完整"""
        for op in EXTENDED_BINARY_OPS:
            root = ExprNode(op=op)
            root.children.append(ExprNode(factor_idx=0))
            root.children.append(ExprNode(factor_idx=1))
            assert root.is_complete is True, f"扩展二元算子 '{op}' 有2子节点时应完整"

    def test_empty_node_is_not_complete(self):
        """空节点应不完整"""
        empty = ExprNode()
        assert empty.is_complete is False
