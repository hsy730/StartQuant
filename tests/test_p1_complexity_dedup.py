"""P1 改造验证测试：算子加权复杂度 + SymPy 规范形去重"""
import sys
sys.path.insert(0, '.')

from backend.services.factor_primitives import (
    create_pset,
    compute_weighted_complexity,
    zobrist_hash,
    sympy_canonical_key,
    simplify_gp_expression,
)
from deap import gp


def test_weighted_complexity():
    """测试算子加权复杂度"""
    print("=== compute_weighted_complexity tests ===")

    pset = create_pset(n_factors=2, extended=True)

    # add(factor_0, factor_1) = 1.0 + 0.5 + 0.5 = 2.0
    tree1 = gp.PrimitiveTree.from_string("add(factor_0, factor_1)", pset)
    c1 = compute_weighted_complexity(tree1)
    assert abs(c1 - 2.0) < 1e-6, f"add(f0,f1) expected 2.0, got {c1}"
    print(f"  OK: add(factor_0, factor_1) = {c1} (expected: 2.0)")

    # ts_corr_5(factor_0, factor_1) = 4.0 + 0.5 + 0.5 = 5.0
    tree2 = gp.PrimitiveTree.from_string("ts_corr_5(factor_0, factor_1)", pset)
    c2 = compute_weighted_complexity(tree2)
    assert abs(c2 - 5.0) < 1e-6, f"ts_corr_5 expected 5.0, got {c2}"
    print(f"  OK: ts_corr_5(factor_0, factor_1) = {c2} (expected: 5.0)")

    # log(factor_0) = 2.0 + 0.5 = 2.5
    tree3 = gp.PrimitiveTree.from_string("log(factor_0)", pset)
    c3 = compute_weighted_complexity(tree3)
    assert abs(c3 - 2.5) < 1e-6, f"log expected 2.5, got {c3}"
    print(f"  OK: log(factor_0) = {c3} (expected: 2.5)")

    # ts_mean_5(factor_0) = 3.0 + 0.5 = 3.5
    tree4 = gp.PrimitiveTree.from_string("ts_mean_5(factor_0)", pset)
    c4 = compute_weighted_complexity(tree4)
    assert abs(c4 - 3.5) < 1e-6, f"ts_mean_5 expected 3.5, got {c4}"
    print(f"  OK: ts_mean_5(factor_0) = {c4} (expected: 3.5)")

    # 嵌套: add(log(factor_0), ts_corr_5(factor_0, factor_1))
    # = 1.0 + (2.0+0.5) + (4.0+0.5+0.5) = 8.5
    tree5 = gp.PrimitiveTree.from_string(
        "add(log(factor_0), ts_corr_5(factor_0, factor_1))", pset
    )
    c5 = compute_weighted_complexity(tree5)
    assert abs(c5 - 8.5) < 1e-6, f"nested expected 8.5, got {c5}"
    print(f"  OK: add(log(f0), ts_corr_5(f0,f1)) = {c5} (expected: 8.5)")

    # 验证加权复杂度 != 朴素节点数
    assert c5 != float(len(tree5)), "weighted complexity should differ from node count"
    print(f"  OK: weighted ({c5}) != node_count ({float(len(tree5))})")


def test_sympy_canonical_key():
    """测试 SymPy 规范形去重"""
    print("\n=== sympy_canonical_key tests ===")

    pset = create_pset(n_factors=2, extended=True)

    # add(factor_0, sub(factor_1, factor_0)) ≡ factor_1
    tree_a = gp.PrimitiveTree.from_string(
        "add(factor_0, sub(factor_1, factor_0))", pset
    )
    tree_b = gp.PrimitiveTree.from_string("factor_1", pset)

    key_a = sympy_canonical_key(tree_a)
    key_b = sympy_canonical_key(tree_b)

    print(f"  add(f0, sub(f1, f0)) canonical: {key_a}")
    print(f"  factor_1 canonical:             {key_b}")

    # 两者 SymPy 规范形应相同
    assert key_a == key_b, (
        f"Algebraic equivalence failed:\n  {key_a}\n  {key_b}"
    )
    print(f"  OK: add(f0, sub(f1, f0)) ≡ factor_1 (same canonical key)")

    # 但 Zobrist hash 应不同（结构不同）
    hash_a = zobrist_hash(tree_a)
    hash_b = zobrist_hash(tree_b)
    assert hash_a != hash_b, "zobrist should differ for structurally different trees"
    print(f"  OK: zobrist differs ({hash_a != hash_b}) — only SymPy detects equivalence")

    # mul(factor_0, div(factor_0, factor_0)) ≡ factor_0
    tree_c = gp.PrimitiveTree.from_string(
        "mul(factor_0, div(factor_0, factor_0))", pset
    )
    tree_d = gp.PrimitiveTree.from_string("factor_0", pset)
    key_c = sympy_canonical_key(tree_c)
    key_d = sympy_canonical_key(tree_d)
    assert key_c == key_d, f"mul(f0, div(f0,f0)) should ≡ factor_0"
    print(f"  OK: mul(f0, div(f0, f0)) ≡ factor_0 (same canonical key)")

    # 非等价表达式应有不同 key
    tree_e = gp.PrimitiveTree.from_string("add(factor_0, factor_1)", pset)
    tree_f = gp.PrimitiveTree.from_string("sub(factor_0, factor_1)", pset)
    key_e = sympy_canonical_key(tree_e)
    key_f = sympy_canonical_key(tree_f)
    assert key_e != key_f, "add(f0,f1) and sub(f0,f1) should have different keys"
    print(f"  OK: add(f0,f1) ≠ sub(f0,f1) (different canonical keys)")


def test_import():
    """测试导入"""
    print("\n=== import test ===")
    from backend.services.genetic_factor_mining_service import GeneticFactorMiningService
    print("  OK: GeneticFactorMiningService imported successfully")


if __name__ == "__main__":
    test_weighted_complexity()
    test_sympy_canonical_key()
    test_import()
    print("\n========================================")
    print("ALL P1 TESTS PASSED")
    print("========================================")
