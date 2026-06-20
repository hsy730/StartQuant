"""Test Zobrist hash and SymPy simplification functions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.factor_primitives import simplify_gp_expression, zobrist_hash
from deap import gp
from backend.services.factor_primitives import create_pset_numpy


def test_simplify():
    """Test SymPy expression simplification."""
    tests = [
        ("add(factor_0, sub(factor_1, factor_0))", "factor_1"),
        ("mul(factor_0, div(factor_0, factor_0))", "factor_0"),
        ("add(factor_0, 0)", "factor_0"),
        ("mul(factor_0, 1)", "factor_0"),
        ("neg(neg(factor_0))", "factor_0"),
        ("ts_mean_5(add(factor_0, 0))", "ts_mean_5(factor_0)"),
    ]

    print("=== simplify_gp_expression tests ===")
    all_ok = True
    for expr, expected in tests:
        result = simplify_gp_expression(expr)
        status = "OK" if result == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {status}: {expr} -> {result} (expected: {expected})")
    return all_ok


def test_zobrist():
    """Test Zobrist hash for commutative/non-commutative detection."""
    pset = create_pset_numpy(3, extended=True)

    tree1 = gp.PrimitiveTree.from_string("add(factor_0, factor_1)", pset)
    tree2 = gp.PrimitiveTree.from_string("add(factor_1, factor_0)", pset)
    tree3 = gp.PrimitiveTree.from_string("sub(factor_0, factor_1)", pset)
    tree4 = gp.PrimitiveTree.from_string("sub(factor_1, factor_0)", pset)

    h1 = zobrist_hash(tree1)
    h2 = zobrist_hash(tree2)
    h3 = zobrist_hash(tree3)
    h4 = zobrist_hash(tree4)

    print("\n=== zobrist_hash tests ===")
    ok1 = h1 == h2
    ok2 = h3 != h4
    ok3 = h1 != h3
    print(f"  {'OK' if ok1 else 'FAIL'}: add(a,b) == add(b,a) (commutative): {ok1}")
    print(f"  {'OK' if ok2 else 'FAIL'}: sub(a,b) != sub(b,a) (non-commutative): {ok2}")
    print(f"  {'OK' if ok3 else 'FAIL'}: add != sub: {ok3}")
    return ok1 and ok2 and ok3


def test_import():
    """Test that the mining service imports correctly."""
    print("\n=== import test ===")
    try:
        from backend.services.genetic_factor_mining_service import GeneticFactorMiningService
        print("  OK: GeneticFactorMiningService imported successfully")
        return True
    except Exception as e:
        print(f"  FAIL: Import error: {e}")
        return False


if __name__ == "__main__":
    r1 = test_simplify()
    r2 = test_zobrist()
    r3 = test_import()
    print(f"\n{'='*40}")
    if r1 and r2 and r3:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
