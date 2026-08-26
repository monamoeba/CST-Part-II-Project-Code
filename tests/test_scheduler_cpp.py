"""Tests for the native `scheduler` extension (native/scheduler.cpp)."""

import pytest
import scheduler


def assert_valid_topological_order(result, operations, edges):
    """`edges` are (before, after) pairs that must hold; Kahn's algorithm doesn't guarantee a unique order, so we check constraints, not one exact sequence."""
    assert sorted(result) == sorted(operations), "result must contain every op exactly once"
    position = {op: i for i, op in enumerate(result)}
    for before, after in edges:
        assert position[before] < position[after], f"expected {before} before {after}, got {result}"


def test_no_shared_components_any_order_is_valid():
    result = scheduler.happensBeforeForOperations([0, 1, 2], [[10], [20], [30]])
    assert_valid_topological_order(result, [0, 1, 2], edges=[])


def test_simple_chain_through_two_components():
    result = scheduler.happensBeforeForOperations([0, 1, 2], [[100], [100, 200], [200]])
    assert_valid_topological_order(result, [0, 1, 2], edges=[(0, 1), (1, 2)])


def test_component_shared_by_many_ops_creates_edge_from_every_earlier_op():
    # op_i depends on every earlier op_j, not just the immediately preceding one - pins down one exact order.
    result = scheduler.happensBeforeForOperations([0, 1, 2, 3], [[5], [5], [5], [5]])
    assert result == [0, 1, 2, 3]


def test_independent_components_can_interleave():
    result = scheduler.happensBeforeForOperations([0, 1, 2, 3], [[1], [1], [2], [2]])
    assert_valid_topological_order(result, [0, 1, 2, 3], edges=[(0, 1), (2, 3)])


def test_duplicate_component_within_one_op_does_not_self_loop():
    # regression: a duplicate in one op's own component list (e.g. Split/Merge listing a trap twice)
    # must not create a self-dependency that makes the op unsatisfiable.
    result = scheduler.happensBeforeForOperations([0], [[5, 5]])
    assert result == [0]


def test_duplicate_component_in_earlier_op_still_creates_one_edge_to_later_op():
    result = scheduler.happensBeforeForOperations([0, 1], [[5, 5], [5]])
    assert_valid_topological_order(result, [0, 1], edges=[(0, 1)])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        scheduler.happensBeforeForOperations([0, 1], [[5]])
