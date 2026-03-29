from hypothesis import given, strategies as st
import numpy as np
import pytest
from src.utils.qccd_nodes import QubitIon
from src.compiler.qccd_color_qubits_to_ions import regularColorPartition, regularColorPartition_vectorised

def make_ions(positions):
    ions = [QubitIon() for _ in positions]
    for idx, (ion, (x, y)) in enumerate(zip(ions, positions)):
        ion.idx = idx
        if hasattr(ion, 'set'):
            ion.set(idx, x, y)
    return ions

@given(
    st.lists(st.tuples(st.floats(0, 100), st.floats(0, 100)), min_size=1, max_size=30),
    st.integers(min_value=2, max_value=10)
)
def test_regular_partition_invariants(positions, trap_capacity):
    ions = make_ions(positions)
    clusters = regularColorPartition(ions, [], trap_capacity)
    all_ions = [ion for group, _ in clusters for ion in group]
    indices = [ion.idx for ion in all_ions]
    assert all(len(group) <= trap_capacity - 1 for group, _ in clusters)
    assert len(all_ions) == len(ions)
    assert set(indices) == set(range(len(ions)))
    assert len(indices) == len(set(indices))

@given(
    st.lists(st.tuples(st.floats(0, 100), st.floats(0, 100)), min_size=1, max_size=30),
    st.integers(min_value=2, max_value=10)
)
def test_vectorised_partition_invariants(positions, trap_capacity):
    ions = make_ions(positions)
    clusters = regularColorPartition_vectorised(ions, [], trap_capacity)
    all_ions = [ion for group, _ in clusters for ion in group]
    indices = [ion.idx for ion in all_ions]
    assert all(len(group) <= trap_capacity - 1 for group, _ in clusters)
    assert len(all_ions) == len(ions)
    assert set(indices) == set(range(len(ions)))
    assert len(indices) == len(set(indices))
