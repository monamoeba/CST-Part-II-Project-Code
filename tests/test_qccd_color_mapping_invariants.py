from hypothesis import given, strategies as st
from src.utils.qccd_nodes import QubitIon
from src.compiler.qccd_color_qubits_to_ions import regularColorPartition, regularColorPartition_vectorised
from src.color_code_utils.color_code_circuits.color_code_circuit_666 import ColorCodeCircuit666
from src.color_code_utils.color_code_circuits.color_code_circuit_488 import ColorCodeCircuit488

def make_ions(positions):
    ions = [QubitIon() for _ in positions]
    for idx, (ion, (x, y)) in enumerate(zip(ions, positions)):
        if hasattr(ion, 'set'):
            ion.set(idx, x, y)
    return ions

@given(
    st.integers(min_value=1, max_value=7).map(lambda x: 2 * x + 1),
    st.integers(min_value=2, max_value=10)
)
def test_color_code_666_partition_invariants(distance, trap_capacity):
    circuit = ColorCodeCircuit666(distance, rounds=2)
    positions = set()
    for tile in circuit._tiles:
        positions.update(tile.qubits)
    ions = make_ions(list(positions))
    clusters = regularColorPartition(ions, [], trap_capacity)
    all_ions = [ion for group, _ in clusters for ion in group]
    indices = [ion.idx for ion in all_ions]
    assert all(len(group) <= trap_capacity - 1 for group, _ in clusters)
    assert len(all_ions) == len(ions)
    assert set(indices) == set(range(len(ions)))
    assert len(indices) == len(set(indices))

@given(
    st.integers(min_value=1, max_value=7).map(lambda x: 2 * x + 1),
    st.integers(min_value=2, max_value=10)
)
def test_color_code_666_vectorised_partition_invariants(distance, trap_capacity):
    circuit = ColorCodeCircuit666(distance, rounds=2)
    positions = set()
    for tile in circuit._tiles:
        positions.update(tile.qubits)
    ions = make_ions(list(positions))
    clusters = regularColorPartition_vectorised(ions, [], trap_capacity)
    all_ions = [ion for group, _ in clusters for ion in group]
    indices = [ion.idx for ion in all_ions]
    assert all(len(group) <= trap_capacity - 1 for group, _ in clusters)
    assert len(all_ions) == len(ions)
    assert set(indices) == set(range(len(ions)))
    assert len(indices) == len(set(indices))

@given(
    st.integers(min_value=1, max_value=7).map(lambda x: 2 * x + 1),
    st.integers(min_value=2, max_value=10)
)
def test_color_code_488_partition_invariants(distance, trap_capacity):
    circuit = ColorCodeCircuit488(distance, rounds=2)
    positions = set()
    for tile in circuit._tiles:
        positions.update(tile.qubits)
    ions = make_ions(list(positions))
    clusters = regularColorPartition(ions, [], trap_capacity)
    all_ions = [ion for group, _ in clusters for ion in group]
    indices = [ion.idx for ion in all_ions]
    assert all(len(group) <= trap_capacity - 1 for group, _ in clusters)
    assert len(all_ions) == len(ions)
    assert set(indices) == set(range(len(ions)))
    assert len(indices) == len(set(indices))

@given(
    st.integers(min_value=1, max_value=7).map(lambda x: 2 * x + 1),
    st.integers(min_value=2, max_value=10)
)
def test_color_code_488_vectorised_partition_invariants(distance, trap_capacity):
    circuit = ColorCodeCircuit488(distance, rounds=2)
    positions = set()
    for tile in circuit._tiles:
        positions.update(tile.qubits)
    ions = make_ions(list(positions))
    clusters = regularColorPartition_vectorised(ions, [], trap_capacity)
    all_ions = [ion for group, _ in clusters for ion in group]
    indices = [ion.idx for ion in all_ions]
    assert all(len(group) <= trap_capacity - 1 for group, _ in clusters)
    assert len(all_ions) == len(ions)
    assert set(indices) == set(range(len(ions)))
    assert len(indices) == len(set(indices))
