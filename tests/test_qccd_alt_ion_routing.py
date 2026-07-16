import os
import pytest
import networkx as nx
from src.compiler.qccd_alt_ion_routing import naiveAltIonRouting, _determineRouted
from src.simulator.qccd_circuit import QCCDCircuit
from src.utils.qccd_operations_on_qubits import TwoQubitMSGate
import numpy as np

STIM_PATH = os.path.join(os.path.dirname(__file__), "..", "color_code_experiments", "superdense_color_code_d5_r3_p1000.stim")
DATA_QUBIT_IDS = "0 2 5 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 35"
QCOUNT = 37
TEST_CAPACITY = 2
ROWS = 10
COLS = 10


def _make_circuit_and_arch(capacity):
    with open(STIM_PATH, "r") as f:
        raw = f.read()
    d_qubits = [int(i) for i in DATA_QUBIT_IDS.strip().split()]
    circuit = QCCDCircuit(raw)
    circuit.dataQubitsIdxs = d_qubits
    arch, (instructions, _) = circuit.processColorCircuitAugmentedGrid(
        rows=ROWS,
        cols=COLS,
        trapCapacity=capacity,
        dataQubitsIdxs=d_qubits,
        cluster_merge_strategy='bounded',
    )
    arch.refreshGraph()
    return arch, instructions


@pytest.fixture
def setup_circuit():
    return _make_circuit_and_arch(TEST_CAPACITY)


def test_no_operations(setup_circuit):
    arch, _ = setup_circuit
    allOps, barriers = naiveAltIonRouting(arch, [], TEST_CAPACITY)
    assert allOps == []
    assert barriers == []


def test_single_operation(setup_circuit):
    arch, instructions = setup_circuit
    allOps, barriers = naiveAltIonRouting(arch, [instructions[0]], TEST_CAPACITY)
    assert len(barriers) == 1
    assert len(allOps) >= 1


def test_with_full_routing(setup_circuit):
    arch, instructions = setup_circuit
    allOps, _ = naiveAltIonRouting(arch, instructions, TEST_CAPACITY)
    for node in arch.nodes.values():
        assert len(node.ions) <= node.capacity
    assert len(allOps) >= len(instructions)


def test_high_trap_capacity():
    high_capacity = 2 * QCOUNT + 1
    arch, instructions = _make_circuit_and_arch(high_capacity)
    allOps, _ = naiveAltIonRouting(arch, instructions, high_capacity)
    assert len(allOps) == len(instructions)


def test_ancilla_ancilla_routing(setup_circuit):
    arch, instructions = setup_circuit
    aa_ops = [
        op for op in instructions
        if isinstance(op, TwoQubitMSGate) and op.ions[0].label[0] == op.ions[1].label[0]
    ]
    assert len(aa_ops) > 0, "superdense circuit should contain ancilla-ancilla gates"
    allOps, barriers = naiveAltIonRouting(arch, aa_ops, TEST_CAPACITY)
    assert len(allOps) >= len(aa_ops)
    for node in arch.nodes.values():
        assert len(node.ions) <= node.capacity


def _first_aa_pair(instructions):
    for op in instructions:
        if isinstance(op, TwoQubitMSGate) and op.ions[0].label[0] == op.ions[1].label[0]:
            return op.ions[0], op.ions[1]
    return None


def _first_asymmetric_aa_pair(arch, instructions):
    for op in instructions:
        if not isinstance(op, TwoQubitMSGate):
            continue
        ion1, ion2 = op.ions
        if ion1.label[0] != ion2.label[0]:
            continue
        d1 = nx.shortest_path_length(arch.graph, ion1.idx, ion2.parent.idx)
        d2 = nx.shortest_path_length(arch.graph, ion2.idx, ion1.parent.idx)
        if d1 != d2:
            return ion1, ion2, d1, d2
    return None


def test_determine_routed_min_distance(setup_circuit):
    arch, instructions = setup_circuit
    pair = _first_aa_pair(instructions)
    assert pair is not None, "superdense circuit should contain ancilla-ancilla gates"
    ion1, ion2 = pair
    d1 = nx.shortest_path_length(arch.graph, ion1.idx, ion2.parent.idx)
    d2 = nx.shortest_path_length(arch.graph, ion2.idx, ion1.parent.idx)
    r, _ = _determineRouted(ion1, ion2, arch)
    assert r is (ion1 if d1 <= d2 else ion2)


def test_determine_routed_commutative(setup_circuit):
    # Equal-distance pairs are intentionally not commutative (first arg wins the tiebreak),
    # so this test only runs when a genuinely asymmetric pair exists in the arch.
    arch, instructions = setup_circuit
    result = _first_asymmetric_aa_pair(arch, instructions)
    if result is None:
        pytest.skip("All same-type pairs are equidistant in this arch; commutativity only applies to asymmetric pairs")
    ion1, ion2, _, _ = result
    r1, s1 = _determineRouted(ion1, ion2, arch)
    r2, s2 = _determineRouted(ion2, ion1, arch)
    assert r1 is r2
    assert s1 is s2
