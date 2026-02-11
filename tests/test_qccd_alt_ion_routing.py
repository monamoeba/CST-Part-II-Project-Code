import pytest
from src.compiler.qccd_alt_ion_routing import naiveAltIonRouting
from src.simulator.qccd_circuit import QCCDCircuit
import numpy as np

TEST_DISTANCE = 5
TEST_CAPACITY = 2

@pytest.fixture
def setup_circuit():
    """
    Generate a QCCDCircuit for testing.
    """

    path = ".\\color_code_experiments\\r=20,d=5,p=0.001,noise=uniform,c=midout_color_code_488_Z,gates=all.stim"

    with open(path, "r") as f:
        raw = f.read()

    circuit = QCCDCircuit(raw)

    s = "17 15 14 13 12 11 10 5 4 3 0"
    d_qubits = list(int(i) for i in s.strip().split(" "))
    circuit.dataQubitsIdxs = list(d_qubits)
    qcount = 21
    nrowsNeeded = int(np.sqrt(qcount))+2

    arch, (instructions, _) = circuit.processColorCircuitAugmentedGrid(
        rows=nrowsNeeded, 
        cols=nrowsNeeded,
        trapCapacity=TEST_CAPACITY, 
        dataQubitIdxs=d_qubits
    )
    arch.refreshGraph()

    return arch, instructions

def test_no_operations(setup_circuit):
    """
    Test `ionRouting` when no operations are provided.
    """
    arch, _ = setup_circuit
    operations = []

    allOps, barriers = naiveAltIonRouting(arch, operations, TEST_CAPACITY)

    assert allOps == []
    assert barriers == []

def test_single_operation(setup_circuit):
    """
    Test `ionRouting` with a single operation.
    """
    arch, instructions = setup_circuit

    # Use the first operation only
    operations = [instructions[0]]

    allOps, barriers = naiveAltIonRouting(arch, operations, TEST_CAPACITY)

    assert len(allOps) == 1
    assert barriers == [1]

def test_with_full_routing(setup_circuit):
    """
    Test `ionRouting` with a realistic set of instructions requiring routing.
    """
    arch, instructions = setup_circuit
   
    allOps, _ = naiveAltIonRouting(arch, instructions, TEST_CAPACITY)
    for op in allOps:
        op.run()
        for node in arch.nodes.values():
            assert len(node.ions)<=node.capacity

    assert len(allOps) >= len(instructions)


def test_high_trap_capacity(setup_circuit):
    """
    Test `ionRouting` with high trap capacity.
    """
    trapCapacity = 2*TEST_DISTANCE**2+1
    path = ".\\color_code_experiments\\r=20,d=5,p=0.001,noise=uniform,c=midout_color_code_488_Z,gates=all.stim"

    with open(path, "r") as f:
        raw = f.read()

    circuit = QCCDCircuit(raw)

    s = "17 15 14 13 12 11 10 5 4 3 0"
    d_qubits = list(int(i) for i in s.strip().split(" "))
    circuit.dataQubitsIdxs = list(d_qubits)
    qcount = 21
    nrowsNeeded = int(np.sqrt(qcount))+2

    arch, (instructions, _) = circuit.processColorCircuitAugmentedGrid(
        rows=nrowsNeeded, 
        cols=nrowsNeeded,
        trapCapacity=TEST_CAPACITY, 
        dataQubitIdxs=d_qubits
    )
    arch.refreshGraph()
    
    allOps, _ = naiveAltIonRouting(arch, instructions, trapCapacity)

    # High capacity means there should be no ion movements
    assert len(allOps) == len(instructions)
