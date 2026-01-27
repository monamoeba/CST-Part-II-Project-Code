import pytest
import numpy as np
# Assuming these imports are correct in your local environment
from src.utils.qccd_nodes import QubitIon
from src.compiler.qccd_qubits_to_ions import arrangeClusters
from src.compiler.qccd_color_qubits_to_ions import regularColorPartition
from src.color_code_utils.color_code_circuits.color_code_circuit_666 import ColorCodeCircuit666
from src.color_code_utils.color_code_circuits.color_code_circuit_488 import ColorCodeCircuit488

@pytest.fixture
def create_ions():
    """Fixture to create ions with specific positions."""
    def _create_ions(positions):
        ions = [QubitIon() for _ in positions]
        for i, (x, y) in enumerate(positions):
            if hasattr(ions[i], 'set'):
                ions[i].set(ions[i].idx, x, y)
        return ions
    return _create_ions

@pytest.fixture
def initialise_positions(create_ions):
    def _initPos(dist):
        CCObj = ColorCodeCircuit666(dist, 2)
        mids = CCObj.ancilla
        qtoid = CCObj.qtoid
        dcoords = []
        mcoords = []
        for coord in qtoid.keys():
            if qtoid[coord] in mids:
                mcoords.append(coord)
            else:
                dcoords.append(coord)
        
        dions = create_ions(dcoords)
        mions = create_ions(mcoords)
        return (mions, dions)
    
    return _initPos

@pytest.fixture
def initialise_488_positions(create_ions):
    def _initPos(dist):
        CCObj = ColorCodeCircuit488(dist,2)
        mids = CCObj.ancilla
        qtoid = CCObj.qtoid
        dcoords = []
        mcoords = []
        for coord in qtoid.keys():
            if qtoid[coord] in mids:
                mcoords.append(coord)
            else:
                dcoords.append(coord)
        
        dions = create_ions(dcoords)
        mions = create_ions(mcoords)
        return (mions, dions)
    
    return _initPos

def test_small_capacity_small_dist_clustering(initialise_positions):
    mions, dions = initialise_positions(3)
    trap_capacity = 2
    expected = len(mions) + len(dions)

    clusters = regularColorPartition(mions, dions, trap_capacity)

    assert len(clusters) == expected


def test_small_capacity_large_dist_clustering(initialise_positions):
    mions, dions = initialise_positions(9)
    trap_capacity = 2
    expected = len(mions) + len(dions)

    clusters = regularColorPartition(mions, dions, trap_capacity)

    assert len(clusters) == expected

def test_high_capacity_small_dist_clustering(initialise_positions):
    mions, dions = initialise_positions(3)
    trap_capacity = 120

    clusters = regularColorPartition(mions, dions, trap_capacity)

    assert len(clusters) == 1

def test_high_capacity_large_dist_clustering(initialise_positions):
    mions, dions = initialise_positions(9)
    trap_capacity = 120

    clusters = regularColorPartition(mions, dions, trap_capacity)

    assert len(clusters) == 1

def test_qubits_in_equals_qubits_out(initialise_positions):
    mions, dions = initialise_positions(5)
    trap_capacity = 3

    clusters = regularColorPartition(mions, dions, trap_capacity)

    total_qubits_in = len(mions) + len(dions)
    total_qubits_out = sum(len(cluster[0]) for cluster in clusters)

    assert total_qubits_in == total_qubits_out

def test_qubits_in_equals_qubits_out_488(initialise_488_positions):
    mions, dions = initialise_488_positions(5)
    trap_capacity = 3

    clusters = regularColorPartition(mions, dions, trap_capacity)

    total_qubits_in = len(mions) + len(dions)
    total_qubits_out = sum(len(cluster[0]) for cluster in clusters)

    assert total_qubits_in == total_qubits_out

def test_arrange_clusters(create_ions):
    grid_positions = [(0,0), (1,1), (2,0), (2,2), (3,1), (3,3), (4,0), (4,2), (5,1) ,(6,0)]
    clusters = [(create_ions([(i, j), (i, j)]), (i,j)) for i, j in grid_positions]
    arranged_positions = arrangeClusters(clusters, grid_positions)
    for (_, clpos), pos in zip(clusters, arranged_positions):
        assert np.array_equal(clpos, pos)