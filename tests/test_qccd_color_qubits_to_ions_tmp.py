import pytest
import numpy as np
from src.utils.qccd_nodes import QubitIon
from src.compiler.qccd_color_qubits_to_ions import (
    regularColorPartition,
    regularColorPartition_vectorised,
    TriangularPartitionIons,
    TriangularPartitionIons_vectorised,
    _mergeUnderfilledClusters,
    _mergeUnderfilledClusters_kdtree,
    _merge_unbounded_nn,
    _merge_knn,
)
from src.color_code_utils.color_code_circuits.color_code_circuit_666 import ColorCodeCircuit666
from src.color_code_utils.color_code_circuits.color_code_circuit_488 import ColorCodeCircuit488

# --- Fixtures ---
@pytest.fixture
def create_ions():
    def _create_ions(positions):
        ions = [QubitIon() for _ in positions]
        for i, (x, y) in enumerate(positions):
            if hasattr(ions[i], 'set'):
                ions[i].set(i, x, y)
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
        CCObj = ColorCodeCircuit488(dist, 2)
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

# --- Parametrized tests for partitioning strategies ---
@pytest.mark.parametrize("partition_func", [
    regularColorPartition,
    regularColorPartition_vectorised,
])
def test_qubits_in_equals_qubits_out_all(partition_func, initialise_positions):
    mions, dions = initialise_positions(5)
    trap_capacity = 3
    clusters = partition_func(mions, dions, trap_capacity)
    total_qubits_in = len(mions) + len(dions)
    total_qubits_out = sum(len(cluster[0]) for cluster in clusters)
    assert total_qubits_in == total_qubits_out

@pytest.mark.parametrize("partition_func", [
    regularColorPartition,
    regularColorPartition_vectorised,
])
def test_high_capacity_single_cluster(partition_func, initialise_positions):
    mions, dions = initialise_positions(5)
    trap_capacity = 100
    clusters = partition_func(mions, dions, trap_capacity)
    assert len(clusters) == 1

@pytest.mark.parametrize("partition_func", [
    regularColorPartition,
    regularColorPartition_vectorised,
])
def test_small_capacity_many_clusters(partition_func, initialise_positions):
    mions, dions = initialise_positions(5)
    trap_capacity = 2
    clusters = partition_func(mions, dions, trap_capacity)
    assert all(len(cluster[0]) <= trap_capacity - 1 for cluster in clusters)

# --- Direct tests for merging methods ---
def make_synthetic_clusters(create_ions):
    # 3 clusters, each with 2 ions, spaced apart
    positions = [
        [(0, 0), (0, 1)],
        [(10, 10), (10, 11)],
        [(20, 20), (20, 21)],
    ]
    clusters = []
    for pos_group in positions:
        ions = create_ions(pos_group)
        coords = np.array(pos_group)
        center = np.mean(coords, axis=0)
        clusters.append((coords, center))
    coordsToIons = {(c[0], c[1]): i for group in positions for c, i in zip(group, create_ions(group))}
    return clusters, coordsToIons

@pytest.mark.parametrize("merge_func", [
    _mergeUnderfilledClusters,
    _mergeUnderfilledClusters_kdtree,
    _merge_unbounded_nn,
    _merge_knn,
])
def test_merge_methods_basic(create_ions, merge_func):
    clusters, coordsToIons = make_synthetic_clusters(create_ions)
    trap_capacity = 4
    merged = merge_func(clusters, trap_capacity, coordsToIons)
    # All ions should be present in output
    total_ions_in = sum(len(c[0]) for c in clusters)
    total_ions_out = sum(len(c[0]) for c in merged)
    assert total_ions_in == total_ions_out
    # No cluster should exceed trap_capacity
    assert all(len(c[0]) <= trap_capacity for c in merged)

# --- Vectorised vs non-vectorised consistency ---
def test_vectorised_vs_nonvectorised_consistency(initialise_positions):
    mions, dions = initialise_positions(5)
    trap_capacity = 3
    clusters1 = regularColorPartition(mions, dions, trap_capacity)
    clusters2 = regularColorPartition_vectorised(mions, dions, trap_capacity)
    # Compare sorted sets of ion indices in clusters
    def all_ion_indices(clusters):
        return set(ion.idx for group in clusters for ion in group[0])
    assert all_ion_indices(clusters1) == all_ion_indices(clusters2)

# --- Test with 488 code ---
def test_qubits_in_equals_qubits_out_488(initialise_488_positions):
    mions, dions = initialise_488_positions(5)
    trap_capacity = 3
    clusters = regularColorPartition(mions, dions, trap_capacity)
    total_qubits_in = len(mions) + len(dions)
    total_qubits_out = sum(len(cluster[0]) for cluster in clusters)
    assert total_qubits_in == total_qubits_out

# --- Arrange clusters test (unchanged) ---
from src.compiler.qccd_qubits_to_ions import arrangeClusters

def test_arrange_clusters(create_ions):
    grid_positions = [(0,0), (1,1), (2,0), (2,2), (3,1), (3,3), (4,0), (4,2), (5,1) ,(6,0)]
    clusters = [(create_ions([(i, j), (i, j)]), (i,j)) for i, j in grid_positions]
    arranged_positions = arrangeClusters(clusters, grid_positions)
    for (_, clpos), pos in zip(clusters, arranged_positions):
        assert np.array_equal(clpos, pos)
