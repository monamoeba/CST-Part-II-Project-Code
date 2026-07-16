import pytest
import numpy as np
from src.utils.qccd_nodes import QubitIon
from src.compiler.qccd_qubits_to_ions import arrangeClusters
from src.compiler.qccd_color_qubits_to_ions import (
    regularColorPartition,
    regularColorPartition_vectorised,
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
    """Fixture to create ions with specific positions, uniquely indexed by list position."""
    def _create_ions(positions):
        ions = [QubitIon() for _ in positions]
        for i, (x, y) in enumerate(positions):
            if hasattr(ions[i], 'set'):
                ions[i].set(i, x, y)
        return ions
    return _create_ions


def _split_data_and_ancilla(circuit_cls, dist, create_ions):
    CCObj = circuit_cls(dist, 2)
    mids = CCObj.ancilla
    qtoid = CCObj.qtoid
    dcoords, mcoords = [], []
    for coord in qtoid.keys():
        (mcoords if qtoid[coord] in mids else dcoords).append(coord)
    return create_ions(mcoords), create_ions(dcoords)


@pytest.fixture
def initialise_positions(create_ions):
    def _initPos(dist):
        return _split_data_and_ancilla(ColorCodeCircuit666, dist, create_ions)
    return _initPos


@pytest.fixture
def initialise_488_positions(create_ions):
    def _initPos(dist):
        return _split_data_and_ancilla(ColorCodeCircuit488, dist, create_ions)
    return _initPos


PARTITION_VARIANTS = [
    (regularColorPartition, None),
    (regularColorPartition_vectorised, "bounded"),
    (regularColorPartition_vectorised, "unbounded_nn"),
    (regularColorPartition_vectorised, "knn"),
]


def _partition(partition_func, merge_strategy, mions, dions, trap_capacity):
    if merge_strategy is None:
        return partition_func(mions, dions, trap_capacity)
    return partition_func(mions, dions, trap_capacity, merge_strategy=merge_strategy)


# --- Qubit-count preservation across partition strategies and code families ---

@pytest.mark.parametrize("partition_func,merge_strategy", PARTITION_VARIANTS)
def test_qubits_in_equals_qubits_out_666(partition_func, merge_strategy, initialise_positions):
    mions, dions = initialise_positions(5)
    clusters = _partition(partition_func, merge_strategy, mions, dions, trap_capacity=3)
    assert len(mions) + len(dions) == sum(len(cluster[0]) for cluster in clusters)


@pytest.mark.parametrize("partition_func,merge_strategy", PARTITION_VARIANTS)
def test_qubits_in_equals_qubits_out_488(partition_func, merge_strategy, initialise_488_positions):
    mions, dions = initialise_488_positions(5)
    clusters = _partition(partition_func, merge_strategy, mions, dions, trap_capacity=3)
    assert len(mions) + len(dions) == sum(len(cluster[0]) for cluster in clusters)


# --- Capacity boundary behaviour, across code distances and partition strategies ---

@pytest.mark.parametrize("distance", [3, 5, 9])
@pytest.mark.parametrize("partition_func,merge_strategy", PARTITION_VARIANTS)
def test_high_capacity_single_cluster(partition_func, merge_strategy, distance, initialise_positions):
    mions, dions = initialise_positions(distance)
    clusters = _partition(partition_func, merge_strategy, mions, dions, trap_capacity=120)
    assert len(clusters) == 1


@pytest.mark.parametrize("distance", [3, 9])
def test_small_capacity_singleton_clusters(distance, initialise_positions):
    """At capacity 2, every ion ends up in its own cluster for these code distances."""
    mions, dions = initialise_positions(distance)
    clusters = regularColorPartition(mions, dions, 2)
    assert len(clusters) == len(mions) + len(dions)


@pytest.mark.parametrize("partition_func,merge_strategy", PARTITION_VARIANTS)
def test_small_capacity_respects_bound(partition_func, merge_strategy, initialise_positions):
    mions, dions = initialise_positions(5)
    trap_capacity = 2
    clusters = _partition(partition_func, merge_strategy, mions, dions, trap_capacity)
    assert all(len(cluster[0]) <= trap_capacity - 1 for cluster in clusters)


# --- Direct tests for the underlying merge strategies ---

def _make_synthetic_clusters(create_ions):
    # 3 clusters, each with 2 ions, spaced far apart
    position_groups = [
        [(0, 0), (0, 1)],
        [(10, 10), (10, 11)],
        [(20, 20), (20, 21)],
    ]
    clusters = []
    coords_to_ions = {}
    for group in position_groups:
        ions = create_ions(group)
        coords = np.array(group)
        clusters.append((coords, np.mean(coords, axis=0)))
        for coord, ion in zip(group, ions):
            coords_to_ions[coord] = ion
    return clusters, coords_to_ions


@pytest.mark.parametrize("merge_func", [
    _mergeUnderfilledClusters,
    _mergeUnderfilledClusters_kdtree,
    _merge_unbounded_nn,
    _merge_knn,
])
def test_merge_methods_preserve_ions_and_capacity(create_ions, merge_func):
    clusters, coords_to_ions = _make_synthetic_clusters(create_ions)
    trap_capacity = 4
    merged = merge_func(clusters, trap_capacity, coords_to_ions)
    assert sum(len(cluster[0]) for cluster in clusters) == sum(len(cluster[0]) for cluster in merged)
    assert all(len(cluster[0]) <= trap_capacity for cluster in merged)


# --- Vectorised implementation must agree with the reference implementation ---

@pytest.mark.parametrize("merge_strategy", ["bounded", "unbounded_nn", "knn"])
def test_vectorised_matches_reference_assignment(initialise_positions, merge_strategy):
    mions, dions = initialise_positions(5)
    trap_capacity = 3
    reference = regularColorPartition(mions, dions, trap_capacity)
    vectorised = regularColorPartition_vectorised(mions, dions, trap_capacity, merge_strategy=merge_strategy)

    def ion_indices(clusters):
        return set(ion.idx for group in clusters for ion in group[0])

    assert ion_indices(reference) == ion_indices(vectorised)


# --- Grid placement ---

def test_arrange_clusters(create_ions):
    grid_positions = [(0, 0), (1, 1), (2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 2), (5, 1), (6, 0)]
    clusters = [(create_ions([(i, j), (i, j)]), (i, j)) for i, j in grid_positions]
    arranged_positions = arrangeClusters(clusters, grid_positions)
    for (_, cluster_pos), pos in zip(clusters, arranged_positions):
        assert np.array_equal(cluster_pos, pos)
