"""Regression tests for the incremental QCCDArch.refreshGraph(dirty=...) path and dirtyComponentsForOp()."""

import pytest
from src.utils.qccd_arch import QCCDArch, dirtyComponentsForOp
from src.utils.qccd_nodes import QubitIon
from src.utils.qccd_operations import Split, Move, Merge, JunctionCrossing
from src.utils.qccd_operations_on_qubits import (
    GateSwap,
    TwoQubitMSGate,
    OneQubitGate,
    Measurement,
    QubitReset,
)

QUBITS_PER_TRAP = 4


def _build_topology():
    """Same shape as test_qccd_parallelisation.py::setup_arch (4 traps, 2 junctions, 5 crossings), nothing run yet. A plain function, not a fixture, so it can be called repeatedly for independent architectures."""
    arch = QCCDArch()
    arch.SIZING = 0.5

    # Capacity well above QUBITS_PER_TRAP so remove-then-add-again sequences
    # within a trap (e.g. GateSwap) never land exactly on the capacity boundary.
    trap_capacity = QUBITS_PER_TRAP + 3

    ions11 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap11 = arch.addManipulationTrap(x=0, y=0, ions=ions11, capacity=trap_capacity)

    ions12 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap12 = arch.addManipulationTrap(x=1, y=0, ions=ions12, capacity=trap_capacity)

    junctionL = arch.addJunction(x=0, y=0.5)
    junctionR = arch.addJunction(x=1, y=0.5)

    ions21 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap21 = arch.addManipulationTrap(x=0, y=1, ions=ions21, capacity=trap_capacity)

    ions22 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap22 = arch.addManipulationTrap(x=1, y=1, ions=ions22, capacity=trap_capacity)

    crossing11 = arch.addEdge(trap11, junctionL)
    crossing12 = arch.addEdge(trap12, junctionR)
    crossing21 = arch.addEdge(trap21, junctionL)
    crossing22 = arch.addEdge(trap22, junctionR)
    crossingLR = arch.addEdge(junctionL, junctionR)

    arch.refreshGraph()  # required initial full refresh

    return {
        "arch": arch,
        "trap11": trap11, "trap12": trap12, "trap21": trap21, "trap22": trap22,
        "junctionL": junctionL, "junctionR": junctionR,
        "ions11": ions11, "ions12": ions12, "ions21": ions21, "ions22": ions22,
        "crossing11": crossing11, "crossing12": crossing12,
        "crossing21": crossing21, "crossing22": crossing22, "crossingLR": crossingLR,
    }


@pytest.fixture
def topology():
    return _build_topology()


def _op_signature(op):
    """Content-based fingerprint (type + referenced component idxs): full and incremental rebuilds each construct fresh Operation objects, so identity comparison can't work."""
    parts = [type(op).__name__]
    for attr in ("_trap", "_crossing", "_junction", "_ion", "_ion1", "_ion2"):
        component = getattr(op, attr, None)
        if component is not None and hasattr(component, "idx"):
            parts.append((attr, component.idx))
    return tuple(parts)


def _snapshot(arch):
    """Fingerprint of everything refreshGraph() owns: graph nodes/edges, per-edge operations, and crossingEdges."""
    graph = arch.graph
    edge_ops = {
        edge: tuple(sorted(_op_signature(op) for op in graph.edges[edge]["operations"]))
        for edge in graph.edges
    }
    crossing_edges = {key: crossing.idx for key, crossing in arch.crossingEdges.items()}
    return {
        "nodes": frozenset(graph.nodes),
        "edges": frozenset(graph.edges),
        "edge_ops": edge_ops,
        "crossing_edges": crossing_edges,
    }


def _assert_incremental_matches_full(arch, dirty):
    """Runs both refresh strategies on the same physical state and compares. Full rebuild is correct by construction, so any mismatch is an incremental-path bug."""
    arch.refreshGraph(dirty=dirty)
    incremental_snapshot = _snapshot(arch)
    arch.refreshGraph()
    full_snapshot = _snapshot(arch)
    assert incremental_snapshot == full_snapshot


class TestDirtyComponentsForOp:
    """Exact dirtyComponentsForOp contract per operation type."""

    def test_split_touches_trap_crossing_and_both_endpoints(self, topology):
        trap, crossing = topology["trap11"], topology["crossing11"]
        op = Split.physicalOperation(trap, crossing)
        assert dirtyComponentsForOp(op) == {trap, crossing, *crossing.connection}

    def test_merge_touches_trap_crossing_and_both_endpoints(self, topology):
        trap, crossing = topology["trap11"], topology["crossing11"]
        op = Merge.physicalOperation(trap, crossing)
        assert dirtyComponentsForOp(op) == {trap, crossing, *crossing.connection}

    def test_move_touches_only_the_crossing(self, topology):
        crossing = topology["crossing11"]
        op = Move.physicalOperation(crossing)
        assert dirtyComponentsForOp(op) == {crossing}

    def test_junction_crossing_touches_junction_and_crossing(self, topology):
        junction, crossing = topology["junctionL"], topology["crossing11"]
        op = JunctionCrossing.physicalOperation(junction, crossing)
        assert dirtyComponentsForOp(op) == {junction, crossing}

    def test_gate_swap_physical_operation_already_includes_its_trap(self, topology):
        trap = topology["trap11"]
        ion1, ion2 = topology["ions11"][0], topology["ions11"][1]
        # physicalOperation() calls setTrap() internally, so trap is already
        # present; the ions are filtered out (not Trap/Junction/Crossing).
        op = GateSwap.physicalOperation(trap=trap, ion1=ion1, ion2=ion2)
        assert dirtyComponentsForOp(op) == {trap}

    def test_two_qubit_gate_built_without_a_trap_has_no_dirty_components_until_settrap(self, topology):
        # mirrors circuit parsing: qubitOperation() before routing picks a trap
        ion1, ion2 = topology["ions22"][0], topology["ions22"][1]
        op = TwoQubitMSGate.qubitOperation(ion1=ion1, ion2=ion2)
        assert dirtyComponentsForOp(op) == set()

        trap = topology["trap22"]
        op.setTrap(trap)
        assert dirtyComponentsForOp(op) == {trap}


class TestIncrementalMatchesFullRebuild:
    """Each op type, run then incrementally refreshed, must match a full rebuild from the same state."""

    def test_split(self, topology):
        arch, trap, crossing = topology["arch"], topology["trap11"], topology["crossing11"]
        op = Split.physicalOperation(trap, crossing)
        op.run()
        _assert_incremental_matches_full(arch, dirtyComponentsForOp(op))

    def test_move(self, topology):
        # Split and Move both run before any refresh - dirty must cover both,
        # not just Move, or Split's effect on trap11 gets silently lost.
        arch, trap, crossing = topology["arch"], topology["trap11"], topology["crossing11"]
        dirty = set()
        split_op = Split.physicalOperation(trap, crossing)
        split_op.run()
        dirty.update(dirtyComponentsForOp(split_op))
        move_op = Move.physicalOperation(crossing)
        move_op.run()
        dirty.update(dirtyComponentsForOp(move_op))
        _assert_incremental_matches_full(arch, dirty)

    def test_junction_crossing(self, topology):
        arch = topology["arch"]
        trap, crossing, junction = topology["trap11"], topology["crossing11"], topology["junctionL"]
        dirty = set()
        for op in (
            Split.physicalOperation(trap, crossing),
            Move.physicalOperation(crossing),
            JunctionCrossing.physicalOperation(junction, crossing),
        ):
            op.run()
            dirty.update(dirtyComponentsForOp(op))
        _assert_incremental_matches_full(arch, dirty)

    def test_merge(self, topology):
        arch = topology["arch"]
        trap11, crossing11, junctionL = topology["trap11"], topology["crossing11"], topology["junctionL"]
        trap21, crossing21 = topology["trap21"], topology["crossing21"]
        # trap11 -> junction -> trap21, matching test_qccd_parallelisation's
        # fixture so the receiving trap's capacity check isn't tripped.
        dirty = set()
        for op in (
            Split.physicalOperation(trap11, crossing11),
            Move.physicalOperation(crossing11),
            JunctionCrossing.physicalOperation(junctionL, crossing11),
            JunctionCrossing.physicalOperation(junctionL, crossing21),
            Move.physicalOperation(crossing21),
            Merge.physicalOperation(trap21, crossing21),
        ):
            op.run()
            dirty.update(dirtyComponentsForOp(op))
        _assert_incremental_matches_full(arch, dirty)

    def test_gate_swap(self, topology):
        arch, trap, ions = topology["arch"], topology["trap11"], topology["ions11"]
        op = GateSwap.physicalOperation(trap=trap, ion1=ions[0], ion2=ions[1])
        op.run()
        _assert_incremental_matches_full(arch, dirtyComponentsForOp(op))

    def test_full_operation_sequence_accumulated_into_one_refresh(self, topology):
        """Mirrors the "doesn't need routing" loop: several ops, one refresh after. Reuses test_qccd_parallelisation.py::setup_arch's op sequence."""
        arch = topology["arch"]
        trap11, crossing11, junctionL = topology["trap11"], topology["crossing11"], topology["junctionL"]
        trap21, crossing21, ions21 = topology["trap21"], topology["crossing21"], topology["ions21"]
        trap12, ions12 = topology["trap12"], topology["ions12"]
        trap22, ions22 = topology["trap22"], topology["ions22"]

        ops = (
            Split.physicalOperation(trap11, crossing11),
            Move.physicalOperation(crossing11),
            JunctionCrossing.physicalOperation(junctionL, crossing11),
            JunctionCrossing.physicalOperation(junctionL, crossing21),
            Move.physicalOperation(crossing21),
            Merge.physicalOperation(trap21, crossing21),
            GateSwap.physicalOperation(trap=trap21, ion1=ions21[0], ion2=ions21[2]),
            TwoQubitMSGate.physicalOperation(ion1=ions22[0], ion2=ions22[2], trap=trap22),
            OneQubitGate.physicalOperation(ion=ions12[0], trap=trap12),
            Measurement.physicalOperation(ion=ions12[0], trap=trap12),
            QubitReset.physicalOperation(ion=ions12[0], trap=trap12),
        )
        dirty = set()
        for op in ops:
            op.run()
            dirty.update(dirtyComponentsForOp(op))

        _assert_incremental_matches_full(arch, dirty)


class TestEmptyDirtySetIsNotAFullRebuild:
    """An empty dirty list means "nothing changed", not "fall back to full rebuild" - these are different code paths."""

    def test_empty_dirty_list_does_not_pick_up_unrelated_changes(self, topology):
        arch, trap = topology["arch"], topology["trap12"]
        extra_ion = QubitIon()
        # simulate an unreported arrival by bypassing addIon
        trap._ions.append(extra_ion)
        extra_ion.set(999, *trap.pos, parent=trap)

        arch.refreshGraph(dirty=set())
        assert 999 not in arch.graph.nodes

        arch.refreshGraph()  # dirty=None -> full rebuild picks it up
        assert 999 in arch.graph.nodes


class TestNumIonsIsAlwaysResynced:
    """numIons is speculatively bumped by the routing loops during path planning; refreshGraph must resync every trap/junction, not just dirty ones, or a stale reservation lingers."""

    def test_numions_reset_on_a_trap_outside_the_dirty_set(self, topology):
        arch = topology["arch"]
        untouched_trap = topology["trap22"]
        real_count = len(untouched_trap.ions)
        untouched_trap.numIons = real_count + 5  # simulate a stale reservation

        arch.refreshGraph(dirty={topology["crossing11"]})  # something unrelated

        assert untouched_trap.numIons == real_count


class TestIncrementalRefreshRequiresAnInitialFullRefresh:
    """No prior state to patch against without an initial full refresh - must fail loudly, not silently."""

    def test_incremental_refresh_before_any_full_refresh_raises(self):
        arch = QCCDArch()
        ions = [QubitIon() for _ in range(2)]
        arch.addManipulationTrap(x=0, y=0, ions=ions)
        # the required initial arch.refreshGraph() call is deliberately skipped

        with pytest.raises(RuntimeError):
            arch.refreshGraph(dirty=set())


class TestInTransitNodeCleanupIsOrderIndependent:
    """
    An ion finishing its journey within one dirty batch must both remove the
    crossing's leftover in-transit marker and get real trap edges, regardless
    of processing order. Set iteration order depends on object identity, which
    varies across separately constructed architectures, so this repeats with
    fresh objects for practical coverage of both orderings.
    """

    def test_merge_after_a_prior_refresh_keeps_real_edges(self):
        for _ in range(8):
            state = _build_topology()
            arch = state["arch"]
            trap11, crossing11, junctionL = state["trap11"], state["crossing11"], state["junctionL"]
            trap21, crossing21 = state["trap21"], state["crossing21"]

            # Split moves whichever ion is geometrically closest, not
            # necessarily ions11[0] - read it back off the crossing.
            split_op = Split.physicalOperation(trap11, crossing11)
            split_op.run()
            ion = crossing11.ion
            travel_dirty = set(dirtyComponentsForOp(split_op))

            # walk to crossing21 and refresh while still mid-transit, so the
            # in-transit marker node actually gets written at least once
            for op in (
                Move.physicalOperation(crossing11),
                JunctionCrossing.physicalOperation(junctionL, crossing11),
                JunctionCrossing.physicalOperation(junctionL, crossing21),
                Move.physicalOperation(crossing21),
            ):
                op.run()
                travel_dirty.update(dirtyComponentsForOp(op))
            arch.refreshGraph(dirty=travel_dirty)
            assert ion.idx in arch.graph.nodes  # in-transit marker exists

            # complete the journey - must clean up the stale marker without
            # destroying the real new edges
            merge_op = Merge.physicalOperation(trap21, crossing21)
            merge_op.run()
            arch.refreshGraph(dirty=dirtyComponentsForOp(merge_op))

            assert arch.graph.has_edge(ion.idx, trap21.idx)
