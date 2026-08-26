
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Dict,
    Set,
    Union,
    FrozenSet,
    Iterable,
)
from matplotlib import pyplot as plt
import networkx as nx
from dataclasses import dataclass
from matplotlib.patches import Ellipse
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *


def dirtyComponentsForOp(op: Operation) -> Set[Union[Trap, Junction, Crossing]]:
    """Traps/Junctions/Crossings touched by `op`; pass to refreshGraph(dirty=...) after op.run()."""
    return {c for c in op.involvedComponents if isinstance(c, (Trap, Junction, Crossing))}


class QCCDArch:
    SIZING = 1
    JUNCTION_SIZE = 800 * SIZING
    ION_SIZE = 800 * SIZING
    FONT_SIZE = 14 * SIZING
    WINDOW_SIZE = 30 * SIZING, 24 * SIZING
    TRAP_WIDTH = 15 * SIZING
    EDGE_WIDTH = SIZING

    HIGHLIGHT_COLOR = "yellow"
    HIGHLIGHT_NODE_SIZE = 4000 * SIZING
    JUNCTION_SHAPE = "s"
    ION_SHAPE = "o"
    DEFAULT_ALPHA = 0.5
    PADDING = 0.6

    N_ITERS = 5000

    def __init__(self):
        self._trapEdges: Mapping[int, Sequence[Tuple[int, int]]] = {}
        self._crossingEdges: Mapping[Tuple[int, int], Crossing] = {}
        self._crossings: List[Crossing] = []
        self._manipulationTraps: List[ManipulationTrap] = []
        self._junctions: List[Junction] = []
        self._nextIdx = 0
        self._routingTable: Mapping[int, Mapping[int, Sequence[Operation]]] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._inActiveEdges: List[int] = []
        self._centralities: Mapping
        self._originalArrangement: Dict[Ion, Trap] = {}
        # Incremental refreshGraph(): last-rebuilt state per component, so a
        # later rebuild knows what to remove before re-adding.
        self._trapMemberIdxs: Dict[Trap, FrozenSet[int]] = {}
        self._crossingBoundaryIdxs: Dict[Crossing, Tuple[int, int]] = {}
        self._crossingInTransitIdx: Dict[Crossing, Optional[int]] = {}
        self._crossingsByComponent: Optional[Dict[QCCDNode, List[Crossing]]] = None
        self._graphInitialized: bool = False

    @property
    def graph(self):
        return self._graph
    
    @property
    def crossingEdges(self):
        return self._crossingEdges

    @property
    def routingTable(self):
        return self._routingTable

    @property
    def ions(self) -> Mapping[int, Ion]:
        ions = {}
        for t in self._manipulationTraps:
            ions.update([(ion.idx, ion) for ion in t.ions])
        for j in self._junctions:
            ions.update([(ion.idx, ion) for ion in j.ions])
        for c in self._crossings:
            if c.ion:
                ions[c.ion.idx] = c.ion
        return ions

    @property
    def nodes(self) -> Mapping[int, QCCDNode]:
        cs = {}
        for t in self._manipulationTraps:
            cs[t.idx] = t
        for j in self._junctions:
            cs[j.idx] = j
        return cs

    def addEdge(self, source: QCCDNode, target: QCCDNode) -> Crossing:
        crossing = Crossing(self._nextIdx, source, target)
        self._crossings.append(crossing)
        self._nextIdx += 1
        return crossing

    def addManipulationTrap(
        self,
        x: int,
        y: int,
        ions: Sequence[Ion],
        color: str = ManipulationTrap.DEFAULT_COLOR,
        isHorizontal: bool = ManipulationTrap.DEFAULT_ORIENTATION,
        spacing: int = ManipulationTrap.DEFAULT_SPACING,
        capacity: int = ManipulationTrap.DEFAULT_CAPACITY,
    ) -> Trap:
        trap = ManipulationTrap(
            self._nextIdx,
            x,
            y,
            ions,
            color=color,
            isHorizontal=isHorizontal,
            spacing=spacing * self.SIZING,
            capacity=capacity,
        )
        for ion in trap.ions:
            self._originalArrangement[ion] = trap
        self._manipulationTraps.append(trap)
        self._nextIdx += len(ions) + 1
        return trap

    def addJunction(
        self,
        x: int,
        y: int,
        color: str = Junction.DEFAULT_COLOR,
        label: str = Junction.DEFAULT_LABEL,
        capacity: int = Junction.DEFAULT_CAPACITY,
    ) -> Junction:
        junction = Junction(
            self._nextIdx, x, y, color=color, label=label, capacity=capacity
        )
        self._junctions.append(junction)
        self._nextIdx += 1
        return junction

    def _removeTrapEdges(self, trap: Trap, g: nx.DiGraph) -> None:
        """Removes trap's edges from the last rebuild. Separate from _addTrapEdges so a batch's removals all run before its additions (see _refreshComponents)."""
        staleIdxs = self._trapMemberIdxs.get(trap)
        if staleIdxs:
            staleEdges = [(idx, trap.idx) for idx in staleIdxs]
            staleEdges += [(a, b) for a in staleIdxs for b in staleIdxs if a != b]
            g.remove_edges_from(staleEdges)

    def _addTrapEdges(self, trap: Trap, g: nx.DiGraph) -> None:
        trapEdges = []
        for ion1 in trap.ions:
            g.add_edge(ion1.idx, trap.idx, operations=[])
            for ion2 in trap.ions:
                if ion1 == ion2:
                    continue
                trapEdges.append((ion1.idx, ion2.idx))
                trapEdges.append((ion2.idx, ion1.idx))
                g.add_edge(
                    ion1.idx,
                    ion2.idx,
                    operations=[GateSwap.physicalOperation(trap=trap, ion1=ion1, ion2=ion2)],
                )
                g.add_edge(
                    ion2.idx,
                    ion1.idx,
                    operations=[GateSwap.physicalOperation(trap=trap, ion1=ion2, ion2=ion1)],
                )
        self._trapEdges[trap.idx] = trapEdges
        self._trapMemberIdxs[trap] = frozenset(ion.idx for ion in trap.ions)

    def _removeCrossingEdges(self, crossing: Crossing, g: nx.DiGraph) -> Optional[int]:
        """Removes crossing's edges from the last rebuild. Returns its previous in-transit-ion node id so the caller can clean it up once the whole batch's additions are done."""
        staleBoundary = self._crossingBoundaryIdxs.get(crossing)
        if staleBoundary is not None:
            staleN1Idx, staleN2Idx = staleBoundary
            g.remove_edges_from([(staleN1Idx, staleN2Idx), (staleN2Idx, staleN1Idx)])
            self._crossingEdges.pop((staleN1Idx, staleN2Idx), None)
            self._crossingEdges.pop((staleN2Idx, staleN1Idx), None)
        return self._crossingInTransitIdx.get(crossing)

    def _addCrossingEdges(self, crossing: Crossing, g: nx.DiGraph) -> None:
        n1, n2 = crossing.connection
        n1Idx = crossing.getEdgeIon(n1).idx if n1.ions else n1.idx
        n2Idx = crossing.getEdgeIon(n2).idx if n2.ions else n2.idx
        self._crossingEdges[(n1Idx, n2Idx)] = crossing
        self._crossingEdges[(n2Idx, n1Idx)] = crossing
        ion1 = crossing.getEdgeIon(n1) if n1.ions else None
        ion2 = crossing.getEdgeIon(n2) if n2.ions else None
        doRotation1 = [GateSwap.physicalOperation(trap=n1,ion1=ion1,ion2=ion1)] if len(n1.ions)==1 else []
        doRotation2 = [GateSwap.physicalOperation(trap=n2,ion1=ion2,ion2=ion2)] if len(n2.ions)==1 else []
        if isinstance(n1, Trap) and isinstance(n2, Junction):
            ops1 = doRotation1+[
                Split.physicalOperation(n1, crossing, ion1),
                Move.physicalOperation(crossing, ion1),
                JunctionCrossing.physicalOperation(n2, crossing, ion1),
            ]
            ops2 = [
                JunctionCrossing.physicalOperation(n2, crossing, ion2),
                Move.physicalOperation(crossing, ion2),
                Merge.physicalOperation(n1, crossing, ion2),
            ]
        elif isinstance(n1, Junction) and isinstance(n2, Trap):
            ops1 = [
                JunctionCrossing.physicalOperation(n1, crossing, ion1),
                Move.physicalOperation(crossing, ion1),
                Merge.physicalOperation(n2, crossing, ion1),
            ]
            ops2 = doRotation2+[
                Split.physicalOperation(n2, crossing, ion2),
                Move.physicalOperation(crossing, ion2),
                JunctionCrossing.physicalOperation(n1, crossing, ion2),
            ]
        elif isinstance(n1, Junction) and isinstance(n2, Junction):
            ops1 = [
                JunctionCrossing.physicalOperation(n1, crossing, ion1),
                Move.physicalOperation(crossing, ion1),
                JunctionCrossing.physicalOperation(n2, crossing, ion1),
            ]
            ops2 = [
                JunctionCrossing.physicalOperation(n2, crossing, ion2),
                Move.physicalOperation(crossing, ion2),
                JunctionCrossing.physicalOperation(n1, crossing, ion2),
            ]
        else:
            ops1 = doRotation1+[
                Split.physicalOperation(n1, crossing, ion1),
                Move.physicalOperation(crossing, ion1),
                Merge.physicalOperation(n2, crossing, ion1),
            ]
            ops2 = doRotation2+[
                Split.physicalOperation(n2, crossing, ion2),
                Move.physicalOperation(crossing, ion2),
                Merge.physicalOperation(n1, crossing, ion2),
            ]
        g.add_edge(n1Idx, n2Idx, operations=ops1)
        g.add_edge(n2Idx, n1Idx, operations=ops2)
        self._crossingBoundaryIdxs[crossing] = (n1Idx, n2Idx)

        if crossing.ion:
            g.add_node(crossing.ion.idx, pos=crossing.ion.pos)
            self._crossingInTransitIdx[crossing] = crossing.ion.idx
        else:
            self._crossingInTransitIdx[crossing] = None

    def _syncCounts(self) -> None:
        """Resets numIons on every trap/junction, unconditionally (not just dirty ones): routing algorithms also use numIons as a speculative reservation counter on nodes elsewhere on a candidate path, which only a full pass reliably clears."""
        for j in self._junctions:
            j.numIons = len(j.ions)
        for t in self._manipulationTraps:
            t.numIons = len(t.ions)
        self._centralities = None
        self._routingTable = {ion.idx: {} for ion in self.ions.values()}

    def _buildCrossingsByComponent(self) -> Dict[QCCDNode, List[Crossing]]:
        """Trap/Junction -> incident Crossings. Topology is fixed after construction, so built once."""
        byComponent: Dict[QCCDNode, List[Crossing]] = {}
        for crossing in self._crossings:
            for node in crossing.connection:
                byComponent.setdefault(node, []).append(crossing)
        return byComponent

    def _refreshComponents(
        self, traps: Sequence[Trap], crossings: Sequence[Crossing], g: nx.DiGraph
    ) -> None:
        """
        Rebuilds `traps` and `crossings` in `g` in three phases: remove all
        their stale edges, then add all fresh edges, then clean up any
        now-unused in-transit marker nodes. Removals must all happen before
        any additions: an ion leaving a trap can become a crossing's new
        boundary node on the exact (u, v) pair the trap used to have an
        edge on, so interleaving removal/addition per-component would let a
        same-batch addition get clobbered by an unrelated same-batch
        removal, depending on processing order. In-transit cleanup runs
        last for the same reason - "still unused" is only answerable once
        every addition in the batch has happened.
        """
        staleInTransitIdxs = [self._removeCrossingEdges(c, g) for c in crossings]
        for trap in traps:
            self._removeTrapEdges(trap, g)

        for trap in traps:
            self._addTrapEdges(trap, g)
        for crossing in crossings:
            self._addCrossingEdges(crossing, g)

        # Only stale if this crossing's in-transit ion actually changed -
        # it may still legitimately be there, in which case "add" above
        # re-added the same (edge-less) node, so degree alone can't tell.
        for crossing, staleIdx in zip(crossings, staleInTransitIdxs):
            if staleIdx is None or staleIdx == self._crossingInTransitIdx.get(crossing):
                continue
            if g.has_node(staleIdx) and g.degree(staleIdx) == 0:
                g.remove_node(staleIdx)

    def _fullRefresh(self) -> None:
        g = nx.DiGraph()

        for j in self._junctions:
            j.subgraph(g)
        for t in self._manipulationTraps:
            t.subgraph(g)

        self._refreshComponents(self._manipulationTraps, self._crossings, g)

        for n2Idx in self._inActiveEdges:
            graphEdges = [
                (u, v)
                for (u, v), crossing in self._crossingEdges.items()
                if self.nodes[n2Idx] in crossing.connection
                and (v in self.nodes[n2Idx].nodes)
            ]
            g.remove_edges_from(graphEdges)

        self._graph = g
        self._crossingsByComponent = self._buildCrossingsByComponent()
        self._graphInitialized = True

    def _incrementalRefresh(self, dirty: Iterable[Union[Trap, Junction, Crossing]]) -> None:
        if self._crossingsByComponent is None:
            self._crossingsByComponent = self._buildCrossingsByComponent()

        # A dirty trap/junction can change which ion a neighbouring crossing
        # treats as its boundary, so its incident crossings must be
        # recomputed too even if their own ion count didn't change.
        expanded: Set[Union[Trap, Junction, Crossing]] = set()
        for component in dirty:
            expanded.add(component)
            if isinstance(component, (Trap, Junction)):
                expanded.update(self._crossingsByComponent.get(component, ()))

        # Sorted by idx rather than left in set-iteration order: default
        # object hashing is address-based and varies run to run, and since
        # processing order changes edge *insertion* order, an unstable
        # order changes how networkx breaks path-length ties in the
        # routing algorithms - making compiled output non-deterministic.
        traps = sorted((c for c in expanded if isinstance(c, Trap)), key=lambda c: c.idx)
        crossings = sorted((c for c in expanded if isinstance(c, Crossing)), key=lambda c: c.idx)
        self._refreshComponents(traps, crossings, self._graph)

    def refreshGraph(self, dirty: Optional[Iterable[Union[Trap, Junction, Crossing]]] = None) -> None:
        """dirty=None: full rebuild. dirty=<iterable, possibly empty>: patch only those components (+ incident crossings) in place."""
        if dirty is not None:
            if not self._graphInitialized:
                raise RuntimeError(
                    "refreshGraph(dirty=...) called before an initial full refreshGraph()"
                )
            self._incrementalRefresh(dirty)
        else:
            self._fullRefresh()
        self._syncCounts()

    def display(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        title: str = "",
        operation: Optional[Operation] = None,
        show_junction: bool = True,
        showEdges: bool = True,
        showIons: bool = True,
        showLabels: bool = True,
        runOps: bool = False,
    ) -> None:
        pos = {}
        labels = {}
        operationNodes: List[List[int]] = []
        involvedIons: List[Sequence[Ion]] = []

        if operation is None:
            operations = []
        elif isinstance(operation, ParallelOperation):
            operations = operation.operations
            if runOps:
                operation.run()
                self.refreshGraph()
        else:
            operations = [operation]
            if runOps:
                for op in operations:
                    op.run()

                self.refreshGraph()

        for op in operations:
            operationNodes.append([])
            involvedIons.append(op.involvedIonsForLabel)

        for junction in self._junctions:
            pos[junction.nodes[0]] = junction.pos
            labels[junction.nodes[0]] = ""
            if show_junction:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=[junction.nodes[0]],
                    node_color=[junction.color],
                    node_shape=self.JUNCTION_SHAPE,
                    node_size=self.JUNCTION_SIZE,
                )
            for n, ion in zip(junction.nodes[1:], junction.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[n],
                        node_color=[ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showLabels:
                x = junction.pos[0]
                y = junction.pos[1]
                ax.text(
                    x,
                    y,
                    junction.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        for c in self._crossings:
            if c.ion:
                pos[c.ion.idx] = c.ion.pos
                labels[c.ion.idx] = c.ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[c.ion.idx],
                        node_color=[c.ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if c.ion in ions:
                        nodes.append(c.ion.idx)

        for t in self._manipulationTraps:
            if not isinstance(t, Trap):
                continue
            pos[t.nodes[0]] = t.pos
            labels[t.nodes[0]] = ""
            colors = {}
            for n, ion in zip(t.nodes[1:], t.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                colors[n] = ion.color
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showIons:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=t.nodes[1:],
                    node_color=colors.values(),
                    node_shape=self.ION_SHAPE,
                    node_size=self.ION_SIZE,
                )

        for trap in self._manipulationTraps:
            if not isinstance(trap, Trap):
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=trap[0],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color='red',
                    width=trap[1],
                )
                continue
            if showIons:
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=self._trapEdges[trap.idx],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color=trap.color,
                    width=self.TRAP_WIDTH,
                )
            if showLabels:
                x = trap.pos[0]
                y = trap.pos[1]
                ax.text(
                    x,
                    y,
                    trap.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showEdges:
            nx.draw_networkx_edges(
                self._graph,
                pos,
                edgelist=self._crossingEdges.keys(),
                ax=ax,
                alpha=self.DEFAULT_ALPHA,
                width=self.EDGE_WIDTH,
            )
        if showLabels:
            for e in self._crossings:
                ax.text(
                    *e.pos,
                    e.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showIons:
            nx.draw_networkx_labels(
                self._graph, pos, ax=ax, labels=labels, font_size=self.FONT_SIZE
            )

        for nodes, op in zip(operationNodes, operations):
            if nodes:
                xVals = [pos[node][0] for node in nodes]
                yVals = [pos[node][1] for node in nodes]
                padding = self.SIZING * self.PADDING
                xMin, xMax = min(xVals) - padding, max(xVals) + padding
                yMin, yMax = min(yVals) - padding, max(yVals) + padding
                width = xMax - xMin
                height = yMax - yMin
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ellip = Ellipse(
                    (xLabel, yLabel),
                    width,
                    height,
                    edgecolor=op.color,
                    alpha=self.DEFAULT_ALPHA,
                    facecolor=op.color,
                )
                ax.add_patch(ellip)
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ax.text(
                    xLabel,
                    yLabel,
                    op.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        ax.set_title(title, fontsize=self.FONT_SIZE*5)
        n = len(fig.axes)
        fig.set_size_inches(self.WINDOW_SIZE[0]*n, self.WINDOW_SIZE[1])

