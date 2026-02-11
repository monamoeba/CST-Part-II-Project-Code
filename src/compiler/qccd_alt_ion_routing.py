
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Set,
    Dict,
)
import networkx as nx
from collections import defaultdict
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *



def _determineRouted(ion1:Ion, ion2:Ion):
    """ Determines the ion that will be routed between the pair """
    if ion1.label[0]==ion2.label[0]:
        # Same type - tiebreak with index
        r_ion, s_ion = sorted(
            (ion1, ion2), key=lambda ion: ion.idx
        )
    else:
        # prioritise routing ancilla otherwise
        r_ion, s_ion = sorted(
            (ion1, ion2), key=lambda ion: ion.label[0]=="D"
        )

    return (r_ion, s_ion)
    
def naiveAltIonRouting(
    qccdArch: QCCDArch,
    operations: Sequence[QubitOperation],
    trapCapacity: int
) -> Tuple[Sequence[Operation], Sequence[int]]:

    twoQubitGates: defaultdict[int, List[TwoQubitMSGate]] = defaultdict(list)

    for op in operations:
        if isinstance(op, TwoQubitMSGate):
            # routed and stationary ion
            # give preference to route ancilla 
            r_ion, s_ion = _determineRouted(*op.ions)
            trap = r_ion.parent
            twoQubitGates[r_ion.idx].append(op)

    opPriorities: Dict[Operation, int] = {op:i for i, op in enumerate(operations)}

    allOps: List[Operation] = []
    barriers: List[int] = []
    operationsRemaining = list(operations)
    toMoveCandidates: Dict[int, TwoQubitMSGate] = {}
    while operationsRemaining:
        
        # Find and run ops that don't need routing
        while True:
            toRemove: List[Operation] = []
            ionsInvolved: Set[Ion] = set()
            for op in operationsRemaining:
                trap = op.getTrapForIons()
                if ionsInvolved.isdisjoint(op.ions) and trap:
                    op.setTrap(trap)
                    toRemove.append(op)
                ionsInvolved = ionsInvolved.union(op.ions)

            for op in toRemove:
                op.run()
                allOps.append(op)
                operationsRemaining.remove(op)
                if isinstance(op, TwoQubitMSGate):
                    r_ion, s_ion = _determineRouted(*op.ions)
                    twoQubitGates[r_ion.idx].remove(op)

            if len(toRemove) == 0:
                break

        for rIdx in twoQubitGates.keys():
            if rIdx in toMoveCandidates:
                continue
            if len(twoQubitGates[rIdx]) == 0:
                continue
            gate = twoQubitGates[rIdx][0]
            trap = gate.getTrapForIons()
            if trap:
                # Skip if ions in the same trap for gate op
                continue
            toMoveCandidates[rIdx] = twoQubitGates[rIdx].pop(0)

        # Move ops with original happens-before priority (oldest first)
        toMove = sorted([(k,o) for k,o in toMoveCandidates.items()], key = lambda ko: opPriorities[ko[1]])
        usedCrossings: Set[Crossing] = set()
        fullQccdNodes: Set[QCCDNode] = set()
        movements: Dict[int, Tuple[TwoQubitMSGate, List[QCCDNode], Trap]] = {}
        # routedIdx: op, pathChosen, destinationTrap, goBackTrap

        #determining path each routing operation should take
        for rIdx, op in toMove:
            r_ion, s_ion = _determineRouted(*op.ions)
            # destination trap (where stationary ion is)
            trap = s_ion.parent
            if not isinstance(trap, Trap):
                raise ValueError(f"Data Ion not in trap {trap}")

            src = rIdx
            dest = trap.idx
            paths = list(nx.all_shortest_paths(qccdArch.graph, src, dest))

            chosenQccdNodes: List[QCCDNode] = []
            chosenCrossings: List[Crossing] = []

            # choosing possible routing path
            for path in paths:
                crossingsInPath: List[Crossing] = []
                # iterate over adjacent pairs in path
                for n1, n2 in zip(path[:-1], path[1:]):
                    if (n1, n2) not in qccdArch.crossingEdges:
                        continue
                    crossingsInPath.append(qccdArch.crossingEdges[(n1,n2)])

                qccdNodesInPath: List[QCCDNode] = []
                for n in path:
                    nd = qccdArch.nodes[n] if n in qccdArch.nodes else qccdArch.ions[n].parent
                    if nd not in qccdNodesInPath:
                        qccdNodesInPath.append(nd)

                qccdNodesInPathFull = {
                    qccdNode
                    for qccdNode in qccdNodesInPath[1:]
                    if (isinstance(qccdNode, Junction) and qccdNode.numIons == 1) or (qccdNode.numIons == trapCapacity)
                }

                # Choose path if no overlap with full nodes + used crossings
                if usedCrossings.isdisjoint(crossingsInPath) and fullQccdNodes.isdisjoint(qccdNodesInPathFull):
                    chosenQccdNodes = qccdNodesInPath
                    chosenCrossings = crossingsInPath
                    break

            # can't do move ops
            if len(chosenQccdNodes) == 0:
                continue

            # able to do move op
            toMoveCandidates.pop(rIdx)
            movements[rIdx] = (op, chosenQccdNodes, trap)

            # remove reserved crossings
            usedCrossings = usedCrossings.union(chosenCrossings)

            # increment traps + juncs by 1 ion + remove traps + juncs at capacity
            for qccdNode in chosenQccdNodes[1:]:
                qccdNode.numIons += 1
                if isinstance(qccdNode, Junction) and qccdNode.numIons == 1:
                    fullQccdNodes.add(qccdNode)
                elif qccdNode.numIons == trapCapacity:
                    fullQccdNodes.add(qccdNode)

        # send back routed ion to og trap
        toForward: Dict[TwoQubitMSGate, Tuple[int, List[QCCDNode], Trap, Optional[Trap]]] = {}
        for rIdx, (op, qccdNodes, destTrap) in movements.items():
            goBackTrap = qccdNodes[0]
            # naive impl reroute ion back to original trap 
            destTrap.numIons -= 1

            toForward[op] = (rIdx, qccdNodes, goBackTrap)

        startedGoingBack = {op:False for op in toForward.keys()}
        # forwarding the ions and sending them back after running operation
        while toForward:
            ionsInvolved = set()
            orderedToForward = sorted([(o, rc) for o, rc in toForward.items()], key=lambda orc: opPriorities[orc[0]])
            for op, (rIdx, qccdNodes, goBackTrap) in orderedToForward:
                if not ionsInvolved.isdisjoint(op.ions):
                    continue

                r_idx, s_idx = (i.idx for i in _determineRouted(*op.ions))
                # include stationary ion (to ensure it is actually stationary
                ionsInvolvedNow = [qccdArch.ions[r_idx], qccdArch.ions[s_idx]]

                n1 = rIdx
                #index of the trap the routing ion is in 
                n1Idx = qccdArch.ions[rIdx].parent.idx
                #index of the destination trap
                trap = qccdNodes[-1]
                srcTrap = goBackTrap

                # arrived at stationary ion's trap and haven't ran operation yet
                if n1Idx == trap.idx and not startedGoingBack[op]:
                    op.setTrap(trap)
                    op.run()
                    allOps.append(op)
                    operationsRemaining.remove(op)
                    ionsInvolved = ionsInvolved.union(op.ions)
                    if goBackTrap is not None:
                        startedGoingBack[op] = True
                    else:
                        toForward.pop(op)
                        continue
                # have ran operation, started going back and arrived at start trap 
                elif startedGoingBack[op] and n1Idx == srcTrap.idx:
                    toForward.pop(op)
                    continue

                # op has run started going back (or hasn't - needs to be routed)
                if startedGoingBack[op]:
                    n2Idx = [dn.idx for sn, dn in zip(qccdNodes[::-1][:-1], qccdNodes[::-1][1:]) if sn.idx == n1Idx][0]
                else:
                    n2Idx = [dn.idx for sn, dn in zip(qccdNodes[:-1], qccdNodes[1:]) if sn.idx == n1Idx][0]
                forwardingPath = nx.shortest_path(qccdArch.graph, n1, n2Idx)
                
                ms: List[Operation] = []
                for n1, n2 in zip(forwardingPath[:-1], forwardingPath[1:]):
                    ms.extend(qccdArch.graph.edges[n1,n2]["operations"])

                for m in ms:
                    if isinstance(m, CrystalOperation):
                        ionsInvolvedNow.extend(m.ionsInfluenced)
                    if isinstance(m, GateSwap) and m._ion1==m._ion2:
                        continue
                    m.run()
                    allOps.append(m)
                    
                qccdArch.refreshGraph()
                ionsInvolved = ionsInvolved.union(ionsInvolvedNow)

        barriers.append(len(allOps))

    return allOps, barriers
