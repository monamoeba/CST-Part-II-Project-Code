import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler.qccd_parallelisation import *
from src.compiler.qccd_qubits_to_ions import *
from src.compiler.qccd_ion_routing import *
from src.compiler.qccd_alt_ion_routing import *
from src.compiler.qccd_WISE_ion_route import *
from src.compiler.qccd_color_qubits_to_ions import *
from src.simulator.qccd_circuit import QCCDCircuit

NDE_LZ = 10
NDE_JZ = 20
NSE_Z = 10

def _simulate_and_collect_errors(circuit, allOps, gate_improvements, num_shots):
    logical, phys_z, phys_x = [], [], []
    for gi in gate_improvements:
        le, px, pz = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gi)
        logical.append(le)
        phys_z.append(pz)
        phys_x.append(px)
    return logical, phys_z, phys_x


def _calculate_electrodes_and_dacs(allOps, capacity, nqubits):
    trapSet, junctionSet = set(), set()
    for op in allOps:
        for c in op.involvedComponents:
            if isinstance(c, Trap):
                trapSet.add(c)
            elif isinstance(c, Junction):
                junctionSet.add(c)
    Njz = len(junctionSet)
    Nlz = len(trapSet) * capacity
    Nde = NDE_LZ * Nlz + NDE_JZ * Njz
    Nse = NSE_Z * (Njz + Nlz)
    Num_electrodes = Nde + Nse
    return Num_electrodes, Num_electrodes  # (electrodes, DACs)


def build_color_code_ops(distance, rounds, tesselation, basis, capacity,
                          cluster_merge_strategy='bounded',
                          cluster_merge_threshold=2.5,
                          placement_strategy='hill_climb'):

    circuit = QCCDCircuit.generate_color_code(distance, rounds=rounds,
                                               tesselation=tesselation, basis=basis)
    if tesselation == (6, 6, 6):
        nqubits = len(circuit.dataQubitsIdxs) + len(circuit.measureQubitsIdxs)
    elif tesselation == (4, 8, 8):
        nqubits = (3 * distance ** 2 + 6 * distance - 5) // 4
    else:
        nqubits = len(circuit.dataQubitsIdxs) + len(circuit.measureQubitsIdxs)

    nrows = int(np.sqrt(nqubits)) + 2

    arch, (instructions, _) = circuit.processColorCircuitAugmentedGrid(
        rows=nrows, cols=nrows,
        trapCapacity=capacity,
        dataQubitsIdxs=circuit.dataQubitsIdxs,
        measureQubitsIdxs=circuit.measureQubitsIdxs,
        cluster_merge_strategy=cluster_merge_strategy,
        cluster_merge_threshold=cluster_merge_threshold,
        placement_strategy=placement_strategy,
    )
    arch.refreshGraph()

    allOps, barriers = ionRouting(arch, instructions, capacity)
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)

    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()
    return arch, circuit, parallelOpsMap


def build_surface_code_ops(distance, capacity, gate_improvements, num_shots):
    circuit = QCCDCircuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
    )
    nqubits = 2 * distance ** 2 - 1
    nrows = int(np.sqrt(nqubits)) + 2

    arch, (instructions, _) = circuit.processCircuitAugmentedGrid(
        rows=nrows, cols=nrows, trapCapacity=capacity
    )
    arch.refreshGraph()

    allOps, barriers = ionRouting(arch, instructions, capacity)
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)

    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()
    return arch, circuit, parallelOpsMap

#animating 
def _make_update_fn(fig, ax, arch, parallel_ops, parallel_times):
    def update(frame):
        ax.clear()
        if frame > 0:
            op = parallel_ops[frame - 1]
            t = round(parallel_times[frame - 1], 6)
            title = f"Operation {t}: {op.label}"
            arch.display(fig, ax, title, operation=op, runOps=True, showLabels=False)
        else:
            arch.display(fig, ax, showLabels=False)
    return update


def animate_qec_cycle(arch, circuit, parallelOpsMap, save_path=None, interval=300):
    arch = circuit.resetArch()
    arch.refreshGraph()

    times = sorted(parallelOpsMap.keys())
    ops = [parallelOpsMap[t] for t in times]
    n_frames = len(ops) + 1  # frame 0 = initial state

    print(f"Animating {len(ops)} parallel operations (total time: {times[-1]:.6f}s)")

    fig, ax = plt.subplots()
    update = _make_update_fn(fig, ax, arch, ops, times)
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=False)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        writer = 'pillow' if ext == '.gif' else 'ffmpeg'
        print(f"Saving animation to {save_path} ...")
        ani.save(save_path, writer=writer)
        print("Done.")
    else:
        plt.show()

    plt.close(fig)


def visualise_gate_interactions(circuit, capacities, rows=10, cols=10):
    fig, axs = plt.subplots(1, len(capacities),
                             figsize=(6 * len(capacities), 5))
    if len(capacities) == 1:
        axs = [axs]

    for ax, capacity in zip(axs, capacities):
        arch, (instructions, _) = circuit.processCircuitAugmentedGrid(
            rows=rows, cols=cols, trapCapacity=capacity
        )
        arch.ION_SIZE = 2000

        for idx, (ion, pos) in circuit._ionMapping.items():
            ion.set(ion.idx, pos[0], pos[1])
        arch.refreshGraph()

        edge_pairs = []
        scores = []
        ions_involved = set()
        score = 0

        for op in instructions:
            if not isinstance(op, TwoQubitMSGate):
                continue
            if not ions_involved.isdisjoint(op.ions):
                score += 1
                ions_involved = set()
            edge_pairs.append((op.ions[0].idx, op.ions[1].idx))
            scores.append(score)
            ions_involved = ions_involved.union(op.ions)

        unique_scores = sorted(set(scores), reverse=True)
        color_map = {s: (5 + i * i) * 2 for i, s in enumerate(unique_scores)}

        arch._manipulationTraps.append(
            (edge_pairs, [color_map[s] for s in scores])
        )
        arch.display(fig, ax, showLabels=False, showEdges=False, show_junction=False)
        ax.set_title(f"Capacity {capacity}")

    plt.tight_layout()
    plt.show()

#run w smth like:
"""
arch, circuit, parallelOpsMap = build_color_code_ops(
    distance, rounds, (6, 6, 6), 'Z', capacity,
    placement_strategy=placement_strategy,
)
visualise_gate_interactions(circuit, capacities=[args.capacity])
or 
animate_qec_cycle(arch, circuit, parallelOpsMap, save_path=args.save, interval=args.interval)

"""
