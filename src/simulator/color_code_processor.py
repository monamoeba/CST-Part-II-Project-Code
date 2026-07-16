from os import path

import numpy as np
import numpy.typing as npt
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Dict,
)
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
import logging
from multiprocessing import get_logger


NDE_LZ = 10
NDE_JZ = 20
NSE_Z = 10

def setup_logger(log_file):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def initialize_results():
    return {"ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, "QubitOperations": {}, "LogicalErrorRates": {}, "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}, "Electrodes": {}, "DACs": {}}

def simulate_and_collect_errors(circuit, allOps, gate_improvements, num_shots):
    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement)
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)
    return logicalErrors, physicalZErrors, physicalXErrors

def populate_results(results, label, parallelOpsMap, allOps, instructions, logicalErrors, physicalZErrors, physicalXErrors, capacity, distance):
    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors

def calculate_electrodes_and_dacs(allOps, capacity, nqubitsNeeded, arch_type):
    if arch_type == 'wise':
        Njz = np.ceil(nqubitsNeeded / capacity)
        Nlz = nqubitsNeeded - Njz
    else:
        trapSet = set()
        junctionSet = set()
        for op in allOps:
            for c in op.involvedComponents:
                if isinstance(c, Trap):
                    trapSet.add(c)
                elif isinstance(c, Junction):
                    junctionSet.add(c)
        Njz = len(junctionSet)
        Nlz = len(trapSet)*capacity
    Nde = NDE_LZ*Nlz + NDE_JZ*Njz
    Nse = NSE_Z*(Njz + Nlz)
    Num_electrodes = Nde + Nse
    if arch_type == 'wise':
        Num_DACs = int(min(100, Nde) + np.ceil(Nse/100))
    else:
        Num_DACs = Num_electrodes
    return Num_electrodes, Num_DACs

def finalize_and_log(results, distance, capacity, label, logger, extra_msg=""):
    logger.info(f"{distance} {capacity} {label} = {results}")
    logger.info(f"Finished processing{extra_msg} for distance {distance} and capacity {capacity}")
    return results

def get_circuit_qubit_count(distance, tesselation):
    if tesselation == (6,6,6):
        return (9*distance**2 -1)//8
    elif tesselation == (4,8,8):
        return (3*distance**2 +6*distance -5)//4
    else:
        raise ValueError("Unsupported tesselation type")

def process_color_code_circuit(
    distance,
    rounds,
    capacity,
    gate_improvements,
    num_shots,
    tesselation,
    basis='Z',
    cluster_merge_strategy='bounded',
    cluster_merge_threshold=2.5,
    placement_strategy='hill_climb',
    routing='original'
):
    logger = setup_logger("process_log_color_code.txt")

    logger.info(f"Starting circuit generation for distance {distance}, capacity {capacity} and tesselation {tesselation}")
  
    circuit = QCCDCircuit.generate_color_code(distance, rounds=rounds, tesselation=tesselation, basis=basis)
    #for single ancilla circuits (6.6.6)
    nqubitsNeeded = get_circuit_qubit_count(distance, tesselation)
    
    #for dual ancilla (6.6.6) circuits
    #nqubitsNeeded = 3*(distance**2 - 1)//4 + (3*distance**2 + 1)//4
    nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    arch, (instructions, _) = circuit.processColorCircuitAugmentedGrid(
        rows=nrowsNeeded,
        cols=nrowsNeeded,
        trapCapacity=capacity,
        dataQubitsIdxs=circuit.dataQubitsIdxs,
        measureQubitsIdxs=circuit.measureQubitsIdxs,
        cluster_merge_strategy=cluster_merge_strategy,
        cluster_merge_threshold=cluster_merge_threshold,
        placement_strategy=placement_strategy,
    )

    arch.refreshGraph()

    results = initialize_results()

    # FIXME legacy formatting!
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    if routing == 'original':
        allOps, barriers = ionRouting(arch, instructions, capacity)
    else:
        allOps, barriers = naiveAltIonRouting(arch, instructions, capacity)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors, physicalZErrors, physicalXErrors = simulate_and_collect_errors(circuit, allOps, gate_improvements, num_shots)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    
    
    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    populate_results(results, label, parallelOpsMap, allOps, instructions, logicalErrors, physicalZErrors, physicalXErrors, capacity, distance)


    Num_electrodes, Num_DACs = calculate_electrodes_and_dacs(allOps, capacity, nqubitsNeeded, 'augmented')
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    return finalize_and_log(results, distance, capacity, label, logger)

def process_color_code_circuit_wise_arch(distance, capacity, gate_improvements, num_shots, tesselation, basis='Z'):
    logger = setup_logger("process_log_color_code_wise.txt")

    logger.info(f"Starting circuit generation for distance {distance}, capacity {capacity} and tesselation {tesselation}")
  
    circuit = QCCDCircuit.generate_color_code(distance, rounds=2, tesselation=tesselation, basis=basis)
    nqubitsNeeded = (9*distance**2 -1)//8

    nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    wiseArch = QCCDWiseArch(m=int(np.sqrt(capacity*nqubitsNeeded/2))+1, n=int(np.sqrt(2*nqubitsNeeded/capacity))+1, k=capacity)
    arch, (instructions, _) = circuit.processCircuitWiseArch(wiseArch=wiseArch)
    
    arch.refreshGraph()

    results = initialize_results()

    # FIXME legacy formatting!
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    allOps, barriers = ionRoutingWISEArch(arch, wiseArch, instructions)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors, physicalZErrors, physicalXErrors = simulate_and_collect_errors(circuit, allOps, gate_improvements, num_shots)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity} and tesselation {tesselation}")

    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    populate_results(results, label, parallelOpsMap, allOps, instructions, logicalErrors, physicalZErrors, physicalXErrors, capacity, distance)

    Num_electrodes, Num_DACs = calculate_electrodes_and_dacs(allOps, capacity, nqubitsNeeded, 'wise')
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    return finalize_and_log(results, distance, capacity, label, logger)

def process_color_code_circuit_linear_arch(
    distance,
    rounds,
    capacity,
    gate_improvements, 
    num_shots,
    tesselation,
    basis='Z',
    cluster_merge_strategy='bounded',
    cluster_merge_threshold=2.5):
    logger = setup_logger("process_log_color_code_linear.txt")

    logger.info(f"Starting circuit generation for distance {distance}, capacity {capacity} and tesselation {tesselation}")
  
    circuit = QCCDCircuit.generate_color_code(distance, rounds=rounds, tesselation=tesselation, basis=basis)
    nqubitsNeeded = get_circuit_qubit_count(distance, tesselation)
    
    #for dual ancilla (6.6.6) circuits
    #nqubitsNeeded = 3*(distance**2 - 1)//4 + (3*distance**2 + 1)//4
    #nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2
    trapsrequired = int(np.ceil(nqubitsNeeded/(capacity-1))) * 2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {trapsrequired} traps in a linear chain")

    arch, (instructions, _) = circuit.processColorCircuitLinearGrid(
        trapCapacity=capacity,
        traps = trapsrequired,
        dataQubitsIdxs=circuit.dataQubitsIdxs,
        measureQubitsIdxs=circuit.measureQubitsIdxs,
        cluster_merge_strategy=cluster_merge_strategy,
        cluster_merge_threshold=cluster_merge_threshold,
    )

    arch.refreshGraph()

    results = initialize_results()
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    allOps, barriers = ionRouting(arch, instructions, capacity)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors, physicalZErrors, physicalXErrors = simulate_and_collect_errors(circuit, allOps, gate_improvements, num_shots)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    
    
    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    populate_results(results, label, parallelOpsMap, allOps, instructions, logicalErrors, physicalZErrors, physicalXErrors, capacity, distance)


    Num_electrodes, Num_DACs = calculate_electrodes_and_dacs(allOps, capacity, nqubitsNeeded, 'augmented')
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    return finalize_and_log(results, distance, capacity, label, logger)

def process_color_code_circuit_switch_arch(
    distance,
    rounds,
    capacity,
    gate_improvements, 
    num_shots,
    tesselation,
    basis='Z',
    cluster_merge_strategy='bounded',
    cluster_merge_threshold=2.5):
    logger = setup_logger("process_log_color_code_switch.txt")

    logger.info(f"Starting circuit generation for distance {distance}, capacity {capacity} and tesselation {tesselation}")
  
    circuit = QCCDCircuit.generate_color_code(distance, rounds=rounds, tesselation=tesselation, basis=basis)
    nqubitsNeeded = get_circuit_qubit_count(distance, tesselation)
    
    #for dual ancilla (6.6.6) circuits
    #nqubitsNeeded = 3*(distance**2 - 1)//4 + (3*distance**2 + 1)//4
    #nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2
    trapsrequired = int(np.ceil(nqubitsNeeded/(capacity-1))) * 2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {trapsrequired} traps in a switch topology")

    arch, (instructions, _) = circuit.processColorCircuitNetworkedGrid(
        trapCapacity=capacity,
        traps = trapsrequired,
        dataQubitsIdxs=circuit.dataQubitsIdxs,
        measureQubitsIdxs=circuit.measureQubitsIdxs,
        cluster_merge_strategy=cluster_merge_strategy,
        cluster_merge_threshold=cluster_merge_threshold,
    )

    arch.refreshGraph()

    results = initialize_results()
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    allOps, barriers = ionRouting(arch, instructions, capacity)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors, physicalZErrors, physicalXErrors = simulate_and_collect_errors(circuit, allOps, gate_improvements, num_shots)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    
    
    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    populate_results(results, label, parallelOpsMap, allOps, instructions, logicalErrors, physicalZErrors, physicalXErrors, capacity, distance)


    Num_electrodes, Num_DACs = calculate_electrodes_and_dacs(allOps, capacity, nqubitsNeeded, 'augmented')
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    return finalize_and_log(results, distance, capacity, label, logger)
