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

def process_color_code_circuit(distance, capacity, gate_improvements, num_shots, tesselation):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("process_log_color_code.txt")
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Starting circuit generation for distance {distance}, capacity {capacity} and tesselation {tesselation}")
  
    circuit = QCCDCircuit.generate_color_code(distance, rounds=2, tesselation=tesselation)
    #for single ancilla circuits (6.6.6)
    nqubitsNeeded = (9*distance**2 -1)//8
    #for dual ancilla circuits
    #nqubitsNeeded = 3*(distance**2 - 1)//4 + (3*distance**2 + 1)//4
    nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    arch, (instructions, _) = circuit.processColorCircuitAugmentedGrid(rows=nrowsNeeded, cols=nrowsNeeded, trapCapacity=capacity, dataQubitIdxs=circuit.dataQubitsIdxs)
   
    arch.refreshGraph()

    results = {"ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, "QubitOperations": {}, "LogicalErrorRates": {}, "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}, "Electrodes": {}, "DACs": {}}

    # FIXME legacy formatting!
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    allOps, barriers = ionRouting(arch, instructions, capacity)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement)
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    
    
    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors


    trapSet = set()
    junctionSet = set()
    for op in allOps:
        for c in op.involvedComponents:
            if isinstance(c, Trap):
                trapSet.add(c)
            elif isinstance(c, Junction):
                junctionSet.add(c)

    # Every zone can contain up to two qubits
    # Njz = 1*Nj, Nlz = Nl*k
    # The number of zones  N = 1*Nj+k*Nl
    Njz = len(junctionSet) # each junction is one zone
    Nlz = len(trapSet)*capacity # each trap is k zones

    # Njz = int(np.ceil(nqubitsNeeded / (2*(capacity-1))) )# 2 traps per junction
    # Nlz = nqubitsNeeded-Njz
    Nde = NDE_LZ*Nlz+NDE_JZ*Njz
    Nse = NSE_Z*(Njz+Nlz)

    Num_electrodes = Nde+Nse
    Num_DACs = Num_electrodes
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    logger.info(f"{distance} {capacity} {label} = {results}")
    
    logger.info(f"Finished processing for distance {distance} and capacity {capacity}")
    return results

def process_model_color_code_circuit(distance, capacity, gate_improvements, num_shots, circtype):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("process_log_color_code.txt")
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Starting circuit generation for distance {distance}, capacity {capacity} and type {circtype}")
    
    """ path = f'.\\color_code_experiments\\r={distance*4},d={distance},p=0.001,noise=uniform,c={circtype},gates=all.stim'
    with open(path, 'r') as f:
        raw = f.read()
    lines = raw.split('\n')
    qcount = 0
    m_qubit_ids = set()
    d_qubit_ids = set()
    for line in lines:
        if line.startswith("QUBIT_COORDS"):
            qcount += 1
            d_qubit_ids.add(int(line.split(' ')[2]))
        if line.startswith("MX(0.001)"):
            splitline = line.split(' ')
            ids = [int(i) for i in splitline[1:]]
            m_qubit_ids.update(ids)
        if line.startswith("M(0.001)"):
            splitline = line.split(' ')
            ids = [int(i) for i in splitline[1:]]
            m_qubit_ids.update(ids)
            break

    for mqubit in m_qubit_ids:
        d_qubit_ids.remove(mqubit) """
    
    path = r".\\color_code_experiments\\superdense_color_code_d5_r20_p1000.stim"

    with open(path, "r") as f:
        raw = f.read()
    s = "0 2 5 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 35"
    d_qubit_ids = list(int(i) for i in s.strip().split(" "))
    qcount = 37

    circuit = QCCDCircuit(raw)
    circuit.dataQubitsIdxs = list(d_qubit_ids)
    circuit.iscolorcode = True #ensure chromobius used when simulate

    #for single ancilla circuits
    nqubitsNeeded = qcount

    nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    arch, (instructions, _) = circuit.processColorCircuitAugmentedGrid(rows=nrowsNeeded, cols=nrowsNeeded, trapCapacity=capacity, dataQubitIdxs=circuit.dataQubitsIdxs)
   
    arch.refreshGraph()

    results = {"ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, "QubitOperations": {}, "LogicalErrorRates": {}, "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}, "Electrodes": {}, "DACs": {}}

    # FIXME legacy formatting!
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance}, capacity {capacity} and type {type}")
    allOps, barriers = naiveAltIonRouting(arch, instructions, capacity)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement)
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity} and type {circtype}")
    
    
    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors


    trapSet = set()
    junctionSet = set()
    for op in allOps:
        for c in op.involvedComponents:
            if isinstance(c, Trap):
                trapSet.add(c)
            elif isinstance(c, Junction):
                junctionSet.add(c)

    # Every zone can contain up to two qubits
    # Njz = 1*Nj, Nlz = Nl*k
    # The number of zones  N = 1*Nj+k*Nl
    Njz = len(junctionSet) # each junction is one zone
    Nlz = len(trapSet)*capacity # each trap is k zones

    # Njz = int(np.ceil(nqubitsNeeded / (2*(capacity-1))) )# 2 traps per junction
    # Nlz = nqubitsNeeded-Njz
    Nde = NDE_LZ*Nlz+NDE_JZ*Njz
    Nse = NSE_Z*(Njz+Nlz)

    Num_electrodes = Nde+Nse
    Num_DACs = Num_electrodes
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    logger.info(f"{distance} {capacity} {label} = {results}")
    
    logger.info(f"Finished processing for model circuit distance {distance}, capacity {capacity}")
    return results

def process_color_code_circuit_wise_arch(distance, capacity, gate_improvements, num_shots, tesselation):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("process_log_color_code_wise.txt")
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Starting circuit generation for distance {distance}, capacity {capacity} and tesselation {tesselation}")
  
    circuit = QCCDCircuit.generate_color_code(distance, rounds=2, tesselation=(6,6,6))
    nqubitsNeeded = (9*distance**2 -1)//8

    nrowsNeeded = int(np.sqrt(nqubitsNeeded))+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    wiseArch = QCCDWiseArch(m=int(np.sqrt(capacity*nqubitsNeeded/2))+1, n=int(np.sqrt(2*nqubitsNeeded/capacity))+1, k=capacity)
    arch, (instructions, _) = circuit.processCircuitWiseArch(wiseArch=wiseArch)
    
    arch.refreshGraph()

    results = {"ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, "QubitOperations": {}, "LogicalErrorRates": {}, "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}, "Electrodes": {}, "DACs": {}}

    # FIXME legacy formatting!
    label ="Forwarding"

    logger.info(f"Processing operations using {label} for distance {distance}, capacity {capacity} and tesselation {tesselation}")
    allOps, barriers = ionRoutingWISEArch(arch, wiseArch, instructions)
 
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    logicalErrors = []
    physicalZErrors = []
    physicalXErrors = []
    
    for gate_improvement in gate_improvements:
        logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement)
        logicalErrors.append(logicalError)
        physicalZErrors.append(physicalZError)
        physicalXErrors.append(physicalXError)

    logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity} and tesselation {tesselation}")

    for op in parallelOpsMap.values():
        op.calculateOperationTime()
        op.calculateFidelity()

    circuit.resetArch()
    arch.refreshGraph()

    results["Capacity"] = capacity
    results["Distance"] = distance
    results["ElapsedTime"][label] = max(parallelOpsMap.keys())
    results["Operations"][label] = len(allOps)
    results["MeanConcurrency"][label] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
    results["QubitOperations"][label] = len(instructions)
    results["LogicalErrorRates"][label] = logicalErrors
    results["PhysicalZErrorRates"][label] = physicalZErrors
    results["PhysicalXErrorRates"][label] = physicalXErrors

    Njz = np.ceil(nqubitsNeeded / capacity)
    Nlz = nqubitsNeeded - Njz # note the difference because we do not have vertical traps

    Nde = NDE_LZ*Nlz+NDE_JZ*Njz
    Nse = NSE_Z*(Njz+Nlz)

    Num_electrodes =int( Nde+Nse)
    Num_DACs = int(min(100, Nde)+np.ceil(Nse/100))
    results["DACs"][label] = Num_DACs
    results["Electrodes"][label] = Num_electrodes

    logger.info(f"{distance} {capacity} {label} = {results}")

    logger.info(f"Finished processing for distance {distance} and capacity {capacity}")
    return results