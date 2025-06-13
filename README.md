High-Level Problem
 We require resource-efficient mappings of QEC cycles for the surface code onto QCCD systems with different architectures to answer the design questions. While several tool flows have been developed to map NISQ workloads on QCCD hardware, they incur large communication overheads and do not scale to high QEC code distances. This is primarily because they do not consider the local structure available in stabiliser circuits. The compiler accepts a QEC code (stabiliser circuits, code distance) and a candidate QCCD device architecture (trap capacity, topology) as input and produces a compiled executable output, which is used for architecture evaluation.


Repository Structure 

src/:


• compiler/: code for the QEC compiler.

– qccd_ion_routing code for ion routing (§3.1.4). [222 lines]

– qccd_parallelisation code for scheduling. [218 lines]

– qccd_qubits_to_ions code for mapping qubits to ions (§3.1.2). [305 lines]

• simulator/:

– qccd_circuit code for QCCD Simulator, QCCD resource estimation and logical error rate calculation (using Stim [33] and the PyMatching decoder [25] (§3.2, §3.3.2). [715 lines]

• utils/:

– qccd_arch, qccd_nodes code for modelling QCCD hardware as a directed graph object using NetworkX, with traps and junctions as nodes (§3.1.4, §3.3.1). [996 lines]

– qccd_operations, qccd_operations_on_qubits code for QCCD primitive instruc- tions, treating them as objects with effects, associated components, operation time, and fidelity (§3.1.3). [1085 lines]

• tests/: code for unit and integration tests that validate the compiler’s functionality using the pytest framework. [225 lines]

• experiments/: Jupyter notebooks for QCCD architectural exploration. [3397 lines] 

• results/: outputs of simulations.

• Makefile: code for build automation, allowing execution in virtual environments. 

• config.yaml: configuration file to specify inputs for experiments
