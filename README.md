# QCCD QEC Compiler — Color Code Extension

A compiler and simulator for mapping Quantum Error Correction (QEC) cycles onto Trapped-Ion Quantum Charge-Coupled Device (QCCD) architectures. Given a QEC code (stabiliser circuit, code distance) and a candidate QCCD device (trap capacity, topology), the tool produces a compiled executable used for architecture evaluation — measuring logical error rates, round times, routing overhead, and parallelisation.

## Attribution

This repository is a fork of [scottjones03/public-material-2025](https://github.com/scottjones03/public-material-2025), which implements the original QCCD compiler and simulator for the **surface code**. All credit for that foundation — the hardware graph model, ion routing, scheduling, and resource-estimation pipeline — belongs to the original authors.

This fork adds a **color code compiler**: qubit-to-ion mapping, ion routing, circuit definitions, and simulation support for the (6,6,6) and (4,8,8) color codes, on top of the original surface-code infrastructure.

## What this fork adds

The original repository compiles and simulates QEC cycles for the surface code only. This fork extends it to color codes, reusing the original hardware model, routing scheduler, and resource-estimation pipeline wherever it already generalised, and adding new components where it didn't:

- **Triangular qubit-to-ion partitioning** (`src/compiler/qccd_color_qubits_to_ions.py`) — a Sierpinski-style recursive partitioning scheme for color code qubit layouts, with three interchangeable cluster-merge strategies (`bounded`/kd-tree, `unbounded_nn`, `k-NN`) plus a direct-coordinate placement mode.
- **Alternative ion routing** (`src/compiler/qccd_alt_ion_routing.py`) — a routing algorithm with distance-based routed/stationary ion selection, aimed at improving intra-trap locality over the original routing scheme.
- **Multi-topology color code processors** (`src/simulator/color_code_processor.py`) — simulation entry points for color code circuits across grid, linear, and switch QCCD topologies, refactored around shared helpers (logging, error collection, result aggregation) rather than duplicated per-topology.
- **Color code circuit definitions** (`src/color_code_utils/`) — Stim-based circuit builders for the (6,6,6) and (4,8,8) color codes, including a [Chromobius](https://github.com/quantumlib/Chromobius)-compatible variant for correlated decoding.
- **Color code support in the core simulator** (`src/simulator/qccd_circuit.py`) — extended with color code circuit generation and Chromobius-based decoding alongside the original surface-code path.
- **Analysis tooling** (`color_code_experiments/`) — threshold fitting, log-ratio heatmaps, and architecture/topology comparison plots for color code experiments, as reusable functions (`analysis_utils.py`) rather than one-off notebook cells.

## Repository Structure

```
.
├── src/
│   ├── compiler/           # Qubit-to-ion mapping and ion routing
│   ├── simulator/          # Circuit execution, resource estimation, error rate calculation
│   ├── utils/              # QCCD hardware graph model and operation primitives
│   └── color_code_utils/   # Color code circuit definitions (Stim-based)
├── configs/                 # All run configurations (see below)
├── color_code_experiments/  # Color code analysis scripts and exploratory notebooks
├── experiments/              # Surface-code architecture exploration scripts
├── scripts/                  # QASM prep scripts for external benchmarking
├── tests/                    # pytest unit and integration tests
├── data/                      # Simulation output JSON files
├── plots/                     
├── results/                   # Misc. benchmark/log outputs
├── main.py                    
├── run_comparison.py          # Runs a fixed set of comparison configs sequentially
├── Makefile
└── requirements.txt
```

### `src/compiler/`

| File | Description |
|---|---|
| `qccd_qubits_to_ions.py` | *(original)* Maps surface code qubits to ions using halving-based partition; also provides `arrangeClusters`/`hillClimbOnArrangeClusters`, reused by the color code path |
| `qccd_color_qubits_to_ions.py` | **(new)** Maps color code qubits using triangular partitioning (Sierpinski/kd-tree variants: unbounded NN, bounded NN, k-NN) |
| `qccd_ion_routing.py` | *(original)* Ion routing for surface code, reused as-is by the color code path |
| `qccd_alt_ion_routing.py` | **(new)** Alternative ion routing with improved intra-trap locality |
| `qccd_parallelisation.py` | *(original)* Scheduling and parallelisation of gate operations |
| `qccd_WISE_ion_route.py` | *(original)* WISE architecture ion routing (surface code only) |

### `src/simulator/`

| File | Description |
|---|---|
| `qccd_circuit.py` | *(original, extended)* QCCD simulator, resource estimation, and logical error rate calculation; extended with color code circuit generation and Chromobius-based decoding |
| `color_code_processor.py` | **(new)** Equivalent processor for color code circuits across grid, linear, and switch topologies |

### `src/utils/`

| File | Description |
|---|---|
| `qccd_arch.py`, `qccd_nodes.py` | *(original)* QCCD hardware model as a directed NetworkX graph (traps and junctions as nodes) |
| `qccd_operations.py`, `qccd_operations_on_qubits.py` | *(original)* QCCD primitive instructions with timing and fidelity |

### `src/color_code_utils/` **(new)**

Color code circuit definitions built on [Stim](https://github.com/quantumlib/Stim):

| File | Description |
|---|---|
| `color_code_circuits/color_code_circuit_666.py` | (6,6,6) color code circuit |
| `color_code_circuits/color_code_circuit_488.py` | (4,8,8) color code circuit |
| `color_code_circuits/color_code_chrom_circuit_666.py` | Chromobius-compatible (6,6,6) circuit |
| `abstract_color_code_circuit.py` | Base class for color code circuits |
| `color_code_tile.py` | Tile geometry representation |

### `color_code_experiments/`

Analysis and plotting scripts for color code experiments, plus the exploratory notebooks they were consolidated from.

- **`analysis_utils.py`** — reusable functions, organised into sections:
  1. Data loading
  2. Threshold fitting (scaling + crossing models, optimal crossing sweep)
  3. Sinter-based threshold plots
  4. Logical error rate plots (JSON data)
  5. Log-ratio heatmaps (single and 1×3 comparative)
  6. Architecture/topology comparison plots and dashboards

  All figure-saving functions write into `plots/` via a shared `PLOTS_DIR` constant, regardless of the working directory the script is run from.
- **`run_analysis.py`** — runner script calling the above with specific data files. Uncomment the relevant section in `__main__` to run a specific analysis. Can be run from the project root or from `color_code_experiments/`.

### `scripts/`

- **`rename_qasm.py`** — reorders qubit indices in QASM circuit files (data qubits before ancillas) and renames them to a `{data}-{k}-{distance}_code{ancilla}Ancilla.qasm` convention.
- **`generate_metadata.py`** — reads the renamed QASM files, matches each against the color code circuit classes, and extracts stabiliser/tile-color metadata for external comparison tooling. Depends on `rename_qasm.py` having been run first, and both scripts run relative to the current working directory (invoke as `python scripts/rename_qasm.py` from the project root).

## Configuration Files

All run configurations live in `configs/`. Fields:

```yaml
hardware:
  trap_capacity: [2, 5, 10, ...]   # ions per trap
  topology: "grid"                  # grid | linear | switch
  placement_strategy: "hill_climb"  # hill_climb | direct_coordinate (optional)

qec:
  code_type: "surface_code"          # or "color_code"
  distances: [3, 5, 7, 9, 11]
  gate_improvements: [1.0, 5.0, 10.0]

simulation:
  rounds: 1
  num_shots: 1000000
  log_file: process_log.txt
  output_dir: data                  # optional
```

## Setup and Usage

Requires Python 3.11.

```bash
make setup                              # create venv and install dependencies
make run                                # run main.py with configs/config.yml
make run CONFIG=configs/config_color.yml  # run with a different config
make test                               # run pytest test suite
make run-analysis                       # run color_code_experiments/run_analysis.py
make run-comparison                     # run run_comparison.py
make clean                              # remove logs, caches, and __pycache__
make reset                              # clean + remove venv
```

`make` auto-detects Windows vs. POSIX venv layouts (`Scripts/` vs `bin/`).
