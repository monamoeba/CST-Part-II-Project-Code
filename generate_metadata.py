import os
import re
import pickle
import sys
sys.path.append('src')
from color_code_utils.color_code_circuits.color_code_circuit_666 import ColorCodeCircuit666
from color_code_utils.color_code_circuits.color_code_circuit_488 import ColorCodeCircuit488

def parse_qasm_ancilla_order(qasm_text, n_data, n_ancilla):
    resets = re.findall(r"reset\s+q\[(\d+)\];", qasm_text)
    ancilla_indices = sorted({int(idx) for idx in resets if resets.count(idx) > 1})
    expected_ancillas = list(range(n_data, n_data + n_ancilla))
    if ancilla_indices != expected_ancillas:
        raise ValueError(
            f"Renamed QASM ancilla ordering mismatch: expected {expected_ancillas}, "
            f"found {ancilla_indices}"
        )
    return ancilla_indices

def generate_metadata():
    renamed_dir = 'renamed_qasm_files'
    metadata_dir = 'stabilizer_metadata'
    os.makedirs(metadata_dir, exist_ok=True)

    for fname in os.listdir(renamed_dir):
        if not fname.endswith('.qasm'):
            continue
        m = re.match(r'(\d+)-(\d+)-(\d+)_code(\d+)Ancilla\.qasm', fname)
        if not m:
            print(f"Skipping {fname}, no match")
            continue
        n_data, k, d_str, ancilla = m.groups()
        n_data = int(n_data)
        d = int(d_str)
        ancilla = int(ancilla)

        code_found = False
        qasm_path = os.path.join(renamed_dir, fname)
        with open(qasm_path, 'r') as qasm_file:
            qasm_text = qasm_file.read()
        parse_qasm_ancilla_order(qasm_text, n_data, ancilla)

        for code_class, code_name in [(ColorCodeCircuit666, '666'), (ColorCodeCircuit488, '488')]:
            try:
                circ = code_class(d, 1)
                total_q = len(circ.qtoid)
                if total_q == n_data + ancilla:
                    code_found = True
                    tiles = circ._tiles
                    all_coords = sorted(circ.qtoid.keys(), key=lambda q: circ.qtoid[q])
                    data_coords_set = {q for tile in tiles for q in tile.qubits}
                    data_coords = [q for q in all_coords if q in data_coords_set]
                    num_data = len(data_coords)
                    if num_data != n_data:
                        raise ValueError(f"Data qubit count mismatch for {fname}: expected {n_data}, found {num_data}")
                    print(f"For {fname}, {code_name} d={d}, data_qubits={num_data}, total_qubits={total_q}, ancilla={ancilla}")

                    stabilizers = []
                    colors = []
                    color_map = {'red': 0, 'green': 1, 'blue': 2}

                    for tile in tiles:
                        # X stabilizer
                        x_bits = [1 if q in tile.qubits else 0 for q in data_coords]
                        z_bits = [0] * num_data
                        stab_str = '[' + ' '.join(map(str, x_bits)) + '|' + ' '.join(map(str, z_bits)) + ']'
                        stabilizers.append(stab_str)
                        colors.append(color_map[tile.color])

                        # Z stabilizer
                        x_bits = [0] * num_data
                        z_bits = [1 if q in tile.qubits else 0 for q in data_coords]
                        stab_str = '[' + ' '.join(map(str, x_bits)) + '|' + ' '.join(map(str, z_bits)) + ']'
                        stabilizers.append(stab_str)
                        colors.append(color_map[tile.color])

                    matrix_str = '\n'.join(stabilizers)
                    codeName = f"{n_data}-{k}-{d_str}"

                    with open(f'{metadata_dir}/{codeName}matrix.pkl', 'wb') as f:
                        pickle.dump(matrix_str, f)
                    with open(f'{metadata_dir}/{codeName}colors.pkl', 'wb') as f:
                        pickle.dump(colors, f)
                    print(f"Generated metadata for {codeName}")
                    break
            except Exception as e:
                print(f"Error instantiating {code_name} d={d}: {e}")
                continue
        if not code_found:
            print(f"No matching code found for {fname}")

if __name__ == '__main__':
    generate_metadata()