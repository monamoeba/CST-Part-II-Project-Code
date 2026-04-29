import os
import re
from collections import Counter

INPUT_DIR = "qasmfiles"        # folder with original files
OUTPUT_DIR = "renamed_qasm_files"    # new folder

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_distance(filename):
    # assumes filenames start with d{distance}
    m = re.match(r"d(\d+)", filename)
    return int(m.group(1)) if m else None

def extract_total_qubits(text):
    m = re.search(r"qreg\s+q\[(\d+)\];", text)
    return int(m.group(1)) if m else None

def find_ancilla_indices(text):
    resets = re.findall(r"reset\s+q\[(\d+)\];", text)
    counts = Counter(resets)
    return sorted(int(q) for q, c in counts.items() if c > 1)

def normalize_qubit_order(text, total_qubits, ancilla_indices):
    if total_qubits is None:
        return text
    data_indices = [i for i in range(total_qubits) if i not in ancilla_indices]
    ordered_indices = data_indices + ancilla_indices
    old_to_new = {old: new for new, old in enumerate(ordered_indices)}
    if all(old_to_new[i] == i for i in range(total_qubits)):
        return text

    out_lines = []
    for line in text.splitlines():
        if re.match(r"\s*qreg\s+q\[\d+\];", line):
            out_lines.append(line)
            continue

        def repl(match):
            old_idx = int(match.group(1))
            return f"q[{old_to_new.get(old_idx, old_idx)}]"

        out_lines.append(re.sub(r"\bq\[(\d+)\]", repl, line))

    return "\n".join(out_lines)

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".qasm"):
        continue

    in_path = os.path.join(INPUT_DIR, fname)

    with open(in_path, "r") as f:
        text = f.read()

    d = extract_distance(fname)
    total_qubits = extract_total_qubits(text)
    k = 1  # assumed
    ancilla_indices = find_ancilla_indices(text)
    ancilla = len(ancilla_indices)

    if None in (d, total_qubits):
        print(f"Skipping {fname} (couldn't parse)")
        continue

    data_qubits = total_qubits - ancilla
    if data_qubits < 0:
        print(f"Skipping {fname} (invalid ancilla count)")
        continue

    normalized_text = normalize_qubit_order(text, total_qubits, ancilla_indices)
    new_name = f"{data_qubits}-{k}-{d}_code{ancilla}Ancilla.qasm"
    out_path = os.path.join(OUTPUT_DIR, new_name)

    with open(out_path, "w") as f:
        f.write(normalized_text)

    print(f"{fname} → {new_name} (data={data_qubits}, ancilla={ancilla})")