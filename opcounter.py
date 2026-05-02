
with open("./qasmfiles/decompqasmfiles/d3_cc_666_decomp.qasm", "r") as f:
    text = f.readlines()

text = text[6:]

opcount = 0
for line in text:
    if line.startswith("cx") or line.startswith("h") or line.startswith("reset") or line.startswith("measure"):
        opcount += 1

print(f"Total operations: {opcount}")