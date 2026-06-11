#!/usr/bin/env python3
SRC = "src/token_optimizer/core/signal_noise.py"
with open(SRC, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "re.search" in line and "Error" in line and "Exception" in line:
        lines[i] = '            if self._re_error_trace.search(stripped):\n'
        print(f"Fixed line {i+1}")
        break

with open(SRC, "w") as f:
    f.writelines(lines)
print("Done")
