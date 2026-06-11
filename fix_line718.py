#!/usr/bin/env python3
"""Fix line 718: HistoryCompressor needs self.classifier._re_error_trace"""
SRC = "src/token_optimizer/core/signal_noise.py"
with open(SRC, "r") as f:
    lines = f.readlines()

# Line 718 is in HistoryCompressor._compress_assistant_reply
# Change self._re_error_trace to self.classifier._re_error_trace
for i, line in enumerate(lines):
    if i >= 700 and "_re_error_trace" in line and "self.classifier" not in line:
        lines[i] = line.replace("self._re_error_trace", "self.classifier._re_error_trace")
        print(f"Fixed line {i+1}")
        break

with open(SRC, "w") as f:
    f.writelines(lines)
print("Done")
