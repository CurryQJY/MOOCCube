"""从 baselines_x_log.txt 中提取所有 FINAL RESULT 表格"""
import re

with open("baselines_x_log.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

in_block = False
block_lines = []

for i, l in enumerate(lines):
    s = l.rstrip()
    if "FINAL" in s and "RESULT" in s or "FINAL TEST REPORT" in s:
        in_block = True
        block_lines = [s]
        continue
    if in_block:
        block_lines.append(s)
        # End of block: line of '=' after metric rows
        if s.strip().startswith("===") and len(block_lines) > 3:
            print("\n".join(block_lines))
            print()
            in_block = False
        # Also capture Saved lines
        if "Saved:" in s:
            in_block = False
            print("\n".join(block_lines))
            print()
