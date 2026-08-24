from pathlib import Path
text = Path('results_mooccubex/train.log').read_bytes().decode('utf-16', errors='ignore')
print(f"Total chars: {len(text)}")
print("=== LAST 3000 chars ===")
print(text[-3000:])
