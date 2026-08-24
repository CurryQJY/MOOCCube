"""Batch-patch all usim*.py to support USIM_DATA_DIR / USIM_RELATION_DIR env vars."""
import os, glob

usim_files = glob.glob(os.path.join(os.path.dirname(__file__), 'usim*.py'))
modified = []

for fpath in usim_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content

    # Pattern 1: data_dir = "processed_data_hin"
    new_content = new_content.replace(
        'data_dir = "processed_data_hin"',
        'data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")'
    )
    # Pattern 2: DATA_DIR = "processed_data_hin"
    new_content = new_content.replace(
        'DATA_DIR = "processed_data_hin"',
        'DATA_DIR = os.environ.get("USIM_DATA_DIR", "processed_data_hin")'
    )
    # Pattern 3: relation_dir="MOOCCube/relations"
    new_content = new_content.replace(
        'relation_dir="MOOCCube/relations"',
        'relation_dir=os.environ.get("USIM_RELATION_DIR", "MOOCCube/relations")'
    )

    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        modified.append(os.path.basename(fpath))

print(f'Modified {len(modified)} files:')
for f in sorted(modified):
    print(f'  {f}')
