"""批量修改基线脚本，支持通过环境变量 USIM_DATA_DIR 切换数据集。"""
import re, os

# Type A: 使用 load_hin_processed("processed_data_hin") 的文件
TYPE_A_FILES = [
    "lightgcn_static_hin.py",
    "lightgcn_full_hin.py",
]

# Type B: 直接硬编码 "processed_data_hin/xxx" 路径的文件
TYPE_B_FILES = [
    "drop_static_hin.py",
    "drop_full_hin.py",
    "gar_static_hin.py",
    "gar_full_hin.py",
    "lightgcl_static_hin.py",
    "lightgcl_full_hin.py",
    "sasrec_static_hin.py",
    "sasrec_full_hin.py",
]

def patch_type_a(filepath):
    """替换 load_hin_processed("processed_data_hin") 为使用环境变量"""
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # 检查是否已经 patch 过
    if 'USIM_DATA_DIR' in code:
        print(f"  [SKIP] {filepath} - already patched")
        return False

    # 替换 load_hin_processed 调用前插入 data_dir
    old = 'meta, df, content_emb = load_hin_processed("processed_data_hin")'
    new = 'data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")\n    meta, df, content_emb = load_hin_processed(data_dir)'
    if old not in code:
        print(f"  [WARN] {filepath} - pattern not found for load_hin_processed")
        return False

    code = code.replace(old, new)

    # 更新 print 语句 (如果有)
    code = code.replace(
        'print("Loading data from processed_data_hin ...")',
        'print(f"Loading data from {data_dir} ...")'
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  [OK] {filepath}")
    return True


def patch_type_b(filepath):
    """替换硬编码的 "processed_data_hin/xxx" 路径"""
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    if 'USIM_DATA_DIR' in code:
        print(f"  [SKIP] {filepath} - already patched")
        return False

    # 1. 在 main() 的 setup_seed 后插入 data_dir 定义
    old_seed = "    setup_seed(2025)\n"
    new_seed = '    setup_seed(2025)\n    data_dir = os.environ.get("USIM_DATA_DIR", "processed_data_hin")\n'
    if old_seed not in code:
        print(f"  [WARN] {filepath} - setup_seed pattern not found")
        return False
    code = code.replace(old_seed, new_seed, 1)  # only first occurrence

    # 2. 替换 print 中的 processed_data_hin
    code = re.sub(
        r'print\("(Loading Data for .+?) from processed_data_hin\.\.\."\)',
        r'print(f"\1 from {data_dir}...")',
        code
    )

    # 3. 替换所有硬编码路径为 f-string
    code = code.replace('"processed_data_hin/stream_data.pkl"', 'f"{data_dir}/stream_data.pkl"')
    code = code.replace('"processed_data_hin/meta.json"', 'f"{data_dir}/meta.json"')
    code = code.replace('"processed_data_hin/content_emb.pt"', 'f"{data_dir}/content_emb.pt"')

    # 4. 替换 error/exists 中残留的 processed_data_hin 字符串
    code = code.replace(
        'print("Error: processed_data_hin/stream_data.pkl not found")',
        'print(f"Error: {data_dir}/stream_data.pkl not found")'
    )
    code = code.replace(
        'print("Error: run data_process_hin.py first")',
        'print(f"Error: {data_dir}/stream_data.pkl not found")'
    )
    code = code.replace(
        'print("错误: 请先运行 data_process_hin.py")',
        'print(f"错误: {data_dir}/stream_data.pkl 未找到")'
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  [OK] {filepath}")
    return True


if __name__ == "__main__":
    print("=== Patching Type A (load_hin_processed) ===")
    for fn in TYPE_A_FILES:
        patch_type_a(fn)

    print("\n=== Patching Type B (hardcoded paths) ===")
    for fn in TYPE_B_FILES:
        patch_type_b(fn)

    print("\nDone! All baselines now support USIM_DATA_DIR env var.")
