import json
import os
import itertools

# ================= 配置 =================
BASE_DIR = "./MOOCCubeX"
USER_VIDEO_FILE = os.path.join(BASE_DIR, "relations/user-video.json")
CCID_MAP_FILE = os.path.join(BASE_DIR, "relations/video_id-ccid.txt")


# =======================================

def main():
    print(">>> 开始诊断 ID 对齐问题...")

    # 1. 采样 映射文件 (生成的 keys)
    print(f"1. 读取映射文件: {CCID_MAP_FILE}")
    map_keys = set()
    map_samples = []

    with open(CCID_MAP_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # 假设格式是 V_ID \t CCID
                ccid = parts[1]
                map_keys.add(ccid)
                if i < 5: map_samples.append(f"'{ccid}' -> {parts[0]}")

    print(f"   [映射表] 样本: {map_samples}")
    print(f"   [映射表] 总数: {len(map_keys)}")

    # 2. 采样 用户日志 (User Logs)
    print(f"\n2. 读取用户日志: {USER_VIDEO_FILE}")
    user_keys = set()
    user_samples = []

    # 统计日志里到底有哪些字段
    field_counter = {}

    with open(USER_VIDEO_FILE, 'r', encoding='utf-8') as f:
        # 只读前 10000 行做统计
        for i, line in enumerate(f):
            if i >= 10000: break
            try:
                obj = json.loads(line)

                # 收集所有可能的 ID
                candidates = []
                if 'video_id' in obj: candidates.append(str(obj['video_id']))
                if 'ccid' in obj: candidates.append(str(obj['ccid']))
                if 'resource_id' in obj: candidates.append(str(obj['resource_id']))

                # 记录存在的字段名
                for k in obj.keys():
                    field_counter[k] = field_counter.get(k, 0) + 1

                for c in candidates:
                    user_keys.add(c)
                    if len(user_samples) < 5: user_samples.append(f"'{c}'")
            except:
                continue

    print(f"   [用户日志] 字段分布: {field_counter}")
    print(f"   [用户日志] ID样本: {user_samples}")
    print(f"   [用户日志] ID去重数(前1w行): {len(user_keys)}")

    # 3. 核心：计算交集
    intersection = map_keys & user_keys
    print(f"\n3. 交集分析")
    print(f"   完全匹配的数量: {len(intersection)}")

    if len(intersection) == 0:
        print("\n!!! 结论: 完全不匹配 !!!")
        print("尝试分析原因:")

        # 取出一个样本对比
        m_sample = list(map_keys)[0] if map_keys else "Empty"
        u_sample = list(user_keys)[0] if user_keys else "Empty"

        print(f"   Map 里的样子: {m_sample} (长度: {len(m_sample)})")
        print(f"   User 里的样子: {u_sample} (长度: {len(u_sample)})")

        # 检查是否大小写问题
        lower_map = {k.lower() for k in map_keys}
        lower_user = {k.lower() for k in user_keys}
        if len(lower_map & lower_user) > 0:
            print("   -> 发现是【大小写】不一致！代码加 .lower() 即可修复。")
        else:
            print("   -> 内容完全不同。可能是 UserLog 用了 V_xxx 而 Map 存的是 CCID？")

            # 检查 Map 的 Value (V_ID) 是否在 UserKeys 里
            # 重新读取 Map 的 V_ID
            map_vids = set()
            with open(CCID_MAP_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1: map_vids.add(parts[0])

            if len(map_vids & user_keys) > 0:
                print("   -> 破案了！UserLog 里直接存的是 V_ID，不需要 CCID 映射！")
    else:
        print("   竟然有交集？那为什么之前代码跑不通？可能是代码逻辑有 Bug。")


if __name__ == "__main__":
    main()