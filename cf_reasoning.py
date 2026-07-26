# -*- coding: utf-8 -*-
"""
CF 不确定性推理程序

题目规则：
RULE1: IF E1 THEN H1 (0.9)
RULE2: IF E2 THEN H1 (0.8)
RULE3: IF E3 THEN H1 (0.9)
RULE4: IF E4 AND E5 THEN E1 (0.9)
RULE5: IF E6 AND (E7 OR E8) THEN E3 (1.0)
RULE6: IF E9 THEN H (0.9)
RULE7: IF H1 THEN H (0.9)

已知证据：
E2 = -0.8
E4 = 0.9
E5 = 0.8
E6 = 0.9
E7 = -0.3
E8 = 0.8
E9 = 0.9

计算目标：
H 的可信度 CF(H)
"""


def cf_and(*values):
    """
    CF 中 AND 运算：取最小值
    """
    return min(values)


def cf_or(*values):
    """
    CF 中 OR 运算：取最大值
    """
    return max(values)


def rule_infer(premise_cf, rule_cf):
    """
    规则推理：
    结论 CF = 前提 CF × 规则 CF
    """
    return premise_cf * rule_cf


def combine_cf(cf1, cf2):
    """
    两个 CF 值的合成规则：

    1. 两个都是正证据：
       CF = cf1 + cf2 * (1 - cf1)

    2. 两个都是负证据：
       CF = cf1 + cf2 * (1 + cf1)

    3. 一个正、一个负：
       CF = (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))
    """
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1 - cf1)

    if cf1 < 0 and cf2 < 0:
        return cf1 + cf2 * (1 + cf1)

    return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))


def combine_many(cf_values):
    """
    多个 CF 值依次合成
    """
    if not cf_values:
        raise ValueError("cf_values 不能为空")

    result = cf_values[0]
    for cf in cf_values[1:]:
        result = combine_cf(result, cf)

    return result


def main():
    """
    主函数：按照题目规则逐步计算 H 的可信度
    """

    # -----------------------------
    # 1. 输入初始证据 CF
    # -----------------------------
    E2 = -0.8
    E4 = 0.9
    E5 = 0.8
    E6 = 0.9
    E7 = -0.3
    E8 = 0.8
    E9 = 0.9

    print("========== 初始证据 ==========")
    print(f"E2 = {E2}")
    print(f"E4 = {E4}")
    print(f"E5 = {E5}")
    print(f"E6 = {E6}")
    print(f"E7 = {E7}")
    print(f"E8 = {E8}")
    print(f"E9 = {E9}")
    print()

    # -----------------------------
    # 2. RULE4: IF E4 AND E5 THEN E1 (0.9)
    # -----------------------------
    E1_premise = cf_and(E4, E5)
    E1 = rule_infer(E1_premise, 0.9)

    print("========== 计算 E1 ==========")
    print("RULE4: IF E4 AND E5 THEN E1 (0.9)")
    print(f"CF(E4 AND E5) = min({E4}, {E5}) = {E1_premise}")
    print(f"CF(E1) = {E1_premise} × 0.9 = {E1}")
    print()

    # -----------------------------
    # 3. RULE5: IF E6 AND (E7 OR E8) THEN E3 (1.0)
    # -----------------------------
    E7_or_E8 = cf_or(E7, E8)
    E3_premise = cf_and(E6, E7_or_E8)
    E3 = rule_infer(E3_premise, 1.0)

    print("========== 计算 E3 ==========")
    print("RULE5: IF E6 AND (E7 OR E8) THEN E3 (1.0)")
    print(f"CF(E7 OR E8) = max({E7}, {E8}) = {E7_or_E8}")
    print(f"CF(E6 AND (E7 OR E8)) = min({E6}, {E7_or_E8}) = {E3_premise}")
    print(f"CF(E3) = {E3_premise} × 1.0 = {E3}")
    print()

    # -----------------------------
    # 4. RULE1、RULE2、RULE3 共同推出 H1
    # -----------------------------
    H1_from_E1 = rule_infer(E1, 0.9)
    H1_from_E2 = rule_infer(E2, 0.8)
    H1_from_E3 = rule_infer(E3, 0.9)

    # 注意：
    # H1 的合成顺序可能影响中间值的显示。
    # 这里先合成两个正证据，再合成负证据，方便和手算过程一致。
    H1_positive = combine_many([H1_from_E1, H1_from_E3])
    H1 = combine_cf(H1_positive, H1_from_E2)

    print("========== 计算 H1 ==========")
    print("RULE1: IF E1 THEN H1 (0.9)")
    print(f"CF(H1_from_E1) = {E1} × 0.9 = {H1_from_E1}")
    print()

    print("RULE2: IF E2 THEN H1 (0.8)")
    print(f"CF(H1_from_E2) = {E2} × 0.8 = {H1_from_E2}")
    print()

    print("RULE3: IF E3 THEN H1 (0.9)")
    print(f"CF(H1_from_E3) = {E3} × 0.9 = {H1_from_E3}")
    print()

    print("先合成正证据 H1_from_E1 和 H1_from_E3：")
    print(f"CF_positive(H1) = {H1_positive}")
    print("再与负证据 H1_from_E2 合成：")
    print(f"CF(H1) = {H1}")
    print()

    # -----------------------------
    # 5. RULE6、RULE7 共同推出 H
    # -----------------------------
    H_from_E9 = rule_infer(E9, 0.9)
    H_from_H1 = rule_infer(H1, 0.9)

    H = combine_many([H_from_E9, H_from_H1])

    print("========== 计算 H ==========")
    print("RULE6: IF E9 THEN H (0.9)")
    print(f"CF(H_from_E9) = {E9} × 0.9 = {H_from_E9}")
    print()

    print("RULE7: IF H1 THEN H (0.9)")
    print(f"CF(H_from_H1) = {H1} × 0.9 = {H_from_H1}")
    print()

    print("合成 H_from_E9 和 H_from_H1：")
    print(f"CF(H) = {H}")
    print()

    # -----------------------------
    # 6. 输出最终结果
    # -----------------------------
    print("========== 最终结果 ==========")
    print(f"CF(H1) = {H1:.6f}")
    print(f"CF(H)  = {H:.6f}")
    print(f"所以 H 的可信度约为：{H:.3f}")


if __name__ == "__main__":
    main()
