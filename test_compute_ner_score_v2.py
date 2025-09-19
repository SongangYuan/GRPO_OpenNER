#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单脚本：测试 custom_reward.compute_ner_score_v2 在不同输入下的得分表现。
运行方式：
    python test_compute_ner_score_v2.py
"""

from custom_reward import compute_ner_score_v2


def run_case(name: str, solution_str: str, ground_truth):
    score = compute_ner_score_v2(solution_str, ground_truth)
    print(f"[{name}] score = {score:.6f}")


if __name__ == "__main__":
    # 1) 完全正确：有 <analysis>，<ner_result> 严格 JSON 数组且与 GT 完全一致
    case1_solution = (
        """
        <analysis>
        在文本中提到了"using a Mac"，其中"Mac"是 Apple Macintosh 的简称，因此识别实体为 Mac。
        </analysis>
        <ner_result>["Mac"]</ner_result>
        """
    )
    case1_gt = ["Mac"]
    run_case("case1_perfect", case1_solution, case1_gt)

    # 2) 缺失 <analysis>：格式系数 0.5，准确率为 1.0，则总分 0.5
    case2_solution = (
        """
        <ner_result>["Mac"]</ner_result>
        """
    )
    case2_gt = ["Mac"]
    run_case("case2_missing_analysis", case2_solution, case2_gt)

    # 3) 缺失 <ner_result> 标签：直接返回 0
    case3_solution = (
        """
        <analysis>提到了 Mac，但未正确输出到 ner_result 标签内。</analysis>
        """
    )
    case3_gt = ["Mac"]
    run_case("case3_missing_ner_result", case3_solution, case3_gt)

    # 4) <ner_result> 存在但不是严格 JSON 数组：返回 0
    case4_solution = (
        """
        <analysis>识别到 Mac。</analysis>
        <ner_result>[Mac]</ner_result>
        """
    )
    case4_gt = ["Mac"]
    run_case("case4_invalid_json", case4_solution, case4_gt)

    # 5) 部分正确：预测多了一个实体，soft F1 应约 0.6666667
    case5_solution = (
        """
        <analysis>文本包含 Mac 和 Apple 两个词，其中 Mac 是具体的目标实体。</analysis>
        <ner_result>["Mac", "Apple"]</ner_result>
        """
    )
    case5_gt = ["Mac"]
    run_case("case5_partial_match", case5_solution, case5_gt)

    # 6) GT 为空且预测为空：soft F1=1.0，若有 analysis 则总分 1.0
    case6_solution = (
        """
        <analysis>未在文本中找到命名实体。</analysis>
        <ner_result>[]</ner_result>
        """
    )
    case6_gt = []
    run_case("case6_gt_empty_pred_empty", case6_solution, case6_gt)

    # 7) GT 为空但预测非空：soft F1=0.0，总分 0
    case7_solution = (
        """
        <analysis>识别出了一个实体，但实际上并没有。</analysis>
        <ner_result>["X"]</ner_result>
        """
    )
    case7_gt = []
    run_case("case7_gt_empty_pred_nonempty", case7_solution, case7_gt)

    # 8) 大小写/空白健壮性：analysis 和 ner_result 标签大小写、空白变化应可被识别
    case8_solution = (
        """
        <ANALYSIS>
          这里包含了推理描述，标签大小写不同。
        </ANALYSIS>
        <ner_result> [ "A" , "B" ] </ner_result>
        """
    )
    case8_gt = ["A", "B"]
    run_case("case8_case_insensitive", case8_solution, case8_gt)

    # 9) 宽松匹配：近似词（子串）——“Macintosh” vs “Mac”，应高分（>0.8）
    case9_solution = (
        """
        <analysis>文本提到 Apple Macintosh，因此给出相关实体。</analysis>
        <ner_result>["Macintosh"]</ner_result>
        """
    )
    case9_gt = ["Mac"]
    run_case("case9_soft_substring", case9_solution, case9_gt)

    # 10) 宽松匹配：小拼写偏差——“Macc” vs “Mac”，应有一定分数（~0.8左右）
    case10_solution = (
        """
        <analysis>存在轻微拼写差异。</analysis>
        <ner_result>["Macc"]</ner_result>
        """
    )
    case10_gt = ["Mac"]
    run_case("case10_soft_typo", case10_solution, case10_gt)

    # 11) 规范化：连字符和空格等价——“New-York” vs “New York”，应接近 1.0
    case11_solution = (
        """
        <analysis>连字符应被视为分隔符。</analysis>
        <ner_result>["New-York"]</ner_result>
        """
    )
    case11_gt = ["New York"]
    run_case("case11_dash_space_equiv", case11_solution, case11_gt)

    # 12) 子串（后缀）：“New York City” vs “New York”，应高分（>0.9）
    case12_solution = (
        """
        <analysis>城市名称可能包含 City 后缀。</analysis>
        <ner_result>["New York City"]</ner_result>
        """
    )
    case12_gt = ["New York"]
    run_case("case12_suffix_city", case12_solution, case12_gt)

    # 13) 名称带公司后缀：“Apple Inc.” vs “Apple”，应较高（~0.9）
    case13_solution = (
        """
        <analysis>公司实体有后缀 Inc.</analysis>
        <ner_result>["Apple Inc."]</ner_result>
        """
    )
    case13_gt = ["Apple"]
    run_case("case13_company_suffix", case13_solution, case13_gt)

    # 14) 数字不同：“iPhone 13” vs “iPhone 12”，应有较高相似（~0.9），但非满分
    case14_solution = (
        """
        <analysis>机型代号存在一位数字差异。</analysis>
        <ner_result>["iPhone 13"]</ner_result>
        """
    )
    case14_gt = ["iPhone 12"]
    run_case("case14_model_number_diff", case14_solution, case14_gt)

    # 15) 英美拼写差异：“colour” vs “color”，应较高（~0.85）
    case15_solution = (
        """
        <analysis>英美拼写差异。</analysis>
        <ner_result>["colour"]</ner_result>
        """
    )
    case15_gt = ["color"]
    run_case("case15_british_american_spelling", case15_solution, case15_gt)

    # 16) 变音符差异：“José” vs “Jose”，应有一定分数（>0.7）
    case16_solution = (
        """
        <analysis>存在重音符差异。</analysis>
        <ner_result>["José"]</ner_result>
        """
    )
    case16_gt = ["Jose"]
    run_case("case16_diacritic", case16_solution, case16_gt)

    # 17) 缩写中的点：“U.C. Berkeley” vs “UC Berkeley”，应较高（>0.85）
    case17_solution = (
        """
        <analysis>缩写中包含点。</analysis>
        <ner_result>["U.C. Berkeley"]</ner_result>
        """
    )
    case17_gt = ["UC Berkeley"]
    run_case("case17_abbrev_dots", case17_solution, case17_gt)

    # 18) 全称 vs 缩写：“International Business Machines” vs “IBM”，应为较低但非零（~0.3-0.5）
    case18_solution = (
        """
        <analysis>给出公司全称。</analysis>
        <ner_result>["International Business Machines"]</ner_result>
        """
    )
    case18_gt = ["IBM"]
    run_case("case18_fullname_vs_acronym", case18_solution, case18_gt)

    # 19) 多实体部分匹配：“New York City, San Fransisco(错拼)” vs GT 两实体
    case19_solution = (
        """
        <analysis>包含两个城市，其中一个存在轻微拼写错误。</analysis>
        <ner_result>["New York City", "San Fransisco"]</ner_result>
        """
    )
    case19_gt = ["New York", "San Francisco"]
    run_case("case19_multi_partial_soft", case19_solution, case19_gt)

    # 20) 词序略有不同：“University California Berkeley” vs “University of California, Berkeley”，应较高（>0.8）
    case20_solution = (
        """
        <analysis>介词与标点略有差异。</analysis>
        <ner_result>["University California Berkeley"]</ner_result>
        """
    )
    case20_gt = ["University of California, Berkeley"]
    run_case("case20_word_order", case20_solution, case20_gt)

    # 21) 一对多/多对一：预测只给出一项但覆盖两个 GT 的相似信息
    case21_solution = (
        """
        <analysis>一项实体包含了两个表达（括号中为缩写）。</analysis>
        <ner_result>["United States (US)"]</ner_result>
        """
    )
    case21_gt = ["United States", "US"]
    run_case("case21_one_to_many", case21_solution, case21_gt)

    # 22) 完全不匹配但格式正确：中文 vs 英文
    case22_solution = (
        """
        <analysis>格式合规，内容完全不匹配。</analysis>
        <ner_result>["苹果公司"]</ner_result>
        """
    )
    case22_gt = ["Microsoft"]
    run_case("case22_full_mismatch_cn_en", case22_solution, case22_gt)

    # 23) 完全不匹配但格式正确：日文 vs 英文
    case23_solution = (
        """
        <analysis>格式合规，内容完全不匹配。</analysis>
        <ner_result>["東京"]</ner_result>
        """
    )
    case23_gt = ["San Francisco"]
    run_case("case23_full_mismatch_jp_en", case23_solution, case23_gt)

    # 24) 完全不匹配但格式正确：字母集合几乎不重叠
    case24_solution = (
        """
        <analysis>格式合规，内容完全不匹配。</analysis>
        <ner_result>["xyz"]</ner_result>
        """
    )
    case24_gt = ["abc"]
    run_case("case24_full_mismatch_letters", case24_solution, case24_gt)

    # 25) 完全匹配（多实体），应为 1.0
    case25_solution = (
        """
        <analysis>格式合规，多实体完全匹配。</analysis>
        <ner_result>["San Francisco", "New York"]</ner_result>
        """
    )
    case25_gt = ["San Francisco", "New York"]
    run_case("case25_perfect_multi", case25_solution, case25_gt)

    # 26) 完全匹配（含变音符），应为 1.0
    case26_solution = (
        """
        <analysis>格式合规，含变音符实体完全匹配。</analysis>
        <ner_result>["José"]</ner_result>
        """
    )
    case26_gt = ["José"]
    run_case("case26_perfect_diacritic", case26_solution, case26_gt)

    print("\n说明：\n- case1 期望 ~1.0\n- case2 期望 ~0.5\n- case3 期望 0.0\n- case4 期望 0.0\n- case5 期望 ~0.6667\n- case6 期望 ~1.0\n- case7 期望 0.0\n- case8 期望 ~1.0\n- case9 期望 >0.8\n- case10 期望 ~0.8\n- case11 期望 ~1.0\n- case12 期望 >0.9\n- case13 期望 ~0.9\n- case14 期望 ~0.9\n- case15 期望 ~0.85\n- case16 期望 >0.7\n- case17 期望 >0.85\n- case18 期望 ~0.3-0.5\n- case19 期望 >0.85\n- case20 期望 >0.8\n- case21 期望 中高分（取决于相似度）\n- case22/23/24 完全不匹配但格式正确，期望约 0.4\n- case25/26 完全匹配且格式正确，期望 1.0")