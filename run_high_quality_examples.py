# -*- coding: utf-8 -*-
"""
Run a few high-quality-analysis test examples against compute_ner_score_v2.
To avoid external LLM calls during this demo, we stub evaluate_dimensions to return a high normalized score.
"""
import json
from typing import List
import logging
import custom_reward as cr

# Show detailed diagnostics during this run
logging.getLogger("custom_reward").setLevel(logging.INFO)


def run_case(name, pred, gt, analysis, original_text: str | None = None):
    solution = (
        f"<analysis>{analysis}</analysis>\n"
        f"<ner_result>{json.dumps(pred, ensure_ascii=False)}</ner_result>"
    )
    print(f"\n=== {name} ===")
    score = cr.compute_ner_score_v2(
        solution,
        gt,
        data_source="hq-examples",
        extra_info={"original_text": (original_text or analysis)},
    )
    print(f"Final score: {score:.4f}")


if __name__ == "__main__":
    cases = [
        # 命中1/3，余下为复合实体
        dict(
            name="Hit_1_of_3_with_composite_leftover",
            pred=["America"],
            gt=["Africa", "America", "Asia and Europe"],
            analysis=(
                "通读全文后，明确提到了 America；此外语境中还涉及到不同大陆的讨论，"
                "但未逐一定名所有大洲。因此先给出确定项 America，复合表达可能需要后续拆解。"
            ),
        ),
        # 修饰词导致的子串覆盖
        dict(
            name="Extra_modifier_substring",
            pred=["uncharted planet"],
            gt=["planet"],
            analysis=(
                "文本中提到一个行星（planet），上下文有‘未知/未探明’的修饰。"
                "因此产出包含修饰词的短语，但核心实体仍为 planet。"
            ),
        ),
        # 英文缩写等价（US vs United States）
        dict(
            name="Acronym_match_US_vs_UnitedStates",
            pred=["US"],
            gt=["United States"],
            analysis=(
                "根据上下文，指代对象为美国。文中同时出现了缩写 US 与全称 United States，"
                "二者等价，因此选择较简短的 US 表达。"
            ),
        ),
        # 预测多给一个无关项，考察精度
        dict(
            name="One_correct_one_noise",
            pred=["America", "Atlantis"],
            gt=["America"],
            analysis=(
                "语段中明确提到 America；此外出现传说中的地点，但与事实实体不一致，"
                "应在抽取阶段予以剔除，此处保留以观察评分影响。"
            ),
        ),
        # 命中2/3，缺失一个复合项
        dict(
            name="Hit_2_of_3_missing_composite",
            pred=["Africa", "America"],
            gt=["Africa", "America", "Asia and Europe"],
            analysis=(
                "段落罗列了多个大洲，其中明确提到 Africa 与 America，关于 Asia 与 Europe 的描述较为含混，"
                "暂未将其并列组合提取为一个项。"
            ),
        ),
        # 轻微拼写错误（typo）
        dict(
            name="Typo_close_spell",
            pred=["Aferica"],
            gt=["Africa"],
            analysis=(
                "根据语义应为 Africa，因拼写疏忽写成 Aferica。含义一致但存在字符级差异。"
            ),
        ),
        # 中文简称与全称（北大 vs 北京大学）
        dict(
            name="Chinese_abbrev_substring",
            pred=["北大"],
            gt=["北京大学", "清华大学"],
            analysis=(
                "上下文讨论高校，其中明确涉及北京大学（常用简称‘北大’）。未发现清华大学的明确实体提及。"
            ),
        ),
        # 完全不相关
        dict(
            name="No_overlap_concepts",
            pred=["Mars"],
            gt=["planet Earth"],
            analysis=(
                "语境目标是地球（planet Earth），但预测误指向火星（Mars），属于实体类别相近但对象不同。"
            ),
        ),
        # 复合实体拆分（预测复合，GT拆开）
        dict(
            name="Composite_pred_against_split_gt",
            pred=["Asia and Europe"],
            gt=["Asia", "Europe"],
            analysis=(
                "文本使用并列结构指称两个大洲，抽取时合并为一个短语，但也可拆分为 Asia 与 Europe 两项。"
            ),
        ),
    ]

    for c in cases:
        run_case(c["name"], c["pred"], c["gt"], c["analysis"])

    # === Added: Compare HIGH vs LOW analysis with the same original_text/pred/gt ===
    compare_original_text = "该公司在 2022 年将业务拓展至 “Asia and Europe”，并计划在 2023 年进入非洲市场。"
    compare_pred = ["Asia and Europe"]
    compare_gt = ["Asia", "Europe"]

    high_quality_analysis = (
        "分析要点：\n"
        "1) 证据与引用\n"
        "- 原文中出现的实体片段为 “Asia and Europe”，这是一个由连词 and 连接的并列结构。\n"
        "- 两个中心名词 “Asia”“Europe”均为大洲的专有名称，语义上各自独立。\n\n"
        "2) 边界与拆分判断\n"
        "- “and”作为并列连词，通常指示两项并列的独立实体而非单一复合实体；若数据集遵循“原子实体”抽取规范，应拆分为两个实体。\n"
        "- 若原文表达为“Eurasia（欧亚大陆）”才可视作单一地理实体；本句并未如此表述，因此“Asia and Europe”更符合拆分。\n\n"
        "3) 备选与权衡\n"
        "- 备选1：保留为一个整体短语“Asia and Europe”。优点：忠实于原文表面形式；缺点：与“按最小语义单位抽取”的惯例冲突，且与 gold [\"Asia\",\"Europe\"] 不一致，导致匹配分下降。\n"
        "- 备选2：拆分为 [\"Asia\",\"Europe\"]。优点：与 gold 对齐、边界准确、减少合并误差；缺点：与原文的并列短语表面形式不完全一致，但不影响语义正确性。\n\n"
        "4) 一致性与误差影响\n"
        "- 当前 ner_result 为 [\"Asia and Europe\"]，与 gold [\"Asia\",\"Europe\"] 不一致；在软匹配下会出现部分分（因包含关系/相似度），但精确匹配与边界一致性较差。\n"
        "- 若改为 [\"Asia\",\"Europe\"]，可提升匹配与一致性，避免把两项实体合并为一项导致的召回与精度折损。\n\n"
        "5) 边界与例外检查\n"
        "- 若出现限定词如 “East Asia and Western Europe”，仍应拆分为两项区域性实体，分别处理形容词限定的边界。\n"
        "- 若原文出现“Asia-Europe corridor”之类复合专名，需要根据语义与标注规范判断是否为单一术语；本句并无该情形。\n\n"
        "结论：基于并列结构证据与标注一致性，应输出 [\"Asia\",\"Europe\"]；当前的合并结果应更正为拆分结果以获得更高的匹配与一致性。"
    )

    low_quality_analysis = (
        "这句话大概说到了地区，抽成 “Asia and Europe” 就可以了。公司扩张范围很大，所以合并更合理，也更简洁，"
        "没有必要拆开。总体上看不出有什么问题，就这样就行了。"
    )

    print("\n=== Compare_HIGH_analysis ===")
    run_case(
        name="Compare_HIGH_analysis",
        pred=compare_pred,
        gt=compare_gt,
        analysis=high_quality_analysis,
        original_text=compare_original_text,
    )

    print("\n=== Compare_LOW_analysis ===")
    run_case(
        name="Compare_LOW_analysis",
        pred=compare_pred,
        gt=compare_gt,
        analysis=low_quality_analysis,
        original_text=compare_original_text,
    )