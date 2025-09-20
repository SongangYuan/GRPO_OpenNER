# -*- coding: utf-8 -*-
"""
Run a few high-quality-analysis test examples against compute_ner_score_v2.
To avoid external LLM calls during this demo, we stub evaluate_dimensions to return a high normalized score.
"""
import json
from typing import List
import logging
import custom_reward as cr

# Silence info logs from custom_reward to keep output concise
logging.getLogger("custom_reward").setLevel(logging.WARNING)

# Stub LLM evaluation to avoid network calls and reflect high-quality analysis
def stub_evaluate_dimensions(text: str, analysis_content: str, ner_result: str, gold_entities: str, max_workers: int | None = None):
    # you can tweak this fixed score per-case if needed
    return {"normalized_score": 0.95}

cr.evaluate_dimensions = stub_evaluate_dimensions  # monkeypatch inside the module


def run_case(name: str, solution_str: str, ground_truth: List[str]):
    score = cr.compute_ner_score_v2(solution_str, ground_truth)
    print(f"[{name}] final_score = {score:.6f}")


# Case A: Perfect match with thorough analysis
caseA_solution = (
    """
    <analysis>
    - Task Understanding: Identify named entities explicitly mentioned in the passage.
    - Evidence: The phrase "using a Mac" appears; here, "Mac" refers to Apple Macintosh (a product line).
    - Boundary Handling: Only the token "Mac" should be extracted; do not include surrounding words or quotes.
    - Error Detection: Ignore brand synonyms that are not present in the text (e.g., "Apple" is implied but not explicitly stated as an entity here).
    - Reasoning: Since the instruction requires literal mentions, "Mac" qualifies while general concepts do not.
    - Final Decision: Output ["Mac"].
    </analysis>
    <ner_result>["Mac"]</ner_result>
    """
)
caseA_gt = ["Mac"]

# Case B: Soft substring match (NYC suffix), great analysis
caseB_solution = (
    """
    <analysis>
    - Context Utilization: The text refers to the city known as "New York City". In common usage, both "New York" and "New York City" denote the same urban entity.
    - BoundaryHandling: Keep the entity token contiguous; avoid including commas or trailing punctuation.
    - ResultConsistency: The main canonical form in GT is "New York"; our prediction includes the suffix "City" but remains semantically equivalent.
    - Multi-perspective Thinking: Consider regional naming conventions and aliases; the match should be high under soft similarity.
    - Final Decision: Output ["New York City"].
    </analysis>
    <ner_result>["New York City"]</ner_result>
    """
)
caseB_gt = ["New York"]

# Case C: Two entities with one slight typo, detailed analysis and self-check
caseC_solution = (
    """
    <analysis>
    - Entities Mentioned: "New York City" and "San Francisco" appear as locations.
    - Error Detection: The second entity may have a minor typo ("San Fransisco" vs "San Francisco").
    - Logical Coherence: Despite the typo, token boundaries are preserved; we do not add extra descriptors.
    - ResultConsistency: Provide both entities as a flat list of strings.
    - Final Decision: Output ["New York City", "San Fransisco"].
    </analysis>
    <ner_result>["New York City", "San Fransisco"]</ner_result>
    """
)
caseC_gt = ["New York", "San Francisco"]

# Case D: Full name vs acronym (moderate soft match), still high-quality analysis
caseD_solution = (
    """
    <analysis>
    - Domain Knowledge Application: "International Business Machines" corresponds to the acronym "IBM".
    - Context Utilization: The passage uses the full corporate name; the GT may list the acronym.
    - BoundaryHandling: Extract the entity without quotes or trailing punctuation.
    - ResultConsistency: Provide a single-item list.
    - Final Decision: Output ["International Business Machines"].
    </analysis>
    <ner_result>["International Business Machines"]</ner_result>
    """
)
caseD_gt = ["IBM"]


if __name__ == "__main__":
    print("Running high-quality analysis examples with stubbed LLM score=0.95...\n")
    run_case("A_perfect_mac", caseA_solution, caseA_gt)
    run_case("B_nyc_suffix_soft", caseB_solution, caseB_gt)
    run_case("C_multi_with_typo", caseC_solution, caseC_gt)
    run_case("D_fullname_vs_acronym", caseD_solution, caseD_gt)