# -*- coding: utf-8 -*-
import json
import logging
import custom_reward as cr

# Show INFO logs for diagnostics
logging.getLogger("custom_reward").setLevel(logging.INFO)

# Stub LLM scoring to avoid any external/model calls
cr.evaluate_dimensions = lambda **kwargs: {"normalized_score": 0.0}


def run_case(name, pred, gt, analysis):
    solution = (
        f"<analysis>{analysis}</analysis>\n"
        f"<ner_result>{json.dumps(pred, ensure_ascii=False)}</ner_result>"
    )
    print(f"\n=== {name} ===")
    score = cr.compute_ner_score_v2(
        solution,
        gt,
        data_source="unit-tests",
        extra_info={"original_text": analysis},
    )
    print(f"Final score: {score:.4f}")


if __name__ == "__main__":
    default_analysis = "Reason about entities list clearly."
    run_case(
        "Example1_missing_two",
        ["America"],
        ["Africa", "America", "Asia and Europe"],
        default_analysis,
    )
    run_case(
        "Example2_extra_modifier",
        ["uncharted planet"],
        ["planet"],
        default_analysis,
    )