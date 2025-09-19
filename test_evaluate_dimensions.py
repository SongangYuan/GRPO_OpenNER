# -*- coding: utf-8 -*-
"""
Smoke/batch tests for evaluate_dimensions in llm_scoring.
- Default: run a single smoke case
- Options: --all to run all builtin cases; --case name1,name2 to run subset
- Single-text mode: --text "..." with optional prompt fields
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from llm_scoring import evaluate_dimensions


CASES: Dict[str, Dict[str, Any]] = {
    "tech_launch": {
        "original_text": "Apple Inc. today released the next-generation iPhone in Cupertino, California, featuring the A18 chip with improved battery life and thermal performance.",
        "ner_query": "Identify products and organizations",
        "analysis_content": "The model identifies Apple Inc. as an organization; iPhone as a product; A18 as a chip model (a product attribute).",
        "ner_result": "ORG: Apple Inc.; PROD: iPhone; CHIP: A18",
        "gold_entities": "ORG: Apple Inc.; PROD: iPhone",
    },
    "medical_note": {
        "original_text": "The patient was diagnosed with non-Hodgkin lymphoma and received the R-CHOP regimen with good response.",
        "ner_query": "Identify diseases and therapies",
        "analysis_content": "non-Hodgkin lymphoma is a disease entity; R-CHOP is an acronym for a chemotherapy regimen.",
        "ner_result": "DISEASE: non-Hodgkin lymphoma; THERAPY: R-CHOP",
        "gold_entities": "DISEASE: non-Hodgkin lymphoma; THERAPY: R-CHOP",
    },
    "finance_report": {
        "original_text": "The company's Q3 revenue increased 12% year-over-year, and net profit margin rose to 18%, mainly driven by the recovery of the North American market.",
        "ner_query": "Identify financial metrics and regions",
        "analysis_content": "Revenue growth and net profit margin are financial metrics; North America is a region entity.",
        "ner_result": "KPI: Revenue; KPI: Net profit margin; REGION: North America",
        "gold_entities": "KPI: Revenue; KPI: Net profit margin; REGION: North America",
    },
    "polysemy": {
        "original_text": "Amazon's river management project is advancing in Peru, and Amazon also announced a new AWS region.",
        "ner_query": "Identify organizations and locations",
        "analysis_content": "'Amazon' may refer to the company or the Amazon River basin; here the contexts correspond to a river management project (geographic) and an AWS region (company).",
        "ner_result": "ORG: Amazon; LOC: Peru; ORG: AWS",
        "gold_entities": "ORG: Amazon; LOC: Peru; ORG: AWS",
    },
    "no_entity": {
        "original_text": "This paragraph mainly comments on writing style and does not mention specific entities.",
        "ner_query": "Identify any entities",
        "analysis_content": "The text is descriptive and lacks extractable named entities.",
        "ner_result": "",
        "gold_entities": "",
    },
    # Combined-input style example: if only 'text' is provided, it will be used as both original_text and ner_query
    "news_politics_zh": {
        "text": "The State Council issued guidelines on promoting the development of the digital economy, proposing to establish a data factor market system by 2025 and to promote the healthy development of platform enterprises.",
        "ner_query": "Identify organizations, document titles, and policy key points",
        "analysis_content": "Contains organization, policy document, and policy key points.",
        "ner_result": "ORG: State Council; DOC: Guiding Opinions; KPI: Digital Economy",
        "gold_entities": "ORG: State Council; DOC: Guiding Opinions",
    },
}


def _assert_scores_structure(result: Dict[str, Any]):
    assert isinstance(result, dict), "result must be dict"
    assert "scores" in result and isinstance(result["scores"], list)
    for item in result["scores"]:
        assert isinstance(item, dict)
        assert "dimension_name" in item and isinstance(item["dimension_name"], str)
        assert "reason" in item and isinstance(item["reason"], str)
        assert "score" in item
        s = item["score"]
        assert isinstance(s, int) and 0 <= s <= 10
    assert "evaluated_dimensions" in result and isinstance(result["evaluated_dimensions"], list)
    assert "normalized_score" in result and isinstance(result["normalized_score"], float)


# If only 'text' is provided, it will be used as both original_text and ner_query
# This function returns a unified 'text' and the four required fields

def _coerce_inputs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "text" in cfg and cfg["text"]:
        t = cfg["text"]
        return {
            "text": t,
            "analysis_content": cfg.get("analysis_content", ""),
            "ner_result": cfg.get("ner_result", ""),
            "gold_entities": cfg.get("gold_entities", ""),
        }
    # Otherwise, use original_text as text; ignore independent ner_query per the new convention
    return {
        "text": cfg.get("original_text", ""),
        "analysis_content": cfg.get("analysis_content", ""),
        "ner_result": cfg.get("ner_result", ""),
        "gold_entities": cfg.get("gold_entities", ""),
    }


def _run_one(name: str, cfg: Dict[str, Any]):
    print(f"\n[CASE] {name}")
    args = _coerce_inputs(cfg)
    res = evaluate_dimensions(
        text=args["text"],
        analysis_content=args["analysis_content"],
        ner_result=args["ner_result"],
        gold_entities=args["gold_entities"],
    )
    _assert_scores_structure(res)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="run all preset cases")
    p.add_argument("--cases", type=str, default="", help="comma-separated case names")
    p.add_argument("--text", type=str, default="", help="custom text (used as both original_text and ner_query)")
    args = p.parse_args()

    if args.all:
        for k, v in CASES.items():
            _run_one(k, v)
        return

    if args.cases:
        selected = [x.strip() for x in args.cases.split(",") if x.strip()]
        for k in selected:
            if k in CASES:
                _run_one(k, CASES[k])
            else:
                print(f"[WARN] case not found: {k}")
        return

    if args.text:
        _run_one("custom", {"text": args.text, "analysis_content": "", "ner_result": "", "gold_entities": ""})
        return

    print("Please run with --all or --cases or --text.")


if __name__ == "__main__":
    main()