# -*- coding: utf-8 -*-
"""
Smoke test for llm_scoring.dimension_detection
- Calls the function with a sample text (or user-provided text via --text)
- Prints the parsed JSON result
- Performs minimal structural assertions
- Supports batch mode with predefined diverse cases via --all or --case
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

from llm_scoring import dimension_detection

# Predefined diverse cases to probe different dimensions
CASES: List[Tuple[str, str]] = [
    (
        "boundary_ambiguity",
        "The report mentions the term 'non-Hodgkin lymphoma', which is sometimes abbreviated as 'non-Hodgkin'.",
    ),
    (
        "medical_domain",
        "The patient was diagnosed with acute myeloid leukemia and received azacitidine treatment.",
    ),
    (
        "finance_domain",
        "Tesla rose 7% at the Nasdaq close, with its market capitalization surpassing $800 billion.",
    ),
    (
        "polysemy",
        "Apple plans to hold a launch event in Cupertino, California.",
    ),
    (
        "complex_long",
        "Between 1997 and 2001, Google's co-founder Larry Page repeatedly mentioned the PageRank algorithm and its role in search ranking.",
    ),
    (
        "no_entity",
        "The sunshine is great today, perfect for a walk in the park.",
    ),
    (
        "geo_ambiguity",
        "Georgia passed a new election law on Tuesday.",
    ),
    (
        "quote_conflict",
        "According to 'John Smith', he is not the same person as 'John Smith'.",
    ),
]


def _assert_structure(res: Dict[str, Any]) -> None:
    assert isinstance(res, dict), "Result must be a dict"
    assert "relevant_dimensions" in res, "Missing key: relevant_dimensions"
    dims = res["relevant_dimensions"]
    assert isinstance(dims, list), "relevant_dimensions must be a list"
    for i, item in enumerate(dims):
        assert isinstance(item, dict), f"Item #{i} must be a dict"
        assert "dimension_name" in item, f"Item #{i} missing 'dimension_name'"
        assert "relevance_reason" in item, f"Item #{i} missing 'relevance_reason'"
        assert isinstance(item["dimension_name"], str), f"Item #{i} 'dimension_name' must be str"
        assert isinstance(item["relevance_reason"], str), f"Item #{i} 'relevance_reason' must be str"


def _run_one(name: str, text: str) -> None:
    print(f"\n=== Case: {name} ===")
    # Pass required fields per new signature; use empty placeholders to satisfy required params
    res = dimension_detection(
        text=text,
        analysis_content="",
        ner_result="",
        gold_entities="",
    )
    _assert_structure(res)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test llm_scoring.dimension_detection")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Raw text to analyze (if provided, runs single case)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all predefined diverse cases",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Comma-separated case names to run (e.g., boundary_ambiguity,medical_domain)",
    )

    args = parser.parse_args()

    try:
        if args.all:
            for name, text in CASES:
                _run_one(name, text)
            return 0
        if args.case:
            wanted = {x.strip() for x in args.case.split(",") if x.strip()}
            found = False
            for name, text in CASES:
                if name in wanted:
                    _run_one(name, text)
                    found = True
            if not found:
                print(f"[Warn] No matched cases for: {args.case}")
            return 0
        # Fallback: single run
        single_text = (
            args.text
            if args.text is not None
            else "Microsoft co-founder Bill Gates met with several Chinese entrepreneurs in Beijing on Wednesday."
        )
        _run_one("single", single_text)
        return 0
    except Exception as e:
        print(f"[Test Error] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())