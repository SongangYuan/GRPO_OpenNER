# -*- coding: utf-8 -*-
"""
Batch reporter for evaluate_dimensions.
Runs multiple diverse cases and logs Markdown report to repoert.md
Usage:
  python test_evaluate_dimensions_report.py [--overwrite]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any, Dict, List

from llm_scoring import evaluate_dimensions

MD_PATH = r"c:\\Users\\admin\\Desktop\\临时文件\\M\\repoert.md"


CASES: Dict[str, Dict[str, Any]] = {
    "news_politics_en": {
        "original_text": "The State Council today issued guiding opinions on further moilaboratory, emphasizing streamlining approval procedures and strengthening digital governance.",
        "ner_query": "Identify organizations, document titles, and key policy points",
        "analysis_content": "'State Council' is an organization; 'guiding opinions' is a policy document; keywords include moilaboratory, streamlining approvals, and digital governance.",
        "ner_result": "ORG: State Council; DOC: Guiding Opinions; KPI: moilaboratory",
        "gold_entities": "ORG: State Council; DOC: Guiding Opinions",
    },
    "legal_contract_en": {
        "original_text": "Party A and Party B signed a purchase contract in Beijing on May 1, 2023, agreeing on a total price of CNY 3,000,000 with delivery due on June 30, 2023.",
        "ner_query": "Identify parties, dates, location, amount, and contract type",
        "analysis_content": "Party A/Party B are contracting parties; there are two dates; the location is Beijing; the amount is CNY 3,000,000; the contract type is a purchase contract.",
        "ner_result": "PARTY: Party A; PARTY: Party B; DATE: 2023-05-01; LOC: Beijing; MONEY: CNY 3,000,000",
        "gold_entities": "PARTY: Party A; PARTY: Party B; DATE: 2023-05-01; LOC: Beijing; MONEY: CNY 3,000,000",
    },
    "sports_news_en": {
        "original_text": "Lionel Messi scored a brace in his debut for Inter Miami, helping the team win 3:1.",
        "ner_query": "Identify person, team, and match stats",
        "analysis_content": "Lionel Messi is a person; Inter Miami is a team; 'brace' means two goals; the score is 3:1.",
        "ner_result": "PER: Lionel Messi; ORG: Inter Miami; SCORE: 3:1",
        "gold_entities": "PER: Lionel Messi; ORG: Inter Miami; SCORE: 3:1",
    },
    "research_abstract_en": {
        "original_text": "We propose a Transformer-based cross-domain entity recognition method that achieves SOTA performance on five public datasets.",
        "ner_query": "Identify method, model, datasets, and metrics",
        "analysis_content": "The method is based on Transformer; it involves cross-domain entity recognition; 'five public datasets' are not named; SOTA is a metric description.",
        "ner_result": "MODEL: Transformer; TASK: NER; CLAIM: SOTA",
        "gold_entities": "MODEL: Transformer; TASK: Cross-domain NER",
    },
    "product_bug_en": {
        "original_text": "Users report that opening the gallery on Android 14 causes a crash; logs show a NullPointerException in the ImageLoader module.",
        "ner_query": "Identify platform, module, and exception type",
        "analysis_content": "Platform: Android 14; Exception: NullPointerException; Module: ImageLoader; Symptom: crash.",
        "ner_result": "PLATFORM: Android 14; MODULE: ImageLoader; EXC: NullPointerException",
        "gold_entities": "PLATFORM: Android 14; MODULE: ImageLoader; EXC: NullPointerException",
    },
    "mixed_language_en": {
        "original_text": "OpenAI released a new model in San Francisco, supports multi-turn reasoning, and improves reasoning speed compared to GPT-4.",
        "ner_query": "Identify organizations, locations, products/models",
        "analysis_content": "OpenAI is an organization; San Francisco is a location; the text contrasts a new model with GPT-4; keyword: multi-turn reasoning.",
        "ner_result": "ORG: OpenAI; LOC: San Francisco; MODEL: new model; MODEL: GPT-4",
        "gold_entities": "ORG: OpenAI; LOC: San Francisco; MODEL: GPT-4",
    },
    "finance_earnings_en": {
        "original_text": "The company's Q4 revenue increased 20% year-over-year, and net profit margin rose to 22%, mainly driven by the recovery of the European market and cost optimization.",
        "ner_query": "Identify financial metrics and regions",
        "analysis_content": "Revenue and net profit margin are financial metrics; Europe is a region; reasons include market recovery and cost optimization.",
        "ner_result": "KPI: Revenue; KPI: Net profit margin; REGION: Europe",
        "gold_entities": "KPI: Revenue; KPI: Net profit margin; REGION: Europe",
    },
    "poetry_no_entity_en": {
        "original_text": "The setting sun and lone ducks fly together; the autumn waters share the same hue with the vast sky.",
        "ner_query": "Identify any entities",
        "analysis_content": "Imagery description, typically lacking standard named entities.",
        "ner_result": "",
        "gold_entities": "",
    },
}


def _truncate(text: str, n: int = 160) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _write(md_lines: List[str], overwrite: bool) -> None:
    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    mode = "w" if overwrite or not os.path.exists(MD_PATH) else "a"
    with open(MD_PATH, mode, encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def _run_and_collect(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    res = evaluate_dimensions(
        text=cfg.get("original_text", ""),
        analysis_content=cfg.get("analysis_content", ""),
        ner_result=cfg.get("ner_result", ""),
        gold_entities=cfg.get("gold_entities", ""),
        max_workers=1,  # not used in single-prompt mode, keep for signature compatibility
    )
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="overwrite repoert.md instead of appending")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md: List[str] = []
    md.append(f"# Evaluation Report - {ts}")
    md.append("")

    for name, cfg in CASES.items():
        res = _run_and_collect(name, cfg)
        # Build markdown section for this case
        md.append(f"## Case: {name}")
        md.append("")
        md.append("**Input Summary**:")
        md.append("")
        md.append(f"- Original text: {cfg.get('original_text','')}")
        md.append(f"- NER query: {cfg.get('ner_query','')}")
        md.append(f"- Analysis content: {cfg.get('analysis_content','')}")
        md.append(f"- NER result: {cfg.get('ner_result','')}")
        md.append(f"- Gold entities: {cfg.get('gold_entities','')}")
        md.append("")
        md.append("**Model Output Overview**:")
        md.append("")
        ev_dims = res.get("evaluated_dimensions", [])
        md.append(f"- Evaluated dimensions ({len(ev_dims)}): {', '.join(ev_dims)}")
        md.append(f"- Normalized score: {res.get('normalized_score')}")
        md.append("")
        md.append("**Dimension Scores**:")
        md.append("")
        md.append("| Source | Dimension | Score | Reason (truncated) |")
        md.append("| --- | --- | ---: | --- |")
        for s in res.get("scores", []):
            src = s.get("source", "")
            dn = s.get("dimension_name", "")
            sc = s.get("score", 0)
            rs = _truncate(s.get("reason", ""), 180).replace("\n", " ")
            md.append(f"| {src} | {dn} | {sc} | {rs} |")
        md.append("")
        md.append("<details><summary>Raw JSON</summary>")
        md.append("")
        md.append("\n" + "```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```" + "\n")
        md.append("</details>")
        md.append("")

    _write(md, overwrite=args.overwrite)
    print(f"[OK] Report written to: {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())