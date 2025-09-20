# -*- coding: utf-8 -*-
"""
Utilities to call LLM and parse scoring-related outputs.
Implements function: dimension_detection(raw_text)
- Reads prompt template from Prompts/prompt_step_1.md
- Calls Qwen via try_qwen.call_qwen(messages, ...)
- Parses strict-JSON result and returns a Python dict
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from try_qwen import call_qwen
from pathlib import Path

# Valid dimension names as defined in the prompt template
VALID_DIMENSION_NAMES = {
    "Boundary Handling",
    "Error Detection and Correction",
    "Domain Knowledge Application",
    # "Language Quality",  # removed per user request
    "Analysis Depth",
    "Multi-perspective Thinking",
}


def _template_path() -> str:
    """Return absolute path to the prompt template file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "Prompts", "prompt_step_1.md")


essential_notice = (
    "Strictly output JSON only, and it must match the structure under 'Output Format'. Do NOT include any extra text, explanations, or Markdown."
)


def _read_prompt_template() -> str:
    """Load prompt template content from disk."""
    path = _template_path()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_messages(
    raw_text: str,
    analysis_content: str,
    ner_result: str,
    gold_entities: str,
) -> List[Dict[str, str]]:
    """
    Build messages for Qwen call using the prompt template.
    raw_text 与 NER 查询相同，不做拆分；占位符 {ner_query} 直接使用 raw_text。
    其它字段为必填参数。
    """
    tpl = _read_prompt_template()

    # Fill placeholders expected by the template
    filled = (
        tpl
        .replace("{original_text}", raw_text)
        .replace("{ner_query}", raw_text)  # 同源输入，不拆分
        .replace("{analysis_content}", analysis_content)
        .replace("{ner_result}", ner_result)
        .replace("{gold_entities}", gold_entities)
    )

    # Compose a single user message that contains the full instructions
    user_content = (
        f"{filled}\n\n"
        f"Important: Base your judgment only on the information provided. If some fields are missing, judge relevance accordingly and explain your evidence in the reasons.\n"
        f"{essential_notice}\n"
    )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a rigorous evaluation assistant.\n"
                "You must strictly follow the user's 'Output Format' and output JSON only. No extra characters are allowed."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages


def _extract_json(text: str) -> Dict[str, Any]:
    """Try best-effort extraction of a JSON object from model output.

    Adds a metadata flag '__parse_ok' to indicate whether JSON parsing was successful.
    This allows callers (e.g., dimension_detection) to decide on retry strategies.
    When parsing fails, returns '__parse_error' with a brief diagnostic message.
    """
    text = (text or "").strip()
    err_msg = ""
    # First, try direct loading
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj.setdefault("__parse_ok", True)
            return obj
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"

    # Heuristic: take the largest {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                obj.setdefault("__parse_ok", True)
                return obj
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            # Try removing trailing commas (common LLM error)
            candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                obj = json.loads(candidate2)
                if isinstance(obj, dict):
                    obj.setdefault("__parse_ok", True)
                    return obj
            except Exception as e2:
                err_msg = f"{type(e2).__name__}: {e2}"
    else:
        if not err_msg:
            err_msg = "No JSON object found in model output."

    # Fallback to empty structure with explicit failure flag and error message
    return {"relevant_dimensions": [], "__parse_ok": False, "__parse_error": err_msg}


def _normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the parsed JSON according to the template schema."""
    dims = data.get("relevant_dimensions", [])
    if not isinstance(dims, list):
        return {"relevant_dimensions": []}

    normalized: List[Dict[str, str]] = []
    for item in dims:
        if not isinstance(item, dict):
            continue
        name = item.get("dimension_name")
        reason = item.get("relevance_reason")
        if not isinstance(name, str) or not isinstance(reason, str):
            continue
        if name not in VALID_DIMENSION_NAMES:
            continue
        normalized.append({
            "dimension_name": name,
            "relevance_reason": reason.strip(),
        })

    return {"relevant_dimensions": normalized}



def dimension_detection(
    text: str,
    analysis_content: str,
    ner_result: str,
    gold_entities: str,
) -> Dict[str, Any]:
    """
    调用大模型，根据 prompt_step_1.md 的规范，识别与当前文本相关的评估维度并返回解析后的 JSON 结果。

    Args:
        text: 单一输入文本，同时作为 {original_text} 与 {ner_query}
        analysis_content: 供模型参考的分析片段（必填）
        ner_result: 模型或系统的NER输出（必填）
        gold_entities: 标注的标准答案实体（必填）

    Returns:
        一个包含 "relevant_dimensions" 列表的字典，元素为：
        {"dimension_name": str, "relevance_reason": str}
    """
    base_messages = _build_messages(
        text,
        analysis_content=analysis_content,
        ner_result=ner_result,
        gold_entities=gold_entities,
    )

    # 当严格 JSON 解析失败时，利用错误信息在下一轮追加纠偏提示，让模型重新生成
    max_retries = 2
    last_parsed: Dict[str, Any] = {"relevant_dimensions": [], "__parse_ok": False}
    feedback_msg: Dict[str, str] | None = None

    for attempt in range(max_retries + 1):
        messages = base_messages if feedback_msg is None else (base_messages + [feedback_msg])
        answer = call_qwen(messages, temperature=0.2, enable_thinking=False)
        parsed = _extract_json(answer)
        last_parsed = parsed
        if parsed.get("__parse_ok", False):
            break
        # 组装面向模型的纠偏提示，强调严格 JSON 与字段结构
        err = parsed.get("__parse_error", "未知错误")
        feedback_content = (
            "上一次回答无法被解析为合法 JSON。\n"
            f"解析错误: {err}\n"
            "请重新仅输出一个严格的 JSON 对象，不要包含任何多余文本、解释或 Markdown。\n"
            "输出结构必须为: {\"relevant_dimensions\": [ { \"dimension_name\": \"...\", \"relevance_reason\": \"...\" } ] }。\n"
            "注意: dimension_name 与 relevance_reason 均为字符串；relevant_dimensions 是数组。"
        )
        feedback_msg = {"role": "user", "content": feedback_content}

    normalized = _normalize_result(last_parsed)
    return normalized


# Step-2 prompt utilities

def _step2_template_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "Prompts", "prompt_step_2.md")


def _read_step2_template() -> str:
    path = _step2_template_path()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Chinese display name -> folder name mapping under Prompts/Dimensions
_DIMENSION_FOLDER_MAP = {
    # Required three
    "Context Utilization": "ContextUtilization",
    "Logical Coherence": "Logical Coherence",
    "Result Consistency": "ResultConsistency",
    # Detected (step-1)
    "Boundary Handling": "BoundaryHandling",
    "Error Detection and Correction": "Error Detection",
    "Domain Knowledge Application": "Domain Knowledge Application",
    # "Language Quality": "Language Quality",  # removed per user request
    "Analysis Depth": "Analysis Depth",
    "Multi-perspective Thinking": "Multi-perspective Thinking",
}


def _dimensions_base_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "Prompts", "Dimensions")


def _read_dimension_def(dimension_name: str) -> Dict[str, str]:
    """Read target_dimension.md and dimension_criteria.md for a given display name.

    Returns dict: {"target_dimension": str, "criteria": str}
    Raises FileNotFoundError if not found.
    """
    folder = _DIMENSION_FOLDER_MAP.get(dimension_name)
    if not folder:
        raise FileNotFoundError(f"Unknown dimension mapping for: {dimension_name}")
    base = _dimensions_base_dir()
    dim_dir = os.path.join(base, folder)
    target_fp = os.path.join(dim_dir, "target_dimension.md")
    criteria_fp = os.path.join(dim_dir, "dimension_criteria.md")
    with open(target_fp, "r", encoding="utf-8") as f:
        target_txt = f.read().strip()
    with open(criteria_fp, "r", encoding="utf-8") as f:
        criteria_txt = f.read().strip()
    return {"target_dimension": target_txt, "criteria": criteria_txt}


def _build_step2_messages(
    dimension_name: str,
    original_text: str,
    ner_query: str,
    analysis_content: str,
    ner_result: str,
    gold_entities: str,
) -> List[Dict[str, str]]:
    tpl = _read_step2_template()
    defs = _read_dimension_def(dimension_name)
    filled = (
        tpl
        .replace("{original_text}", original_text)
        .replace("{ner_query}", ner_query)
        .replace("{analysis_content}", analysis_content)
        .replace("{ner_result}", ner_result)
        .replace("{gold_entities}", gold_entities)
        .replace("{target_dimension}", defs["target_dimension"])
        .replace("{__dimension_criteria__}", defs["criteria"])
    )
    user_content = (
        f"{filled}\n\n"
        f"Important: Strictly follow the output format and output JSON only; do not include any extra text or Markdown. {essential_notice}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous evaluation assistant. You must strictly follow the output format, output JSON only, and ensure field names and types match."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _build_step2_messages_batch(
    dimension_names: List[str],
    original_text: str,
    ner_query: str,
    analysis_content: str,
    ner_result: str,
    gold_entities: str,
) -> List[Dict[str, str]]:
    """Build a single prompt to score multiple dimensions together for holistic judgment.
    It reuses the step-2 template's background fields and inlines each dimension's definition and criteria.
    The required output is a strict JSON object:
    {
      "scores": [ {"dimension_name": "...", "reason": "...", "score": 0-10}, ... ]
    }
    """
    # Read and fill the background fields once from template
    tpl = _read_step2_template()
    # Only keep the background part by replacing placeholders; we will add multi-dim task below
    filled_bg = (
        tpl
        .replace("{original_text}", original_text)
        .replace("{ner_query}", ner_query)
        .replace("{analysis_content}", analysis_content)
        .replace("{ner_result}", ner_result)
        .replace("{gold_entities}", gold_entities)
    )

    # Compose dimension sections
    sections: List[str] = []
    for idx, dim_name in enumerate(dimension_names, 1):
        defs = _read_dimension_def(dim_name)
        section = (
            f"### Dimension {idx}: {dim_name}\n"
            f"[Definition]\n{defs['target_dimension']}\n\n"
            f"[Scoring Criteria]\n{defs['criteria']}\n"
        )
        sections.append(section)

    multi_task_instr = (
        "## Your Task (Holistic multi-dimension scoring)\n"
        "- Score EACH dimension above independently with an integer score from 0 to 10, and provide evidence-based reasons grounded in the <analysis> content.\n"
        "- Strictly follow the corresponding scoring criteria of each dimension. Do not let dimensions influence each other, but you may ensure global consistency based on the shared context.\n"
        "- Output a single JSON object with fields:\n"
        "  {\n    \"scores\": [\n      { \"dimension_name\": \"...\", \"reason\": \"...\", \"score\": 0-10 }\n    ]\n  }\n"
        "- Output JSON only. Do NOT include any other text or Markdown."
    )

    # Avoid f-string expressions containing backslashes by precomputing the joined text
    sections_text = "\n\n".join(sections)


    user_content = (
        f"{filled_bg}\n\n"
        f"## Dimensions to Evaluate\n"
        f"{sections_text}\n\n"
        f"{multi_task_instr}\n\n"
        f"{essential_notice}"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous evaluation assistant. You must strictly follow the output format, output JSON only, and ensure field names and types match."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _extract_step2_json(text: str) -> Dict[str, Any]:
    data = _extract_json(text)
    # normalize expected fields
    name = data.get("dimension_name")
    reason = data.get("reason")
    score = data.get("score")
    if not isinstance(name, str):
        name = ""
    if not isinstance(reason, str):
        reason = ""
    try:
        # allow str numbers
        if isinstance(score, str):
            score = float(score)
        score = int(round(float(score)))
    except Exception:
        score = None
    if score is None or score < 0:
        score = 0
    if score > 10:
        score = 10
    return {"dimension_name": name.strip(), "reason": reason.strip(), "score": score}


def _extract_step2_batch_json(text: str, expected_names: List[str]) -> List[Dict[str, Any]]:
    """Parse batch scoring JSON. Accepts either {"scores": [...]} or a top-level list.
    Ensures each item has dimension_name, reason, score (0-10 int). When missing or mismatched,
    it falls back to the expected_names by position.
    """
    raw = _extract_json(text)
    items = None
    if isinstance(raw, dict) and isinstance(raw.get("scores"), list):
        items = raw.get("scores")
    elif isinstance(raw, list):
        items = raw
    else:
        # try to detect a dict-of-dimension mapping form
        if isinstance(raw, dict):
            guessed: List[Dict[str, Any]] = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    guessed.append({"dimension_name": k, **v})
            if guessed:
                items = guessed
    if not isinstance(items, list):
        items = []

    norm: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            it = {}
        name = it.get("dimension_name") if isinstance(it.get("dimension_name"), str) else ""
        reason = it.get("reason") if isinstance(it.get("reason"), str) else ""
        score = it.get("score")
        try:
            if isinstance(score, str):
                score = float(score)
            score = int(round(float(score)))
        except Exception:
            score = 0
        score = max(0, min(10, score))
        # fallback to expected by position
        exp_name = expected_names[i] if i < len(expected_names) else (name or "")
        model_name = name or exp_name
        norm.append({
            "dimension_name": exp_name,
            "model_dimension_name": exp_name,
            "reason": reason.strip(),
            "score": score,
        })
    # If model missed some dimensions, pad with defaults
    if len(norm) < len(expected_names):
        for j in range(len(norm), len(expected_names)):
            exp_name = expected_names[j]
            norm.append({
                "dimension_name": exp_name,
                "model_dimension_name": exp_name,
                "reason": "Evaluation missing; default score 0 used.",
                "score": 0,
            })
    return norm


def evaluate_dimensions(
    text: str,
    analysis_content: str,
    ner_result: str,
    gold_entities: str,
    max_workers: int = 6,
) -> Dict[str, Any]:
    """
    使用单次提示词，对三个必要维度（上下文利用度、逻辑连贯性、结果一致性）以及第一步检测出的其它相关维度进行整体性评估。
    仍使用 Prompts/prompt_step_2.md 的背景与维度定义/标准内容，但一次性汇总到一个提示中，请模型统一输出 JSON 列表。

    输入 text 同时作为 {original_text} 与 {ner_query} 提供给模板。

    Returns:
        {
          "scores": [ {"dimension_name": str, "score": int, "reason": str, "source": "required|detected", "model_dimension_name": str} ],
          "evaluated_dimensions": [str],
          "normalized_score": float  # 所有维度得分之和 / (10 * 维度数)，范围[0,1]
        }
    """
    required_names = {"Context Utilization", "Logical Coherence", "Result Consistency"}

    # Step-1 detect other relevant dimensions（text 同用于 original_text 与 ner_query）
    detected = dimension_detection(
        text,
        analysis_content=analysis_content,
        ner_result=ner_result,
        gold_entities=gold_entities,
    ).get("relevant_dimensions", [])
    detected_names = {item.get("dimension_name") for item in detected if isinstance(item, dict) and isinstance(item.get("dimension_name"), str)}

    # Keep only those we can map to a folder
    mapped_detected = {name for name in detected_names if name in _DIMENSION_FOLDER_MAP and name not in required_names}

    # Stable order: required first (fixed order), then detected (sorted)
    required_order = ["Context Utilization", "Logical Coherence", "Result Consistency"]
    detected_order = sorted(mapped_detected)
    all_targets = required_order + detected_order

    # Parallel per-dimension evaluation using multiple threads
    def _score_one(dn: str) -> Dict[str, Any]:
        msgs = _build_step2_messages(
            dn, text, text, analysis_content, ner_result, gold_entities
        )
        ans_local = call_qwen(msgs, temperature=0.2, enable_thinking=False)
        parsed = _extract_step2_json(ans_local)
        # Normalize and pin the expected dimension name for deterministic downstream handling
        reason = (parsed.get("reason") or "").strip()
        try:
            sc = parsed.get("score", 0)
            if isinstance(sc, str):
                sc = float(sc)
            score_int = int(round(float(sc)))
        except Exception:
            score_int = 0
        score_int = max(0, min(10, score_int))
        model_name = parsed.get("dimension_name") or dn
        return {
            "dimension_name": dn,
            "model_dimension_name": model_name,
            "reason": reason,
            "score": score_int,
        }
    
    future_map = {}
    results_map: Dict[str, Dict[str, Any]] = {}
    if all_targets:
        with ThreadPoolExecutor(max_workers=max_workers or 1) as ex:
            for dn in all_targets:
                future_map[dn] = ex.submit(_score_one, dn)
            for dn in all_targets:
                try:
                    results_map[dn] = future_map[dn].result()
                except Exception as e:
                    results_map[dn] = {
                        "dimension_name": dn,
                        "model_dimension_name": dn,
                        "reason": f"Evaluation failed: {e}",
                        "score": 0,
                    }
    
    # Assemble results in stable target order and annotate source
    results: List[Dict[str, Any]] = []
    for dn in all_targets:
        item = results_map.get(dn, {
            "dimension_name": dn,
            "model_dimension_name": dn,
            "reason": "Missing result; default score 0 used.",
            "score": 0,
        })
        item["source"] = "required" if dn in required_names else "detected"
        results.append(item)

    # Normalized total score in [0,1]
    total_possible = 10 * max(1, len(all_targets))
    total_score = sum(int(x.get("score", 0)) for x in results)
    normalized = total_score / float(total_possible) if total_possible > 0 else 0.0

    return {
        "scores": results,
        "evaluated_dimensions": all_targets,
        "normalized_score": normalized,
    }