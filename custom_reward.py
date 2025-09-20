#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom reward functions for DAPO training.
Includes: math answer correctness, solution step quality, clarity assessment,
and NER entity extraction accuracy.
"""

import re
import numpy as np
from typing import Union, Dict, Any, List
import logging
from llm_scoring import evaluate_dimensions

# 初始化模块级日志记录器（避免依赖外部工程的日志配置）
logger = logging.getLogger("custom_reward")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def extract_final_answer(text: str) -> Union[float, None]:
    """
    Extract the final numeric answer from the response text.
    Returns a float if found, otherwise None.
    """
    # Try patterns like "答案是", "答案：", or English variants such as "Therefore, the answer is"/"The answer is"
    answer_patterns = [
        r"答案是\s*([-+]?\d*\.?\d+)",
        r"答案：\s*([-+]?\d*\.?\d+)",
        r"最终答案是\s*([-+]?\d*\.?\d+)",
        r"所以答案是\s*([-+]?\d*\.?\d+)",
        r"Therefore,?\s*the\s*answer\s*is\s*([-+]?\d*\.?\d+)",
        r"The\s*answer\s*is\s*([-+]?\d*\.?\d+)",
        r"=$\s*([-+]?\d*\.?\d+)",
        r"=\s*([-+]?\d*\.?\d+)$"
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # 如果上述模式都没找到，尝试找到最后出现的数字
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def extract_ner_entities(text: str) -> List[str]:
    """
    从NER回答文本中提取识别的实体列表
    支持格式：
    - <ner_result>["entity1", "entity2"]</ner_result>
    - 任意空白/大小写/换行的灵活匹配
    - 回退：从最内层的方括号中提取带引号的项
    """
    if not isinstance(text, str) or not text:
        return []

    # 首选：提取 <ner_result> ... </ner_result>（不区分大小写，允许空白）
    m = re.search(r"<\s*ner_result\s*>\s*(.*?)\s*<\s*/\s*ner_result\s*>",
                  text, flags=re.IGNORECASE | re.DOTALL)
    candidate = None
    if m:
        candidate = m.group(1).strip()
    else:
        # 回退1：从方括号中抓取（取最内层匹配）
        bracket = re.findall(r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]", text, flags=re.DOTALL)
        if bracket:
            candidate = bracket[-1]

    if candidate is None:
        return []

    # 尝试严格JSON解析
    try:
        import json
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # 回退：从candidate中的引号抓取
    items = re.findall(r'"([^"]*)"', candidate)
    return [s.strip() for s in items if s.strip()]


def evaluate_ner_accuracy(predicted_entities: List[str], ground_truth_entities: List[str]) -> float:
    """
    评估NER任务的准确性
    返回0-1之间的分数
    """
    if not ground_truth_entities:
        # 如果标准答案为空，预测为空得满分，否则得0分
        return 1.0 if not predicted_entities else 0.0
    
    if not predicted_entities:
        # 如果预测为空，标准答案不为空，得0分
        return 0.0
    
    # 计算精确率、召回率和F1分数
    predicted_set = set(predicted_entities)
    ground_truth_set = set(ground_truth_entities)
    
    # 精确率 = 正确预测的实体数 / 预测的实体总数
    precision = len(predicted_set & ground_truth_set) / len(predicted_set) if predicted_set else 0.0
    
    # 召回率 = 正确预测的实体数 / 标准答案的实体总数
    recall = len(predicted_set & ground_truth_set) / len(ground_truth_set) if ground_truth_set else 0.0
    
    # F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1_score


def evaluate_ner_analysis_quality(text: str) -> float:
    """
    评估NER分析过程的质量
    返回0-1之间的分数
    """
    score = 0.0
    
    # 检查是否包含分析部分
    has_analysis = bool(re.search(r'<analysis>(.*?)</analysis>', text, re.DOTALL))
    if has_analysis:
        score += 0.4
        
        # 分析分析部分的质量
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', text, re.DOTALL)
        if analysis_match:
            analysis_text = analysis_match.group(1).strip()
            
            # 检查分析是否包含推理过程
            reasoning_keywords = [
                # Chinese
                "因为", "由于", "根据", "分析", "识别", "找到", "包含", "提到",
                # English
                "because", "due to", "according to", "analyze", "analysis", "identify", "identified", "find", "found", "includes", "include", "mentioned", "mention"
            ]
            has_reasoning = any(keyword in analysis_text for keyword in reasoning_keywords)
            if has_reasoning:
                score += 0.3
            
            # Length heuristic
            if 20 <= len(analysis_text) <= 200:
                score += 0.2
            elif len(analysis_text) > 200:
                score += 0.1
    
    # Check result tag presence
    has_result_tag = bool(re.search(r'<ner_result>', text))
    if has_result_tag:
        score += 0.1
    
    return min(score, 1.0)


def evaluate_solution_steps(text: str) -> float:
    """
    Evaluate the completeness and logical flow of solution steps.
    Returns a score in [0, 1].
    """
    score = 0.0

    # Step markers (Chinese + English)
    step_markers = [
        # Chinese
        "首先", "然后", "接下来", "所以", "因此", "第一步", "第二步", "最后",
        # English
        "first", "then", "next", "so", "therefore", "step one", "step two", "finally", "in the end", "consequently"
    ]
    found_markers = sum(1 for marker in step_markers if marker.lower() in text.lower())
    step_score = min(found_markers / 3, 1.0)  # Need at least 3 markers for full score

    # Presence of math operators
    operators = ["+", "-", "*", "/", "×", "÷", "="]
    has_operators = any(op in text for op in operators)

    # Reasoning indicators (Chinese + English)
    has_reasoning = any(marker in text for marker in ["因为", "由于", "根据", "because", "due to", "according to"])

    # Intermediate calculations
    has_calculations = bool(re.findall(r"\d+\s*[\+\-\*\/]\s*\d+", text))

    # Aggregate score
    score = 0.4 * step_score  # steps: 40%
    score += 0.2 if has_operators else 0  # operators: 20%
    score += 0.2 if has_reasoning else 0  # reasoning: 20%
    score += 0.2 if has_calculations else 0  # calculations: 20%

    return score


def evaluate_clarity(text: str) -> float:
    """
    Evaluate clarity of expression.
    Returns a score in [0, 1].
    """
    score = 0.0

    # Length heuristic
    length = len(text)
    if 50 <= length <= 500:
        score += 0.3
    elif length > 500:
        score += 0.2
    else:
        score += 0.1

    # Paragraph separation
    has_paragraphs = text.count("\n\n") > 0 or text.count("。") > 2
    if has_paragraphs:
        score += 0.2

    # Math symbols
    has_math = bool(re.search(r"[\+\-\*\/\=\(\)\[\]\{\}]", text))
    if has_math:
        score += 0.2

    # Summary markers (Chinese + English)
    has_summary = any(marker in text for marker in ["所以", "因此", "总之", "答案是", "so", "therefore", "in conclusion", "the answer is"])
    if has_summary:
        score += 0.3

    return min(score, 1.0)


def check_answer_correctness(response_answer: float, reference_answer: float, tolerance: float = 1e-6) -> float:
    """
    Check the correctness of the numeric answer with tolerance.
    Returns a score in [0, 1].
    """
    if response_answer is None:
        return 0.0

    # 计算相对误差
    if abs(reference_answer) < tolerance:
        # 如果参考答案接近0，使用绝对误差
        error = abs(response_answer - reference_answer)
        return 1.0 if error < tolerance else 0.0
    else:
        # 否则使用相对误差
        relative_error = abs(
            (response_answer - reference_answer) / reference_answer)
        if relative_error < tolerance:
            return 1.0
        elif relative_error < 0.01:  # 1%以内的误差
            return 0.8
        elif relative_error < 0.05:  # 5%以内的误差
            return 0.5
        elif relative_error < 0.1:   # 10%以内的误差
            return 0.2
        else:
            return 0.0


# 简单的日志文本截断工具，避免日志过长

def _truncate_for_log(obj, max_len: int = 1200) -> str:
    try:
        s = obj if isinstance(obj, str) else repr(obj)
    except Exception:
        s = str(obj)
    if s is None:
        return "None"
    if len(s) > max_len:
        return f"{s[:max_len]}... (truncated, total_len={len(s)})"
    return s


def compute_ner_score_v2(solution_str: str, ground_truth: Union[List[str], str, Dict[str, Any]], data_source: str="", extra_info=None) -> float:
    """
    计算NER任务的奖励分数（v2：加权求和，宽松匹配）
    变更点：
    - 严格校验格式：必须存在 <ner_result>[...]</ner_result>（中括号内为JSON数组），否则得分=0
    - <analysis>...</analysis> 缺失则格式得分为0.5；存在则为1.0（仅在 ner_result 合规时生效）
    - 最终分数 = 0.6 * match_score + 0.4 * format_score；两部分加总最高为1
    """
    import json
    import difflib

    logger.info("== NER Reward Input Logging V2 (soft match) ==")
    logger.info(f"Input parameters:")
    logger.info(f"  solution_str: {_truncate_for_log(solution_str, 800)}")
    logger.info(f"  ground_truth: {_truncate_for_log(ground_truth, 400)}")
    logger.info(f"  data_source: {data_source}")
    logger.info(f"  extra_info: {_truncate_for_log(extra_info, 600)}")
    logger.info(f"  extra_info_type: {type(extra_info).__name__}")

    # 1) 解析 <ner_result> 标签；先检查标签是否存在
    m = re.search(r"<\s*ner_result\s*>\s*(.*?)\s*<\s*/\s*ner_result\s*>", str(solution_str), flags=re.IGNORECASE | re.DOTALL)
    if not m:
        format_coeff = 0.0
        logger.warning("v2 format check: <ner_result> tag NOT found. format_coeff=0.0, return 0.")
        return 0.0

    # 2) 校验标签内为严格 JSON 列表
    candidate = m.group(1).strip()
    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, list):
            format_coeff = 0.0
            logger.warning("v2 format check: content inside <ner_result> is NOT a JSON array. format_coeff=0.0, return 0.")
            return 0.0
        predicted_entities = [str(x).strip() for x in parsed if str(x).strip()]
    except Exception as e:
        format_coeff = 0.0
        logger.warning(f"v2 format check: JSON parse error inside <ner_result>, error={e}. format_coeff=0.0, return 0.")
        return 0.0

    # 3) 解析 ground truth（与 v1 保持健壮性）
    ground_truth_entities: List[str] = []
    if isinstance(ground_truth, dict):
        for k in ("entities", "labels", "items", "gt", "ground_truth"):
            v = ground_truth.get(k)
            if isinstance(v, list):
                ground_truth_entities = [str(x).strip() for x in v if str(x).strip()]
                break
        else:
            merged: List[str] = []
            for v in ground_truth.values():
                if isinstance(v, list):
                    merged.extend([str(x).strip() for x in v if str(x).strip()])
            ground_truth_entities = merged
    elif isinstance(ground_truth, str):
        s = ground_truth.strip()
        try:
            parsed_gt = json.loads(s)
            if isinstance(parsed_gt, list):
                ground_truth_entities = [str(x).strip() for x in parsed_gt if str(x).strip()]
            elif isinstance(parsed_gt, dict):
                tmp: List[str] = []
                for k in ("entities", "labels", "items", "gt", "ground_truth"):
                    v = parsed_gt.get(k)
                    if isinstance(v, list):
                        tmp = [str(x).strip() for x in v if str(x).strip()]
                        break
                if not tmp:
                    for v in parsed_gt.values():
                        if isinstance(v, list):
                            tmp.extend([str(x).strip() for x in v if str(x).strip()])
                ground_truth_entities = tmp
            else:
                ground_truth_entities = [s] if s else []
        except Exception:
            parts = [p.strip().strip('"').strip("'") for p in s.strip('[]').split(",")]
            ground_truth_entities = [p for p in parts if p]
    elif isinstance(ground_truth, list):
        ground_truth_entities = [str(x).strip() for x in ground_truth if str(x).strip()]
    else:
        ground_truth_entities = []

    logger.info("Processed entities (v2):")
    logger.info(f"  predicted_entities: {predicted_entities}")
    logger.info(f"  ground_truth_entities: {ground_truth_entities}")

    # 4) 计算宽松匹配的相似度（match_score）
    def _normalize_entity(s: str) -> str:
        s = (s or "").lower().strip()
        # 统一连字符/下划线/多空白为单空格
        s = re.sub(r"[\-_]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        # 去除首尾标点
        s = re.sub(r"^[^0-9a-z\u4e00-\u9fff]+|[^0-9a-z\u4e00-\u9fff]+$", "", s)
        return s

    def _sim(a: str, b: str) -> float:
        a_n, b_n = _normalize_entity(a), _normalize_entity(b)
        if not a_n and not b_n:
            return 1.0
        if not a_n or not b_n:
            return 0.0
        # 如果其中一个是另一个的子串，提升一点分数（更宽松）
        if a_n in b_n or b_n in a_n:
            # 找到更长/更短的串以便判断是否为“合并/复合”实体
            longer, shorter = (a_n, b_n) if len(a_n) >= len(b_n) else (b_n, a_n)
            ratio_len = len(shorter) / len(longer)
            base = 0.8 + 0.2 * ratio_len  # [0.8, 1.0]
            # 复合实体惩罚：当更长的串包含括号/斜杠/“and”/&等并列或别名指示符时，下调匹配
            if re.search(r"[()\/]|\b(and|&|aka)\b", longer):
                base *= 0.75  # 对合并表达（如 “United States (US)”）进行惩罚，避免过高得分
            return min(1.0, base)
        return difflib.SequenceMatcher(None, a_n, b_n).ratio()

    def _soft_precision_recall(pred: List[str], gt: List[str]) -> (float, float):
        if not pred and not gt:
            return 1.0, 1.0
        # soft precision: 每个预测与其最相近的 GT 相似度的平均
        if pred:
            p_scores = []
            for p in pred:
                best = max((_sim(p, g) for g in gt), default=0.0)
                p_scores.append(best)
            p_soft = float(np.mean(p_scores)) if p_scores else (1.0 if not gt else 0.0)
        else:
            p_soft = 1.0 if not gt else 0.0
        # soft recall: 每个 GT 与其最相近的 预测 相似度的平均
        if gt:
            r_scores = []
            for g in gt:
                best = max((_sim(g, p) for p in pred), default=0.0)
                r_scores.append(best)
            r_soft = float(np.mean(r_scores)) if r_scores else (1.0 if not pred else 0.0)
        else:
            r_soft = 1.0
        return p_soft, r_soft

    p_soft, r_soft = _soft_precision_recall(predicted_entities, ground_truth_entities)
    if (p_soft + r_soft) > 0:
        match_score = 2 * p_soft * r_soft / (p_soft + r_soft)
    else:
        match_score = 0.0

    # 对非完美匹配做轻微全局下调，以加大与完全匹配的区分度（不影响满分=1.0）
    if 0.0 < match_score < 1.0:
        PARTIAL_CALIBRATION = 0.93
        match_score *= PARTIAL_CALIBRATION

    # 5) 计算格式得分：analysis 缺失 => 0.5；存在 => 1.0（仅在 ner_result 合规时）
    has_analysis = bool(re.search(r"<\s*analysis\s*>.*?<\s*/\s*analysis\s*>", str(solution_str), flags=re.IGNORECASE | re.DOTALL))
    format_score = 1.0 if has_analysis else 0.5

    # 新增逻辑：若缺失 analysis，则不计入匹配得分（强制通过分析过程）
    if not has_analysis:
        match_score = 0.0

    # 计算 LLM 维度综合评分（evaluate_dimensions），容错处理；当没有 analysis 时，不调用大模型
    if has_analysis:
        try:
            ana_match = re.search(r"<\s*analysis\s*>(.*?)<\s*/\s*analysis\s*>", str(solution_str), flags=re.IGNORECASE | re.DOTALL)
            analysis_text = ana_match.group(1).strip() if ana_match else ""

            # 原文文本优先从 extra_info 中提取（如果提供），否则回退到 analysis 或 solution_str
            original_text = ""
            if isinstance(extra_info, dict):
                for key in ("original_text", "text", "input", "query", "ner_query", "context", "content"):
                    v = extra_info.get(key)
                    if isinstance(v, str) and v.strip():
                        original_text = v.strip()
                        break
            elif isinstance(extra_info, str):
                original_text = extra_info.strip()
            if not original_text:
                original_text = analysis_text or str(solution_str)

            # 构造传递给 LLM 的 ner_result 和 gold_entities（字符串形式）
            llm_ner = json.dumps(predicted_entities, ensure_ascii=False)
            llm_gt = json.dumps(ground_truth_entities, ensure_ascii=False)

            llm_out = evaluate_dimensions(
                text=original_text,
                analysis_content=analysis_text,
                ner_result=llm_ner,
                gold_entities=llm_gt,
            )
            llm_score = float(llm_out.get("normalized_score", 0.0))
            if not (0.0 <= llm_score <= 1.0):
                llm_score = max(0.0, min(1.0, llm_score))
        except Exception as e:
            logger.warning(f"evaluate_dimensions failed: {e}")
            llm_score = 0.0
    else:
        llm_score = 0.0

    # 6) 打印诊断日志
    logger.info("诊断信息（v2-soft）：")
    logger.info(f"  soft_precision={p_soft:.4f}, soft_recall={r_soft:.4f}, match_score(soft_f1)={match_score:.4f}")
    logger.info(f"  analysis标签存在={has_analysis}, format_score={format_score:.2f}")
    logger.info(f"  LLM维度归一化得分 normalized_score={llm_score:.4f}")

    # 7) 最终得分：按新权重线性组合（LLM 0.35 + 结果匹配 0.35 + 格式 0.30）
    final_score = 0.35 * llm_score + 0.35 * match_score + 0.3 * format_score

    final_score = min(max(final_score, 0.0), 1.0)

    logger.info("最终得分（v2-soft）：")
    logger.info(f"  llm_score={llm_score:.4f}, match_score={match_score:.4f}, 格式得分={format_score:.2f}, 最终得分={final_score:.4f}")

    return final_score
