#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate NER F1 over datasets under the specified model and data root (TestData).
- Reuse the training/adherence prompt construction: concatenate non-assistant turns and append <|assistant|>
- Reuse entity extraction and F1 functions from custom_reward
- Outputs:
  1) Print Macro/Micro F1 for each data file to console and log file
  2) Write details_*.jsonl for each data file (per-sample predictions and F1)
  3) Write summary results_*.json

Example (Windows PowerShell):
python .\test_eval_ner_f1.py ^
  --model_name_or_path "C:\\your\\model\\path" ^
  --data_root "C:\\Users\\admin\\Desktop\\临时文件\\M\\TestData" ^
  --max_samples_per_file 200 ^
  --output_path .\ner_eval_logs\eval.log
"""
import re
import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# ====== 新增：格式合规检查（与 test_format_adherence.py 保持一致的核心逻辑） ======

def _extract_blocks(text: str):
    if not isinstance(text, str):
        return None, None
    analysis = re.search(r"<\s*analysis\s*>\s*(.*?)\s*<\s*/\s*analysis\s*>", text, re.DOTALL | re.IGNORECASE)
    ner = re.search(r"<\s*ner_result\s*>\s*(.*?)\s*<\s*/\s*ner_result\s*>", text, re.DOTALL | re.IGNORECASE)
    return analysis, ner

def check_adherence(text: str) -> bool:
    """Return True if both <analysis> and <ner_result> exist with non-empty content and analysis appears before ner_result."""
    a, n = _extract_blocks(text)
    if a is None or n is None:
        return False
    if len(a.group(1).strip()) == 0 or len(n.group(1).strip()) == 0:
        return False
    return a.start() < n.start()


def setup_logging(output_path: str) -> str:
    """Support directory or file path:
    - If directory or path without extension: create a timestamped log file inside the directory
    - If a file path: ensure parent directory exists and write to that file
    Return: actual log file path
    """
    looks_like_file = os.path.splitext(output_path)[1] != ""
    is_existing_dir = os.path.isdir(output_path)
    if is_existing_dir or not looks_like_file:
        out_dir = output_path
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(out_dir, f"ner_eval_{ts}.log")
    else:
        parent = os.path.dirname(output_path) or "."
        os.makedirs(parent, exist_ok=True)
        log_file = output_path

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

    logging.info(f"Logs will be written to: {log_file}")
    return log_file


def build_prompt_from_conversation(conversation: List[Dict[str, str]]) -> str:
    """Consistent with training: concatenate all non-assistant turns and append <|assistant|> at the end."""
    parts: List[str] = []
    for turn in conversation:
        role = str(turn.get("role", "user"))
        content = str(turn.get("content", ""))
        if role.lower() == "assistant":
            continue
        parts.append(f"<|{role}|>\n{content}\n")
    prompt = "".join(parts).strip() + "\n<|assistant|>\n"
    return prompt


def iter_json_items(file_path: str) -> Iterable[Dict[str, Any]]:
    """Iterate samples by file extension. Supports .json (array) and .jsonl (one JSON per line).
    Expected each sample to contain:
      { "input": [ {"role":..., "content":...}, ... ], "label": [ ... ] }
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                except Exception:
                    continue
    else:  # default .json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                yield obj
        else:
            # If not a list, try to wrap
            yield data


def collect_json_files(root: str) -> List[str]:
    ret: List[str] = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".json", ".jsonl")):
                ret.append(os.path.join(r, fn))
    ret.sort()
    return ret


def decode_new_text(tokenizer: AutoTokenizer, full_text: str) -> str:
    # Keep only the newly generated text after the assistant segment
    cut = full_text.rfind("<|assistant|>")
    return full_text[cut + len("<|assistant|>"):] if cut != -1 else full_text


def eval_file(model, tokenizer, file_path: str, max_samples: int, out_dir: str) -> Dict[str, Any]:
    items = list(iter_json_items(file_path))
    total = len(items) if max_samples <= 0 else min(len(items), max_samples)
    logging.info(f"Evaluating file: {file_path} | total_samples={len(items)} | evaluating={total}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(file_path)
    base_noext = os.path.splitext(base)[0]
    details_path = os.path.join(out_dir, f"details_{base_noext}_{ts}.jsonl")

    macro_f1_sum = 0.0
    # micro stats
    tp_sum = 0
    pred_sum = 0
    gt_sum = 0
    # Format adherence aggregate
    format_ok_cnt = 0

    t0 = time.time()
    with open(details_path, "w", encoding="utf-8") as wf:
        for i in range(total):
            obj = items[i]
            conversation = obj.get("input") or obj.get("conversation") or []
            labels = obj.get("label") or obj.get("labels") or []
            if not isinstance(conversation, list):
                # Try to be compatible: if dict, wrap as a single turn
                conversation = [conversation]
            if not isinstance(labels, list):
                labels = [labels]

            prompt = build_prompt_from_conversation(conversation)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                do_sample = True
                gen_kwargs = dict(
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=do_sample,
                    temperature=0.2,
                    top_p=0.9,
                )
                gen_out = model.generate(**inputs, **gen_kwargs)
            out_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            pred_text = decode_new_text(tokenizer, out_text)

            # Adherence check
            format_ok = check_adherence(pred_text)
            if format_ok:
                format_ok_cnt += 1

            pred_entities = extract_ner_entities(pred_text)
            gt_entities = [str(x).strip() for x in labels if str(x).strip()]

            # per-sample F1 (set-based)
            f1 = evaluate_ner_accuracy(pred_entities, gt_entities)
            macro_f1_sum += f1

            # micro aggregation
            p_set = set(pred_entities)
            g_set = set(gt_entities)
            tp = len(p_set & g_set)
            tp_sum += tp
            pred_sum += len(p_set)
            gt_sum += len(g_set)

            record = {
                "index": i,
                "pred_entities": pred_entities,
                "gt_entities": gt_entities,
                "f1": f1,
                "format_ok": format_ok,
                "prompt_preview": prompt[:400],
                "output_preview": pred_text[:400],
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0 or (i + 1) == total:
                curr_macro = macro_f1_sum / max(1, i + 1)
                curr_prec = (tp_sum / pred_sum) if pred_sum > 0 else 0.0
                curr_rec = (tp_sum / gt_sum) if gt_sum > 0 else 0.0
                curr_micro = (2 * curr_prec * curr_rec / (curr_prec + curr_rec)) if (curr_prec + curr_rec) > 0 else 0.0
                curr_format = format_ok_cnt / max(1, i + 1)
                logging.info(
                    f"  Progress {i + 1}/{total} | Running Macro F1={curr_macro:.4f}  Running Micro F1={curr_micro:.4f}  Format adherence={curr_format:.2%}"
                )

    macro_f1 = macro_f1_sum / max(1, total)
    precision_micro = (tp_sum / pred_sum) if pred_sum > 0 else 0.0
    recall_micro = (tp_sum / gt_sum) if gt_sum > 0 else 0.0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)) if (precision_micro + recall_micro) > 0 else 0.0
    format_rate = format_ok_cnt / max(1, total)

    dt = time.time() - t0
    logging.info(f"Done: {file_path} | time={dt:.1f}s | samples={total} | Macro F1={macro_f1:.4f} | Micro F1={f1_micro:.4f} | Format adherence={format_rate:.2%}")

    return {
        "file": file_path,
        "samples": total,
        "macro_f1": macro_f1,
        "micro_f1": f1_micro,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "format_ok_rate": format_rate,
        "details_path": details_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=False,
                        default="/private/DAPO/Model/MyModel/LLAMA_7B_SFT_DeepSpeed_Half_Merged",
                        help="HF model path or name")
    parser.add_argument("--data_root", type=str, required=False,
                        default="./TestData", help="Test data root; may contain multiple JSON/JSONL files")
    parser.add_argument("--max_samples_per_file", type=int, default=0, help="Max samples per file; 0 or negative means no limit")
    parser.add_argument("--output_path", type=str, default="./ner_eval_logs", help="Log file path or directory")
    args = parser.parse_args()

    log_file = setup_logging(args.output_path)

    # Device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model: {args.model_name_or_path}")
    logging.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # Output directory
    out_dir = os.path.dirname(log_file) or "."

    # Collect files and evaluate
    files = collect_json_files(args.data_root)
    if not files:
        logging.warning(f"No .json/.jsonl files found under {args.data_root}")

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()
    for fp in files:
        try:
            res = eval_file(model, tokenizer, fp, args.max_samples_per_file, out_dir)
            all_results.append(res)
        except Exception as e:
            logging.exception(f"Error evaluating file: {fp} | {e}")

    # Save summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"results_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as wf:
        json.dump(all_results, wf, ensure_ascii=False, indent=2)

    # Print overall metrics per dataset (file): Micro F1, Macro F1, format adherence rate
    logging.info("=" * 60)
    logging.info(f"Evaluation complete, {len(all_results)} files. Overall metrics by dataset:")

    def _rel(p: str) -> str:
        try:
            return os.path.relpath(p, args.data_root)
        except Exception:
            return os.path.basename(p)

    for r in sorted(all_results, key=lambda x: _rel(x["file"])):
        rel = _rel(r["file"]) 
        logging.info(
            f" - {rel} | samples={r['samples']} | Macro F1={r['macro_f1']:.4f} | Micro F1={r['micro_f1']:.4f} | format_adherence={r.get('format_ok_rate', 0.0):.2%}"
        )
    logging.info(f"Results saved: {summary_path}")


if __name__ == "__main__":
    main()