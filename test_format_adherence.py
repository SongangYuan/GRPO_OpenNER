#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate whether model outputs adhere to the required format on a given dataset:

- Must contain paired <analysis>...</analysis> and <ner_result>...</ner_result>
- The <analysis> block must appear before <ner_result>
- Additionally, count whether a JSON list can be parsed inside the ner_result block

Usage (Windows PowerShell example):

python .\test_format_adherence.py \
  --model_name_or_path "C:\\path\\to\\model" \
  --dataset_path "C:\\path\\to\\split_dataset_2.json" \
  --output_dir ".\\adherence_results" \
  --max_samples 200

Notes:
- The dataset format should match split_dataset_2.json (typically an array of multi-turn conversations, each turn with role/content)
- Prompt construction is consistent with training: concatenate all non-assistant turns and append <|assistant|> at the end to guide the model output
- Compliance: must contain paired <analysis>...</analysis> and <ner_result>...</ner_result>, with analysis before ner_result; also track whether a JSON list can be parsed inside ner_result
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging(output_path: str) -> str:
    # 同时兼容：
    # 1) 传入目录：在目录下自动创建带时间戳的日志文件
    # 2) 传入文件：将其视为日志文件路径，创建父目录
    looks_like_file = os.path.splitext(output_path)[1] != ""
    is_existing_dir = os.path.isdir(output_path)
    if is_existing_dir or not looks_like_file:
        # 目录模式
        out_dir = output_path
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(out_dir, f"adherence_{ts}.log")
    else:
        # 文件模式
        parent = os.path.dirname(output_path) or "."
        os.makedirs(parent, exist_ok=True)
        log_file = output_path

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
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

    logging.info(f"Log will be written to: {log_file}")
    return log_file


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load data with the same structure as split_dataset_2.json and output unified as [{"conversation": [...]}]."""
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 情况A：顶层即为“多条对话”的数组，每条对话为数组（多轮）
    if isinstance(content, list) and len(content) > 0 and all(isinstance(item, list) for item in content):
        return [{"conversation": conv} for conv in content]

    # 情况B：顶层为“单条对话”的数组（多轮），元素是字典且含role/content
    if isinstance(content, list) and len(content) > 0 and all(isinstance(item, dict) and "role" in item for item in content):
        return [{"conversation": content}]

    # 情况C：已是我们期望的格式
    if isinstance(content, dict) and "conversation" in content:
        return [content]

    # 兜底：包装成单条对话
    return [{"conversation": content if isinstance(content, list) else [content]}]


def build_prompt_from_conversation(conversation: List[Dict[str, str]]) -> str:
    """Build prompt consistent with training: concatenate all turns except assistant, and append <|assistant|> at the end."""
    parts: List[str] = []
    for turn in conversation:
        role = str(turn.get("role", "user"))
        content = str(turn.get("content", ""))
        if role.lower() == "assistant":
            # 跳过assistant以避免泄露答案
            continue
        parts.append(f"<|{role}|>\n{content}\n")
    prompt = "".join(parts).strip() + "\n<|assistant|>\n"
    return prompt


def extract_blocks(text: str) -> Tuple[re.Match | None, re.Match | None]:
    if not isinstance(text, str):
        return None, None
    analysis = re.search(r"<\s*analysis\s*>\s*(.*?)\s*<\s*/\s*analysis\s*>", text, re.DOTALL | re.IGNORECASE)
    ner = re.search(r"<\s*ner_result\s*>\s*(.*?)\s*<\s*/\s*ner_result\s*>", text, re.DOTALL | re.IGNORECASE)
    return analysis, ner


def try_parse_ner_list(ner_inner_text: str) -> Tuple[bool, int]:
    """Try to parse a JSON list inside <ner_result>. Return: (parsed_ok, list_length or -1)."""
    if not isinstance(ner_inner_text, str):
        return False, -1
    # 优先查找方括号数组
    m = re.search(r"\[\s*.*?\s*\]", ner_inner_text, re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            arr = json.loads(snippet)
            if isinstance(arr, list):
                return True, len(arr)
        except Exception:
            pass
    # 直接尝试整体解析
    try:
        arr = json.loads(ner_inner_text)
        if isinstance(arr, list):
            return True, len(arr)
    except Exception:
        pass
    return False, -1


def check_adherence(text: str) -> Dict[str, Any]:
    a, n = extract_blocks(text)
    analysis_ok = a is not None and len(a.group(1).strip()) > 0
    ner_ok = n is not None and len(n.group(1).strip()) > 0
    order_ok = False
    if a is not None and n is not None:
        order_ok = a.start() < n.start()
    ner_json_ok, ner_list_len = (False, -1)
    if n is not None:
        ner_json_ok, ner_list_len = try_parse_ner_list(n.group(1))
    full_ok = analysis_ok and ner_ok and order_ok
    return {
        "analysis_ok": analysis_ok,
        "ner_ok": ner_ok,
        "order_ok": order_ok,
        "ner_json_list_ok": ner_json_ok,
        "ner_list_len": ner_list_len,
        "full_ok": full_ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model path or name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset JSON file path (same structure as split_dataset_2.json)")
    parser.add_argument("--output_dir", type=str, default="./adherence_results", help="Log file path or directory (if directory, a timestamped log file will be created)")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate (sequentially truncated)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)

    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    logging.info(f"Loading model: {args.model_name_or_path}")
    logging.info(f"Using device: {device}")

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

    # 加载数据
    data = load_dataset(args.dataset_path)
    total = min(len(data), args.max_samples)
    logging.info(f"Data items: {len(data)}, evaluated items: {total}")

    # 输出明细文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_dir = os.path.dirname(log_file) or "."
    detail_path = os.path.join(detail_dir, f"details_{ts}.jsonl")

    stats = {
        "total": 0,
        "analysis_ok": 0,
        "ner_ok": 0,
        "order_ok": 0,
        "full_ok": 0,
        "ner_json_list_ok": 0,
    }

    t0 = time.time()
    with open(detail_path, "w", encoding="utf-8") as wf:
        for i in range(total):
            conv = data[i]["conversation"]
            prompt = build_prompt_from_conversation(conv)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                do_sample = True if args.temperature and args.temperature > 0 else False
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if do_sample:
                    gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
                else:
                    gen_kwargs.update(dict(do_sample=False))
                gen_out = model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            output_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            # 仅保留 assistant 段之后的新生成文本作为候选（尽量避免把提示重复算入）
            # 简单做法：查找最后一个 "<|assistant|>" 的位置
            cut = output_text.rfind("<|assistant|>")
            if cut != -1:
                candidate = output_text[cut + len("<|assistant|>"):]
            else:
                candidate = output_text

            res = check_adherence(candidate)

            stats["total"] += 1
            for k in ("analysis_ok", "ner_ok", "order_ok", "full_ok", "ner_json_list_ok"):
                if res.get(k):
                    stats[k] += 1

            rec = {
                "index": i,
                "prompt_preview": prompt[:400],
                "output_preview": candidate[:400],
                **res,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logging.info(f"Progress {i + 1}/{total} | full_ok={stats['full_ok']} analysis={stats['analysis_ok']} ner={stats['ner_ok']} order={stats['order_ok']} json_list={stats['ner_json_list_ok']}")

    dt = time.time() - t0
    logging.info("=" * 60)
    logging.info(f"Evaluation completed in {dt:.1f}s, total {stats['total']} samples")
    logging.info(f"- Samples containing <analysis>...</analysis>: {stats['analysis_ok'] / max(1, stats['total']):.2%}")
    logging.info(f"- Samples containing <ner_result>...</ner_result>: {stats['ner_ok'] / max(1, stats['total']):.2%}")
    logging.info(f"- Correct order (analysis before ner_result): {stats['order_ok'] / max(1, stats['total']):.2%}")
    logging.info(f"- ner_result can be parsed as JSON list: {stats['ner_json_list_ok'] / max(1, stats['total']):.2%}")
    logging.info(f"- Full adherence rate (analysis+ner both exist and correct order): {stats['full_ok'] / max(1, stats['total']):.2%}")
    logging.info(f"Details written to: {detail_path}")


if __name__ == "__main__":
    main()