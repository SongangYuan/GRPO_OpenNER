#!/usr/bin/env python3
"""
TRL-GRPO 训练入口（四卡 + Accelerate + LoRA）
完全基于TRL，不依赖verl框架
数据格式：{"prompt": "...", "answer": "..."} 或 {"prompt": "...", "entities": [...]}
"""

import json
import re
import torch
import logging
import numpy as np
import os
from datetime import datetime
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

# 导入自定义奖励函数
import sys
from pathlib import Path
# 确保项目根目录在Python路径中
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from verl.utils.reward_score.custom_reward import compute_ner_score_v2

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF 模型路径或名称"})
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)

@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "JSONL 数据文件"})

@dataclass
class LoggingArguments:
    metrics_output_dir: Optional[str] = field(default=None, metadata={"help": "保存指标JSON的目录（默认使用 --output_dir）"})

def load_jsonl(path):
    """加载JSON Lines数据文件（支持嵌套数组格式）"""
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    
    # 如果是嵌套数组，展平为一维列表
    if isinstance(content, list) and all(isinstance(item, list) for item in content):
        return [{"conversation": conv} for conv in content]
    else:
        return [{"conversation": content}]  # 单条对话

# ===== 使用 custom_reward.py 中的高级奖励函数 =====
# 所有奖励计算逻辑现在都在 verl/utils/reward_score/custom_reward.py 中

# 新增：从assistant回答中解析<ner_result>为实体列表（用于构造ground truth）
def _extract_entities_from_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    # 抓取标签内容
    m = re.search(r"<\s*ner_result\s*>\s*(.*?)\s*<\s*/\s*ner_result\s*>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        # 兜底：尝试从方括号中提取
        b = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if not b:
            return []
        content = b.group(1)
        # 先尝试解析JSON
        try:
            arr = json.loads("[" + content + "]")
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        # 再退回到引号提取
        items = re.findall(r'"([^"]*)"', content)
        return [s.strip() for s in items if s.strip()]
    ner_content = m.group(1).strip()
    # 优先在标签内部查找方括号数组并解析
    try:
        bracket_json = re.search(r"\[\s*.*?\s*\]", ner_content, re.DOTALL)
        if bracket_json:
            arr = json.loads(bracket_json.group(0))
            if isinstance(arr, list):
                return [str(e).strip() for e in arr if str(e).strip()]
    except Exception:
        pass
    # 先尝试严格JSON解析
    try:
        entities = json.loads(ner_content)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if str(e).strip()]
    except Exception:
        pass
    # 回退：从引号提取
    items = re.findall(r'"([^"]*)"', ner_content)
    if items:
        return [s.strip() for s in items if s.strip()]
    # 最后回退：尝试用逗号分割（去除包裹的方括号与引号）
    raw = ner_content.strip().strip('[]')
    if raw:
        parts = re.split(r"\s*,\s*", raw)
        ents = [p.strip().strip('"').strip("'") for p in parts if p.strip().strip('"').strip("'")]
        if ents:
            return ents
    return []

def unified_reward_function(completions, **kwargs):
     """使用自定义奖励函数计算分数"""
     rewards = []
     
     # 计算与completions对齐所需的索引映射（处理每个prompt可能生成多条completion的情况）
     prompts_list = kwargs.get("prompts")
     if not isinstance(prompts_list, list):
         prompts_list = kwargs.get("prompt")
     num_prompts = len(prompts_list) if isinstance(prompts_list, list) else (1 if isinstance(prompts_list, str) else 0)
 
     for idx, completion in enumerate(completions):
         try:
             # 计算样本索引：当每个prompt生成多条completion时，使用取模对齐
             sample_idx = (idx % num_prompts) if num_prompts else idx
 
             # 优先从 reward_kwargs 中提取 ground_truth（TRL会将自定义字段打包在此处）
             rk = kwargs.get("reward_kwargs")
             rk_item = None
             if isinstance(rk, list) and len(rk) > 0:
                 rk_index = sample_idx % len(rk)
                 rk_item = rk[rk_index]
             elif isinstance(rk, dict):
                 rk_item = rk
 
             def pick_val(container, key):
                 if not isinstance(container, dict):
                     return None
                 val = container.get(key)
                 if isinstance(val, list):
                     if len(val) == 0:
                         return None
                     return val[sample_idx] if sample_idx < len(val) else val[0]
                 return val
 
             entities = pick_val(rk_item, "entities")
             answer = pick_val(rk_item, "answer")
             user_prompt_text = pick_val(rk_item, "user_prompt")
 
             # 如果 reward_kwargs 中没有，再从顶层 kwargs 兜底
             if entities is None:
                 top_entities = kwargs.get("entities")
                 if isinstance(top_entities, list):
                     entities = top_entities[sample_idx] if sample_idx < len(top_entities) else (top_entities[0] if top_entities else None)
                 else:
                     entities = top_entities
             if answer is None:
                 top_answer = kwargs.get("answer")
                 if isinstance(top_answer, list):
                     answer = top_answer[sample_idx] if sample_idx < len(top_answer) else (top_answer[0] if top_answer else None)
                 else:
                     answer = top_answer
             if user_prompt_text is None:
                 top_user_prompt = kwargs.get("user_prompt")
                 if isinstance(top_user_prompt, list):
                     user_prompt_text = top_user_prompt[sample_idx] if sample_idx < len(top_user_prompt) else (top_user_prompt[0] if top_user_prompt else None)
                 else:
                     user_prompt_text = top_user_prompt
 
             if entities is not None:
                 ground_truth = entities
                 data_source = "ner_dapo"
                 gt_source = "reward_kwargs.entities" if rk_item and "entities" in rk_item else "kwargs.entities"
             elif answer is not None:
                 ground_truth = {"answer": answer}
                 data_source = "openai/gsm8k"
                 gt_source = "reward_kwargs.answer" if rk_item and "answer" in rk_item else "kwargs.answer"
             else:
                 ground_truth = {}
                 data_source = "dapo"
                 gt_source = "none"
             
             # 获取与当前样本对齐的 prompt（长度统计用，不再记录内容）
             if isinstance(prompts_list, list):
                 prompt_text = prompts_list[sample_idx] if sample_idx < len(prompts_list) else (prompts_list[0] if prompts_list else None)
             elif isinstance(prompts_list, str):
                 prompt_text = prompts_list
             else:
                 prompt_text = None
 
             # 构造精简且安全的 extra_info
             extra_info = {
                 "batch_idx": idx,              # 当前completion在批次中的索引
                 "sample_idx": sample_idx,      # 对应的样本索引（对齐prompts/reward_kwargs）
                 "completion_length": len(completion),
                 "has_reward_kwargs": isinstance(rk, (list, dict)),
                 "reward_kwargs_type": type(rk).__name__,
                 "reward_kwargs_keys": list(rk_item.keys()) if isinstance(rk_item, dict) else [],
                 "num_prompts": num_prompts,
                 "prompt_length": (len(prompt_text) if isinstance(prompt_text, str) else None),
                 # 仅记录NER抽取目标文本（user侧内容），便于对齐分析
                 "user_prompt_preview": (user_prompt_text[:1000] if isinstance(user_prompt_text, str) else None),
                 "user_prompt_length": (len(user_prompt_text) if isinstance(user_prompt_text, str) else None),
                 "gt_source": gt_source,
             }
             
             # 使用自定义奖励函数计算分数
             reward = compute_ner_score_v2(completion, ground_truth, data_source, extra_info)
             
         except Exception as e:
             logging.error(f"❌ 奖励函数计算错误: {e}")
             reward = 0.0
         
         rewards.append(reward)
     
     # 只记录批次级别的奖励统计（减少日志频率）
     if len(rewards) > 0:
         avg_reward = sum(rewards) / len(rewards)
         min_reward = min(rewards)
         max_reward = max(rewards)
         logging.info(f"🎯 批次奖励统计: 平均={avg_reward:.3f}, 最小={min_reward:.3f}, 最大={max_reward:.3f}")
     
     return rewards

def setup_logging():
    """设置日志配置，包括控制台输出和文件输出"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    
    # 检查并创建logs目录
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"创建日志目录: {logs_dir}")
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"grpo_log_{timestamp}.log")
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    print(f"日志将保存到: {log_file}")
    logging.info(f"开始GRPO训练 - 日志文件: {log_file}")
    
    return log_file

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOConfig, LoggingArguments))
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()

    # 设置日志（包括文件输出）
    log_file = setup_logging()

    # 记录训练参数
    logging.info("=" * 50)
    logging.info("训练参数配置:")
    logging.info(f"模型路径: {model_args.model_name_or_path}")
    logging.info(f"数据集路径: {data_args.dataset_path}")
    logging.info(f"输出目录: {training_args.output_dir}")
    if logging_args.metrics_output_dir:
        logging.info(f"指标JSON目录: {logging_args.metrics_output_dir}")
    logging.info(f"LoRA r: {model_args.lora_r}")
    logging.info(f"LoRA alpha: {model_args.lora_alpha}")
    logging.info(f"学习率: {training_args.learning_rate}")
    logging.info(f"批次大小: {training_args.per_device_train_batch_size}")
    logging.info(f"梯度累积步数: {training_args.gradient_accumulation_steps}")
    logging.info(f"最大训练步数: {training_args.max_steps}")
    logging.info("=" * 50)

    # 加载tokenizer
    logging.info("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("tokenizer加载完成")

    # 加载模型（多GPU分布式训练版本）
    logging.info("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,  # 修复deprecation警告
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # 减少CPU内存使用
    )
    logging.info("模型加载完成")

    # 应用LoRA
    logging.info("正在应用LoRA配置...")
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    logging.info("LoRA配置应用完成")

    # 加载数据
    logging.info("正在加载数据集...")
    raw_data = load_jsonl(data_args.dataset_path)
    dataset = Dataset.from_list(raw_data)
    logging.info(f"数据集加载完成，共 {len(dataset)} 条数据")

    # 数据预处理
    logging.info("正在进行数据预处理...")
    def preprocess_function(examples):
        prompts = []
        gt_entities_per_sample = []
        user_prompts_per_sample = []
        for conversation in examples["conversation"]:
            # 构造提示（跳过assistant以避免泄露答案）
            prompt_parts = []
            # 收集该对话的ground truth（优先使用最后一个assistant回答）
            last_assistant_entities = []
            last_user_content = ""
            for turn in conversation:
                 role = turn.get("role", "user")
                 content = turn.get("content", "")
                 if role.lower() == "assistant":
                     # 解析assistant中的<ner_result>
                     last_assistant_entities = _extract_entities_from_text(content)
                     continue
                 if role.lower() == "user":
                     # 记录用于NER抽取的原始用户文本（取最后一条user）
                     last_user_content = content
                 prompt_parts.append(f"<|{role}|>\n{content}\n")
            # 在末尾追加assistant起始标签，引导模型生成
            prompt = "".join(prompt_parts).strip() + "\n<|assistant|>\n"
            prompts.append(prompt)
            gt_entities_per_sample.append(last_assistant_entities)
            user_prompts_per_sample.append(last_user_content)
        
        # 为每个样本分别构造 reward_kwargs，确保与 prompts 一一对应
        reward_kwargs = []
        n = len(prompts)
        for i in range(n):
            item = {}
            # 将解析得到的ground truth实体放入reward_kwargs
            item["entities"] = gt_entities_per_sample[i]
            # 同时携带user侧原文，便于在extra_info中按需记录
            item["user_prompt"] = user_prompts_per_sample[i]
            reward_kwargs.append(item)
        
        return {"prompt": prompts, "reward_kwargs": reward_kwargs}

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    logging.info("数据预处理完成")

    # 创建Trainer
    logging.info("正在创建GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        reward_funcs=[unified_reward_function],
    )
    logging.info("GRPO Trainer创建完成")

    # ========== 回调：在保存检查点时写出区间均值 JSON ==========
    # 按运行区分：放在 metrics_output_dir/<run_id>/ 下，run_id 取 output_dir 的最后一级目录名
    base_metrics_dir = logging_args.metrics_output_dir or training_args.output_dir
    run_id = os.path.basename(os.path.normpath(training_args.output_dir)) or "run"
    metrics_dir = os.path.join(base_metrics_dir, run_id)
    logging.info(f"指标JSON最终目录: {metrics_dir}")

    class SaveMetricsCallback(TrainerCallback):
        def __init__(self, metrics_output_dir: str):
            self.reset()
            self.metrics_output_dir = metrics_output_dir

        def reset(self):
            self.loss_sum = 0.0
            self.loss_cnt = 0
            self.reward_sum = 0.0
            self.reward_cnt = 0

        def _maybe_get(self, logs: Dict[str, Any], keys):
            for k in keys:
                v = logs.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            return None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            # 兼容不同键名
            loss_val = self._maybe_get(logs, [
                "loss",
                "train_loss",
                "objective/loss",
            ])
            if loss_val is not None and np.isfinite(loss_val):
                self.loss_sum += float(loss_val)
                self.loss_cnt += 1

            reward_val = self._maybe_get(logs, [
                "rewards/mean",
                "reward/mean",
                "train/reward",
                "avg_reward",
            ])
            if reward_val is not None and np.isfinite(reward_val):
                self.reward_sum += float(reward_val)
                self.reward_cnt += 1

        def _flush_json(self, args, state):
            try:
                os.makedirs(self.metrics_output_dir, exist_ok=True)
                step = int(state.global_step)
                payload = {
                    "global_step": step,
                    "avg_batch_loss": (self.loss_sum / self.loss_cnt) if self.loss_cnt > 0 else None,
                    "avg_batch_reward": (self.reward_sum / self.reward_cnt) if self.reward_cnt > 0 else None,
                }
                fname = os.path.join(self.metrics_output_dir, f"metrics_step_{step}.json")
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                logging.info(f"📦 保存区间指标: {fname}")
                self.reset()
            except Exception as e:
                logging.warning(f"写出区间指标JSON失败: {e}")

        def on_save(self, args, state, control, **kwargs):
            self._flush_json(args, state)

        def on_train_end(self, args, state, control, **kwargs):
            # 训练结束兜底再写一次，避免最后一段未命中 on_save
            if self.loss_cnt > 0 or self.reward_cnt > 0:
                self._flush_json(args, state)

    trainer.add_callback(SaveMetricsCallback(metrics_output_dir=metrics_dir))

    # 开始训练
    logging.info("开始GRPO训练...")
    logging.info(f"训练将进行 {training_args.max_steps} 步")
    
    # 检查是否需要从检查点恢复
    resume_from_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        logging.info(f"从检查点恢复训练: {resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logging.info("训练完成！")

    # 保存模型
    logging.info(f"正在保存模型到: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logging.info("模型保存完成")
    logging.info(f"训练日志已保存到: {log_file}")

if __name__ == "__main__":
    main()