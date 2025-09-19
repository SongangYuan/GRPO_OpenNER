#!/usr/bin/env bash
# 简版：模型输出格式遵从率评估（参考 run_grpo_single.sh 风格）
# 评估 <analysis>...</analysis><ner_result>...</ner_result> 的遵从率

set -euo pipefail

# ====== 路径配置 ======
MODEL_NAME_OR_PATH="/private/DAPO/Model/MyModel/LLAMA_7B_SFT_DeepSpeed_Half_Merged"
DATASET_JSONL="/private/DAPO/Data/AnaPileNER/data/Train/RLHF/split_dataset_2.json"
OUTPUT_DIR="/private/DAPO/Model/MyModel/LLAMA_7B_SFT_DeepSpeed_Half_Merged-Adherence-$(date +%Y%m%d_%H%M%S).log"

# ====== 生成参数（可按需改） ======
MAX_SAMPLES=100
MAX_NEW_TOKENS=256
TEMPERATURE=0.2
TOP_P=0.9

# ====== 关键环境变量 ======
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ====== 启动命令 ======
python "/private/DAPO/verl/verl-main/recipe/dapo/test_format_adherence.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_path "$DATASET_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P"