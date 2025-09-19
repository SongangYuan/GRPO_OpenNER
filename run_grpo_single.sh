#!/usr/bin/env bash
# 单卡高性能GRPO训练（修复参数错误）

# ====== 路径配置 ======
MODEL_NAME_OR_PATH="/private/DAPO/Model/MyModel/LLAMA_7B_SFT_DeepSpeed_Half_Merged"
DATASET_JSONL="/private/DAPO/Data/AnaPileNER/data/Train/RLHF/split_dataset_2.json"
OUTPUT_DIR="/private/DAPO/Model/MyModel/LLAMA_7B_SFT_DeepSpeed_Half_Merged-GRPO-LoRA-FIX-$(date +%Y%m%d_%H%M%S)"

# ====== 高性能参数 ======
PER_DEVICE_BATCH_SIZE=8
GRAD_ACC=2
MAX_STEP=25000
LORA_R=32
LORA_ALPHA=64
LR=1e-4

# ====== 关键环境变量 ======
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ====== 启动命令 ======
python /private/DAPO/verl/verl-main/recipe/dapo/main_grpo.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_path "$DATASET_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --max_steps "$MAX_STEP" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --learning_rate "$LR" \
  --bf16 true \
  --fp16 false \
  --tf32 true \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --save_total_limit 5 \
  --logging_steps 500 \
  --save_steps 5000 \
  --dataloader_num_workers 16 \
  --gradient_checkpointing false \
  --dataloader_pin_memory true \
  --report_to none