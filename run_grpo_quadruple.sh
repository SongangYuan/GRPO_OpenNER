#!/usr/bin/env bash
# 单卡高性能GRPO训练（修复参数错误）
# 982724
# ====== 路径配置 ======
MODEL_NAME_OR_PATH="/private/DAPO/Model/MyModel/LLAMA_7B_SFT_DeepSpeed_Half_Merged"
DATASET_JSONL="/private/DAPO/Data/AnaPileNER/data/Train/RLHF/split_dataset_2.json"
OUTPUT_DIR="/private/DAPO/Model/MyModel/LLAMA_7B_GRPO-double-1-$(date +%Y%m%d_%H%M%S)"
                   
TS=$(date +%Y%m%d_%H%M%S)
TEE_LOG_DIR="/private/DAPO/Data/AnaPileNER/data/Train/RLHF/var_value"
TEE_LOG="$TEE_LOG_DIR/run_grpo_double_${TS}_$$_${RANDOM}.log"
mkdir -p "$TEE_LOG_DIR"

# 新增：DeepSpeed 配置（脚本同目录下的 ds_config_zero2.json）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_CONFIG="$SCRIPT_DIR/ds_config_zero2.json"

# ====== 高性能参数（降低显存占用） ======
PER_DEVICE_BATCH_SIZE=1         # 减小显存占用
GRAD_ACC=32                     # 保持有效批次不变可先不改
MAX_STEP=25000
LORA_R=16                       # 更小LoRA，训练更快
LORA_ALPHA=32                   # 对应调整
LR=5e-5                         # 更保守学习率
MAX_PROMPT_LEN=384              # 限制输入长度，降低KV缓存和内存
MAX_COMPLETION_LEN=128          # 限制生成长度，降低KV缓存和内存
NUM_GENERATIONS=2               # GRPO最小需要2条生成以计算优势

# ====== 关键环境变量 ======
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# 减少碎片化导致的OOM（来自错误提示建议）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 启动四卡训练 (GPU 0,1,2,3)..."
echo "LoRA: R=$LORA_R, Alpha=$LORA_ALPHA"
echo "学习率: $LR"
echo "单卡批次: ${PER_DEVICE_BATCH_SIZE} x ${GRAD_ACC} = $((PER_DEVICE_BATCH_SIZE*GRAD_ACC))"
echo "总批次大小: $((PER_DEVICE_BATCH_SIZE*GRAD_ACC*4)) (四卡)"
echo "使用 DeepSpeed 配置: $DS_CONFIG"
# 做法A：由 Trainer/Accelerate 通过 --gradient_accumulation_steps 指定，DeepSpeed 配置用 auto 跟随
if [ -f "$DS_CONFIG" ]; then
  if grep -q '"gradient_accumulation_steps"\s*:\s*"auto"' "$DS_CONFIG"; then
    echo "DeepSpeed 梯度累积步数: auto（将跟随 --gradient_accumulation_steps=$GRAD_ACC）"
  else
    echo "⚠️ 警告：$DS_CONFIG 中未设置 \"gradient_accumulation_steps\": \"auto\"，可能覆盖为固定值并触发不一致警告"
  fi
fi

echo "长度设置: prompt=${MAX_PROMPT_LEN}, completion=${MAX_COMPLETION_LEN}, generations=${NUM_GENERATIONS}"

# ====== 启动命令 ======
torchrun --nproc_per_node=4 --master_port 29501 "$SCRIPT_DIR/main_grpo_quadruple.py" \
--deepspeed "$DS_CONFIG" \
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
--logging_steps 50 \
--save_steps 2000 \
--dataloader_num_workers 0 \
--gradient_checkpointing true \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--dataloader_pin_memory true \
--max_prompt_length "$MAX_PROMPT_LEN" \
--max_completion_length "$MAX_COMPLETION_LEN" \
--num_generations "$NUM_GENERATIONS" \
--ds3_gather_for_generation true \
--dataloader_drop_last true \
--report_to none 2>&1 | tee "$TEE_LOG"
