#!/usr/bin/env bash
# 一键评测 NER F1（与 run_grpo_single.sh 相同风格：上方集中配置 + 下方单条启动命令）

# ====== 路径配置 ======
MODEL_NAME_OR_PATH="/private/DAPO/Model/MyModel/LLAMA_7B_SFT_DeepSpeed_Half_Merged" # TODO: 修改为你的本地HF模型目录或名称
DATA_ROOT="/private/DAPO/Data/AnaPileNER/data/TestData/"
# 输出到具体日志文件（也可改为目录路径，脚本会在内部生成带时间戳日志）
LOG_OUTPUT_PATH="$/private/DAPO/Data/AnaPileNER/data/TestData/ner_eval_logs/NER_EVAL_$(date +%Y%m%d_%H%M%S).log"

# ====== 评测参数 ======
MAX_SAMPLES_PER_FILE=1 # 取0表示不限制；冒烟可改为 30

# ====== 关键环境变量 ======
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# ====== 目录准备 ======
# 为 tee 输出准备动态日志名（日期 + 进程ID + 随机数）
TS=$(date +%Y%m%d_%H%M%S)
TEE_LOG_DIR="/private/DAPO/Data/AnaPileNER/data/Train/RLHF"
TEE_LOG="$TEE_LOG_DIR/run_grpo_double_${TS}_$$_${RANDOM}.log"
mkdir -p "$TEE_LOG_DIR"

# ====== 启动命令 ======
python "/private/DAPO/verl/verl-main/recipe/dapo/test_eval_ner_f1.py" \
--model_name_or_path "$MODEL_NAME_OR_PATH" \
--data_root "$DATA_ROOT" \
--output_path "$LOG_OUTPUT_PATH" \
--max_samples_per_file "$MAX_SAMPLES_PER_FILE" \
2>&1 | tee "$TEE_LOG"
