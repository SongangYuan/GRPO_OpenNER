# 一键评测 NER F1（与 run_grpo_single.sh 相同风格：上方集中配置 + 下方单条启动命令）

# ====== 路径配置 ======
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODEL_NAME_OR_PATH = "C:\\path\\to\\your\\model"  # TODO: 修改为你的本地HF模型目录或名称
$DATA_ROOT = Join-Path $SCRIPT_DIR "TestData"
# 输出到具体日志文件（也可改为目录路径，脚本会在内部生成带时间戳日志）
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OUTPUT_PATH = Join-Path $SCRIPT_DIR "ner_eval_logs/NER_EVAL_${timestamp}.log"

# ====== 评测参数 ======
$MAX_SAMPLES_PER_FILE = 0  # 取0表示不限制；冒烟可改为 30

# ====== 关键环境变量 ======
$env:CUDA_VISIBLE_DEVICES = "0"
$env:TOKENIZERS_PARALLELISM = "false"

# ====== 目录准备 ======
$newOutDir = Split-Path -Path $OUTPUT_PATH -Parent
if (-not (Test-Path -LiteralPath $newOutDir)) { New-Item -ItemType Directory -Path $newOutDir -Force | Out-Null }

# ====== 启动命令 ======
python "$SCRIPT_DIR/test_eval_ner_f1.py" `
  --model_name_or_path "$MODEL_NAME_OR_PATH" `
  --data_root "$DATA_ROOT" `
  --output_path "$OUTPUT_PATH" `
  --max_samples_per_file "$MAX_SAMPLES_PER_FILE"