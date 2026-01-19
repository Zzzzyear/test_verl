#!/bin/bash
set -e

# ================= 1. 用户配置区 (在此修改) =================

# [任务] mixed | math | code | dryrun
TASK="open-r1-math"

# [模式] debug | debug_10g | 4gpu | 8gpu | limit_35g
MODE="debug"

# [显卡] 指定 GPU ID (逗号分隔), 如 "0" 或 "0,1,2,3"
GPU_IDS="0"

# [模型] 相对路径 (相对于 ROOT_CANDIDATES)
MODEL_REL_PATH="models/Qwen/Qwen3-1.7B"

# ==========================================================


# --- 2. 智能路径探测---
echo "🔍 Detecting Model Path..."

ROOT_CANDIDATES=(
    "/data-store/zhaoqiannian"  # 训练服务器
    "/data/zhaoqn"              # 测试服务器
)

DETECTED_ROOT=""
for root in "${ROOT_CANDIDATES[@]}"; do
    if [ -d "$root" ]; then
        DETECTED_ROOT="$root"
        break
    fi
done

if [ -z "$DETECTED_ROOT" ]; then
    echo "   ❌ Error: Could not find any known user directories!"
    exit 1
fi

MODEL_PATH="$DETECTED_ROOT/$MODEL_REL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "   ❌ Error: Model not found at expected path: $MODEL_PATH"
    echo "      Please check 'MODEL_REL_PATH' configuration."
    exit 1
fi
echo "   ✅ Target Model: $MODEL_PATH"


# --- 3. 模式选择策略 ---
if [ -n "$MODE" ]; then
    echo "   👉 Using User-Specified Mode: $MODE"
else
    if [[ "$DETECTED_ROOT" == *"/data/zhaoqn"* ]]; then
        MODE="debug_10g"
        echo "   🛡️  Safety Policy: Test Server detected. Auto-setting MODE='debug_10g'."
    else
        MODE="debug"
        echo "   💡 Safety Policy: Training Server detected. Auto-setting MODE='debug'."
    fi
fi


# --- 4. 基础环境准备 ---
PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
CONFIG_FILE="$PROJECT_ROOT/src/config/egpo_train_config.yaml"
UTILS_SCRIPT="$PROJECT_ROOT/src/scripts/utils/generate_cmd.py"

# 强制添加 verl 源码目录到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/verl:$PYTHONPATH"

# vLLM & PyTorch 性能环境变量
export VLLM_USE_V1=1
export VLLM_NO_USAGE_STATS=1
export RAY_DEDUP_LOGS=0
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_VISIBLE_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)


# --- 5. 参数生成与检查 ---
TIMESTAMP=$(date +%m%d_%H%M)
EXP_NAME="${TASK}_${MODE}_grpo_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$EXP_NAME"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "🚀 GRPO Launcher (vanilla baseline)"
echo "========================================================"
echo "   Task         : $TASK"
echo "   Mode         : $MODE"
echo "   GPUs         : $GPU_IDS (Count: $NUM_VISIBLE_GPUS)"
echo "   Adv Estimator: grpo"
echo "   Config       : src/config/egpo_train_config.yaml"
echo "========================================================"

CMD_ARGS=$(python3 "$UTILS_SCRIPT" \
    --config "$CONFIG_FILE" \
    --task "$TASK" \
    --mode "$MODE" \
    --project_root "$PROJECT_ROOT" \
    --exp_name "$EXP_NAME" \
    --model_path "$MODEL_PATH")

REQUIRED_GPUS=$(echo "$CMD_ARGS" | grep -o "trainer.n_gpus_per_node=[0-9]*" | cut -d= -f2)
if [ "$NUM_VISIBLE_GPUS" -lt "$REQUIRED_GPUS" ]; then
    echo "❌ ERROR: Mode '$MODE' requires $REQUIRED_GPUS GPUs, but you provided $NUM_VISIBLE_GPUS ($GPU_IDS)."
    exit 1
fi


# --- 6. 启动训练 ---
export WANDB_PROJECT="EGPO_Unified"
export WANDB_NAME="$EXP_NAME"
export WANDB_DIR="$LOG_DIR"
export WANDB_MODE="online"

# 记录最终执行命令（用于自证确实跑的是 GRPO）
echo "python3 -u -m verl.trainer.main_ppo $CMD_ARGS algorithm.adv_estimator=grpo" \
  | tee "$LOG_DIR/launch_cmd.txt"

echo "   > Executing Training..."
python3 -u -m verl.trainer.main_ppo $CMD_ARGS algorithm.adv_estimator=grpo 2>&1 | tee "$LOG_DIR/train.log"
