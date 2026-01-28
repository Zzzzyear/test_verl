#!/bin/bash
set -e

# ================= 1. 用户配置区 (在此修改) =================
TASK="open-r1-math-pmtlth1024"
MODE="debug"                 # debug | debug_10g | 4gpu | 8gpu | limit_35g
GPU_IDS="0"
MODEL_REL_PATH="models/Qwen/Qwen2.5-Math-1.5B-Instruct"

# EDGE-GRPO (EDA) 超参（论文默认用 ratio）
EDGE_WEIGHT_MODE="${EDGE_WEIGHT_MODE:-ratio}"     # ratio | zscore
EDGE_LAMBDA_MIN="${EDGE_LAMBDA_MIN:-0.5}"
EDGE_LAMBDA_MAX="${EDGE_LAMBDA_MAX:-2.0}"
EDGE_EPS="${EDGE_EPS:-1e-6}"
EDGE_BETA="${EDGE_BETA:-1.0}"                     # only zscore
EDGE_ZCLIP="${EDGE_ZCLIP:-3.0}"                   # only zscore

# 需要能拿到 token Shannon entropy
CALCULATE_ENTROPY="${CALCULATE_ENTROPY:-True}"    # True | False
# ============================================================

echo "🔍 Detecting Model Path..."

ROOT_CANDIDATES=(
  "/data-store/zhaoqiannian"
  "/data/zhaoqn"
)

DETECTED_ROOT=""
for root in "${ROOT_CANDIDATES[@]}"; do
  if [ -d "$root" ]; then
    DETECTED_ROOT="$root"
    break
  fi
done

if [ -z "$DETECTED_ROOT" ]; then
  echo "❌ Error: Could not find any known user directories!"
  exit 1
fi

MODEL_PATH="$DETECTED_ROOT/$MODEL_REL_PATH"
if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ Error: Model not found at: $MODEL_PATH"
  exit 1
fi
echo "✅ Target Model: $MODEL_PATH"

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
CONFIG_FILE="$PROJECT_ROOT/src/config/egpo_train_config.yaml"
UTILS_SCRIPT="$PROJECT_ROOT/src/scripts/utils/generate_cmd.py"

export PYTHONPATH="${PROJECT_ROOT}/verl:$PYTHONPATH"

export VLLM_USE_V1=1
export VLLM_NO_USAGE_STATS=1
export RAY_DEDUP_LOGS=0
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_VISIBLE_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

TIMESTAMP=$(date +%m%d_%H%M)
EXP_NAME="${TASK}_${MODE}_edge_grpo_${EDGE_WEIGHT_MODE}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$EXP_NAME"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "🚀 EDGE-GRPO Launcher (EDA-only in verl)"
echo "========================================================"
echo "   Task       : $TASK"
echo "   Mode       : $MODE"
echo "   GPUs       : $GPU_IDS (Count: $NUM_VISIBLE_GPUS)"
echo "   Model      : $MODEL_PATH"
echo "   weight_mode: $EDGE_WEIGHT_MODE"
echo "   lambda     : [$EDGE_LAMBDA_MIN, $EDGE_LAMBDA_MAX]"
echo "   calc_entropy: $CALCULATE_ENTROPY"
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

# 计算 gen_batch_size（跟你 dapo 脚本一致：默认 = train_bsz * 4）
TRAIN_BSZ=$(echo "$CMD_ARGS" | grep -o "data.train_batch_size=[0-9]*" | cut -d= -f2)
if [ -z "$TRAIN_BSZ" ]; then
  echo "❌ ERROR: failed to parse data.train_batch_size from CMD_ARGS"
  exit 1
fi
GEN_BSZ=$((TRAIN_BSZ * 4))

EDGE_OVERRIDES="
algorithm.adv_estimator=edge_grpo
++data.gen_batch_size=${GEN_BSZ}

++algorithm.edge_grpo.weight_mode=${EDGE_WEIGHT_MODE}
++algorithm.edge_grpo.lambda_min=${EDGE_LAMBDA_MIN}
++algorithm.edge_grpo.lambda_max=${EDGE_LAMBDA_MAX}
++algorithm.edge_grpo.eps=${EDGE_EPS}
++algorithm.edge_grpo.beta=${EDGE_BETA}
++algorithm.edge_grpo.z_clip=${EDGE_ZCLIP}

++actor_rollout_ref.actor.calculate_entropy=${CALCULATE_ENTROPY}
"
EDGE_OVERRIDES=$(echo "$EDGE_OVERRIDES" | tr '\n' ' ' | xargs)


export WANDB_PROJECT="EGPO_Unified"
export WANDB_NAME="$EXP_NAME"
export WANDB_DIR="$LOG_DIR"
export WANDB_MODE="online"

echo "python3 -u -m verl.trainer.main_ppo $CMD_ARGS $EDGE_OVERRIDES" | tee "$LOG_DIR/launch_cmd.txt"
python3 -u -m verl.trainer.main_ppo $CMD_ARGS $EDGE_OVERRIDES 2>&1 | tee "$LOG_DIR/train.log"
