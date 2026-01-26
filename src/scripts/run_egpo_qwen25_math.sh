#!/bin/bash
set -e

# ================= 1. 用户配置区 (在此修改) =================

# [任务] mixed | math | code | dryrun | open-r1-math-pmtlth1024
# 你在 egpo_train_config.yaml 新增的 task：
TASK="open-r1-math-pmtlth1024"

# [模式] debug | debug_10g | 4gpu | 8gpu | limit_35g
MODE="debug_10g"

# [显卡] 指定 GPU ID (逗号分隔), 如 "0" 或 "0,1,2,3"
GPU_IDS="0"

# [模型] 相对路径 (相对于 ROOT_CANDIDATES)
# 例如：
#   models/Qwen/Qwen2.5-Math-1.5B-Instruct
#   models/Qwen/Qwen2.5-Math-7B-Instruct
MODEL_REL_PATH="models/Qwen/Qwen2.5-Math-1.5B-Instruct"

# [Qwen3 thinking 一键开关] auto | on | off
# - auto(默认): 仅当模型是 Qwen3 时 -> enable_thinking=True；其它模型不注入（完全不影响）
# - on        : 对 Qwen3 强制 enable_thinking=True（非 Qwen3 也不会注入，避免影响）
# - off       : 对 Qwen3 强制 enable_thinking=False（非 Qwen3 也不会注入，避免影响）
THINKING_MODE="${THINKING_MODE:-auto}"

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
export VLLM_NO_USAGE_STATS=1  # 禁止 vLLM 上报统计，加快启动
export RAY_DEDUP_LOGS=0       # 禁止 Ray 折叠重复日志，便于调试
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# 关键兜底：避免 vLLM 放行超长 max_model_len
unset VLLM_ALLOW_LONG_MAX_MODEL_LEN

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NUM_VISIBLE_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)


# --- 5. 参数生成与检查 ---
TIMESTAMP=$(date +%m%d_%H%M)
EXP_NAME="${TASK}_${MODE}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$EXP_NAME"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "🚀 EGPO Launcher (Qwen2.5-Math)"
echo "========================================================"
echo "   Task        : $TASK"
echo "   Mode        : $MODE"
echo "   GPUs        : $GPU_IDS (Count: $NUM_VISIBLE_GPUS)"
echo "   Model       : $MODEL_PATH"
echo "   Config      : src/config/egpo_train_config.yaml"
echo "   Thinking    : $THINKING_MODE"
echo "========================================================"

CMD_ARGS=$(python3 "$UTILS_SCRIPT" \
    --config "$CONFIG_FILE" \
    --task "$TASK" \
    --mode "$MODE" \
    --project_root "$PROJECT_ROOT" \
    --exp_name "$EXP_NAME" \
    --model_path "$MODEL_PATH")


# --- 5.1 Qwen3 thinking 开关（最小改动：只在 CMD_ARGS 末尾追加 override） ---
IS_QWEN3=0
if [[ "$MODEL_PATH" == *"Qwen3"* || "$MODEL_REL_PATH" == *"Qwen3"* ]]; then
  IS_QWEN3=1
fi

ENABLE_THINKING_OVERRIDE=""

case "$THINKING_MODE" in
  auto)
    if [ "$IS_QWEN3" -eq 1 ]; then
      ENABLE_THINKING_OVERRIDE="++data.apply_chat_template_kwargs.enable_thinking=True"
    fi
    ;;
  on|1|true|True)
    if [ "$IS_QWEN3" -eq 1 ]; then
      ENABLE_THINKING_OVERRIDE="++data.apply_chat_template_kwargs.enable_thinking=True"
    fi
    ;;
  off|0|false|False)
    if [ "$IS_QWEN3" -eq 1 ]; then
      ENABLE_THINKING_OVERRIDE="++data.apply_chat_template_kwargs.enable_thinking=False"
    fi
    ;;
  *)
    echo "❌ ERROR: THINKING_MODE must be auto|on|off (got '$THINKING_MODE')"
    exit 1
    ;;
esac

if [ -n "$ENABLE_THINKING_OVERRIDE" ]; then
  CMD_ARGS="$CMD_ARGS $ENABLE_THINKING_OVERRIDE"
  echo "   🧠 apply_chat_template.enable_thinking -> $ENABLE_THINKING_OVERRIDE"
else
  if [ "$IS_QWEN3" -eq 1 ]; then
    echo "   🧠 apply_chat_template.enable_thinking -> (no override)"
  else
    echo "   🧠 non-Qwen3 model detected; thinking override skipped (won't affect other models)"
  fi
fi


REQUIRED_GPUS=$(echo "$CMD_ARGS" | grep -o "trainer.n_gpus_per_node=[0-9]*" | cut -d= -f2)
if [ -n "$REQUIRED_GPUS" ]; then
  if [ "$NUM_VISIBLE_GPUS" -lt "$REQUIRED_GPUS" ]; then
      echo "❌ ERROR: Mode '$MODE' requires $REQUIRED_GPUS GPUs, but you provided $NUM_VISIBLE_GPUS ($GPU_IDS)."
      exit 1
  fi
fi

# --- 6. 启动训练 ---
export WANDB_PROJECT="EGPO_Unified"
export WANDB_NAME="$EXP_NAME"
export WANDB_DIR="$LOG_DIR"
export WANDB_MODE="online"

# 建议打开完整错误，vLLM/Ray 崩溃时能看到根因
export HYDRA_FULL_ERROR=1

echo "   > Executing Training..."
echo "   > python3 -u -m verl.trainer.main_ppo $CMD_ARGS"
python3 -u -m verl.trainer.main_ppo $CMD_ARGS 2>&1 | tee "$LOG_DIR/train.log"
