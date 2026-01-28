#!/bin/bash
set -e
ulimit -n 1048576

export NCCL_IB_TIMEOUT=22
export NCCL_IB_TC=160
export NCCL_NET_GDR_LEVEL=2
export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# ================= 1. 用户配置区 (在此修改) =================

# [任务] mixed | math | code | dryrun | open-r1-math-pmtlth1024
# 你在 egpo_train_config.yaml 新增的 task：
TASK="open-r1-math-pmtlth1024"

# [模式] debug | debug_10g | 4gpu | 8gpu | limit_35g
MODE="4gpu"

# [显卡] 指定 GPU ID (逗号分隔), 如 "0" 或 "0,1,2,3"
GPU_IDS="0,1,2,3"
nnodes=1

# # [模型] 相对路径 (相对于 ROOT_CANDIDATES)
# MODEL_REL_PATH="models/Qwen/Qwen3-1.7B"

# [Qwen3 thinking 一键开关] auto | on | off
# - auto(默认): 仅当模型是 Qwen3 时 -> enable_thinking=True；其它模型不注入（完全不影响）
# - on        : 对 Qwen3 强制 enable_thinking=True（非 Qwen3 也不会注入，避免影响）
# - off       : 对 Qwen3 强制 enable_thinking=False（非 Qwen3 也不会注入，避免影响）
THINKING_MODE="${THINKING_MODE:-auto}"

# [Reward Manager]
# - 默认用 hybrid：可直接跑通你当前 openr1_math 数据（因为 hybrid 自己实现了 math 判分）
REWARD_MANAGER="${REWARD_MANAGER:-hybrid}"   # hybrid | dapo

# [DAPO 核心超参]
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"

# 动态采样过滤（DAPO 的关键）
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-True}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_reward}"   # 推荐 seq_reward；也可 acc/score（但要求 reward_extra_info 里有）
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"

# Overlong buffer（仅当 REWARD_MANAGER=dapo 时真正生效）
OVERLONG_ENABLE="${OVERLONG_ENABLE:-False}"
OVERLONG_LEN="${OVERLONG_LEN:-128}"
OVERLONG_PENALTY="${OVERLONG_PENALTY:-1.0}"
# ==========================================================

# --- 2. 智能路径探测---
# echo "🔍 Detecting Model Path..."

# ROOT_CANDIDATES=(
#     "/data-store/zhaoqiannian"  # 训练服务器
#     "/data/zhaoqn"              # 测试服务器
# )

# DETECTED_ROOT=""
# for root in "${ROOT_CANDIDATES[@]}"; do
#     if [ -d "$root" ]; then
#         DETECTED_ROOT="$root"
#         break
#     fi
# done

# if [ -z "$DETECTED_ROOT" ]; then
#     echo "   ❌ Error: Could not find any known user directories!"
#     exit 1
# fi

# MODEL_PATH="$DETECTED_ROOT/$MODEL_REL_PATH"

# if [ ! -d "$MODEL_PATH" ]; then
#     echo "   ❌ Error: Model not found at expected path: $MODEL_PATH"
#     echo "      Please check 'MODEL_REL_PATH' configuration."
#     exit 1
# fi
# echo "   ✅ Target Model: $MODEL_PATH"

MODEL_PATH="/opt/nas/p/achen/open_models/Qwen_Qwen2.5-Math-1.5B"
MODEL_NAME="Qwen_Qwen2.5-Math-1.5B"

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
CONFIG_FILE="$PROJECT_ROOT/src/config/egpo_train_config_exp24_20260128.yaml"
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
# TIMESTAMP=$(TZ='Asia/Shanghai' date +%m%d_%H%M)
ADV_ESTIMATOR="dapo"
TIMESTAMP=$(date -d "UTC +8 hours" +%m%d_%H%M)
EXP_NAME="${TASK}_${MODEL_NAME}_${MODE}_${ADV_ESTIMATOR}_${TIMESTAMP}"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$EXP_NAME"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "🚀 DAPO Launcher (RayDAPOTrainer)"
echo "========================================================"
echo "   Task        : $TASK"
echo "   Mode        : $MODE"
echo "   GPUs        : $GPU_IDS (Count: $NUM_VISIBLE_GPUS)"
echo "   RewardMgr    : $REWARD_MANAGER"
echo "   Adv Estimator: grpo (DAPO uses GRPO advantage + sampling/clip tweaks)"
echo "   Clip         : low=$CLIP_RATIO_LOW high=$CLIP_RATIO_HIGH"
echo "   FilterGroups : enable=$FILTER_GROUPS_ENABLE metric=$FILTER_GROUPS_METRIC max_gen_batches=$MAX_NUM_GEN_BATCHES"
echo "   Config      : $CONFIG_FILE"
echo "   PROJECT_ROOT: $PROJECT_ROOT"
echo "   EXP_NAME    : $EXP_NAME"
echo "   Thinking    : $THINKING_MODE"
echo "========================================================"

CMD_ARGS=$(python3 "$UTILS_SCRIPT" \
    --config "$CONFIG_FILE" \
    --task "$TASK" \
    --mode "$MODE" \
    --project_root "$PROJECT_ROOT" \
    --exp_name "$EXP_NAME" \
    --model_path "$MODEL_PATH" \
    --nnodes "$nnodes")


# --- 5.1 Qwen3 thinking 开关（最小改动：只在 CMD_ARGS 末尾追加 override） ---
IS_QWEN3=0
if [[ "$MODEL_PATH" == *"Qwen3"* || "$MODEL_REL_PATH" == *"Qwen3"* ]]; then
  IS_QWEN3=1
fi

ENABLE_THINKING_OVERRIDE=""

case "$THINKING_MODE" in
  auto)
    if [ "$IS_QWEN3" -eq 1 ]; then
      # 用 ++ 更稳：未来如果 yaml 里预先定义了 enable_thinking，也不会报 “key already exists”
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


# --- 5.2 计算一个更合适的 gen_batch_size（默认 = train_batch_size * 4） ---
TRAIN_BSZ=$(echo "$CMD_ARGS" | grep -o "data.train_batch_size=[0-9]*" | cut -d= -f2)
if [ -z "$TRAIN_BSZ" ]; then
  echo "❌ ERROR: failed to parse data.train_batch_size from CMD_ARGS"
  exit 1
fi
GEN_BSZ=$((TRAIN_BSZ * 4))


REQUIRED_GPUS=$(echo "$CMD_ARGS" | grep -o "trainer.n_gpus_per_node=[0-9]*" | cut -d= -f2)
if [ "$NUM_VISIBLE_GPUS" -lt "$REQUIRED_GPUS" ]; then
    echo "❌ ERROR: Mode '$MODE' requires $REQUIRED_GPUS GPUs, but you provided $NUM_VISIBLE_GPUS ($GPU_IDS)."
    exit 1
fi


# --- 6. DAPO 专属 overrides（覆盖 generate_cmd 的默认值） ---
# 关键点：
# - 换入口：recipe.dapo.main_dapo （内部用 RayDAPOTrainer）
# - adv_estimator=grpo（DAPO 不是新 advantage estimator）
# - asymmetric clip
# - filter_groups 动态采样（DAPO 最关键）
# - reward_manager 可选：hybrid(先跑通) 或 dapo(严格复现 + overlong)

DAPO_OVERRIDES="
algorithm.adv_estimator=grpo
reward_model.reward_manager=${REWARD_MANAGER}
data.gen_batch_size=${GEN_BSZ}
actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE}
algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}
algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}
actor_rollout_ref.actor.use_kl_loss=False
algorithm.use_kl_in_reward=False
reward_model.overlong_buffer.enable=${OVERLONG_ENABLE}
reward_model.overlong_buffer.len=${OVERLONG_LEN}
reward_model.overlong_buffer.penalty_factor=${OVERLONG_PENALTY}
"

# 清理成一行
DAPO_OVERRIDES=$(echo "$DAPO_OVERRIDES" | tr '\n' ' ' | xargs)

# --- 7. 启动训练 ---
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=e5eabf51ce79203f59fe61312c26901ca0e24d1a
export WANDB_PROJECT="EGPO_Unified"
export WANDB_ENTITY="egpo-paper"
export WANDB_NAME="$EXP_NAME"
export WANDB_DIR="$LOG_DIR"
export WANDB_MODE="online"

# 记录最终执行命令
echo "python3 -u -m recipe.dapo.main_dapo $CMD_ARGS $DAPO_OVERRIDES" | tee "$LOG_DIR/launch_cmd.txt"

echo "   > Executing Training..."
python3 -u -m recipe.dapo.main_dapo $CMD_ARGS $DAPO_OVERRIDES 2>&1 | tee "$LOG_DIR/train.log"

