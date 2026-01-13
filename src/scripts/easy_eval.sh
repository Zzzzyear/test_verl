#!/bin/bash

# ========================================================================
# 🚀 EGPO 通用一键评测工具 (Easy Eval Wrapper)
# ========================================================================
#
# 【功能描述】
# 这个脚本是 `evaluate_benchmarks.py` 的傻瓜式封装。
# 它能自动处理路径解析、环境变量配置、多卡并行设置（TP）以及任务分组。
# 无论是原始下载的模型 (/models) 还是训练出的 Checkpoint，都可以用它一键跑分。
#
# 【使用语法】
# bash src/scripts/easy_eval.sh <模型路径或名称> <显卡编号> [任务类型] [自定义别名]
#
# 【参数详解】
# 1. <模型路径或名称> (必填) : 
#    - 可以是绝对路径 (用于评测 Checkpoint)。
#    - 也可以是相对于 /models/ 的目录名 (用于评测 Base 模型)。
# 2. <显卡编号> (必填) : 
#    - 指定使用哪些 GPU。脚本会根据 GPU 数量自动计算 Tensor Parallel (TP) 大小。
#    - 单卡示例: "0"  (TP=1)
#    - 多卡示例: "0,1,2,3" (TP=4)
# 3. [任务类型] (可选，默认 "all") :
#    - "all"     : 跑全量 (Math + Code + General 共 9 个测试集)
#    - "math"    : 只跑数学 (MATH-500, AIME 24/25, Olympiad)
#    - "code"    : 只跑代码 (HumanEval, LCB, LeetCode)
#    - "general" : 只跑通用 (GPQA, BBH)
#    - 自定义    : 也可以直接传 "math500,humaneval" 这种逗号分隔的字符串
# 4. [自定义别名] (可选) :
#    - 指定输出文件夹和日志的前缀。如果不填，脚本会根据路径自动生成。
#
# ========================================================================
# 【使用案例 (Examples)】
#
# 🌟 场景 1: 跑全量 Benchmark (最常用)
#    说明: 评测 Qwen3-1.7B，使用 GPU 0，跑所有数学、代码和通用任务。
#    命令: bash src/scripts/easy_eval.sh Qwen/Qwen3-1.7B 0 all
#
# 🌟 场景 2: 评测训练出的 Checkpoint (只测数学)
#    说明: 评测 step-100 的模型，使用 GPU 0 和 1 加速 (TP=2)，只跑数学题。
#    命令: bash src/scripts/easy_eval.sh /data-store/zhaoqiannian/workspace/EGPO/checkpoints/my_exp/step-100 0,1 math
#
# 🌟 场景 3: 评测 70B 大模型 (多卡并行)
#    说明: 模型太大单卡放不下，使用 4 张卡 (TP=4) 跑分。
#    命令: bash src/scripts/easy_eval.sh Qwen/Qwen3-72B-Instruct 0,1,2,3 all
#
# ========================================================================

# --- 0. 参数检查 ---
if [ $# -lt 2 ]; then
    # 如果参数不够，打印脚本头部的帮助信息
    grep "^#" "$0" | head -n 50
    exit 1
fi

INPUT_MODEL=$1
GPU_IDS=$2
TASK_MODE=${3:-"all"}  # 默认跑全量
CUSTOM_ALIAS=$4

# --- 1. 环境自动适配 (无需手动修改) ---
# 自动判断是在训练服务器还是测试服务器
if [ -d "/data-store/zhaoqiannian" ]; then
    export BASE_ROOT="/data-store/zhaoqiannian"
else
    export BASE_ROOT="/data/zhaoqn" # 测试服 fallback
fi

PROJECT_ROOT="$BASE_ROOT/workspace/EGPO"
DATA_ROOT="$PROJECT_ROOT/datasets/raw"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/evaluate_benchmarks.py"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/baselines/manual_evals" # 手动评测结果存放处

# 关键环境变量配置
export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com
unset PYTORCH_CUDA_ALLOC_CONF

# 检查 Python 脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Evaluation script not found at $SCRIPT_PATH"
    exit 1
fi

# --- 2. 智能路径解析 ---
# 逻辑：优先检查是否为绝对路径，如果不是，则去 /models 目录下查找
if [ -d "$INPUT_MODEL" ]; then
    MODEL_PATH="$INPUT_MODEL"
    echo "🔍 Detected Checkpoint/Absolute Path: $MODEL_PATH"
elif [ -d "$BASE_ROOT/models/$INPUT_MODEL" ]; then
    MODEL_PATH="$BASE_ROOT/models/$INPUT_MODEL"
    echo "🔍 Detected Base Model Path: $MODEL_PATH"
else
    echo "❌ Error: Model path not found!"
    echo "   Checked: $INPUT_MODEL"
    echo "   Checked: $BASE_ROOT/models/$INPUT_MODEL"
    exit 1
fi

# --- 3. 别名自动生成 ---
if [ -z "$CUSTOM_ALIAS" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
    # 如果路径包含 'step' 或 'checkpoint'，为了防止重名，加上父目录名
    # 例如: my_experiment/global_step_500 -> my_experiment_global_step_500
    if [[ "$MODEL_NAME" == *"step"* ]] || [[ "$MODEL_NAME" == *"checkpoint"* ]]; then
        PARENT_DIR=$(basename $(dirname "$MODEL_PATH"))
        MODEL_ALIAS="${PARENT_DIR}_${MODEL_NAME}"
    else
        MODEL_ALIAS="$MODEL_NAME"
    fi
else
    MODEL_ALIAS="$CUSTOM_ALIAS"
fi

# --- 4. 任务组装 ---
TASKS_MATH="math500,aime24,aime25,olympiad"
TASKS_CODE="humaneval,lcb,leetcode"
TASKS_GENERAL="gpqa,bbh"

case "$TASK_MODE" in
    "math")    TARGET_TASKS="$TASKS_MATH" ;;
    "code")    TARGET_TASKS="$TASKS_CODE" ;;
    "general") TARGET_TASKS="$TASKS_GENERAL" ;;
    "all")     TARGET_TASKS="$TASKS_MATH,$TASKS_CODE,$TASKS_GENERAL" ;;
    *)         TARGET_TASKS="$TASK_MODE" ;; # 允许用户传入 "math500,humaneval"
esac

# --- 5. 准备输出目录 ---
CURRENT_DATE=$(date +%Y%m%d_%H%M)
FINAL_OUTPUT_DIR="$OUTPUT_ROOT/${CURRENT_DATE}_${MODEL_ALIAS}"
mkdir -p "$FINAL_OUTPUT_DIR"
LOG_FILE="$FINAL_OUTPUT_DIR/eval.log"

# --- 6. 计算 Tensor Parallel (TP) ---
# 根据传入 GPU 的逗号数量计算 TP Size
# 例如 "0" -> TP=1; "0,1" -> TP=2
TP_SIZE=$(echo $GPU_IDS | tr -cd ',' | wc -c)
TP_SIZE=$((TP_SIZE + 1))

# --- 7. 执行评测 ---
echo "========================================================"
echo "🚀 Submitting Evaluation Job"
echo "   Model Path : $MODEL_PATH"
echo "   Model Alias: $MODEL_ALIAS"
echo "   GPUs       : $GPU_IDS (TP=$TP_SIZE)"
echo "   Tasks      : $TARGET_TASKS"
echo "   Log File   : $LOG_FILE"
echo "========================================================"

# 设置显卡
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 启动后台任务 (nohup)
# k_values="1,4,8,16" 是固定的，一次生成16个样本计算所有指标
# gpu_memory_utilization=0.85 预留一些显存防止碎片化 OOM
nohup python3 -u $SCRIPT_PATH \
    --model_path "$MODEL_PATH" \
    --model_alias "$MODEL_ALIAS" \
    --data_root "$DATA_ROOT" \
    --tasks "$TARGET_TASKS" \
    --output_dir "$FINAL_OUTPUT_DIR" \
    --k_values "1,4,8,16" \
    --template_type "chat" \
    --tp_size $TP_SIZE \
    --gpu_memory_utilization 0.85 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Job submitted successfully! PID: $PID"
echo "   View progress: tail -f $LOG_FILE"
echo "   Results will be saved to: $FINAL_OUTPUT_DIR"