#!/bin/bash
# Usage: nohup bash src/scripts/run_baseline_gpu.sh 0,1,2,3 > outputs/logs/baseline_async.log 2>&1 &
# bash src/scripts/run_baseline_gpu.sh 0

if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_IDS>"
    echo "Example: bash $0 0,1,2,3"
    exit 1
fi

# 1. 解析 GPU 列表
IFS=',' read -r -a GPU_ARRAY <<< "$1"
NUM_GPUS=${#GPU_ARRAY[@]}

# ================= 2. 环境配置 =================
if [ -d "/data-store/zhaoqiannian" ]; then
    export BASE_ROOT="/data-store/zhaoqiannian"
else
    export BASE_ROOT="/data/zhaoqn"
fi

PROJECT_ROOT="$BASE_ROOT/workspace/EGPO"
DATA_ROOT="$PROJECT_ROOT/datasets/raw"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/evaluate_benchmarks.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/baselines/all_origin_models_async_$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export HF_ENDPOINT=https://hf-mirror.com

# ================= 3. 模型清单 =================
MODELS=(
      "$BASE_ROOT/models/Qwen/Qwen3-1.7B|Qwen3-1.7B|chat"
)

ALL_TASKS="math500,aime24,aime25,olympiad,gpqa,bbh,humaneval,leetcode,lcb"
K_VALS="1,4,8,16"

echo "========================================================"
echo "🚀 Starting EGPO Asynchronous Evaluation (INFINITE LOOP)"
echo "   Strategy: FIFO Token Bucket (Non-blocking + Loop)"
echo "   GPUs Available: ${GPU_ARRAY[*]}"
echo "   Total Models per Loop: ${#MODELS[@]}"
echo "   Loop Start Time: $(date)"
echo "   Stop with: kill -9 $PPID (or kill the script PID)"
echo "========================================================"

# ================= 4. 初始化 GPU 令牌桶 (FIFO) =================
FIFO_FILE="/tmp/egpo_gpu_fifo_$$"
mkfifo "$FIFO_FILE"
exec 6<>"$FIFO_FILE"
rm "$FIFO_FILE"

# 预先填入 GPU 令牌
for gpu in "${GPU_ARRAY[@]}"; do
    echo "$gpu" >&6
done

# ================= 5. 无限循环执行任务（核心修改） =================
LOOP_COUNT=0
while true; do  # 无限循环开关：true 表示一直运行
    LOOP_COUNT=$((LOOP_COUNT + 1))
    echo -e "\n========================================================"
    echo "🔄 Starting Loop $LOOP_COUNT at $(date)"
    echo "========================================================"

    # 每轮循环执行所有模型
    for i in "${!MODELS[@]}"; do
        ITEM="${MODELS[$i]}"
        IFS='|' read -r M_PATH M_ALIAS M_TYPE <<< "$ITEM"

        # 申请 GPU 令牌（无空闲则阻塞）
        read -u 6 AVAILABLE_GPU

        echo ">>> [Loop $LOOP_COUNT] Assigning GPU $AVAILABLE_GPU to $M_ALIAS"

        # 启动后台任务
        (
            # 日志按轮次区分（避免覆盖）
            LOG_FILE="$OUTPUT_DIR/${M_ALIAS}_loop${LOOP_COUNT}.log"
            export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU

            echo -e "\n\n=== Loop $LOOP_COUNT: Run Started at $(date) on GPU $AVAILABLE_GPU ===" >> "$LOG_FILE"

            python3 -u $SCRIPT_PATH \
                --model_path "$M_PATH" \
                --model_alias "$M_ALIAS" \
                --data_root "$DATA_ROOT" \
                --tasks "$ALL_TASKS" \
                --output_dir "$OUTPUT_DIR" \
                --k_values "$K_VALS" \
                --template_type "$M_TYPE" \
                --gpu_memory_utilization 0.9 \
                >> "$LOG_FILE" 2>&1

            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ [Loop $LOOP_COUNT] Finished $M_ALIAS on GPU $AVAILABLE_GPU"
            else
                echo "❌ [Loop $LOOP_COUNT] Failed $M_ALIAS on GPU $AVAILABLE_GPU (Exit: $EXIT_CODE)"
            fi

            # 归还 GPU 令牌
            echo "$AVAILABLE_GPU" >&6
        ) &
    done

    # 等待当前轮次所有模型完成，再进入下一轮
    wait
    echo -e "\n✅ [Loop $LOOP_COUNT] All Models Completed at $(date)"
done

# ================= 6. 收尾（无限循环下不会执行到这里） =================
exec 6>&-
echo "🎉 All Loops Completed (This line will not be reached in infinite mode)"