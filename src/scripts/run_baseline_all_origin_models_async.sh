#!/bin/bash
# Usage: nohup bash src/scripts/run_baseline_all_origin_models_async.sh 0,1,2,3 > outputs/logs/baseline_async.log 2>&1 &

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
# OUTPUT_DIR="$PROJECT_ROOT/outputs/baselines/all_origin_models_v3_20251210"


mkdir -p "$OUTPUT_DIR"
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export HF_ENDPOINT=https://hf-mirror.com

 ================= 3. 模型清单 =================
#MODELS=(
#     "$BASE_ROOT/models/Qwen/Qwen3-1.7B|Qwen3-1.7B|chat"
#     "$BASE_ROOT/models/Qwen/Qwen3-4B|Qwen3-4B|chat"
#     "$BASE_ROOT/models/Qwen/Qwen3-8B|Qwen3-8B|chat"
#     "$BASE_ROOT/models/Llama/Llama-3.1-8B-Instruct|Llama3.1-8B-Inst|chat"
#     "$BASE_ROOT/models/Llama/Llama-3.2-3B-Instruct|Llama3.2-3B-Inst|chat"
#     "$BASE_ROOT/models/DeepSeek/deepseek-math-7b-rl|DS-Math-RL|chat"
#     "$BASE_ROOT/models/DeepSeek/deepseek-math-7b-instruct|DS-Math-Inst|chat"
#     "$BASE_ROOT/models/DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B|DS-R1-Distill|chat"
#)

MODELS=(
     "$BASE_ROOT/models/Llama/Llama-3.1-8B-Instruct|Llama3.1-8B-Inst|chat"
     "$BASE_ROOT/models/Llama/Llama-3.2-3B-Instruct|Llama3.2-3B-Inst|chat"
     "$BASE_ROOT/models/DeepSeek/deepseek-math-7b-rl|DS-Math-RL|chat"
     "$BASE_ROOT/models/DeepSeek/deepseek-math-7b-instruct|DS-Math-Inst|chat"
     "$BASE_ROOT/models/DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B|DS-R1-Distill|chat"
)

ALL_TASKS="math500,aime24,aime25,olympiad,gpqa,bbh,humaneval,leetcode,lcb"
K_VALS="1,4,8,16"

echo "========================================================"
echo "🚀 Starting EGPO Asynchronous Evaluation"
echo "   Strategy: FIFO Token Bucket (Non-blocking)"
echo "   GPUs Available: ${GPU_ARRAY[*]}"
echo "   Total Models: ${#MODELS[@]}"
echo "========================================================"

# ================= 4. 初始化 GPU 令牌桶 (FIFO) =================
# 创建一个临时命名管道
FIFO_FILE="/tmp/egpo_gpu_fifo_$$"
mkfifo "$FIFO_FILE"

# 将文件描述符 6 绑定到管道（读写模式）
exec 6<>"$FIFO_FILE"
rm "$FIFO_FILE" # 删除文件路径，但文件描述符依然有效

# 向管道中预先填入 GPU ID (这就是令牌)
for gpu in "${GPU_ARRAY[@]}"; do
    echo "$gpu" >&6
done

# ================= 5. 异步任务循环 =================
for i in "${!MODELS[@]}"; do
    ITEM="${MODELS[$i]}"
    IFS='|' read -r M_PATH M_ALIAS M_TYPE <<< "$ITEM"

    # --- 关键步骤：申请 GPU 令牌 ---
    # read -u 6 会尝试从管道读取一行。
    # 如果管道为空（所有 GPU 都在忙），这里会阻塞（等待），直到有 GPU 被归还。
    read -u 6 AVAILABLE_GPU

    echo ">>> [Job Start] Assigning GPU $AVAILABLE_GPU to $M_ALIAS"

    # --- 启动后台任务 ---
    (
        LOG_FILE="$OUTPUT_DIR/${M_ALIAS}.log"
        export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU

        echo -e "\n\n=== Run Started at $(date) on GPU $AVAILABLE_GPU ===" >> "$LOG_FILE"

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
            echo "✅ [Finished] $M_ALIAS on GPU $AVAILABLE_GPU"
        else
            echo "❌ [Failed] $M_ALIAS on GPU $AVAILABLE_GPU (Exit: $EXIT_CODE)"
        fi

        # --- 关键步骤：归还 GPU 令牌 ---
        # 任务结束后，把自己的 GPU ID 写回管道
        # 这样主循环里的 read -u 6 就能读到它，并启动下一个任务
        echo "$AVAILABLE_GPU" >&6
    ) &
done

# ================= 6. 等待收尾 =================
# 等待所有后台子进程结束
wait
echo "🎉 All Async Jobs Completed."

# 关闭文件描述符
exec 6>&-