#!/bin/bash
# 路径：/data-store/zhaoqiannian/workspace/EGPO/src/scripts/run_entropy_flexible.sh

# ================= 1. 输入参数解析 =================
if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_IDS_COMMA_SEPARATED>"
    echo "Examples:"
    echo "  Single Card:  bash $0 1"
    echo "  Two Cards:    bash $0 0,1"
    echo "  Three Cards:  bash $0 0,1,2"
    exit 1
fi

# 解析 GPU 列表
GPU_STRING=$1
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_STRING"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "========================================================"
echo "   🚀 EGPO Entropy Analysis Launcher"
echo "   Detected GPUs: ${GPU_ARRAY[*]} (Total: $NUM_GPUS)"
echo "========================================================"

# ================= 2. 环境与路径配置 =================
# 黄金环境配置
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF

# 路径定义
BASE_DIR="/data-store/zhaoqiannian"
PROJECT_ROOT="$BASE_DIR/workspace/EGPO"
DATA_PATH="$PROJECT_ROOT/datasets/processed/math_single.parquet"
# 根据 GPU 组合创建独立的输出目录，防止混淆
OUTPUT_DIR="$PROJECT_ROOT/outputs/analysis/full_experiment_b1_flexible"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/analyze_entropy.py"

mkdir -p "$OUTPUT_DIR"

# 实验参数
SAMPLE_SIZE=800
N_RETURN=8

# ================= 3. 定义任务函数 =================
# 参数: 1.模型路径 2.模型显示名 3.日志文件名 4.分配的GPU_ID
run_task() {
    local m_path=$1
    local m_name=$2
    local log_file=$3
    local gpu_id=$4

    echo ">>> [GPU $gpu_id] Starting $m_name ..."
    
    # 显式指定当前子进程可见的 GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    python3 -u $SCRIPT_PATH \
        --model_path "$m_path" \
        --model_name "$m_name" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --sample_size $SAMPLE_SIZE \
        --n_return $N_RETURN \
        --tp_size 1 \
        > "$OUTPUT_DIR/$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ [GPU $gpu_id] $m_name Finished. Log: $OUTPUT_DIR/$log_file"
    else
        echo "❌ [GPU $gpu_id] $m_name Failed! Check Log: $OUTPUT_DIR/$log_file"
    fi
}

# ================= 4. 动态调度逻辑 =================

# 定义模型信息
P_1B="$BASE_DIR/models/Qwen/Qwen3-1.7B"
P_4B="$BASE_DIR/models/Qwen/Qwen3-4B"
P_8B="$BASE_DIR/models/Qwen/Qwen3-8B"

if [ "$NUM_GPUS" -eq 1 ]; then
    # --- 单卡模式 (串行) ---
    GPU=${GPU_ARRAY[0]}
    echo "Mode: Serial Execution on GPU $GPU"
    
    run_task "$P_1B" "Qwen3-1.7B" "qwen1.7b.log" $GPU
    run_task "$P_4B" "Qwen3-4B"   "qwen4b.log"   $GPU
    run_task "$P_8B" "Qwen3-8B"   "qwen8b.log"   $GPU

elif [ "$NUM_GPUS" -eq 2 ]; then
    # --- 双卡模式 (并行) ---
    # 策略：小模型(1.7B+4B)共用一张卡串行，大模型(8B)独占一张卡
    GPU_A=${GPU_ARRAY[0]}
    GPU_B=${GPU_ARRAY[1]}
    echo "Mode: Balanced Parallel Execution (Small models on $GPU_A, Large on $GPU_B)"

    # 任务组 A (后台运行)
    (
        run_task "$P_1B" "Qwen3-1.7B" "qwen1.7b.log" $GPU_A
        run_task "$P_4B" "Qwen3-4B"   "qwen4b.log"   $GPU_A
    ) &

    # 任务组 B (后台运行)
    (
        run_task "$P_8B" "Qwen3-8B"   "qwen8b.log"   $GPU_B
    ) &

    wait # 等待两组都完成

else
    # --- 三卡及以上模式 (全并行) ---
    echo "Mode: Full Parallel Execution"
    
    GPU_A=${GPU_ARRAY[0]}
    GPU_B=${GPU_ARRAY[1]}
    GPU_C=${GPU_ARRAY[2]}

    ( run_task "$P_1B" "Qwen3-1.7B" "qwen1.7b.log" $GPU_A ) &
    ( run_task "$P_4B" "Qwen3-4B"   "qwen4b.log"   $GPU_B ) &
    ( run_task "$P_8B" "Qwen3-8B"   "qwen8b.log"   $GPU_C ) &

    wait
fi

echo "========================================================"
echo "🎉 All Analysis Tasks Completed."
echo "========================================================"

# GPU 1 nohup bash src/scripts/run_entropy_flexible.sh 1 > outputs/logs/run_entropy.log 2>&1 &
# GPU 1,2 nohup bash src/scripts/run_entropy_flexible.sh 1,2 > outputs/logs/run_entropy.log 2>&1 &
# GPU 0,1,3 nohup bash src/scripts/run_entropy_flexible.sh 0,1,3 > outputs/logs/run_entropy.log 2>&1 &