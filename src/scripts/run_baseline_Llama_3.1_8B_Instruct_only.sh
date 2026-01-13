#!/bin/bash
# Usage: nohup bash src/scripts/run_qwen3_8b_only.sh <GPU_ID> > outputs/logs/qwen3_8b.log 2>&1 &

if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID>"
    echo "Example: bash $0 0"
    exit 1
fi

# 1. 获取单卡 ID
GPU_ID=$1
NUM_GPUS=1 

# ================= 配置 =================
if [ -d "/data-store/zhaoqiannian" ]; then
    export BASE_ROOT="/data-store/zhaoqiannian"
else
    export BASE_ROOT="/data/zhaoqn"
fi

PROJECT_ROOT="$BASE_ROOT/workspace/EGPO"
DATA_ROOT="$PROJECT_ROOT/datasets/raw"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/evaluate_benchmarks.py"

OUTPUT_DIR="$PROJECT_ROOT/outputs/baselines/all_origin_models_v2_20251210"
# OUTPUT_DIR="$PROJECT_ROOT/outputs/baselines/all_origin_models_v1_20251210"
mkdir -p "$OUTPUT_DIR"

export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export HF_ENDPOINT=https://hf-mirror.com

# ================= 模型清单=================
MODELS=(
    # 格式: 路径|别名|模板
    "$BASE_ROOT/models/Llama/Llama-3.1-8B-Instruct|Llama3.1-8B-Inst|chat"
)

# 全量任务
# ALL_TASKS="math500,aime24,aime25,olympiad,bbh,humaneval,leetcode,gpqa,lcb"
ALL_TASKS="bbh,humaneval,leetcode,gpqa,lcb"
# ALL_TASKS="math500,aime24,aime25,olympiad"
K_VALS="1,4,8,16"

echo "========================================================"
echo "🚀 Single Run: Llama-3.1-8B-Instruct  Baseline"
echo "   GPU: $GPU_ID"
echo "   Output: $OUTPUT_DIR"
echo "========================================================"

# ================= 执行循环 =================
for i in "${!MODELS[@]}"; do
    ITEM="${MODELS[$i]}"
    IFS='|' read -r M_PATH M_ALIAS M_TYPE <<< "$ITEM"
    
    LOG_FILE="$OUTPUT_DIR/${M_ALIAS}.log"
    
    echo ">>> Launching $M_ALIAS on GPU $GPU_ID ..."
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    echo -e "\n\n=== Run Started at $(date) ===" >> "$LOG_FILE"
    
    python3 -u $SCRIPT_PATH \
        --model_path "$M_PATH" \
        --model_alias "$M_ALIAS" \
        --data_root "$DATA_ROOT" \
        --tasks "$ALL_TASKS" \
        --output_dir "$OUTPUT_DIR" \
        --k_values "$K_VALS" \
        --template_type "$M_TYPE" \
        --gpu_memory_utilization 0.7 \
        --tp_size 1 \
        >> "$LOG_FILE" 2>&1
        
    if [ $? -eq 0 ]; then
        echo "✅ [Finished] $M_ALIAS"
    else
        echo "❌ [Failed] $M_ALIAS (See $LOG_FILE)"
    fi
done

echo "🎉 Done."