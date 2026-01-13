#!/bin/bash
# 路径：/data/zhaoqn/workspace/EGPO/src/scripts/test_entropy_mini.sh

export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
# 测试机只有一张卡，通常是 0
export CUDA_VISIBLE_DEVICES=0 

BASE_DIR="/data/zhaoqn"
PROJECT_ROOT="$BASE_DIR/workspace/EGPO"
# 使用 Qwen3-1.7B 快速验证
MODEL_PATH="$BASE_DIR/models/Qwen/Qwen3-1.7B"
DATA_PATH="$PROJECT_ROOT/datasets/processed/math_single.parquet"
OUTPUT_DIR="$PROJECT_ROOT/outputs/analysis/mini_check"

echo "=== Quick Test on Test Server ==="

python3 $PROJECT_ROOT/src/scripts/analyze_entropy.py \
    --model_path "$MODEL_PATH" \
    --model_name "Qwen3-1.7B-Mini" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sample_size 50 \
    --n_return 4 \
    --tp_size 1

echo "Check plots in $OUTPUT_DIR"