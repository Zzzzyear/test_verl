#!/bin/bash
set -x
export BASE_ROOT="/data/zhaoqn"
PROJECT_ROOT="$BASE_ROOT/workspace/EGPO"
DATA_ROOT="$PROJECT_ROOT/datasets/raw"
OUT_DIR="$PROJECT_ROOT/outputs/baselines/test_final_exec"
mkdir -p "$OUT_DIR"

export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export HF_ENDPOINT=https://hf-mirror.com

MODEL="/data/zhaoqn/models/Qwen/Qwen3-1.7B"

# Limit=5, K=4 (小测)
CUDA_VISIBLE_DEVICES=0 python3 $PROJECT_ROOT/src/scripts/evaluate_benchmarks.py \
    --model_path "$MODEL" \
    --model_alias "Qwen3-Test-Exec" \
    --data_root "$DATA_ROOT" \
    --tasks "math500,humaneval" \
    --output_dir "$OUT_DIR" \
    --k_values "1,4" \
    --limit 5 \
    --template_type "chat" \
    --gpu_memory_utilization 0.2

if [ $? -eq 0 ]; then
    echo "✅ Test Passed! Files generated:"
    ls -lh $OUT_DIR/*.jsonl
else
    echo "❌ Test Failed!"
fi