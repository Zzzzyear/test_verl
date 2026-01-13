#!/bin/bash
# 路径：/data-store/zhaoqiannian/workspace/EGPO/src/scripts/run_entropy_full.sh

# ================= 1. 环境配置 =================
# 显式指定使用 GPU K
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF

# 路径定义 (训练服务器环境)
BASE_DIR="/data-store/zhaoqiannian"
PROJECT_ROOT="$BASE_DIR/workspace/EGPO"
DATA_PATH="$PROJECT_ROOT/datasets/processed/math_single.parquet"
OUTPUT_DIR="$PROJECT_ROOT/outputs/analysis/full_experiment_b1_serial"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/analyze_entropy.py"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ================= 2. 实验参数 =================
# 采样数量：SAMPLE_SIZE 条 Prompt * N_RETURN 条回复 
# 这个量级足够画出非常平滑的 KDE 分布图
SAMPLE_SIZE=1000
N_RETURN=8

echo "========================================================"
echo "   Starting Entropy Analysis on GPU 1 (Sequential)      "
echo "   Batch Size: $SAMPLE_SIZE Prompts x $N_RETURN Returns "
echo "========================================================"

# ================= 3. 串行任务执行 =================

# --- Model 1: Qwen3-1.7B ---
echo "[1/4] Processing Qwen3-1.7B..."
python3 $SCRIPT_PATH \
    --model_path "$BASE_DIR/models/Qwen/Qwen3-1.7B" \
    --model_name "Qwen3-1.7B" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sample_size $SAMPLE_SIZE \
    --n_return $N_RETURN \
    --tp_size 1 \
    > "$OUTPUT_DIR/qwen1.7b.log" 2>&1
echo "Done. (Log: $OUTPUT_DIR/qwen1.7b.log)"

# --- Model 2: Qwen3-4B ---
echo "[2/4] Processing Qwen3-4B..."
python3 $SCRIPT_PATH \
    --model_path "$BASE_DIR/models/Qwen/Qwen3-4B" \
    --model_name "Qwen3-4B" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sample_size $SAMPLE_SIZE \
    --n_return $N_RETURN \
    --tp_size 1 \
    > "$OUTPUT_DIR/qwen4b.log" 2>&1
echo "Done. (Log: $OUTPUT_DIR/qwen4b.log)"

# --- Model 3: DeepSeek-R1-Distill-Qwen-1.5B (Strong Baseline) ---
# echo "[3/4] Processing DeepSeek-R1-Distill-1.5B..."
# python3 $SCRIPT_PATH \
#     --model_path "$BASE_DIR/models/DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B" \
#     --model_name "DS-R1-Distill-1.5B" \
#     --data_path "$DATA_PATH" \
#     --output_dir "$OUTPUT_DIR" \
#     --sample_size $SAMPLE_SIZE \
#     --n_return $N_RETURN \
#     --tp_size 1 \
#     > "$OUTPUT_DIR/ds_distill.log" 2>&1
# echo "Done. (Log: $OUTPUT_DIR/ds_distill.log)"

# --- Model 4: Qwen3-8B (Core Model) ---
echo "[3/3] Processing Qwen3-8B..."
# A800 80G 单卡跑 8B 推理绰绰有余，不需要 TP=2
python3 $SCRIPT_PATH \
    --model_path "$BASE_DIR/models/Qwen/Qwen3-8B" \
    --model_name "Qwen3-8B" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sample_size $SAMPLE_SIZE \
    --n_return $N_RETURN \
    --tp_size 1 \
    > "$OUTPUT_DIR/qwen8b.log" 2>&1
echo "Done. (Log: $OUTPUT_DIR/qwen8b.log)"

echo "========================================================"
echo "   All tasks completed successfully."
echo "   Results saved to: $OUTPUT_DIR"
echo "========================================================"