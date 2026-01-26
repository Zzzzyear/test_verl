#!/bin/bash

# ================= 基础配置 =================
export HF_ENDPOINT=https://hf-mirror.com
WORKSPACE_DIR="/data/zhaoqn/workspace/EGPO"
DATA_RAW_DIR="${WORKSPACE_DIR}/datasets/raw"

# ================= 下载函数 =================
download_dataset() {
    local REPO_ID=$1
    local DIR_NAME=$2
    local TARGET_PATH="${DATA_RAW_DIR}/${DIR_NAME}"

    echo "----------------------------------------------------------------"
    echo "⬇️  [Target] ${DIR_NAME}"
    echo "📦 Source: ${REPO_ID}"
    echo "📂 Path:   ${TARGET_PATH}"
    echo "----------------------------------------------------------------"

    mkdir -p "${TARGET_PATH}"

    huggingface-cli download "${REPO_ID}" \
        --repo-type dataset \
        --local-dir "${TARGET_PATH}" \
        --local-dir-use-symlinks False \
        --resume-download

    if [ $? -eq 0 ]; then echo "✅ Done."; else echo "❌ Failed!"; fi
    echo ""
}

echo "🚀 开始下载精选数据集..."
echo "================================================================"

# ================= 1. Training Sets (原始数据) =================
# download_dataset "open-r1/Mixture-of-Thoughts" "Mixture-of-Thoughts"

# download_dataset "AI-MO/NuminaMath-CoT" "NuminaMath-CoT"

# download_dataset "PRIME-RL/Eurus-2-RL-Data" "Eurus-2-RL-Data"

# ================= 2. Evaluation Sets =================

# --- Math ---
# download_dataset "HuggingFaceH4/MATH-500" "MATH-500"
# download_dataset "AI-MO/aimo-validation-aime" "AIME-2024"
# download_dataset "opencompass/AIME2025" "AIME-2025"
# download_dataset "Hothan/OlympiadBench" "OlympiadBench"
download_dataset "open-r1/OpenR1-Math-220k" "OpenR1-Math"

# --- Code ---
# download_dataset "livecodebench/code_generation_lite" "LiveCodeBench"
# download_dataset "newfacade/LeetCodeDataset" "LeetCodeDataset"
# download_dataset "openai/openai_humaneval" "openai_humaneval"

# --- Science ---
# download_dataset "allenai/ai2_arc" "ARC-Easy"
# download_dataset "allenai/ai2_arc" "ARC-Challenge"


# --- General ---
# download_dataset "fingertap/GPQA-Diamond" "GPQA-Diamond"
# download_dataset "Joschka/big_bench_hard" "big_bench_hard"

echo "🎉 所有原始数据集下载完成！"
# 运行命令：nohup bash scripts/download_all_datasets.sh > outputs/logs/download_all_data_$(date +%Y%m%d_%H%M%S).log 2>&1 &