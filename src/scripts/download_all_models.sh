#!/bin/bash
# 保存为 /data/zhaoqn/download_all_complete.sh

# ================= 配置区 =================
# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 基础模型存放根目录
BASE_DIR="/data/zhaoqn/models"

# 定义模型映射列表： ["本地存储路径"]="HuggingFace仓库ID"
# 格式：["厂商/模型名"]="Repo_ID"
declare -A models=(
    # --- Qwen 系列 
    ["Qwen/Qwen3-1.7B"]="Qwen/Qwen3-1.7B"
    ["Qwen/Qwen3-4B"]="Qwen/Qwen3-4B"
    ["Qwen/Qwen3-8B"]="Qwen/Qwen3-8B"
    ["Qwen/Qwen2.5-Math-7B"]="Qwen/Qwen2.5-Math-7B"
    
    # --- Llama 系列 (必须先 hugginface-cli login 且有权限) ---
    ["Llama/Llama-3.1-8B-Instruct"]="meta-llama/Llama-3.1-8B-Instruct"
    ["Llama/Llama-3.2-3B-Instruct"]="meta-llama/Llama-3.2-3B-Instruct"
    
    # --- DeepSeek 系列 ---
    ["DeepSeek/deepseek-math-7b-rl"]="deepseek-ai/deepseek-math-7b-rl"
    ["DeepSeek/deepseek-math-7b-instruct"]="deepseek-ai/deepseek-math-7b-instruct"
    ["DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# ================= 执行逻辑 =================

# 遍历下载
for local_subpath in "${!models[@]}"; do
    repo_id=${models[$local_subpath]}
    full_path="$BASE_DIR/$local_subpath"
    
    echo "========================================================"
    echo "🚀 正在处理: $local_subpath"
    echo "🔗 仓库 ID: $repo_id"
    echo "📂 本地路径: $full_path"
    echo "========================================================"

    # 1. 安全检查：如果目录存在但里面全是几KB的指针文件，可以直接覆盖。
    # hf-cli 会自动校验，如果本地文件大小和远程不一致（指针文件肯定不一致），它会自动重新下载。
    # 这里我们不手动 rm -rf，依赖 cli 的断点续传和校验能力。

    mkdir -p "$full_path"

    # 2. 执行全量下载
    # 不加 --include/--exclude 默认下载仓库内所有文件（tokenizer, config, weights 等）
    # --local-dir-use-symlinks False: 强制把真实文件下载到目录下，而不是软链接
    huggingface-cli download "$repo_id" \
        --local-dir "$full_path" \
        --local-dir-use-symlinks False \
        --resume-download \
        --token $(cat ~/.cache/huggingface/token 2>/dev/null || echo "") 
    
    # 检查上一步的退出代码
    if [ $? -eq 0 ]; then
        echo "✅ 下载成功: $local_subpath"
        
        # 3. 简单验证：检查是否包含大文件
        # 查找大于 100MB 的文件，如果没有，可能下载有问题（或者是纯代码库）
        large_files=$(find "$full_path" -type f -size +100M | wc -l)
        if [ "$large_files" -eq 0 ]; then
            echo "⚠️ 警告: 在 $full_path 下未检测到大文件，请检查是否下载了真实权重！"
        else
            echo "📦 检测到 $large_files 个大权重文件，验证通过。"
        fi
    else
        echo "❌ 下载失败: $local_subpath (请检查 Repo ID 是否正确，或是否有权限)"
    fi
    echo ""
done

echo "🎉 所有下载任务执行完毕！"