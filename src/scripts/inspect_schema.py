import os
import glob
import pandas as pd
import json

# ================= 配置 =================
BASE_DIR = "/data/zhaoqn/workspace/EGPO"
RAW_DIR = os.path.join(BASE_DIR, "datasets/raw")

# 定义你想查看的所有数据集文件夹名
TARGET_DIRS = [
    "Mixture-of-Thoughts",
    "NuminaMath-CoT",
    "Eurus-2-RL-Data",
    "MATH-500",
    "AIME-2024",
    "AIME-2025",
    "OlympiadBench",
    "GPQA-Diamond",
    "big_bench_hard",
    "LiveCodeBench",
    "LeetCodeDataset",
    "openai_humaneval"
]

def truncate(text, length=1000):
    """截断长文本，方便在终端显示"""
    s = str(text)
    if len(s) > length:
        return s[:length] + "..."
    return s

def inspect_dataset(folder_name):
    path = os.path.join(RAW_DIR, folder_name)
    if not os.path.exists(path):
        print(f"❌ {folder_name}: 文件夹不存在")
        return

    # 1. 寻找文件 (优先找 parquet，其次找 jsonl)
    parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
    jsonl_files = glob.glob(os.path.join(path, "**/*.jsonl"), recursive=True)
    
    file_path = None
    file_type = None
    
    if parquet_files:
        file_path = parquet_files[0]
        file_type = "parquet"
    elif jsonl_files:
        file_path = jsonl_files[0]
        file_type = "jsonl"
    
    if not file_path:
        print(f"⚠️  {folder_name}: 没找到 parquet 或 jsonl 文件")
        return

    try:
        # 2. 读取第一行
        df = None
        if file_type == "parquet":
            # 只读第一行
            df = pd.read_parquet(file_path).head(1)
        else:
            # 只读第一行
            df = pd.read_json(file_path, lines=True, nrows=1)
            
        # 3. 打印报告
        print(f"\n{'='*20} 📂 {folder_name} ({file_type}) {'='*20}")
        print(f"📄 文件路径: .../{os.path.basename(file_path)}")
        print(f"🔑 字段列表: {list(df.columns)}")
        print("-" * 60)
        
        # 打印第一行示例
        row = df.iloc[0].to_dict()
        for col, val in row.items():
            print(f"   • {col:<15}: {truncate(val)}")
            
    except Exception as e:
        print(f"❌ {folder_name}: 读取失败 - {e}")

def main():
    print(f"🚀 开始检查数据结构 (Root: {RAW_DIR})...\n")
    for target in TARGET_DIRS:
        inspect_dataset(target)
    print("\n✅ 检查完成。请复制以上内容。")

if __name__ == "__main__":
    main()