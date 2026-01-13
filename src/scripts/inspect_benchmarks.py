import os
import glob
import pandas as pd
import json

# ================= 配置 =================
BASE_ROOT = "/data/zhaoqn/workspace/EGPO"
RAW_DIR = os.path.join(BASE_ROOT, "datasets/raw")

TARGETS = [
    "MATH-500", 
    "AIME-2024", 
    "AIME-2025", 
    "OlympiadBench", 
    "GPQA-Diamond", 
    "big_bench_hard", 
    "openai_humaneval", 
    "LiveCodeBench", 
    "LeetCodeDataset",
    # 同时也看一眼训练源数据，确保预处理脚本没问题
    "NuminaMath",
    "Eurus",
    "Mixture-of-Thoughts"
]

def inspect_folder(target_name):
    # 模糊匹配文件夹，因为不确定具体名字后面有没有后缀
    search_path = os.path.join(RAW_DIR, target_name + "*")
    matched_dirs = glob.glob(search_path)
    
    print(f"\n{'='*20} Inspecting: {target_name} {'='*20}")
    
    if not matched_dirs:
        print(f"❌ Directory not found matching: {search_path}")
        return

    # 这里的逻辑是：只取第一个匹配的文件夹进行深入分析
    target_dir = matched_dirs[0]
    print(f"📁 Dir: {target_dir}")

    # 找所有数据文件
    all_files = glob.glob(os.path.join(target_dir, "**/*.parquet"), recursive=True) + \
                glob.glob(os.path.join(target_dir, "**/*.jsonl"), recursive=True) + \
                glob.glob(os.path.join(target_dir, "**/*.json"), recursive=True)
    
    # 过滤掉非数据文件（如 metadata.json）
    data_files = [f for f in all_files if "metadata" not in f.split("/")[-1]]
    
    if not data_files:
        print("   ⚠️ No data files found.")
        return

    # 1. 打印文件列表（最多打印 3 个，防止刷屏）
    print(f"   📄 Found {len(data_files)} files. Examples:")
    for f in data_files[:3]:
        rel_path = os.path.relpath(f, RAW_DIR)
        print(f"      - {rel_path}")

    # 2. 读取第一个文件看结构
    first_file = data_files[0]
    try:
        if first_file.endswith(".parquet"):
            df = pd.read_parquet(first_file)
        elif first_file.endswith(".jsonl"):
            df = pd.read_json(first_file, lines=True)
        else: # json
            try:
                df = pd.read_json(first_file)
            except:
                # 有些 json 是 list of dicts
                with open(first_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])

        print(f"   📊 Schema (Columns): {list(df.columns)}")
        print(f"   📐 Shape: {df.shape}")
        
        # 3. 打印一行样本 (如果是 prompt/problem 相关列)
        # 智能探测：打印可能包含题目内容的列的前50个字符
        sample = df.iloc[0]
        interesting_cols = [c for c in df.columns if any(k in c.lower() for k in ['prob', 'quest', 'prom', 'inpu', 'sol', 'ans', 'test'])]
        print("   👀 Sample Content (First Row):")
        for c in interesting_cols:
            val = str(sample[c])[:100].replace("\n", "\\n")
            print(f"      {c}: {val}...")
            
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")

if __name__ == "__main__":
    for t in TARGETS:
        inspect_folder(t)