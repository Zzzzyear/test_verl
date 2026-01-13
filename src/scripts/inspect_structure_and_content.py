import os
import glob
import pandas as pd
import json

# ================= 配置 =================
BASE_DIR = "/data/zhaoqn/workspace/EGPO"
RAW_DIR = os.path.join(BASE_DIR, "datasets/raw")

# 你想要重点检查的数据集（文件夹名）
TARGETS = [
    "Mixture-of-Thoughts",
    "NuminaMath-CoT",
    "Eurus-2-RL-Data",
    "MATH-500", 
    "AIME-2024",
    "AIME-2025",
    "big_bench_hard",
    "OlympiadBench",
    "LiveCodeBench",
    "LeetCodeDataset",
    "openai_humaneval",
    "GPQA-Diamond"

]

def print_tree(startpath, depth=2):
    """打印文件夹层级结构"""
    print(f"\n📂 目录结构: {os.path.basename(startpath)}/")
    startpath = startpath.rstrip(os.sep)
    num_sep_start = startpath.count(os.sep)
    
    for root, dirs, files in os.walk(startpath):
        num_sep = root.count(os.sep)
        if num_sep - num_sep_start >= depth:
            continue
            
        level = num_sep - num_sep_start
        indent = "    " * level
        print(f"{indent}📁 {os.path.basename(root)}/   (包含 {len(files)} 个文件)")

def inspect_file_content(filepath):
    """读取文件并展示列名和第一行"""
    try:
        filename = os.path.basename(filepath)
        df = None
        
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath).head(1)
        elif filepath.endswith('.jsonl'):
            df = pd.read_json(filepath, lines=True, nrows=1)
            
        if df is not None:
            print(f"   📄 采样文件: {filename}")
            print(f"   🔑 列名列表: {df.columns.tolist()}")
            
            # 打印第一行的内容示例（截断长文本）
            first_row = df.iloc[0].to_dict()
            for k, v in first_row.items():
                val_str = str(v)
                if len(val_str) > 100: val_str = val_str[:100] + "..."
                print(f"      • {k:<12}: {val_str}")
                
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")

def main():
    print(f"🚀 开始全方位检查数据 (Root: {RAW_DIR})")
    
    for target in TARGETS:
        target_path = os.path.join(RAW_DIR, target)
        if not os.path.exists(target_path):
            print(f"\n❌ 未找到数据集: {target}")
            continue

        print("\n" + "="*60)
        print(f"🔍 正在分析: {target}")
        print("="*60)
        
        # 1. 打印目录树 (看看有没有 math/ code/ 这种子文件夹)
        print_tree(target_path, depth=2)
        
        # 2. 深入每个子目录读取一个文件看结构
        # 查找该目录下所有子目录中的第一个 parquet/jsonl
        print("\n📋 数据内容采样:")
        
        # 策略：找到该文件夹下直接包含数据的子目录
        # 如果是 Mixture-of-Thoughts，我们希望能看到 math/xxx.parquet, code/xxx.parquet
        
        # 获取所有包含数据文件的路径
        sample_files = []
        for root, dirs, files in os.walk(target_path):
            # 找一个 parquet 或 jsonl
            valid_files = [f for f in files if f.endswith('.parquet') or f.endswith('.jsonl')]
            if valid_files:
                # 记录这个目录下的第一个文件
                sample_files.append(os.path.join(root, valid_files[0]))
        
        # 如果子目录太多（比如 split 了几百个 shard），我们只取前 3 个和后 3 个展示，避免刷屏
        # 但对于 Mixture-of-Thoughts，我们希望看到 math/code/science 各一个
        
        seen_parents = set()
        for f in sample_files:
            parent = os.path.dirname(f)
            # 简单的去重逻辑：每个子文件夹只看一个文件
            if parent in seen_parents: continue
            seen_parents.add(parent)
            
            # 打印 relative path header
            rel_path = os.path.relpath(parent, target_path)
            print(f"\n   📂 子目录: {rel_path}")
            inspect_file_content(f)

    print("\n✅ 检查完成。")

if __name__ == "__main__":
    main()