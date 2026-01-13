import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
BASE_DIR = "/data/zhaoqn/workspace/EGPO"
RAW_DIR = os.path.join(BASE_DIR, "datasets/raw")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "outputs/analysis")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# 本地模型路径
TOKENIZER_PATH = "/data/zhaoqn/models/Qwen/Qwen3-8B"

# 采样数量 (设为 None 则分析全量，建议先跑 10000 条快速看结果)
SAMPLE_SIZE = 10000 

# 定义我们要分析的数据集及其格式特征
TARGETS = {
    "Mixture-of-Thoughts": {
        "path": "Mixture-of-Thoughts/**/*.parquet",
        "format": "parquet",
        "type": "messages" # 字段是 messages 列表
    },
    "NuminaMath": {
        "path": "NuminaMath-CoT/**/*.parquet",
        "format": "parquet",
        "type": "col_prob_sol" # 字段是 problem + solution
    },
    "Eurus-2": {
        "path": "Eurus-2-RL-Data/**/*.parquet",
        "format": "parquet",
        "type": "eurus" # 特殊: prompt(list) + response/solution
    },
    "MATH-500 (Test)": {
        "path": "MATH-500/**/*.jsonl",
        "format": "json",
        "type": "col_prob_sol"
    },
    "AIME-2024 (Test)": {
        "path": "AIME-2024/**/*.parquet",
        "format": "parquet",
        "type": "col_prob_sol"
    }
}

# ================= 工具函数 =================

def get_tokenizer():
    try:
        print(f"🔄 Loading tokenizer from {TOKENIZER_PATH} ...")
        return AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Load local tokenizer failed: {e}. Trying gpt2...")
        return AutoTokenizer.from_pretrained("gpt2")

tokenizer = get_tokenizer()

def extract_text(example, data_type):
    """根据数据集类型提取完整的 prompt + response 文本"""
    text = ""
    
    try:
        # Type 1: Standard Messages (Mixture-of-Thoughts)
        if data_type == "messages":
            if 'messages' in example and isinstance(example['messages'], list):
                for msg in example['messages']:
                    if isinstance(msg, dict) and 'content' in msg:
                        text += str(msg['content']) + " "
        
        # Type 2: Problem/Solution Columns (Numina, MATH-500, AIME)
        elif data_type == "col_prob_sol":
            # 尝试寻找常见的题目列名
            p = example.get('problem') or example.get('question') or example.get('input') or ""
            # 尝试寻找常见的答案列名
            r = example.get('solution') or example.get('answer') or example.get('output') or ""
            text = str(p) + " " + str(r)

        # Type 3: Eurus Special (Prompt is List[Dict])
        elif data_type == "eurus":
            # Eurus Prompt
            if 'prompt' in example and isinstance(example['prompt'], list):
                 for msg in example['prompt']:
                    if isinstance(msg, dict) and 'content' in msg:
                        text += str(msg['content']) + " "
            elif 'prompt' in example:
                text += str(example['prompt']) + " "
            
            # Eurus Response (Training set usually has it)
            if 'response' in example:
                text += str(example['response'])
            elif 'solution' in example:
                text += str(example['solution'])

    except Exception as e:
        return "" # 解析失败返回空

    return text

def calc_len_batch(examples, data_type):
    """批量计算长度"""
    batch_texts = []
    # examples 是一个 dict: {'col1': [v1, v2], 'col2': [v1, v2]}
    # 我们需要将其转回 row 格式来处理
    keys = list(examples.keys())
    num_rows = len(examples[keys[0]])
    
    for i in range(num_rows):
        # 构造单行 example dict
        row = {k: examples[k][i] for k in keys}
        txt = extract_text(row, data_type)
        batch_texts.append(txt)
    
    # 批量 Tokenize (速度快)
    encodings = tokenizer(batch_texts, truncation=False, add_special_tokens=False)
    lengths = [len(ids) for ids in encodings['input_ids']]
    
    return {'num_tokens': lengths}

# ================= 主逻辑 =================

def main():
    results = {}

    for name, config in TARGETS.items():
        print(f"\n📊 [Analyzing] {name} ...")
        path_pattern = os.path.join(RAW_DIR, config['path'])
        files = glob.glob(path_pattern, recursive=True)
        
        if not files:
            print(f"   ❌ File not found: {path_pattern}")
            continue

        try:
            # 加载数据
            ds = load_dataset(config['format'], data_files=files, split="train")
            
            if SAMPLE_SIZE and len(ds) > SAMPLE_SIZE:
                ds = ds.select(range(SAMPLE_SIZE))
            
            # 计算长度
            # 使用 fn_kwargs 传递 data_type
            ds = ds.map(
                lambda x: calc_len_batch(x, config['type']), 
                batched=True, 
                batch_size=1000,
                desc="   Tokenizing"
            )
            
            # 提取有效长度
            lens = [l for l in ds['num_tokens'] if l > 0]
            if lens:
                results[name] = pd.Series(lens, name=name)
            else:
                print(f"   ⚠️  No valid tokens found (schema mismatch?)")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # --- 打印报告 ---
    print("\n" + "="*85)
    print(f"{'Dataset':<25} | {'Count':<8} | {'Min':<6} | {'Median':<6} | {'Max':<8} | {'P5':<6} | {'P95':<6}")
    print("-" * 85)

    for name, s in results.items():
        if len(s) == 0: continue
        p5 = int(s.quantile(0.05))
        p95 = int(s.quantile(0.95))
        median = int(s.median())
        print(f"{name:<25} | {len(s):<8} | {s.min():<6} | {median:<6} | {s.max():<8} | {p5:<6} | {p95:<6}")
    print("="*85 + "\n")

    # --- 绘图 ---
    if results:
        print(f"🎨 Drawing distribution plot to {OUTPUT_IMG_DIR} ...")
        plt.figure(figsize=(14, 7))
        
        for name, s in results.items():
            # 截断极值以便绘图清晰 (只画 98% 的数据)
            cutoff = s.quantile(0.98)
            subset = s[s < cutoff]
            sns.kdeplot(subset, label=f"{name} (Med: {int(s.median())})", fill=True, alpha=0.3)
        
        plt.title(f"Token Length Distribution (Truncated at P98) - Tokenizer: {os.path.basename(TOKENIZER_PATH)}")
        plt.xlabel("Number of Tokens")
        plt.xlim(0, None) # 从 0 开始
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(OUTPUT_IMG_DIR, "token_distribution_v3.png")
        plt.savefig(save_path)
        print(f"✅ Plot saved: {save_path}")

if __name__ == "__main__":
    main()