import os
import glob
import pandas as pd
import numpy as np
import json
from datasets import Dataset, load_dataset
from tqdm import tqdm

# ================= 配置 =================
BASE_DIR = "/data/zhaoqn/workspace/EGPO/datasets/processed"
RAW_DIR = "/data/zhaoqn/workspace/EGPO/datasets/raw"
OUTPUT_FILE = os.path.join(BASE_DIR, "mixed_reasoning.parquet")

TARGET_COUNTS = {
    "math": 24000,
    "code": 24000,
    "science": 12000
}

def clean_record(record):
    """
    清洗单条数据，确保是纯净的 Python 对象。
    """
    try:
        # 清洗 messages 列表
        def fix_msgs(msgs):
            # 1. 解包 numpy/arrow 容器
            if isinstance(msgs, np.ndarray): msgs = msgs.tolist()
            if not isinstance(msgs, list): return []
            
            cleaned = []
            for m in msgs:
                # 2. 强制将内容转为字符串 (防止 None 报错)
                role = str(m.get('role', ''))
                content = str(m.get('content', ''))
                cleaned.append({"role": role, "content": content})
            return cleaned

        return {
            "data_source": str(record['data_source']),
            "ability": str(record['ability']),
            "prompt": fix_msgs(record['prompt']),
            "response": fix_msgs(record['response']),
            # [Fix 1] 必须透传 ground_truth！
            "ground_truth": str(record.get('ground_truth', '')) 
        }
    except Exception as e:
        return None

def get_science_data():
    print("🔹 Extracting Science from Mixture-of-Thoughts...")
    files = glob.glob(os.path.join(RAW_DIR, "Mixture-of-Thoughts", "**/*.parquet"))
    # 使用 HF 加载以保持一致性
    ds = load_dataset("parquet", data_files=files, split="train")
    
    # 过滤出 Science
    ds_sci = ds.filter(
        lambda x: 'math' not in str(x['source']).lower() and 'code' not in str(x['source']).lower(),
        num_proc=16
    )
    
    science_list = []
    for row in tqdm(ds_sci, desc="Formatting Science"):
        msgs = row['messages']
        if isinstance(msgs, np.ndarray): msgs = msgs.tolist()
        
        u, a = "", ""
        for m in msgs:
            if m['role'] == 'user': u = m['content']
            if m['role'] == 'assistant': a = m['content']
            
        if u and a:
            science_list.append({
                "data_source": "mixture",
                "ability": "science",
                "prompt": [{"role": "user", "content": u}],
                "response": [{"role": "assistant", "content": a}],
                # [Fix 2] 新增 ground_truth，Science 题的 GT 通常就是 assistant 的完整回复
                "ground_truth": a 
            })
            
    # 采样
    import random
    if len(science_list) > TARGET_COUNTS['science']:
        random.shuffle(science_list)
        science_list = science_list[:TARGET_COUNTS['science']]
        
    return science_list

def main():
    print("🚀 Starting Pandas-Bridge Rebuild (Robust Mode)...")
    
    # 1. 加载已验证的 Math/Code 数据
    print("--> Loading Math Single...")
    df_math = pd.read_parquet(os.path.join(BASE_DIR, "math_single.parquet"))
    # 确保 math_single 已经由 prep_dataset.py 生成了 ground_truth，否则这里会报错或丢数据
    if 'ground_truth' not in df_math.columns:
        print("⚠️ Warning: 'ground_truth' missing in math_single.parquet! Please run prep_dataset.py first.")
        
    math_data = df_math.to_dict('records')
    if len(math_data) > TARGET_COUNTS['math']:
        import random
        random.shuffle(math_data)
        math_data = math_data[:TARGET_COUNTS['math']]

    print("--> Loading Code Single...")
    df_code = pd.read_parquet(os.path.join(BASE_DIR, "code_single.parquet"))
    code_data = df_code.to_dict('records')
    if len(code_data) > TARGET_COUNTS['code']:
        import random
        random.shuffle(code_data)
        code_data = code_data[:TARGET_COUNTS['code']]

    # 2. 提取 Science
    science_data = get_science_data()
    print(f"    Science Count: {len(science_data)}")

    # 3. 合并与深度清洗
    print("--> Merging & Deep Cleaning...")
    raw_list = math_data + code_data + science_data
    import random
    random.shuffle(raw_list)
    
    cleaned_list = []
    for r in tqdm(raw_list, desc="Sanitizing"):
        c = clean_record(r)
        if c: cleaned_list.append(c)

    # 4. 使用 Pandas 作为中间桥梁
    print("--> Converting to Pandas DataFrame...")
    df_final = pd.DataFrame(cleaned_list)
    
    print("--> Converting to HuggingFace Dataset...")
    # Dataset.from_pandas 会自动推断最完美的 Schema
    hf_dataset = Dataset.from_pandas(df_final)
    
    # 5. 保存
    print(f"--> Saving to {OUTPUT_FILE}...")
    hf_dataset.to_parquet(OUTPUT_FILE)
    
    print("\n✅ DONE! Mixed Dataset Rebuilt Successfully.")
    print(f"   Total: {len(hf_dataset)}")
    print("   Now run check_data.py one last time.")

if __name__ == "__main__":
    main()