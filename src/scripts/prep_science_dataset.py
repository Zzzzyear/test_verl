import os
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset
from tqdm import tqdm
import multiprocessing

# ================= 1. 配置区 =================
# 自动寻找项目根目录
def find_project_root():
    candidates = [
        "/data-store/zhaoqiannian/workspace/EGPO",
        "/data/zhaoqn/workspace/EGPO",
        os.getcwd()
    ]
    for path in candidates:
        if os.path.exists(path): return path
    return os.getcwd()

BASE_DIR = find_project_root()
RAW_DIR = os.path.join(BASE_DIR, "datasets/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "datasets/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 目标设定
TARGET_SCIENCE_COUNT = 200000  # 目标 20万
VAL_SIZE = 256                 # 验证集大小 (保持与其他大数据集一致)
DUMMY_MESSAGE = [{"role": "user", "content": "DUMMY_SCHEMA_FIX"}]

# ================= 2. 核心处理逻辑 (复用 prep_dataset.py) =================

def to_chat_list(sys, user, asst=None):
    msgs = []
    if sys: msgs.append({"role": "system", "content": str(sys)})
    if user: msgs.append({"role": "user", "content": str(user)})
    if asst: msgs.append({"role": "assistant", "content": str(asst)})
    return msgs

def extract_boxed_answer(text):
    import re
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1] 
    match_mc = re.search(r"[Aa]nswer is\s?[:\s]?\s?([A-D])", text)
    if match_mc: return match_mc.group(1)
    return None

def process_row_science(ex):
    """专门处理 Mixture 中的 Science 数据"""
    res = {
        "data_source": "mixture",
        "ability": "science",
        "prompt": DUMMY_MESSAGE,
        "response": DUMMY_MESSAGE,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": "train", "index": 0, "ability": "science"},
        "is_valid": False,
    }

    # 提取原始消息
    msgs = ex.get('messages', [])
    if hasattr(msgs, 'tolist'): msgs = msgs.tolist()
    
    user_msg, asst_msg = "", ""
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                role = m.get('role', '')
                content = m.get('content', '')
                if role == 'user': user_msg = str(content) if content else ""
                elif role == 'assistant': asst_msg = str(content) if content else ""
    
    # 核心校验
    if user_msg and asst_msg:
        # 1. 确保是 Science 来源 (虽然我们会先 filter，但这里再防一道)
        src = str(ex.get('source', '')).lower()
        
        # 2. 提取 Ground Truth
        clean_gt = extract_boxed_answer(asst_msg)
        
        if clean_gt:
            res.update({
                "prompt": to_chat_list(None, user_msg),
                "response": to_chat_list(None, None, asst_msg),
                "is_valid": True,
                "reward_model": {"style": "rule", "ground_truth": clean_gt}
            })

    return res

# ================= 3. PyArrow 安全切分工具 (复用 split_all_datasets.py) =================

def get_large_type(data_type):
    """递归升级为 64位 Large 类型"""
    if pa.types.is_string(data_type): return pa.large_string()
    if pa.types.is_binary(data_type): return pa.large_binary()
    if pa.types.is_list(data_type): return pa.large_list(get_large_type(data_type.value_type))
    if pa.types.is_fixed_size_list(data_type): return pa.fixed_size_list(get_large_type(data_type.value_type), data_type.list_size)
    if pa.types.is_struct(data_type):
        return pa.struct([field.with_type(get_large_type(field.type)) for field in data_type])
    if pa.types.is_map(data_type):
        return pa.map_(get_large_type(data_type.key_type), get_large_type(data_type.item_type))
    return data_type

def get_safe_schema(original_schema):
    new_fields = []
    for field in original_schema:
        new_type = get_large_type(field.type)
        new_fields.append(field.with_type(new_type))
    return pa.schema(new_fields)

def process_and_write(table, indices, output_path, chunk_size=1000, desc="Writing"):
    total = len(indices)
    if total == 0: return
    with pq.ParquetWriter(output_path, table.schema) as writer:
        for start in tqdm(range(0, total, chunk_size), desc=desc, unit="chunk"):
            end = min(start + chunk_size, total)
            batch_indices = indices[start:end]
            writer.write_table(table.take(batch_indices))

# ================= 4. 主流程 =================

def main():
    print(f"🚀 Starting Science Dataset Prep & Split")
    print(f"   Target: {TARGET_SCIENCE_COUNT} samples")
    print("=" * 60)

    # --- Step 1: 加载 Mixture 数据 ---
    raw_files = glob.glob(os.path.join(RAW_DIR, "Mixture-of-Thoughts", "**/*.parquet"), recursive=True)
    if not raw_files:
        print("❌ Error: Mixture-of-Thoughts raw files not found!")
        return

    print("📖 Loading Mixture-of-Thoughts...")
    ds = load_dataset("parquet", data_files=raw_files, split="train")
    
    # --- Step 2: 筛选 Science 数据 ---
    print("🔍 Filtering for 'science' ability...")
    # 注意：Mixture 数据集里 ability 列通常是 'math', 'code' 等，我们需要确认 'science' 标签
    # 根据 prep_dataset.py 逻辑：if 'math' or 'numina' -> math, else -> science (排除 code)
    
    def is_science(x):
        src = str(x.get('source', '')).lower()
        if 'code' in src: return False
        if 'math' in src or 'numina' in src: return False
        # 排除掉明确不是 Science 的，剩下的当作 Science (包含 physics, chem, bio 等)
        return True

    ds_sci = ds.filter(is_science, num_proc=16, desc="Filtering Science")
    print(f"   Found {len(ds_sci)} raw science candidates.")

    # --- Step 3: 标准化处理 ---
    print("⚙️  Standardizing format...")
    ds_processed = ds_sci.map(
        process_row_science,
        num_proc=16,
        remove_columns=ds.column_names,
        desc="Formatting"
    )
    ds_valid = ds_processed.filter(lambda x: x['is_valid'], desc="Dropping Invalid")
    print(f"   Valid Science Samples: {len(ds_valid)}")

    # --- Step 4: 采样至 20万 ---
    final_ds = ds_valid
    if len(final_ds) > TARGET_SCIENCE_COUNT:
        print(f"✂️  Downsampling to {TARGET_SCIENCE_COUNT}...")
        final_ds = final_ds.shuffle(seed=42).select(range(TARGET_SCIENCE_COUNT))
    
    # --- Step 5: 保存 Single Parquet (清洗 Numpy) ---
    single_path = os.path.join(PROCESSED_DIR, "science_single.parquet")
    print(f"💾 Saving intermediate: {single_path}")
    
    # 递归清洗 Numpy (借用 prep_dataset 的逻辑)
    def clean_obj_recursive(obj):
        if isinstance(obj, np.ndarray): return [clean_obj_recursive(x) for x in obj.tolist()]
        elif isinstance(obj, np.generic): return obj.item()
        elif isinstance(obj, list): return [clean_obj_recursive(x) for x in obj]
        elif isinstance(obj, dict): return {k: clean_obj_recursive(v) for k, v in obj.items()}
        return obj

    cleaned_list = [clean_obj_recursive(row) for row in tqdm(final_ds, desc="Sanitizing")]
    ds_safe = Dataset.from_list(cleaned_list)
    ds_safe.to_parquet(single_path)
    
    # --- Step 6: 执行安全切分 (Train/Val) ---
    print("\n🔪 Performing Safe Split (Train/Val)...")
    
    # 使用 PyArrow 原生读取 + Schema 升级
    try:
        original_schema = pq.read_schema(single_path)
        safe_schema = get_safe_schema(original_schema)
        table = pq.read_table(single_path, schema=safe_schema)
        
        total_len = table.num_rows
        # 索引操作
        indices = np.arange(total_len)
        rng = np.random.default_rng(seed=42)
        rng.shuffle(indices)
        
        val_indices = indices[:VAL_SIZE]
        train_indices = indices[VAL_SIZE:]
        
        base_name = "science_single"
        train_out = os.path.join(PROCESSED_DIR, f"{base_name}_train_final.parquet")
        val_out = os.path.join(PROCESSED_DIR, f"{base_name}_val_fixed.parquet")
        
        print(f"   Writing Validation ({len(val_indices)})...")
        process_and_write(table, val_indices, val_out)
        
        print(f"   Writing Training ({len(train_indices)})...")
        process_and_write(table, train_indices, train_out)
        
        print("\n✅ Science Dataset Pipeline Completed Successfully!")
        print(f"   -> {train_out}")
        print(f"   -> {val_out}")
        
    except Exception as e:
        print(f"❌ Critical Error during split: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()