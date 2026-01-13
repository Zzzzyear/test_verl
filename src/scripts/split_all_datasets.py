import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ================= 配置区 =================
DATA_ROOT = "/data/zhaoqn/workspace/EGPO/datasets/processed"

TASKS = [
    ("mixed_reasoning.parquet", 256),
    ("math_single.parquet", 128),
    ("code_single.parquet", 128)
]

def get_large_type(data_type):
    """递归将类型升级为 64位 Large 类型"""
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
    """根据原始 Schema 生成全 64位 的安全 Schema"""
    new_fields = []
    for field in original_schema:
        new_type = get_large_type(field.type)
        new_fields.append(field.with_type(new_type))
    return pa.schema(new_fields)

def process_and_write(table, indices, output_path, chunk_size=1000, desc="Writing"):
    """分块提取并写入"""
    total = len(indices)
    if total == 0: return

    # 使用 table 自身的 schema (已经是 safe 的了)
    with pq.ParquetWriter(output_path, table.schema) as writer:
        for start in tqdm(range(0, total, chunk_size), desc=desc, unit="chunk"):
            end = min(start + chunk_size, total)
            batch_indices = indices[start:end]
            
            # 提取
            batch = table.take(batch_indices)
            
            # 写入
            writer.write_table(batch)

def split_and_save_ultimate():
    print(f"🚀 Starting Read-Time Schema Evolution split in: {DATA_ROOT}")
    print("=" * 60)

    for filename, val_size in TASKS:
        source_path = os.path.join(DATA_ROOT, filename)
        
        if not os.path.exists(source_path):
            print(f"⚠️  File not found: {filename}, skipping...")
            continue

        print(f"📖 Analyzing {filename}...")
        try:
            # 1. [关键] 只读取 Metadata (Schema)，不读取数据
            original_schema = pq.read_schema(source_path)
            
            # 2. [关键] 构建目标 Safe Schema (全 Large 类型)
            safe_schema = get_safe_schema(original_schema)
            
            # 3. [核心修复] 使用 safe_schema 读取文件
            # 这一步会迫使 PyArrow 在从磁盘加载数据时，直接构建 64位 数组
            # 从而跳过了那个脆弱的 32位 转换过程，彻底根治 Overflow
            table = pq.read_table(source_path, schema=safe_schema)
            
        except Exception as e:
            print(f"❌ Error during safe load: {e}")
            continue

        total_len = table.num_rows
        if total_len <= val_size:
            print(f"❌ Too small ({total_len} <= {val_size}), skipping.")
            continue

        # 索引操作
        indices = np.arange(total_len)
        rng = np.random.default_rng(seed=42)
        rng.shuffle(indices)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # 路径
        base_name = filename.replace(".parquet", "")
        train_out = os.path.join(DATA_ROOT, f"{base_name}_train_final.parquet")
        val_out = os.path.join(DATA_ROOT, f"{base_name}_val_fixed.parquet")

        print(f"   Task: {base_name} | Total: {total_len}")
        print(f"   Schema upgraded to Large types? Yes.")
        
        try:
            # 写入验证集
            process_and_write(table, val_indices, val_out, desc="   [1/2] Validation")
            
            # 写入训练集
            process_and_write(table, train_indices, train_out, desc="   [2/2] Training  ")
            
        except Exception as e:
            print(f"\n❌ Error during write: {e}")
            continue

        print(f"   ✅ Done.\n")

    print("🎉 All tasks finished. You are now Overflow-Proof.")

if __name__ == "__main__":
    split_and_save_ultimate()