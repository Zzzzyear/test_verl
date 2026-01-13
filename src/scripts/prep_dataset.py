import os
import glob
import json
import re
import numpy as np
import multiprocessing
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# ================= 1. 基础配置 =================

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

# 过滤配置
MIN_TOKENS = 20             
MAX_TOKENS = 4096           
NUM_PERM = 128              
THRESHOLD = 0.8             

# 目标规模
TARGET_TOTAL_MIXED = 60000 
RATIOS = {"math": 0.4, "code": 0.4, "science": 0.2}
COUNTS = {k: int(TARGET_TOTAL_MIXED * v) for k, v in RATIOS.items()}
TARGET_SINGLE_TASK = 30000

DUMMY_MESSAGE = [{"role": "user", "content": "DUMMY_SCHEMA_FIX"}]

# ================= 2. Tokenizer =================

def find_tokenizer_path():
    candidates = [
        "/data-store/zhaoqiannian/models/Qwen/Qwen3-1.7B", 
        "/data/zhaoqn/models/Qwen/Qwen3-1.7B",
        "/data/zhaoqn/models/Qwen/Qwen3-8B"
    ]
    for path in candidates:
        if os.path.exists(path): 
            print(f"✅ Found local tokenizer: {path}")
            return path
    print("⚠️  No local Qwen found. Will fallback to gpt2.")
    return "gpt2"

TOKENIZER_PATH = find_tokenizer_path()

def get_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    except:
        return AutoTokenizer.from_pretrained("gpt2")

tokenizer = get_tokenizer()

# ================= 3. LSH 工具 =================

def get_minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    tokens = set(str(text).lower().split())
    for t in tokens:
        m.update(t.encode('utf8'))
    return m

def build_decontamination_index():
    print(f"\n🔒 [Step 1] Building Decontamination Index (Test Sets)...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    count = 0
    test_configs = [
        ("MATH-500", "problem", "jsonl"), 
        ("AIME-2024", "problem", "parquet"),
        ("AIME-2025", "question", "jsonl"), 
        ("GPQA-Diamond", "question", "parquet"),
        ("big_bench_hard", "question", "parquet"), 
        ("LiveCodeBench", "question_content", "jsonl"),
        ("LeetCodeDataset", "query", "jsonl"), 
        ("openai_humaneval", "prompt", "parquet"),
        ("OlympiadBench", "question", "parquet") 
    ]

    for name, col, fmt in test_configs:
        pattern = os.path.join(RAW_DIR, name, f"**/*.{fmt}")
        files = glob.glob(pattern, recursive=True)
        if not files:
            alt = "parquet" if fmt == "jsonl" else "jsonl"
            files = glob.glob(os.path.join(RAW_DIR, name, f"**/*.{alt}"), recursive=True)
        
        for f in files:
            try:
                if f.endswith('jsonl'): df = pd.read_json(f, lines=True)
                else: df = pd.read_parquet(f)
                
                target_col = col if col in df.columns else next((c for c in df.columns if c in ['question','prompt','input','problem']), None)
                if target_col:
                    texts = df[target_col].dropna().astype(str).tolist()
                    for txt in texts:
                        if len(txt) > 15: 
                            lsh.insert(f"test_{name}_{count}", get_minhash(txt))
                            count += 1
            except: pass
    print(f"✅ Decontamination Index Built: {count} samples.")
    return lsh

# ================= 4. 核心解析逻辑 =================

def to_chat_list(sys, user, asst=None):
    msgs = []
    if sys: msgs.append({"role": "system", "content": str(sys)})
    if user: msgs.append({"role": "user", "content": str(user)})
    if asst: msgs.append({"role": "assistant", "content": str(asst)})
    return msgs

def extract_boxed_answer(text):
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1] 
    match_mc = re.search(r"[Aa]nswer is\s?[:\s]?\s?([A-D])", text)
    if match_mc: return match_mc.group(1)
    return None

def parse_eurus_gt(gt_str, debug_mode=False):
    if not gt_str or not isinstance(gt_str, str): return None
    sanitized_str = gt_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    try:
        try: data = json.loads(gt_str)
        except json.JSONDecodeError: data = json.loads(sanitized_str)
        if not isinstance(data, dict): return None
        inputs = data.get('inputs', [])
        outputs = data.get('outputs', [])
        if not isinstance(inputs, list) or not isinstance(outputs, list): return None
        if len(inputs) == 0 or len(inputs) != len(outputs): return None
        cases = []
        for i, o in zip(inputs, outputs):
            i_val = str(i[0]) if isinstance(i, list) and len(i)==1 else (i if isinstance(i, str) else json.dumps(i))
            o_val = str(o[0]) if isinstance(o, list) and len(o)==1 else (o if isinstance(o, str) else json.dumps(o))
            cases.append({"input": str(i_val), "output": str(o_val)})
        return json.dumps(cases)
    except: return None

def extract_prompt_fixed(p_raw):
    """提取 Prompt (兼容 Numpy/List/Str)"""
    user_txt, sys_txt = "", ""
    if p_raw is None: return "", ""
    
    p_list = []
    if isinstance(p_raw, np.ndarray):
        try: p_list = p_raw.tolist()
        except: pass
    elif isinstance(p_raw, list): p_list = p_raw
    elif isinstance(p_raw, str):
        try:
            parsed = json.loads(p_raw)
            if isinstance(parsed, list): p_list = parsed
            else: user_txt = p_raw 
        except: user_txt = p_raw

    if p_list:
        for msg in p_list:
            if isinstance(msg, dict):
                role = msg.get('role', '').lower()
                content = msg.get('content', '')
                if role == 'system': sys_txt = str(content) if content else ""
                elif role in ['user', 'human']: user_txt = str(content) if content else ""
    return user_txt, sys_txt

def process_row(ex, idx, source_type, default_ability=None, debug_mode=False):
    ability_val = default_ability or "unknown"
    res = {
        "data_source": source_type,
        "ability": ability_val,
        "prompt": DUMMY_MESSAGE,
        "response": DUMMY_MESSAGE,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": "train", "index": 0, "ability": ability_val},
        "is_valid": False,
        "search_text": ""
    }

    # --- Numina ---
    if source_type == "numina": 
        p, s = ex.get('problem'), ex.get('solution')
        clean_gt = extract_boxed_answer(s)
        if p and s and clean_gt:
            res.update({
                "ability": "math",
                "prompt": to_chat_list(None, p),
                "response": to_chat_list(None, None, s),
                "is_valid": True,
                "search_text": p,
                "reward_model": {"style": "rule", "ground_truth": clean_gt}
            })

    # --- Eurus ---
    elif source_type == "eurus": 
        p_raw = ex.get('prompt')
        user_txt, sys_txt = extract_prompt_fixed(p_raw)
        raw_gt = ex.get('reward_model', {}).get('ground_truth', '')
        clean_gt = parse_eurus_gt(raw_gt, debug_mode=debug_mode)
        
        if user_txt and clean_gt:
            res.update({
                "ability": "code",
                "prompt": to_chat_list(sys_txt, user_txt),
                "response": to_chat_list(None, None, "REF"),
                "is_valid": True,
                "search_text": user_txt,
                "reward_model": {"style": "rule", "ground_truth": clean_gt}
            })
        
        if not res['is_valid'] and debug_mode:
            print(f"   [Trace ID:{idx}] Invalid Eurus -> UserTxt: {bool(user_txt)}, CleanGT: {bool(clean_gt)}")

    # --- Mixture ---
    elif source_type == "mixture": 
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
        
        if user_msg and asst_msg:
            src = str(ex.get('source', '')).lower()
            if 'code' in src: 
                res['is_valid'] = False 
                return res
            
            clean_gt = extract_boxed_answer(asst_msg)
            if not clean_gt: 
                res['is_valid'] = False
                return res

            if 'math' in src or 'numina' in src: res["ability"] = "math"
            else: res["ability"] = "science"
            
            res.update({
                "prompt": to_chat_list(None, user_msg),
                "response": to_chat_list(None, None, asst_msg),
                "is_valid": True,
                "search_text": user_msg,
                "reward_model": {"style": "rule", "ground_truth": clean_gt}
            })

    if res['is_valid']:
        res['extra_info']['ability'] = res['ability']
        
    return res

# ================= 5. Pipeline =================

def find_files(base_path, pattern):
    files = glob.glob(os.path.join(base_path, pattern), recursive=True)
    if files: return files
    simple_pattern = pattern.replace("**/", "")
    files = glob.glob(os.path.join(base_path, simple_pattern))
    return files

def run_processing(ds_raw, name, source_type, lsh_test, default_ability=None):
    print(f"\n🚀 Processing {name} (Raw: {len(ds_raw)})...")
    
    cpu_cores = multiprocessing.cpu_count()
    proc_count = min(4, cpu_cores) if source_type == "eurus" else min(16, cpu_cores)

    ds = ds_raw.map(
        lambda x, idx: process_row(x, idx, source_type, default_ability, debug_mode=(idx < 10)),
        num_proc=proc_count,
        remove_columns=ds_raw.column_names,
        desc="Formatting",
        load_from_cache_file=False,
        with_indices=True
    )
    
    ds = ds.filter(lambda x: x['is_valid'], num_proc=proc_count, desc="Filtering", load_from_cache_file=False)
    print(f"   -> Struct Valid: {len(ds)}")
    
    if len(ds) == 0: return None

    keep_indices = []
    search_texts = ds['search_text'] 
    lsh_internal = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    stats = {"short": 0, "leak": 0, "dup": 0, "kept": 0}
    
    batch_size = 2000
    for i in tqdm(range(0, len(ds), batch_size), desc="Scanning"):
        batch_txt = search_texts[i : i+batch_size]
        encs = tokenizer(batch_txt, add_special_tokens=False, truncation=True, max_length=MAX_TOKENS)
        
        for j, (txt, ids) in enumerate(zip(batch_txt, encs['input_ids'])):
            idx = i + j
            if len(ids) < MIN_TOKENS: 
                stats["short"] += 1; continue
            m = get_minhash(txt)
            if len(lsh_test.query(m)) > 0:
                stats["leak"] += 1; continue 
            if len(lsh_internal.query(m)) > 0:
                stats["dup"] += 1; continue 
                
            lsh_internal.insert(f"id_{idx}", m)
            keep_indices.append(idx)
            stats["kept"] += 1
            
    ds_clean = ds.select(keep_indices)
    print(f"   📊 Stats for {name}:")
    print(f"      Kept  : {stats['kept']}")
    print(f"      Short : {stats['short']}")
    print(f"      Leak  : {stats['leak']}")
    print(f"      Dup   : {stats['dup']}")
    
    return ds_clean.select_columns(['data_source', 'ability', 'prompt', 'response', 'reward_model', 'extra_info'])

# ================= 6. 安全保存 (核武器级清洗) =================

def save_safe(ds, filename):
    """
    [最终修复] 导出为 Python 对象清洗，彻底去除 Numpy 污染
    返回清洗后的 Dataset 对象以供后续使用
    """
    path = os.path.join(PROCESSED_DIR, filename)
    print(f"💾 Saving to {filename}...")
    
    # 内部清洗函数 (递归处理)
    def clean_obj_recursive(obj):
        if isinstance(obj, np.ndarray):
            return [clean_obj_recursive(x) for x in obj.tolist()]
        elif isinstance(obj, np.generic): 
            return obj.item()
        elif isinstance(obj, list):
            return [clean_obj_recursive(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: clean_obj_recursive(v) for k, v in obj.items()}
        return obj

    # 1. 彻底转为纯 Python List (物理隔离 Arrow/Numpy)
    print("   Extracting to Python list for deep cleaning...")
    raw_list = ds.to_list()
    
    # 2. 递归清洗
    print("   Recursively sanitizing data structures...")
    cleaned_list = [clean_obj_recursive(row) for row in raw_list]
    
    # 3. 重建 Dataset (Arrow 会根据纯 Python 对象推断正确 Schema)
    ds_safe = Dataset.from_list(cleaned_list)
    
    # 4. 保存
    ds_safe.to_parquet(path)
    print(f"✅ Saved: {len(ds_safe)} samples (Verified Clean)")
    
    return ds_safe

# ================= 7. Main =================

def main():
    lsh_test = build_decontamination_index()
    datasets_pool = {"math": [], "code": [], "science": []}

    files = find_files(os.path.join(RAW_DIR, "NuminaMath-CoT"), "**/*.parquet")
    if files:
        ds = load_dataset("parquet", data_files=files, split="train")
        ds_out = run_processing(ds, "Numina", "numina", lsh_test)
        if ds_out: datasets_pool["math"].append(ds_out)

    files = find_files(os.path.join(RAW_DIR, "Eurus-2-RL-Data"), "**/*.parquet")
    if files:
        ds = load_dataset("parquet", data_files=files, split="train")
        if 'ability' in ds.column_names:
            ds = ds.filter(lambda x: x.get('ability') == 'code', num_proc=16, load_from_cache_file=False)
        ds_out = run_processing(ds, "Eurus", "eurus", lsh_test)
        if ds_out: datasets_pool["code"].append(ds_out)
    else:
        print("❌ Warning: Eurus file not found!")

    files = find_files(os.path.join(RAW_DIR, "Mixture-of-Thoughts"), "**/*.parquet")
    if files:
        ds = load_dataset("parquet", data_files=files, split="train")
        ds_out = run_processing(ds, "Mixture", "mixture", lsh_test)
        if ds_out:
            ds_sci = ds_out.filter(lambda x: x['ability'] == 'science', load_from_cache_file=False)
            ds_math = ds_out.filter(lambda x: x['ability'] == 'math', load_from_cache_file=False)
            if len(ds_sci) > 0: datasets_pool["science"].append(ds_sci)
            if len(ds_math) > 0: datasets_pool["math"].append(ds_math)

    # 保存单任务 (带采样和返回)
    def process_and_save_single(domain, filename, target_cnt):
        if not datasets_pool[domain]: return None
        ds = concatenate_datasets(datasets_pool[domain])
        if len(ds) > target_cnt:
            print(f"   Sampling {domain} to {target_cnt}...")
            ds = ds.shuffle(seed=42).select(range(target_cnt))
        else:
            ds = ds.shuffle(seed=42)
        # 调用 safe save 并获取返回值 (清洗后的 Dataset)
        return save_safe(ds, filename)

    ds_math = process_and_save_single("math", "math_single.parquet", TARGET_SINGLE_TASK)
    ds_code = process_and_save_single("code", "code_single.parquet", TARGET_SINGLE_TASK)
    ds_sci = process_and_save_single("science", "science_single.parquet", TARGET_SINGLE_TASK)
    
    print("\n🔄 Building Mixed Dataset...")
    mix_components = []
    
    def get_balanced(ds, target):
        if not ds: return None
        if len(ds) >= target:
            return ds.shuffle(seed=42).select(range(target))
        else:
            print(f"   Upsampling {target}...")
            repeat = (target // len(ds)) + 1
            return concatenate_datasets([ds] * repeat).shuffle(seed=42).select(range(target))

    if ds_math: mix_components.append(get_balanced(ds_math, COUNTS["math"]))
    if ds_code: mix_components.append(get_balanced(ds_code, COUNTS["code"]))
    if ds_sci: mix_components.append(get_balanced(ds_sci, COUNTS["science"]))
        
    if mix_components:
        ds_mixed = concatenate_datasets(mix_components).shuffle(seed=42)
        # 混合后再次调用 save_safe，做最后一道防线
        save_safe(ds_mixed, "mixed_reasoning.parquet")
        
        abilities = ds_mixed['ability']
        print(f"   Dist: Math={abilities.count('math')}, Code={abilities.count('code')}, Sci={abilities.count('science')}")
    else:
        print("❌ Mixed dataset failed.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()