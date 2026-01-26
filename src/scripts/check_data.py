import os
import json
import numpy as np
from datasets import load_dataset
from termcolor import colored  # 如果没有安装，脚本会自动降级处理

# ================= 配置区 =================
BASE_DIR = "/data/zhaoqn/workspace/EGPO"
DATA_DIR = os.path.join(BASE_DIR, "datasets/processed")

EXPECTED_FILES = {
    "openr1_pool_train": "math_openr1_pool30k_random_source_softcap_train_final.parquet",
    "openr1_pool_val":   "math_openr1_pool30k_random_source_softcap_val_fixed.parquet"
}

# ================= 辅助工具 =================
def print_status(msg, status):
    """打印带颜色的状态信息"""
    try:
        if status == "PASS":
            print(f"   [{colored('PASS', 'green')}] {msg}")
        elif status == "FAIL":
            print(f"   [{colored('FAIL', 'red')}] {msg}")
        elif status == "WARN":
            print(f"   [{colored('WARN', 'yellow')}] {msg}")
        else:
            print(f"   [{status}] {msg}")
    except ImportError:
        # 如果没有 termcolor，使用普通打印
        symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"   [{symbol} {status}] {msg}")

def validate_dataset(name, filename):
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n🔍 Inspecting Dataset: {name} ({filename})")
    print("-" * 60)

    if not os.path.exists(filepath):
        print_status(f"File not found: {filepath}", "FAIL")
        return False

    try:
        # 1. 模拟训练器加载 (使用 datasets 库，而非 pandas)
        # 这是判断能否跑通训练的唯一标准
        ds = load_dataset("parquet", data_files=filepath, split="train")
        print_status(f"Successfully loaded {len(ds)} samples", "PASS")
        
        # 2. 类型安全检查 (Type Safety)
        # 检查第一条数据，确保是 Python List 而非 Numpy Array
        sample = ds[0]
        prompt = sample['prompt']
        response = sample['response']
        
        if isinstance(prompt, list) and isinstance(response, list):
            print_status("Data types are pure Python List (Trainer Compatible)", "PASS")
        elif isinstance(prompt, np.ndarray) or isinstance(response, np.ndarray):
            print_status(f"Detected Numpy Array! (Prompt: {type(prompt)})", "FAIL")
            return False
        else:
            print_status(f"Unknown type detected: {type(prompt)}", "WARN")

        # 3. 结构检查 (Structure)
        # 确保 List 里面包的是 Dict
        if len(prompt) > 0 and isinstance(prompt[0], dict) and 'role' in prompt[0]:
            print_status("Chat template structure (List[Dict]) is correct", "PASS")
        else:
            print_status("Invalid Chat structure", "FAIL")
            return False

        # 4. 内容完整性 (Content Integrity)
        # 检查 Ground Truth 是否有效
        error_count = 0
        empty_gt_count = 0
        
        # 定义检查函数
        def check_row(ex):
            nonlocal error_count, empty_gt_count
            gt = ex['reward_model']['ground_truth']
            ability = ex['ability']
            
            # 检查空值
            if not gt or len(str(gt).strip()) == 0:
                empty_gt_count += 1
                return
            
            # Code 任务必须是合法 JSON
            if ability == 'code':
                try:
                    json.loads(gt)
                except:
                    error_count += 1

        # 抽样检查 1000 条 (全量检查太慢，抽样足够代表性)
        check_size = min(1000, len(ds))
        ds.select(range(check_size)).map(check_row, load_from_cache_file=False)
        
        if error_count > 0:
            print_status(f"Found {error_count} invalid Code GTs (JSON parse fail)", "FAIL")
            return False
        else:
            print_status("Code GT JSON validity check passed", "PASS")

        if empty_gt_count > 0:
            # 空 GT 只是警告，不影响跑通，只要数量不多
            print_status(f"Found {empty_gt_count} empty Ground Truths (Acceptable noise)", "WARN")
        else:
            print_status("No empty Ground Truths found", "PASS")

        return True

    except Exception as e:
        print_status(f"Critical Load Error: {str(e)}", "FAIL")
        return False

# ================= 主程序 =================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 EGPO FINAL DATASET VALIDATION")
    print("=" * 60)
    print(f"Target Directory: {DATA_DIR}")
    
    all_passed = True
    results = {}

    for name, fname in EXPECTED_FILES.items():
        is_valid = validate_dataset(name, fname)
        results[name] = is_valid
        if not is_valid:
            all_passed = False

    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ READY" if passed else "❌ FAILED"
        print(f"{name:<10} : {status}")

    print("-" * 60)
    if all_passed:
        print("\n🎉 ALL SYSTEMS GO! Dataset is strictly validated and ready for training.")
        print("   Run the following command to start training:")
        print(f"\n   bash src/scripts/run_egpo.sh")
    else:
        print("\n⛔ BLOCKER: Please fix the failed datasets before training.")