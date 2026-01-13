import os
import argparse
import json
import re
import glob
import pandas as pd
import numpy as np
import subprocess
import sys
import io
import contextlib
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================= 1. 路径映射 (BBH Top-3 极速版) =================
LOCAL_DATA_MAP = {
    "math500":   "{data_root}/MATH-500/test.jsonl",
    "aime24":    "{data_root}/AIME-2024/data/*.parquet",
    "aime25":    "{data_root}/AIME-2025/aime2025-*.jsonl", 
    "olympiad":  "{data_root}/OlympiadBench/OlympiadBench/OE_TO_maths_en_COMP/*.parquet",
    "gpqa":      "{data_root}/GPQA-Diamond/test/gpqa_diamond.parquet",
    
    # BBH Top-3
    "bbh": [
        "{data_root}/big_bench_hard/logical_deduction_seven_objects/*.parquet",
        "{data_root}/big_bench_hard/date_understanding/*.parquet",
        "{data_root}/big_bench_hard/boolean_expressions/*.parquet"
    ],
    
    "humaneval": "{data_root}/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
    "lcb":       "{data_root}/LiveCodeBench/test5.jsonl",
    "leetcode":  "{data_root}/LeetCodeDataset/LeetCodeDataset-test.jsonl"
}

# ================= 2. 代码沙箱 =================
def run_code_in_subprocess(code, test_code, entry_point, timeout=3.0):
    script_template = """
import sys
import math
import collections
import itertools
import random
import heapq
import functools
import re
from typing import List, Dict, Tuple, Optional, Any, Union, Set

# 1. Generated Code
{generated_code}

# 2. Test Code
{test_code}

# 3. Execution Entry
try:
    if '{entry_point}' != 'None':
        if 'check' in globals():
            check({entry_point})
        else:
            pass 
    print("EXECUTION_PASSED")
except Exception as e:
    print(f"EXECUTION_FAILED: {{e}}")
"""
    full_script = script_template.format(
        generated_code=code,
        test_code=test_code,
        entry_point=entry_point
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True, text=True, timeout=timeout
        )
        if "EXECUTION_PASSED" in result.stdout:
            return True, "Passed"
        else:
            err = result.stderr if result.stderr else result.stdout
            return False, f"Failed: {err.strip()[-200:]}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"System Error: {str(e)}"

# ================= 3. 清洗与提取 (增强版: 支持 True/False) =================
def clean_cot(text):
    if not text: return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def extract_code_block(text):
    text = clean_cot(text)
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match: return match.group(1)
    return text

def extract_math_answer(text):
    text = clean_cot(text)
    candidates = [m.start() for m in re.finditer(r"\\?boxed\s*\{", text)]
    if candidates:
        start = candidates[-1] + text[candidates[-1]:].find("{") + 1
        balance, content = 1, ""
        for char in text[start:]:
            if char == "{": balance += 1
            elif char == "}": balance -= 1
            if balance == 0: break
            content += char
        return content.strip()
    
    # 兜底：尝试提取最后一行
    lines = text.split('\n')
    if lines:
        last = lines[-1].strip()
        if len(last) < 20 and any(c.isdigit() for c in last):
            return last
    return ""

def extract_choice_answer(text):
    """
    [增强] 提取 (A) 或 True/False/Yes/No
    """
    text = clean_cot(text)
    
    # 1. 显式声明模式 (Option or Boolean)
    # 匹配: "The answer is (A)", "The answer is True", "Answer: False"
    patterns = [
        r"[Tt]he answer is:?\s*\(?([A-Ja-j])\)?",
        r"[Tt]he answer is:?\s*(True|False|TRUE|FALSE|Yes|No|YES|NO)",
        r"[Aa]nswer:?\s*\(?([A-Ja-j])\)?",
        r"[Aa]nswer:?\s*(True|False|TRUE|FALSE|Yes|No|YES|NO)",
    ]
    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match: return match.group(1).strip()
    
    # 2. Boxed 内容
    boxed = extract_math_answer(text)
    if boxed:
        # 如果 Boxed 是单个字母或布尔词
        if len(boxed) == 1 and boxed.isalpha(): return boxed
        if boxed.lower() in ['true', 'false', 'yes', 'no']: return boxed

    # 3. 弱匹配：文末的选项或布尔词
    last_part = text[-200:]
    
    # 匹配 (A)
    match_letter = re.findall(r"\(?([A-Ja-j])\)", last_part)
    if match_letter: return match_letter[-1]
    
    # 匹配 True/False (单词边界)
    match_bool = re.findall(r"\b(True|False|Yes|No)\b", last_part, re.IGNORECASE)
    if match_bool: return match_bool[-1]
    
    return ""

def normalize_math(s):
    if s is None: return ""
    s = str(s).strip().replace(" ", "").replace("\n", "").replace(r"\left", "").replace(r"\right", "").replace(r"\\", "\\")
    return s

def check_sample(response, item):
    task_type = item['type']
    
    # --- Code ---
    if task_type == 'code':
        code = extract_code_block(response)
        if not code.strip(): return False, "No code extracted"
        if item['gt']: 
            return run_code_in_subprocess(code, item['gt'], item['entry'])
        return False, "No test cases found"

    # --- Match / MCQ (BBH, GPQA) ---
    if task_type == 'match' or task_type == 'mcq':
        pred = extract_choice_answer(response)
        # GT 处理：去掉括号，统一转小写比对
        gt = str(item['gt']).strip().replace("(", "").replace(")", "")
        
        if not pred: pred = extract_math_answer(response) # Fallback
        if not pred: return False, "No answer extracted"
        
        # 鲁棒比对 (A == a, True == true)
        is_correct = (pred.lower() == gt.lower())
        return is_correct, f"Pred: {pred} | GT: {gt}"

    # --- Math ---
    pred = extract_math_answer(response)
    gt = str(item['gt'])
    if not pred: return False, "No answer extracted"
    
    norm_pred = normalize_math(pred)
    norm_gt = normalize_math(gt)
    is_correct = (norm_pred == norm_gt) or (norm_pred in norm_gt and len(norm_pred) > 0)
    return is_correct, f"Pred: {norm_pred} | GT: {norm_gt}"

# ================= 5. 数据加载 (List修复版) =================
def load_local_data(task, data_root, limit=-1):
    raw_template = LOCAL_DATA_MAP.get(task)
    if not raw_template: return []
    
    if isinstance(raw_template, list):
        path_list = [p.format(data_root=data_root) for p in raw_template]
    else:
        path_list = [raw_template.format(data_root=data_root)]
    
    files = []
    for full_path in path_list:
        if '*' in full_path:
            found = sorted(glob.glob(full_path, recursive=True))
            files.extend(found)
        elif os.path.exists(full_path):
            files.append(full_path)
            
    if not files:
        print(f"❌ No files found for {task}")
        return []
    
    print(f"[{task}] Loading {len(files)} file(s)...")
    
    dfs = []
    for f in files:
        try:
            if f.endswith('.parquet'): dfs.append(pd.read_parquet(f))
            elif f.endswith('.jsonl'): dfs.append(pd.read_json(f, lines=True))
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            
    if not dfs: return []
    df = pd.concat(dfs, ignore_index=True)
    
    data = []
    for idx, row in df.iterrows():
        try:
            item = {"id": f"{task}_{idx}", "type": "unknown", "gt": "", "entry": "None"}
            
            if task == "math500": 
                item.update({"prompt": row['problem'], "gt": row['solution'], "type": "math"})
            elif task == "aime24": 
                item.update({"prompt": row['problem'], "gt": row['solution'], "type": "math"})
            elif task == "aime25": 
                item.update({"prompt": row['question'], "gt": str(row['answer']), "type": "math"})
            elif task == "olympiad":
                if row.get('question_type') != 'Open-ended': continue
                gt = row['solution']
                if isinstance(row.get('final_answer'), list) and len(row['final_answer']) > 0: gt = row['final_answer'][0]
                item.update({"prompt": row['question'], "gt": gt, "type": "math"})
            
            elif task == "gpqa":
                if 'Incorrect Answer 1' in row:
                    opts = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
                    gt_val = row['Correct Answer']
                    q_val = row['Question']
                    np.random.shuffle(opts)
                    chars = ['A', 'B', 'C', 'D']
                    correct_char = chars[opts.index(gt_val)]
                    p_text = f"Question: {q_val}\nOptions:\n"
                    for i, o in enumerate(opts): p_text += f"({chars[i]}) {o}\n"
                    p_text += "Select the correct option letter (A, B, C, or D)."
                    item.update({"prompt": p_text, "gt": correct_char, "type": "mcq"})
                elif 'question' in row and 'answer' in row:
                    item.update({"prompt": row['question'], "gt": row['answer'], "type": "math"})
            
            elif task == "bbh": 
                item.update({"prompt": row['question'], "gt": row['target'], "type": "match"})
            
            elif task == "humaneval": 
                item.update({"prompt": row['prompt'], "gt": row['test'], "entry": row['entry_point'], "type": "code"})
            elif task == "lcb": 
                item.update({"prompt": row['question_content'], "gt": "sandbox", "type": "code"}) 
            elif task == "leetcode": 
                item.update({"prompt": row['prompt'], "gt": row.get('test',''), "type": "code"})
            
            data.append(item)
        except Exception:
            pass
            
    if limit > 0: data = data[:limit]
    return data

def estimate_pass_k(n, c, k):
    if n < k: return 0.0
    if c == 0: return 0.0
    if c == n: return 1.0
    prod = 1.0
    for i in range(k): prod *= (n - c - i) / (n - i)
    return 1.0 - prod

# ================= 6. 主程序 =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_alias", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--tasks", default="math500")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k_values", default="1,4,8,16")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--template_type", default="chat")
    parser.add_argument("--tp_size", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    k_list = sorted([int(x) for x in args.k_values.split(",")])
    max_k = max(k_list)

    print(f"🚀 Loading {args.model_alias} | Mem: {args.gpu_memory_utilization}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size, 
              trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    summary = []

    for task in args.tasks.split(","):
        detail_path = os.path.join(args.output_dir, f"{args.model_alias}_{task}_details.jsonl")
        
        if os.path.exists(detail_path):
            print(f"\n⏩ Found existing results for {task}, skipping...")
            try:
                df_old = pd.read_json(detail_path, lines=True)
                acc_g = df_old['greedy_ok'].mean() if 'greedy_ok' in df_old else 0.0
                
                # 如果 BBH 是 0%，说明上次提取失败，强制重跑
                if task == 'bbh' and acc_g == 0.0:
                    print("   [Reset] BBH score is 0.00%, assuming extraction failure. Re-running...")
                else:
                    pass_k_recovered = {}
                    if 'sample_correct_cnt' in df_old.columns:
                        for k in k_list:
                            pk_list = []
                            for _, row in df_old.iterrows():
                                pk_list.append(estimate_pass_k(max_k, row['sample_correct_cnt'], k))
                            pass_k_recovered[k] = np.mean(pk_list)
                    
                    row = {"model": args.model_alias, "task": task, "greedy": acc_g}
                    for k in k_list:
                        row[f"pass@{k}"] = pass_k_recovered.get(k, 0.0)
                    summary.append(row)
                    continue 
            except:
                pass

        try:
            data = load_local_data(task, args.data_root, args.limit)
            if not data: 
                print(f"⚠️ Skipping {task}: No data loaded.")
                continue
            
            print(f"\nEvaluating {task} ({len(data)} items)...")
            prompts = []
            for item in data:
                sys_msg = "You are a helpful assistant."
                if item['type'] == 'code': sys_msg = "Write Python code to solve the problem."
                if item['type'] == 'math': sys_msg = "Reason step by step and put answer in \\boxed{}."
                
                # [Prompt 优化] 兼容选项和布尔值
                if item['type'] == 'mcq' or item['type'] == 'match': 
                    sys_msg = "Think step by step. Finally, answer with the option letter like (A) or the answer phrase (e.g., True/False)."
                
                if args.template_type == 'chat':
                    if getattr(tokenizer, 'chat_template', None):
                        p = tokenizer.apply_chat_template([
                            {"role":"system","content":sys_msg},
                            {"role":"user","content":item['prompt']}
                        ], tokenize=False, add_generation_prompt=True)
                    else:
                        p = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    p = f"{sys_msg}\n\nQ: {item['prompt']}\nA:"
                prompts.append(p)

            # 1. Greedy Pass
            print("  > Greedy Pass...")
            out_g = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4096))
            
            greedy_correct = 0
            details = []
            for i, o in enumerate(out_g):
                res = o.outputs[0].text
                ok, msg = check_sample(res, data[i])
                if ok: greedy_correct += 1
                details.append({
                    "id": data[i]['id'],
                    "prompt": data[i]['prompt'], 
                    "ground_truth": data[i]['gt'],
                    "greedy_res": res,
                    "greedy_ok": ok,
                    "info": msg,
                    "sample_correct_cnt": 0 
                })
                
            acc_g = greedy_correct / len(data)
            print(f"    Greedy: {acc_g:.2%}")

            # 2. Sampling Pass
            print(f"  > Sampling N={max_k}...")
            out_s = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, n=max_k, max_tokens=4096))
            
            pass_k_scores = {k: [] for k in k_list}
            for i, o in enumerate(out_s):
                c_cnt = 0
                samples_log = []
                for j, sample in enumerate(o.outputs):
                    res_s = sample.text
                    ok_s, _ = check_sample(res_s, data[i])
                    if ok_s: c_cnt += 1
                    if j < 3: samples_log.append(res_s[:200] + "...")
                
                for k in k_list:
                    pass_k_scores[k].append(estimate_pass_k(max_k, c_cnt, k))
                
                details[i].update({
                    "sample_correct_cnt": c_cnt,
                    "samples_preview": samples_log
                })
            
            # Save
            pd.DataFrame(details).to_json(detail_path, orient='records', lines=True)
            print(f"    💾 Saved details to: {detail_path}")
            
            row = {"model": args.model_alias, "task": task, "greedy": acc_g}
            for k in k_list:
                avg = np.mean(pass_k_scores[k])
                row[f"pass@{k}"] = avg
                print(f"    Pass@{k}: {avg:.2%}")
            summary.append(row)
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {task}: {e}")
            import traceback
            traceback.print_exc()

    pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, f"{args.model_alias}_summary.csv"), index=False)

if __name__ == "__main__":
    main()