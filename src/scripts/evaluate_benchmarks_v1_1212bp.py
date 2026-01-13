import os
import argparse
import json
import re
import glob
import pandas as pd
import numpy as np
import subprocess
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ================= 1. 深度归一化工具 =================

def normalize_math_str(s):
    """
    [增强版] 数学公式归一化，对齐学术界标准 (MATH/GSM8K)
    """
    if not s: return ""
    s = str(s).strip()

    # 1. 基础清理
    s = s.replace("\n", " ").replace("\r", "")

    # 2. LaTeX 命令标准化
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")

    # 3. 移除所有空格 (数学表达式通常对空格不敏感)
    # 注意：在文本题中可能要保留，但在 Boxed Answer 中通常应移除
    s = s.replace(" ", "")

    # 4. 处理货币符号和单位
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("%", "")

    # 5. 科学计数法统一 (1.2e-3 -> 1.2e-3, 保持原样，依赖后续 float 尝试)
    return s


def is_equiv_math(pred, gt):
    """
    [严谨] 数学等价性判定
    1. 字符串 Exact Match (归一化后)
    2. 数值近似 (对于纯数字)
    """
    norm_pred = normalize_math_str(pred)
    norm_gt = normalize_math_str(gt)

    # 策略 A: 严格字符串匹配
    if norm_pred == norm_gt:
        return True

    # 策略 B: 数值转换匹配 (允许 1e-6 误差)
    try:
        # 移除常见干扰词再尝试转换
        def to_float(x):
            x = x.replace("{", "").replace("}", "").replace("\\", "")
            return float(x)

        if abs(to_float(norm_pred) - to_float(norm_gt)) < 1e-6:
            return True
    except:
        pass

    return False


# ================= 2. 答案提取器 (学术范式) =================

def clean_cot(text):
    """移除 <think> 标签及其内容"""
    if not text: return ""
    # non-greedy match for <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_boxed_content(text):
    """
    [修复] 基于计数器的 Boxed 提取，完美支持嵌套
    寻找最后一个 \boxed{...}
    """
    candidates = [m.start() for m in re.finditer(r"\\?boxed\s*\{", text)]
    if not candidates:
        return None

    # 取最后一个 boxed
    start_idx = candidates[-1]
    # 找到 { 的位置
    brace_start = text.find("{", start_idx)
    if brace_start == -1: return None

    balance = 0
    content = []
    started = False

    for i in range(brace_start, len(text)):
        char = text[i]
        if char == "{":
            balance += 1
            started = True
        elif char == "}":
            balance -= 1

        if started:
            content.append(char)
            if balance == 0:
                break

    if len(content) >= 2:  # 至少 "{}"
        return "".join(content[1:-1])  # 去掉外层 {}
    return None


def extract_math_answer(text):
    """Math 提取策略：优先 Boxed，兜底取最后一行"""
    text = clean_cot(text)

    # 1. 尝试 Boxed (Standard)
    boxed = extract_boxed_content(text)
    if boxed: return boxed

    # 2. 兜底：尝试提取最后一行数值或简短结论
    # 这在 GSM8K Base 模型中常见
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line: continue
        # 如果包含 "The answer is", 截取后面
        match = re.search(r"[Tt]he answer is[:\s]*(.+)", line)
        if match:
            return match.group(1).strip(" .")
        # 如果是很短的数字/公式
        if len(line) < 20 and any(c.isdigit() for c in line):
            return line
        break
    return ""


def extract_choice_answer(text):
    """
    [增强] MCQ 提取策略：显式声明 > Boxed > 文末字符
    支持中文 "答案是"
    """
    text = clean_cot(text)

    # 1. 显式声明 (High Priority) - 取第一个匹配 (防止后面的解释干扰)
    # 增加了中文支持
    patterns = [
        r"(?:The|the) answer is[:\s]*\(?([A-Ja-j])\)?",
        r"(?:The|the) answer is[:\s]*(True|False|TRUE|FALSE|Yes|No|YES|NO)",
        r"答案(?:是|为)[:\s]*\(?([A-Ja-j])\)?",
        r"选项[:\s]*\(?([A-Ja-j])\)?",
    ]
    for p in patterns:
        match = re.search(p, text)
        if match: return match.group(1).strip()

    # 2. Boxed
    boxed = extract_boxed_content(text)
    if boxed:
        boxed = boxed.strip()
        if len(boxed) == 1 and boxed.isalpha(): return boxed
        if boxed.lower() in ['true', 'false', 'yes', 'no']: return boxed

    # 3. 文末弱匹配 (Low Priority) - 取最后一个匹配
    # 仅在末尾 500 字符搜索
    last_part = text[-500:]
    match_letter = re.findall(r"\(?([A-Ja-j])\)", last_part)
    if match_letter: return match_letter[-1]

    return ""


def extract_code_block(text):
    """
    Code 提取策略：取第一个完整代码块 (学术标准)
    """
    text = clean_cot(text)

    # 优先 Python
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match: return match.group(1)

    # 其次通用代码块
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match: return match.group(1)

    # 兜底：如果没 Markdown，返回全文 (Base 模型可能直接输出代码)
    return text


# ================= 3. 代码执行沙箱 (增强版) =================

def run_python_io(code, test_input, timeout=3.0):
    """
    [新增] 针对 LCB/Codeforces 的 IO 模式执行
    通过 stdin 注入 input，捕获 stdout
    """
    try:
        # 使用 sys.executable 启动子进程
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=test_input, timeout=timeout)
        if process.returncode != 0:
            return False, f"Error: {stderr.strip()}"
        return True, stdout.strip()
    except subprocess.TimeoutExpired:
        process.kill()
        return False, "Timeout"
    except Exception as e:
        return False, f"System Error: {e}"


def run_python_script(full_script, timeout=3.0):
    """
    针对 HumanEval 的 Script 模式执行
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True, text=True, timeout=timeout
        )
        if "EXECUTION_PASSED" in result.stdout:
            return True, "Passed"
        else:
            err = result.stderr if result.stderr else result.stdout
            return False, f"Failed: {err.strip()[-500:]}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"System Error: {str(e)}"


# ================= 4. 判分路由 (核心) =================

def check_sample(response, item):
    task_type = item['type']

    # --- 1. Code IO (LCB, Codeforces) ---
    if task_type == 'code_io':
        code = extract_code_block(response)
        if not code.strip(): return False, "No code extracted"

        inputs = item.get('test_inputs', [])
        outputs = item.get('test_outputs', [])

        if not inputs or not outputs:
            # 如果没有测试用例，无法判分 (LCB常见情况)
            # 返回 False 但标记为 Skipped，方便分析
            return False, "Skipped (No test cases found in dataset)"

        # 这是一个简化版评测：只跑前 3 个 Case 防止太慢
        # 正式评测建议用专门的评测库
        passed_cnt = 0
        for inp, outp in zip(inputs[:3], outputs[:3]):
            # 格式化输入：如果是 list，通常 LCB 输入是多行的，或者是 JSON string
            # 这里做一个简单的 str 转换，具体视数据集格式而定
            input_str = str(inp)
            ok, model_out = run_python_io(code, input_str)

            # 宽松比对 (去掉空白)
            if ok and model_out.split() == str(outp).strip().split():
                passed_cnt += 1
            else:
                return False, f"WA: Expected '{outp}', Got '{model_out}'"

        return True, "Passed samples"

    # --- 2. Code Script (HumanEval) ---
    elif task_type == 'code_script':
        code = extract_code_block(response)
        entry = item.get('entry', 'None')
        test_code = item['gt']

        # 构造完整的测试脚本
        # 增加 assert 逻辑防止 entry_point 空转
        script = f"""
import sys
import math
import collections
import itertools
import random
import heapq
import functools
import re
from typing import *

{code}

{test_code}

try:
    # HumanEval 标准 Check
    if '{entry}' != 'None' and 'check' in globals():
        check({entry})
    # LeetCode 风格: 往往没有 check 函数，只有 assert
    # 所以只要上面代码没抛出异常，就算通过
    print("EXECUTION_PASSED")
except Exception as e:
    print(f"EXECUTION_FAILED: {{e}}")
"""
        return run_python_script(script)

    # --- 3. MCQ / Match ---
    elif task_type in ['mcq', 'match']:
        pred = extract_choice_answer(response)
        gt = str(item['gt']).strip().replace("(", "").replace(")", "")

        if not pred:
            # Fallback to math extraction just in case
            pred = extract_math_answer(response)

        is_correct = (pred.lower() == gt.lower())
        return is_correct, f"Pred: {pred} | GT: {gt}"

    # --- 4. Math ---
    else:  # math
        pred = extract_math_answer(response)
        gt = str(item['gt'])

        # 使用严格等价性判定
        is_correct = is_equiv_math(pred, gt)
        return is_correct, f"Pred: {pred} | GT: {gt}"


# ================= 5. 数据加载与路径 =================

LOCAL_DATA_MAP = {
    "math500": "{data_root}/MATH-500/test.jsonl",
    "aime24": "{data_root}/AIME-2024/data/*.parquet",
    "aime25": "{data_root}/AIME-2025/aime2025-*.jsonl",
    "olympiad": "{data_root}/OlympiadBench/OlympiadBench/OE_TO_maths_en_COMP/*.parquet",
    "gpqa": "{data_root}/GPQA-Diamond/test/gpqa_diamond.parquet",
    # BBH Top-3
    "bbh": [
        "{data_root}/big_bench_hard/logical_deduction_seven_objects/*.parquet",
        "{data_root}/big_bench_hard/date_understanding/*.parquet",
        "{data_root}/big_bench_hard/boolean_expressions/*.parquet"
    ],
    "humaneval": "{data_root}/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
    "lcb": "{data_root}/LiveCodeBench/test5.jsonl",
    "leetcode": "{data_root}/LeetCodeDataset/LeetCodeDataset-test.jsonl"
}


def load_local_data(task, data_root, limit=-1):
    raw_template = LOCAL_DATA_MAP.get(task)
    if not raw_template:
        print(f"Unknown task: {task}")
        return []

    if isinstance(raw_template, list):
        path_list = [p.format(data_root=data_root) for p in raw_template]
    else:
        path_list = [raw_template.format(data_root=data_root)]

    files = []
    for full_path in path_list:
        if '*' in full_path:
            files.extend(sorted(glob.glob(full_path, recursive=True)))
        elif os.path.exists(full_path):
            files.append(full_path)

    if not files:
        print(f"❌ No files found for {task}")
        return []

    dfs = []
    for f in files:
        try:
            if f.endswith('.parquet'):
                dfs.append(pd.read_parquet(f))
            elif f.endswith('.jsonl'):
                dfs.append(pd.read_json(f, lines=True))
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    if not dfs: return []
    df = pd.concat(dfs, ignore_index=True)

    data = []
    for idx, row in df.iterrows():
        try:
            item = {"id": f"{task}_{idx}", "type": "unknown", "gt": ""}

            # --- Math ---
            if task == "math500":
                item.update({"prompt": row['problem'], "gt": row['solution'], "type": "math"})
            elif task == "aime24":
                item.update({"prompt": row['problem'], "gt": row['solution'], "type": "math"})
            elif task == "aime25":
                item.update({"prompt": row['question'], "gt": str(row['answer']), "type": "math"})
            elif task == "olympiad":
                if row.get('question_type') != 'Open-ended': continue
                gt = row['solution']
                if isinstance(row.get('final_answer'), list) and len(row['final_answer']) > 0:
                    gt = row['final_answer'][0]
                item.update({"prompt": row['question'], "gt": gt, "type": "math"})

            # --- MCQ ---
            elif task == "gpqa":
                item.update({"prompt": row['Question'], "gt": row['Correct Answer'], "type": "mcq"})
            elif task == "bbh":
                item.update({"prompt": row['question'], "gt": row['target'], "type": "match"})

            # --- Code ---
            elif task == "humaneval":
                item.update(
                    {"prompt": row['prompt'], "gt": row['test'], "entry": row['entry_point'], "type": "code_script"})

            # [LBC IO Fix]
            elif task == "lcb":
                # 尝试解析 Input/Output
                # LCB 数据格式各异，这里假设有 public_test_cases 字段 (常见格式)
                inputs, outputs = [], []
                if 'public_test_cases' in row:
                    try:
                        # 假设它是 JSON 字符串或者已经是 list
                        cases = row['public_test_cases']
                        if isinstance(cases, str): cases = json.loads(cases)
                        for c in cases:  # c is dict {input:..., output:...}
                            inputs.append(c.get('input', ''))
                            outputs.append(c.get('output', ''))
                    except:
                        pass

                item.update({
                    "prompt": row['question_content'],
                    "gt": "io_check",
                    "type": "code_io",
                    "test_inputs": inputs,
                    "test_outputs": outputs
                })

            elif task == "leetcode":
                item.update({"prompt": row['prompt'], "gt": row.get('test', ''), "type": "code_script"})

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


# ================= 6. 多进程评估 Wrapper =================

def process_single_item(args):
    """用于多进程的 worker 函数"""
    response, item = args
    ok, msg = check_sample(response, item)
    return ok, msg


# ================= 7. 主程序 =================

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
        print("Warning: Using GPT2 tokenizer fallback")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    summary = []

    for task in args.tasks.split(","):
        detail_path = os.path.join(args.output_dir, f"{args.model_alias}_{task}_details.jsonl")

        # 断点续传：简单的跳过逻辑
        if os.path.exists(detail_path):
            print(f"⏩ Skipping {task}, file exists: {detail_path}")
            # 这里简单处理，如果需要合并 summary 需额外读取
            continue

        try:
            data = load_local_data(task, args.data_root, args.limit)
            if not data:
                print(f"⚠️ Skipping {task}: No data loaded.")
                continue

            print(f"\nEvaluating {task} ({len(data)} items)...")
            prompts = []
            for item in data:
                sys_msg = "You are a helpful assistant."
                if item['type'] in ['code_script', 'code_io']:
                    sys_msg = "Write Python code to solve the problem. Wrap code in ```python ... ```."
                if item['type'] == 'math':
                    sys_msg = "Reason step by step. Finally, enclose the answer in \\boxed{}."
                if item['type'] in ['mcq', 'match']:
                    sys_msg = "Think step by step. Finally, answer with the option letter (e.g., (A)) or statement."

                # Chat Template 构造
                if getattr(tokenizer, 'chat_template', None):
                    p = tokenizer.apply_chat_template([
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": item['prompt']}
                    ], tokenize=False, add_generation_prompt=True)
                else:
                    p = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(p)

            # ----------------------------------------------------
            # 1. Greedy Pass
            # ----------------------------------------------------
            print("  > Greedy Inference...")
            out_g = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4096))

            greedy_res_list = [o.outputs[0].text for o in out_g]

            # 并行判分
            print("  > Grading Greedy...")
            details = []
            greedy_correct = 0

            # 构造任务列表
            grade_tasks = list(zip(greedy_res_list, data))

            with ProcessPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
                results = list(tqdm(executor.map(process_single_item, grade_tasks), total=len(grade_tasks)))

            for i, (ok, msg) in enumerate(results):
                if ok: greedy_correct += 1
                details.append({
                    "id": data[i]['id'],
                    "prompt": data[i]['prompt'],
                    "gt": str(data[i]['gt'])[:100],  # 截断一下防止过大
                    "greedy_res": greedy_res_list[i],
                    "greedy_ok": ok,
                    "info": msg
                })

            acc_g = greedy_correct / len(data)
            print(f"    Greedy: {acc_g:.2%}")

            # ----------------------------------------------------
            # 2. Sampling Pass
            # ----------------------------------------------------
            print(f"  > Sampling N={max_k}...")
            out_s = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, n=max_k, max_tokens=4096))

            pass_k_scores = {k: [] for k in k_list}

            print("  > Grading Samples...")
            # 由于数据量大，这里需要把 flatten 后的 sample 都送入 executor
            # 结构: [ (sample_text, item), ... ]
            flat_tasks = []
            task_indices = []  # 记录属于哪个题目

            for i, o in enumerate(out_s):
                for sample in o.outputs:
                    flat_tasks.append((sample.text, data[i]))
                    task_indices.append(i)

            with ProcessPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
                flat_results = list(tqdm(executor.map(process_single_item, flat_tasks), total=len(flat_tasks)))

            # 聚合结果
            sample_correct_counts = [0] * len(data)
            for idx, (ok, _) in zip(task_indices, flat_results):
                if ok: sample_correct_counts[idx] += 1

            # 计算 Pass@K
            for i, c_cnt in enumerate(sample_correct_counts):
                for k in k_list:
                    pass_k_scores[k].append(estimate_pass_k(max_k, c_cnt, k))

                # 更新 details
                details[i].update({
                    "sample_correct_cnt": c_cnt,
                    "samples_preview": [flat_tasks[j][0][:100] for j in range(len(flat_tasks)) if task_indices[j] == i][
                                       :3]
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
    # 防止多进程死锁
    multiprocessing.set_start_method('spawn', force=True)
    main()