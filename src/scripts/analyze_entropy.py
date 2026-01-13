import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# 设置无头模式绘图，防止在服务器上报错
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--n_return", type=int, default=8)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    return parser.parse_args()

def extract_answer_content(text):
    """
    提取 \boxed{...} 内部的内容，支持嵌套括号。
    """
    if not text: return None
    candidates = [m.start() for m in re.finditer(r"\\?boxed\s*\{", text)]
    if not candidates: return None
    
    start_idx = candidates[-1]
    brace_start = text.find("{", start_idx)
    if brace_start == -1: return None

    content = ""
    balance = 0
    started = False
    for char in text[brace_start:]:
        if char == "{":
            balance += 1
            started = True
        elif char == "}":
            balance -= 1
        
        if started:
            content += char
            if balance == 0:
                break
    
    if content.startswith("{") and content.endswith("}"):
        return content[1:-1].strip()
    return None

def check_correctness(pred_str, gt_str):
    """判题逻辑"""
    if not pred_str or not gt_str: return False
    def normalize(s):
        s = str(s).strip().replace(" ", "").replace("\n", "")
        s = s.replace(r"\left", "").replace(r"\right", "")
        s = s.replace(r"\\", "\\")
        return s
    return normalize(pred_str) == normalize(gt_str)

def get_token_logprobs(logprobs_list):
    """提取 logprob 数值"""
    vals = []
    for step in logprobs_list:
        if step:
            tid = list(step.keys())[0]
            if hasattr(step[tid], 'logprob'):
                vals.append(step[tid].logprob)
            elif hasattr(step[tid], 'log_prob'):
                vals.append(step[tid].log_prob)
            else:
                vals.append(step[tid])
    return vals

def split_logprobs(token_ids, logprobs_list, tokenizer):
    """
    分割 Thinking 和 Answer，返回对应的 logprobs 列表
    """
    full_text = tokenizer.decode(token_ids)
    vals = get_token_logprobs(logprobs_list)
    
    split_index = -1
    
    # 策略 1: 查找 </think>
    if "</think>" in full_text:
        think_part = full_text.split("</think>")[0] + "</think>"
        think_tokens = tokenizer.encode(think_part, add_special_tokens=False)
        split_index = len(think_tokens)
    # 策略 2: 查找 \boxed
    elif "boxed" in full_text:
        match = re.search(r"\\?boxed", full_text)
        if match:
            ratio = match.start() / len(full_text)
            split_index = int(len(vals) * ratio)
    
    if split_index != -1 and split_index < len(vals):
        thinking_vals = vals[:split_index]
        answer_vals = vals[split_index:]
    else:
        # Fallback: 找不到分割点时，为了数据安全，假设大部分是 Thinking
        # (通常数学题 CoT 很长，答案很短)
        split_point = max(0, len(vals) - 15) # 假设最后15个token是答案相关
        thinking_vals = vals[:split_point]
        answer_vals = vals[split_point:]
        
    return thinking_vals, answer_vals

def calculate_entropy(logprob_vals):
    """计算 NLL (Entropy Proxy)"""
    if not logprob_vals: return np.nan
    return -np.mean(logprob_vals)

def print_statistics(df, model_name):
    """打印详细统计信息到控制台和日志"""
    print(f"\n{'='*20} Statistics for {model_name} {'='*20}")
    
    total = len(df)
    correct = df[df['is_correct'] == True]
    incorrect = df[df['is_correct'] == False]
    
    acc = len(correct) / total if total > 0 else 0
    
    print(f"Total Samples: {total}")
    print(f"Correct:       {len(correct)} ({acc:.2%})")
    print(f"Incorrect:     {len(incorrect)} ({1-acc:.2%})")
    
    print("-" * 50)
    print(f"{'Metric':<20} | {'Correct (Mean±Std)':<20} | {'Incorrect (Mean±Std)':<20}")
    print("-" * 50)
    
    metrics = [
        ('answer_entropy', 'Ans Entropy'),
        ('thinking_entropy', 'Think Entropy'),
        ('answer_tokens', 'Ans Length'),
        ('thinking_tokens', 'Think Length')
    ]
    
    for col, name in metrics:
        if col in df.columns:
            c_mean = correct[col].mean()
            c_std = correct[col].std()
            i_mean = incorrect[col].mean()
            i_std = incorrect[col].std()
            print(f"{name:<20} | {c_mean:.4f} ± {c_std:.4f}   | {i_mean:.4f} ± {i_std:.4f}")
            
    print("=" * 60 + "\n")

def plot_analysis(df, output_dir, model_name):
    """绘制全套分析图表"""
    required_cols = ['thinking_entropy', 'answer_entropy']
    df = df.dropna(subset=required_cols)
    if df.empty: return

    df['Label'] = df['is_correct'].map({True: 'Correct', False: 'Incorrect'})
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # -----------------------------
    # 图 1: 熵分布图 (KDE)
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.kdeplot(data=df, x='thinking_entropy', hue='Label', fill=True, 
                palette={'Correct': 'green', 'Incorrect': 'red'}, ax=axes[0], alpha=0.3, warn_singular=False)
    axes[0].set_title(f"Thinking Process Entropy Distribution\n({model_name})")
    
    sns.kdeplot(data=df, x='answer_entropy', hue='Label', fill=True,
                palette={'Correct': 'green', 'Incorrect': 'red'}, ax=axes[1], alpha=0.3, warn_singular=False)
    axes[1].set_title(f"Final Answer Entropy Distribution\n({model_name})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_dist_comparison.png"), dpi=300)
    plt.close()
    
    # -----------------------------
    # 图 2: ROC 曲线 (双向)
    # -----------------------------
    plt.figure(figsize=(8, 8))
    y_true_error = (~df['is_correct']).astype(int) # 1=Incorrect
    y_true_correct = df['is_correct'].astype(int)  # 1=Correct
    
    if len(y_true_error.unique()) > 1:
        # Curve 1: Entropy Predicts Error (越高越错)
        fpr, tpr, _ = roc_curve(y_true_error, df['answer_entropy'].fillna(0))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='red', label=f'Predict Error (by High Entropy) AUC={roc_auc:.3f}', lw=2)
        
        # Curve 2: Negative Entropy Predicts Correctness (越低越对)
        # 数学上 AUC(Predict Correct) = AUC(Predict Error) 如果使用 -Score
        # 但画出来展示给读者看更直观
        fpr_c, tpr_c, _ = roc_curve(y_true_correct, -df['answer_entropy'].fillna(0))
        roc_auc_c = auc(fpr_c, tpr_c)
        plt.plot(fpr_c, tpr_c, color='green', linestyle='--', label=f'Predict Correct (by Low Entropy) AUC={roc_auc_c:.3f}', lw=2)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Analysis: Answer Entropy vs Correctness\n({model_name})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{model_name}_roc.png"), dpi=300)
        plt.close()
        
    # -----------------------------
    # 图 3: 散点图 (Thinking vs Answer Entropy)
    # -----------------------------
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='answer_entropy', y='thinking_entropy', 
                    hue='Label', style='Label', palette={'Correct': 'green', 'Incorrect': 'red'},
                    alpha=0.6, s=30)
    plt.title(f"Entropy Correlation: Thinking vs Answer\n({model_name})")
    plt.xlabel("Answer Entropy")
    plt.ylabel("Thinking Entropy")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{model_name}_scatter_entropy.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 图 4: 长度 vs 熵 (排除长度偏见)
    # -----------------------------
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='answer_tokens', y='answer_entropy', 
                    hue='Label', palette={'Correct': 'green', 'Incorrect': 'red'},
                    alpha=0.6)
    plt.title(f"Answer Length vs. Entropy\n({model_name})")
    plt.xlabel("Answer Length (Tokens)")
    plt.ylabel("Answer Entropy")
    plt.savefig(os.path.join(output_dir, f"{model_name}_scatter_len_bias.png"), dpi=300)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    if args.sample_size > 0 and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)
        
    prompts, gts = [], []
    for _, row in df.iterrows():
        p = next((x['content'] for x in row['prompt'] if x['role'] == 'user'), "")
        r = next((x['content'] for x in row['response'] if x['role'] == 'assistant'), "")
        gt = extract_answer_content(r)
        if gt:
            prompts.append(p)
            gts.append(gt)
            
    print(f"Eval samples: {len(prompts)} / {len(df)}")
    if len(prompts) == 0:
        print("Error: No valid samples found!")
        return

    # 初始化模型
    print(f"Loading Model: {args.model_name}...")
    os.environ['VLLM_USE_V1'] = '1'
    # 使用 0.6 利用率以防 OOM
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size, 
              gpu_memory_utilization=0.6, trust_remote_code=True, enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    sampling_params = SamplingParams(n=args.n_return, temperature=1.0, max_tokens=args.max_tokens, logprobs=1)
    chat_prompts = [tokenizer.apply_chat_template([{"role":"user", "content":p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    
    print("Generating...")
    outputs = llm.generate(chat_prompts, sampling_params)

    records = []
    print("Processing outputs...")
    for i, output in enumerate(tqdm(outputs)):
        gt = gts[i]
        for sample in output.outputs:
            pred = extract_answer_content(sample.text)
            is_correct = check_correctness(pred, gt)
            
            think_vals, ans_vals = split_logprobs(sample.token_ids, sample.logprobs, tokenizer)
            
            records.append({
                "prompt_id": i,
                "is_correct": is_correct,
                "thinking_entropy": calculate_entropy(think_vals),
                "answer_entropy": calculate_entropy(ans_vals),
                "thinking_tokens": len(think_vals),
                "answer_tokens": len(ans_vals)
            })

    res_df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, f"{args.model_name}_metrics.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # 打印统计并绘图
    print_statistics(res_df, args.model_name)
    plot_analysis(res_df, args.output_dir, args.model_name)

if __name__ == "__main__":
    main()