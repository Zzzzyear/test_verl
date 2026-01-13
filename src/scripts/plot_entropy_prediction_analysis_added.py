# 文件路径: /data-store/zhaoqiannian/workspace/EGPO/src/scripts/plot_entropy_prediction_analysis_added.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

# 设置无头模式，防止服务器报错
import matplotlib
matplotlib.use('Agg')

def plot_roc_comparison(df, output_dir, model_name):
    """
    图1：ROC 曲线对比 (Thinking vs Answer)
    目的：直观展示哪个熵更能区分"错误"和"正确"。
    """
    plt.figure(figsize=(8, 8))

    # 目标：预测样本是否是"错误"的 (Incorrect)
    # 假设：熵越高，越可能是错误的
    y_true = (~df['is_correct']).astype(int)

    # 1. Answer Entropy ROC
    if 'answer_entropy' in df.columns:
        fpr_ans, tpr_ans, _ = roc_curve(y_true, df['answer_entropy'].fillna(0))
        roc_auc_ans = auc(fpr_ans, tpr_ans)
        plt.plot(fpr_ans, tpr_ans, color='#1f77b4', label=f'Answer Entropy (AUC = {roc_auc_ans:.3f})', lw=2.5)

    # 2. Thinking Entropy ROC
    if 'thinking_entropy' in df.columns:
        fpr_think, tpr_think, _ = roc_curve(y_true, df['thinking_entropy'].fillna(0))
        roc_auc_think = auc(fpr_think, tpr_think)
        plt.plot(fpr_think, tpr_think, color='#ff7f0e', linestyle='--', label=f'Thinking Entropy (AUC = {roc_auc_think:.3f})', lw=2.5)

    # 基准线 (随机猜测)
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5, alpha=0.6, label='Random Guess (AUC = 0.5)')
    
    plt.xlabel('False Positive Rate (误报率)', fontsize=12)
    plt.ylabel('True Positive Rate (召回率)', fontsize=12)
    plt.title(f'ROC Comparison: Predicting "Incorrect" Answers\nModel: {model_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"{model_name}_roc_comparison_added.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Generated] ROC Comparison: {save_path}")
    plt.close()

def plot_length_bias_thinking(df, output_dir, model_name):
    """
    图2：思考长度 vs 思考熵 (Thinking Length Bias Analysis)
    目的：分析 Thinking Entropy 是否只是单纯受长度影响，以及红绿点是否混杂。
    """
    plt.figure(figsize=(10, 8))
    
    # 转换标签用于图例
    df['Label'] = df['is_correct'].map({True: 'Correct', False: 'Incorrect'})

    # 绘制散点
    sns.scatterplot(
        data=df, 
        x='thinking_tokens', 
        y='thinking_entropy',
        hue='Label', 
        style='Label',
        palette={'Correct': '#2ca02c', 'Incorrect': '#d62728'}, # 绿对红错
        alpha=0.6, 
        s=40
    )
    
    plt.title(f"Thinking Process: Length vs. Entropy Analysis\n({model_name})", fontsize=14)
    plt.xlabel("Thinking Process Length (Tokens)", fontsize=12)
    plt.ylabel("Thinking Entropy (NLL)", fontsize=12)
    plt.legend(title="Outcome")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"{model_name}_scatter_len_bias_thinking_added.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Generated] Thinking Scatter: {save_path}")
    plt.close()

def main():
    # 配置区
    csv_path = "/data-store/zhaoqiannian/workspace/EGPO/outputs/analysis/full_experiment_b1_flexible/Qwen3-8B_metrics.csv"
    output_dir = os.path.dirname(csv_path)
    model_name = "Qwen3-8B"

    # 1. 加载数据
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 数据清洗：确保关键列不为空
    required_cols = ['thinking_entropy', 'answer_entropy', 'is_correct', 'thinking_tokens']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing columns. Found: {df.columns}")
        return
        
    df_clean = df.dropna(subset=required_cols)
    print(f"Original samples: {len(df)}, Valid samples: {len(df_clean)}")

    # 2. 打印统计信息
    total = len(df_clean)
    correct_count = len(df_clean[df_clean['is_correct']==True])
    incorrect_count = len(df_clean[df_clean['is_correct']==False])
    
    print("\n" + "="*40)
    print(f"DATA STATISTICS FOR {model_name}")
    print("="*40)
    print(f"Total Samples    : {total}")
    print(f"Correct Answers  : {correct_count} ({correct_count/total:.2%})")
    print(f"Incorrect Answers: {incorrect_count} ({incorrect_count/total:.2%})")
    print("-" * 40)
    print(f"{'Metric':<20} | {'Correct (Mean)':<15} | {'Incorrect (Mean)':<15}")
    print("-" * 40)
    print(f"{'Answer Entropy':<20} | {df_clean[df_clean['is_correct']==True]['answer_entropy'].mean():.4f}          | {df_clean[df_clean['is_correct']==False]['answer_entropy'].mean():.4f}")
    print(f"{'Thinking Entropy':<20} | {df_clean[df_clean['is_correct']==True]['thinking_entropy'].mean():.4f}          | {df_clean[df_clean['is_correct']==False]['thinking_entropy'].mean():.4f}")
    print("="*40 + "\n")

    # 3. 绘制补救图表
    # 设置风格
    sns.set_theme(style="whitegrid")
    
    # 图 1: ROC 对比 (Thinking vs Answer)
    plot_roc_comparison(df_clean, output_dir, model_name)
    
    # 图 2: Thinking Length vs Entropy 散点图
    plot_length_bias_thinking(df_clean, output_dir, model_name)

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()