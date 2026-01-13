# EGPO: Entropy-Guided Policy Optimization for LLM Reasoning

**EGPO (Entropy-Guided Policy Optimization)** is a novel reinforcement learning algorithm designed to calibrate Large Language Models' (LLMs) confidence during reasoning. It addresses the "马虎错" and "真不会" problems by dynamically scaling the advantage function based on the entropy of the generated response.

This folder implements EGPO by performing an **invasive modification** on the [Verl](https://github.com/volcengine/verl) framework, elevating EGPO to a first-class citizen alongside PPO and GRPO.

---

## 🧠 Core Methodology

The core idea of EGPO is to reward models that are "confident and correct" while penalizing those that are "blindly confident" or "hesitant."

### Mathematical Formulation
The advantage function is modified as follows:

$$
A_{final}^{(i)} = A_{GRPO}^{(i)} \cdot \text{Clip}\left(\frac{\bar{H}_{group}}{H_i + \epsilon}, \lambda_{min}, \lambda_{max}\right)
$$

Where:
- $A_{GRPO}^{(i)}$: The standard outcome-based advantage (normalized).
- $H_i$: The entropy of the $i$-th response (specific to the **Answer** or **Thinking** segment).
- $\bar{H}_{group}$: The average entropy of the group (N samples).
- $\lambda_{min}, \lambda_{max}$: Clipping thresholds to prevent gradient explosion.

---

## 🛠️ Implementation Details

We implemented EGPO by deeply integrating it into the `verl` framework's core. The modifications ensure robust data flow, correct configuration parsing, and comprehensive monitoring.

### 1. Configuration Layer
* **Files**: 
    * `verl/trainer/config/algorithm.py`
    * `verl/trainer/config/ppo_trainer.yaml`
* **Action**: 
    * Defined the `EgpoConfig` dataclass in Python.
    * Explicitly added the `egpo` config block to the default YAML file to satisfy Hydra's strict structural validation.
* **Parameters**:
    * `entropy_mode`: `"answer"` (default) or `"thinking"`.
    * `lambda_min` / `lambda_max`: Dynamic weight clipping range.
    * `entropy_epsilon`: Stability term.

### 2. Core Algorithm
* **File**: `verl/trainer/ppo/core_algos.py`
* **Action**: 
    * **First-Class Support**: Registered `EGPO` in the `AdvantageEstimator` enum.
    * **Logic Implementation**: Implemented `compute_egpo_advantage` to calculate token-level entropy and apply the scaling formula.
    * **Metric Exposure**: Updated the function to return a dictionary of debug metrics (`egpo/weight/mean`, etc.) alongside advantages.
    * **Robustness**: Added type checks to handle `index` grouping variables regardless of whether they are PyTorch Tensors or Numpy arrays.

### 3. Data Pipeline & Trainer
* **File**: `verl/trainer/ppo/ray_trainer.py`
* **Action**: 
    * **Signature Update**: Modified `compute_advantage` to accept `tokenizer`.
    * **Explicit Branching**: Added a dedicated `elif adv_estimator == AdvantageEstimator.EGPO:` branch.
    * **Data Flow**: Explicitly extracted `input_ids` and `old_log_probs` from the batch and passed them to the core algorithm.
    * **Metric Propagation**: Updated the training loop (`fit`) to unpack the returned metrics and log them to WandB/Console.

### 4. Entropy Utility
* **File**: `verl/utils/entropy.py` (New)
* **Action**: Implemented `EntropyMaskGenerator`. It uses the tokenizer to locate `<think>` and `</think>` tags, generating precise masks to isolate specific reasoning segments for entropy calculation.

### 5. Hybrid Reward System
* **Files**: 
    * `verl/experimental/reward/reward_loop/hybrid.py` (Worker Side)
    * `verl/workers/reward_manager/hybrid.py` (Driver Side)
* **Action**: 
    * **Unified Logic**: Implemented reward calculation that supports mixed datasets (Math + Code) via the `ability` metadata field.
    * **Type Robustness**: Implemented a recursive flattening function (`to_flat_int_list`) to force-convert nested Lists, Tensors, or Numpy arrays into flat Python `int` lists. This resolves `TypeError` crashes during tokenizer decoding.
    * **Numpy Safety**: Added checks to handle cases where `extra_info` is wrapped inside a 0-d Numpy array.

---

## 🚀 Quick Start

### 1. Environment Setup
Ensure you have the modified `verl1` environment installed.
```bash
conda activate verl1
```

### 2. Data Preparation
Generate the debug dataset with mixed Math and Code samples:
```bash
python3 src/scripts/gen_mixed_debug.py
```
This will create datasets/processed/mixed_debug.parquet.

### 3. Run Training (Dry Run)
Use the provided shell script to start a training run with EGPO enabled.
```bash
#!/bin/bash
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF

# Configuration
DATA_DIR="/data/zhaoqn/workspace/EGPO/datasets/processed"
MODEL_PATH="/data/zhaoqn/models/Qwen/Qwen3-1.7B"

# Enable WandB (Optional but recommended)
export WANDB_PROJECT="EGPO_DryRun"
export WANDB_MODE="online"

python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/mixed_debug.parquet" \
    data.val_files="$DATA_DIR/mixed_debug.parquet" \
    data.train_batch_size=16 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    algorithm.adv_estimator=egpo \
    algorithm.egpo.entropy_mode=answer \
    algorithm.egpo.lambda_min=0.5 \
    algorithm.egpo.lambda_max=2.0 \
    reward_model.reward_manager=hybrid \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='EGPO_Test' \
    trainer.experiment_name='dry_run_v1' \
    trainer.total_training_steps=5
```

## 📊 Monitoring
The training logs will show specific metrics for EGPO:

- actor/entropy: The average entropy of the generated responses.

- actor/pg_loss: Policy gradient loss (modified by EGPO weights).

- critic/score: The average reward score from the Hybrid Reward Manager.

- egpo/weight/mean: Average scaling factor applied to advantages.

- egpo/weight/std: Standard deviation of weights (measures differentiation between confident/hesitant samples).

- egpo/entropy/seq_mean: Average entropy of the generated responses.

- egpo/ratio/max: The maximum raw scaling ratio before clipping.

## 📂 Project Structure (Modified Files Only)

```plaintext
verl/
├── experimental/reward/reward_loop/
│   └── hybrid.py            # [NEW] Worker-side reward calculation (Math+Code) with type fixes
├── trainer/
│   ├── config/
│   │   ├── algorithm.py     # [MOD] Added EgpoConfig dataclass
│   │   └── ppo_trainer.yaml # [MOD] Registered egpo config struct for Hydra
│   ├── ppo/
│   │   ├── core_algos.py    # [MOD] Implemented compute_egpo_advantage & metric return
│   │   └── ray_trainer.py   # [MOD] Passed tokenizer/input_ids & unpacked metrics
├── utils/
│   └── entropy.py           # [NEW] Mask generator for thinking/answer segments
└── workers/reward_manager/
    └── hybrid.py            # [NEW] Driver-side reward manager for validation
```

## ⚠️ Notes
Tokenizer: Ensure your tokenizer correctly handles <think> and </think> tags. The EntropyMaskGenerator attempts to find their IDs; if not found, it falls back to treating the entire response as the answer.

OOM Killer: If you encounter Killed errors at the very end of the script (after "Final validation metrics"), it is likely due to Ray's cleanup process releasing resources. Checkpoints are usually saved successfully before this happens.