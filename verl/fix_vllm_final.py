import os

file_path = "/data/zhaoqn/workspace/EGPO/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"

print(f"Fixing {file_path}...")
with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # 修复 process_weights_after_loading 导入
    if "from vllm.model_executor.model_loader.utils import process_weights_after_loading" in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f"{indent}try:\n")
        new_lines.append(f"{indent}    from vllm.model_executor.model_loader.utils import process_weights_after_loading\n")
        new_lines.append(f"{indent}except ImportError:\n")
        new_lines.append(f"{indent}    try:\n")
        new_lines.append(f"{indent}        # Try alternative location for older/newer vLLM versions\n")
        new_lines.append(f"{indent}        from vllm.model_executor.model_loader.loader import process_weights_after_loading\n")
        new_lines.append(f"{indent}    except ImportError:\n")
        new_lines.append(f"{indent}        # Define dummy function if not found\n")
        new_lines.append(f"{indent}        def process_weights_after_loading(*args, **kwargs): pass\n")
        continue

    # 修复 CompilationConfig 导入 (以防万一)
    if "from vllm.config import CompilationConfig" in line:
         indent = line[:len(line) - len(line.lstrip())]
         new_lines.append(f"{indent}try:\n")
         new_lines.append(f"{indent}    from vllm.config import CompilationConfig\n")
         new_lines.append(f"{indent}except ImportError:\n")
         new_lines.append(f"{indent}    CompilationConfig = None\n")
         continue

    new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("✅ vLLM compatibility patch applied successfully.")
