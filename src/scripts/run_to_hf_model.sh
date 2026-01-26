python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /opt/nas/p/achen/codes/EGPO/outputs/checkpoints/open-r1-math_Qwen_Qwen3-8B-Base_8gpu_0122_1225/global_step_560/actor \
    --target_dir /opt/nas/p/achen/codes/EGPO/outputs/checkpoints/open-r1-math_Qwen_Qwen3-8B-Base_8gpu_0122_1225/global_step_560/actor/huggingface