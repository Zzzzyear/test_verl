python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /opt/nas/p/achen/codes/EGPO/outputs/checkpoints/open-r1-math-pmtlth1024_Qwen_Qwen2.5-Math-7B_4gpu_0126_1503/global_step_720/actor \
    --target_dir /opt/nas/p/achen/codes/EGPO/outputs/checkpoints/open-r1-math-pmtlth1024_Qwen_Qwen2.5-Math-7B_4gpu_0126_1503/global_step_720/actor/huggingface