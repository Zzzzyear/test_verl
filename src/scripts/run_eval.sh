python scripts/eval_parallel.py \
 --model-path /opt/nas/p/achen/open_models/Qwen_Qwen2.5-Math-1.5B \
 --model-name Qwen_Qwen2.5-Math-1.5B \
 --dataset aime25 \
 --gpu-ids 0 1 2 3 4 5 6 7 \
 --tensor-parallel-size 1 \
 --vllm-conda-env ac-egpo \
 --eval-conda-env ac-eval \
 --base-port 8100 \
 --startup-timeout 10000 \
 --generation-config '{"max_tokens": 3000,"temperature":0.6,"top_p":0.95,"top_k":20,"n":1,"seed":42}'


python scripts/eval_parallel.py \
 --model-path /opt/nas/p/achen/open_models/Qwen_Qwen2.5-Math-7B \
 --model-name Qwen_Qwen2.5-Math-7B \
 --dataset aime25 \
 --gpu-ids 0 1 2 3 4 5 6 7 \
 --tensor-parallel-size 1 \
 --vllm-conda-env ac-egpo \
 --eval-conda-env ac-eval \
 --base-port 8100 \
 --startup-timeout 10000 \
 --generation-config '{"max_tokens": 3000,"temperature":0.6,"top_p":0.95,"top_k":20,"n":1,"seed":42}'



python scripts/eval_parallel.py \
 --model-path /opt/nas/p/achen/open_models/Qwen_Qwen3-8B-Base \
 --model-name Qwen_Qwen3-8B-Base \
 --dataset aime25 \
 --gpu-ids 0 1 2 3 4 5 6 7 \
 --tensor-parallel-size 1 \
 --vllm-conda-env ac-egpo \
 --eval-conda-env ac-eval \
 --base-port 8100 \
 --startup-timeout 10000