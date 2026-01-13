import os
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    print(f"🕵️ Probing model: {args.model_path}")

    # 1. 加载模型
    llm = LLM(model=args.model_path, 
              trust_remote_code=True, 
              gpu_memory_utilization=0.7,
              enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 2. 定义一个需要推理的问题
    # 这个问题如果不思考直接回答，很容易错（比如直接回答 9.11）
    question = "Which number is larger, 9.11 or 9.8?"

    # ================= 模式 A: Chat Template =================
    try:
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        print("\n" + "="*20 + " MODE A: Chat Template " + "="*20)
        print(f"Input Prompt:\n{repr(chat_prompt)}")
    except Exception as e:
        print(f"\n[!] Chat template failed (Model might not have one): {e}")
        chat_prompt = None

    # ================= 模式 B: Base Completion =================
    base_prompt = f"Question: {question}\nAnswer:"
    print("\n" + "="*20 + " MODE B: Base Completion " + "="*20)
    print(f"Input Prompt:\n{repr(base_prompt)}")

    # 3. 生成
    prompts = []
    if chat_prompt: prompts.append(chat_prompt)
    prompts.append(base_prompt)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    outputs = llm.generate(prompts, sampling_params)

    # 4. 结果对比
    print("\n" + "="*30 + " RESULTS " + "="*30)
    
    current_idx = 0
    if chat_prompt:
        out = outputs[current_idx].outputs[0].text
        print(f"\n🟢 [Chat Mode Output]:\n{out}")
        current_idx += 1
    
    out = outputs[current_idx].outputs[0].text
    print(f"\n🔵 [Base Mode Output]:\n{out}")

if __name__ == "__main__":
    main()