from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#model_path = "/home/jupyter/MBPP/qwen2.5-final"  # or checkpoint-3
model_path = "/home/jupyter/MBPP/FineTunedModels/qwen2.5-easy-heavy/v2-20250707-182332/checkpoint-48"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")

prompt = """
Please solve the following Python programming problem:
Write a function to find length of the string.

Please provide a complete Python function that solves this problem. Write only the function code without any explanations or comments. 

"""

# Format prompt using chat template
messages = [
    {"role": "user", "content": prompt}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)

# Decode full text
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(decoded)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
code = decoded.split('\nassistant\n')
print(code[-1])

