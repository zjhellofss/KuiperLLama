# 此文件主要用来比对 c++/cuda 推理的正确性
from transformers import AutoModelForCausalLM, AutoTokenizer

model = "Qwen/Qwen2.5-0.5B" # 也可以填本地模型绝对路径

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model, device_map="auto", trust_remote_code=True
).eval()

print(model)

inputs = tokenizer("你好", return_tensors="pt")
inputs = inputs.to(model.device)
pred = model.generate(
    **inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.0
)
test = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(test)
