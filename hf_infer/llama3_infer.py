from transformers import AutoModelForCausalLM, AutoTokenizer

model = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    trust_remote_code=True,
).eval()
print(model)
inputs = tokenizer("hello", return_tensors="pt")
inputs = inputs.to(model.device)
pred = model.generate(
    **inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.0
)
test = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
print(test)
