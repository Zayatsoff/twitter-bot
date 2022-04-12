from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

model_name = "mahaamami/distilgpt2-finetuned-wikitext2"
min_length = 10
temperature = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
torch.save(model, f"{model_name}.pt")
model = torch.load(f"{model_name}.pt").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

gen = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, framework="pt", device=0
)
output = gen(
    "Fun fact:", do_sample=True, min_length=min_length, temperature=temperature,
)
print(output[0]["generated_text"])
