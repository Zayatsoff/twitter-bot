from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# from torch.functional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "hivemind/gpt-j-6B-8bit"
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name).to(device)

# generator = pipeline("question-answering", model=model, tokenizer=tokenizer).to(device)

# response = generator("What is your name?").to(device)

# Tokenize the response
input = "What is your name?"
batch = tokenizer(
    input, padding=True, truncation=True, max_length=512, return_tensors="pt"
).to(device)

if __name__ == "__main__":
    with torch.no_grad():
        output = model(**batch)
        print(output)
