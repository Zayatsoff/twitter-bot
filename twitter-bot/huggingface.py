from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from torch.functional

model_name = "hakurei/c1-6B-8bit"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline("question-answering", model=model, tokenizer=tokenizer)

response = generator("What is your name?")

# Tokenize the response
input ="What is your name?"
batch = tokenizer(input,padding=True,truncation=True,max_length=512,return_tensors="pt")

with torch.no_grad():
    output=model(**batch)
    print(output)