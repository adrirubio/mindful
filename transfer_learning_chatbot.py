import torch
torch.cuda.empty_cache()
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load DailyDialog dataset
dataset = load_dataset("daily_dialog")

# Load the pre-trained OpenAssistant model and tokenizer
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a sample user input
user_input = "Hello! How are you?"
input_ids = tokenizer.encode(user_input, return_tensors="pt")
output = model.generate(input_ids, max_length=40, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)

# Access the dialog sections
train_dialog = dataset["train"]["dialog"]
test_dialog = dataset["test"]["dialog"]

print(f"Train set size: {len(train_dialog)}")
print(f"Test set size: {len(test_dialog)}")

