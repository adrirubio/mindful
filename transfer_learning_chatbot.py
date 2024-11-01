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

class DailyDialogDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return(len(self.dialogues))

    def __getitem__(self, idx):
        dialogue = " ".join(self.dialogues[idx])
        encoded = self.tokenizer(dialogue, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return input_ids, input_ids  # Returning input_ids as both input and target

# Create train and test datasets instances
train_dataset = DailyDialogDataset(train_dialog, tokenizer)
test_dataset = DailyDialogDataset(test_dialog, tokenizer)

# Create batches for improved computational efficiency
batch_size = 8
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
