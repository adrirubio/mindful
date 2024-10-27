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
model_name = "OpenAssistant/oasst-sft-1-pythia-12b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a sample user input
user_input = "Hello! How are you today?"
input_ids = tokenizer.encode(user_input, return_tensors="pt")
output = model.generate(input_ips, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
