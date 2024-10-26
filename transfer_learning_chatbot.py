import torch
torch.cuda.empty.cache()
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matpotlib.pyplot as plt
from datetime import datetime

# Load DailyDialog dataset
dataset = load_dataset("daily_dialog")

# Load the pre-trained OpenAssistant model and tokenizer
model_name = "OpenAssistant/oasst-sft-1-pythia-12b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

