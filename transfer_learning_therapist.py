import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load chatbot tokenizer and model
chatbot_tokenizer_path = "facebook/opt-350m"
chatbot_model_path = "transfer_learning_chatbot.pth"
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_tokenizer_path)
chatbot_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
chatbot_model.load_state_dict(torch.load(chatbot_model_path))
