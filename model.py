# inference.py 
import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the trained weights
model.load_state_dict(torch.load("transfer_learning_therapist.pth", map_location=torch.device('cpu')))
model.eval()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

