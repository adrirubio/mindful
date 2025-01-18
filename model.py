# inference.py 
import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

# Load the trained weights
#model.load_state_dict(torch.load("transfer_learning_therapist.pth", map_location=torch.device('cpu')))
#model.eval()

print("Hi there. What brings you here today?")
patient_context = input(":")



