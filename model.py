# inference.py 
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import login

# Load tokenizer and model
tokenizer_path = "facebook/opt-2.7b"
model_path = "/home/adrian/Documents/model-weights/ai-therapist/transfer_learning_therapist.pth"

# Create model and tokenizer instance
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(tokenizer_path)

# Load the fine-tuned weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Validate weights before loading
for key, tensor in state_dict.items():
    # Check for NaN or inf values
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Found NaN or inf in {key}")
        continue
        
    # Check for unusually large values
    max_val = torch.max(torch.abs(tensor))
    if max_val > 100:
        print(f"Large values found in {key}: max abs value = {max_val}")

# Load state dict
model.load_state_dict(state_dict)
model.eval()

print("Hi there. What brings you here today?")
patient_context = input("-")

# Tokenize the input
patient_input = tokenizer(patient_context, return_tensors="pt")

# Inference 
with torch.no_grad():
    outputs = model.generate(
        **patient_input,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )

# Decode the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# response = response.encode('utf-8').decode('unicode_escape')

print("Therapist:", response)

# Print disclamer at the end
print("""IMPORTANT: I am an AI project created to demonstrate therapeutic conversation patterns and am not a licensed mental health professional. If you're struggling with any emotional, mental health, or personal challenges, please seek help from a qualified therapist. You can find licensed therapists at BetterHelp.com.
Remember, there's no substitute for professional mental healthcare. This is just a demonstration project.""")