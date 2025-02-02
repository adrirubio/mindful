# inference.py 
import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token="Token")

# Load the model and tokenizer
model_path = "/home/adrian/Documents/model_weights/ai_therapist/transfer_learning_therapist.pth"
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print("Hi there. What brings you here today?")
patient_context = input(":")

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