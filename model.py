# Inference code
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Model and tokenizer paths
tokenizer_path = "facebook/opt-2.7b"
model_path = "/home/adrian/Documents/model-weights/ai-therapist/transfer_learning_chatbot.pth"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(tokenizer_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def generate_response(model, tokenizer, user_input, device):
    prompt_template = f"You: {user_input}\nTherapist:"
    
    inputs = tokenizer(
        prompt_template, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=256  
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,  
            do_sample=True,  
            temperature=0.7,  
            top_k=50,  
            top_p=0.9,  
            repetition_penalty=1.5,  
            no_repeat_ngram_size=4,  
            early_stopping=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-process output to remove unnecessary duplication
    response = response.replace(prompt_template, "").strip()
    return response

# Chat loop
print("AI Therapist is ready.")
user_input = input("- ")
response = generate_response(model, tokenizer, user_input, device)
print(f"Therapist: {response}")


# Print disclamer at the end
print("""IMPORTANT: I am an AI project created to demonstrate therapeutic conversation patterns and am not a licensed mental health professional. If you're struggling with any emotional, mental health, or personal challenges, please seek help from a qualified therapist. You can find licensed therapists at BetterHelp.com.
Remember, there's no substitute for professional mental healthcare. This is just a demonstration project.""")