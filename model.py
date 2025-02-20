# inference code
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model and tokenizer paths
tokenizer_path = "facebook/opt-2.7b"
model_path = "/transfer_learning_chatbot.pth"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set device (you can force CPU by uncommenting the next line)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Load base model and then custom weights
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

    # Pre-generation check: Run a forward pass and fix any NaNs in logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if torch.isnan(logits).any():
            print("[WARNING] NaN values detected in logits. Applying nan_to_num.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e10, neginf=-1e10)
            # Note: This workaround only addresses the forward pass check.
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=256,
                num_beams=3,         # Lowered beams for stability
                do_sample=True,
                temperature=0.7,
                top_k=30,            # Lowered top_k
                top_p=0.8,           # Lowered top_p
                repetition_penalty=1.2,  # Lowered repetition penalty
                no_repeat_ngram_size=3,  # Adjusted n-gram size
                early_stopping=True
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.replace(prompt_template, "").strip()
        return response
    except Exception as e:
        print(f"[ERROR] Exception during generation: {e}")
        return "An error occurred while generating the response."

# Chat loop
print("AI Therapist is ready.")
user_input = input("- ")
response = generate_response(model, tokenizer, user_input, device)
print(f"Therapist: {response}")

# Print disclamer at the end
print("""IMPORTANT: I am an AI project created to demonstrate therapeutic conversation patterns and am not a licensed mental health professional. If you're struggling with any emotional, mental health, or personal challenges, please seek help from a qualified therapist. You can find licensed therapists at BetterHelp.com.
Remember, there's no substitute for professional mental healthcare. This is just a demonstration project.""")