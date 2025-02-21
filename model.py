# Inference code
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model and tokenizer paths
tokenizer_path = "facebook/opt-1.3b"
model_path = "transfer_learning_therapist.pth"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load base model 
model = AutoModelForCausalLM.from_pretrained(tokenizer_path)

# Properly load the saved checkpoint
checkpoint = torch.load(model_path, map_location=device)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model.to(device)
model.eval()

# Improved inference function
def generate_response(model, tokenizer, user_input, device, max_new_tokens=100):
    model.eval()
    # Enable KV cache for inference
    model.config.use_cache = True
    
    formatted_input = f"User: {user_input.strip()}\nTherapist:"
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    # Disable KV cache again
    model.config.use_cache = False
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the first therapist response
    parts = response.split("User:")
    therapist_part = parts[0].split("Therapist:")[-1].strip()
    
    return therapist_part

# Single response generation
user_input = input("User: ")
response = generate_response(model, tokenizer, user_input, device)
print(f"Therapist: {response}")

# Print disclaimer
print("""IMPORTANT: I am an AI project created to demonstrate therapeutic conversation patterns and am not a licensed mental health professional. If you're struggling with any emotional, mental health, or personal challenges, please seek help from a qualified therapist. You can find licensed therapists at BetterHelp.com.
Remember, there's no substitute for professional mental healthcare. This is just a demonstration project.""")