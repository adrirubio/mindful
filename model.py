# Inference code
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

# Inference function
def generate_response(model, tokenizer, user_input, device, max_new_tokens=150, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    # Ensure model is in evaluation mode and using cache for inference
    model.eval()
    model.config.use_cache = True

    # Clean the user input and check for emptiness
    user_input = user_input.strip()
    if not user_input:
        return "Please provide a valid input."

    # Format the prompt to ensure we get only the therapist response.
    # Using a delimiter that is unique and unlikely to appear in natural text.
    prompt = f"<<START>>\nUser: {user_input}\nTherapist:"
    
    # Tokenize and move the prompt to the specified device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty
            )
    except Exception as e:
        return f"Error generating response: {e}"
    finally:
        model.config.use_cache = False

    # Decode the generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove everything before the therapist's reply based on the known prompt structure.
    # First, remove our unique start delimiter.
    if "<<START>>" in full_output:
        full_output = full_output.split("<<START>>", 1)[1].strip()
    
    # Now, isolate the therapist's response by removing everything up to "Therapist:"
    if "Therapist:" in full_output:
        therapist_response = full_output.split("Therapist:", 1)[1].strip()
    else:
        therapist_response = full_output.strip()
    
    # Optional: If the response contains extra markers or additional user prompts, remove them.
    if "User:" in therapist_response:
        therapist_response = therapist_response.split("User:")[0].strip()
    
    return therapist_response

# Chat loop
print("AI Therapist is ready.")
user_input = input("- ")
response = generate_response(model, tokenizer, user_input, device)
print(f"Therapist: {response}")

# Print disclaimer at the end
print("""IMPORTANT: I am an AI project created to demonstrate therapeutic conversation patterns and am not a licensed mental health professional. If you're struggling with any emotional, mental health, or personal challenges, please seek help from a qualified therapist. You can find licensed therapists at BetterHelp.com.
Remember, there's no substitute for professional mental healthcare. This is just a demonstration project.""")