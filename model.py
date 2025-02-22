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

def generate_response(model, tokenizer, user_input, device, max_new_tokens=150, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    # Ensure the model is in evaluation mode and use cache
    model.eval()
    model.config.use_cache = True

    # Preprocess the input and ensure it's clean
    user_input = user_input.strip()
    if not user_input:  # Handle empty input case
        return "Please provide a valid input."

    # Format the input with a clear separator and role indicator
    formatted_input = f"Human: {user_input}\nTherapist:"
    
    # Tokenize the formatted input
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            # Generate the response with flexible parameters
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                # Add parameters to encourage more unique responses
                num_beams=1,  # Use greedy decoding
                no_repeat_ngram_size=3,  # Avoid repeating 3-grams
            )
    except Exception as e:
        return f"Error generating response: {e}"
    finally:
        model.config.use_cache = False
    
    # Decode the output and extract only the therapist's response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Split the response and get only the therapist's part
    response_parts = full_response.split("Therapist:")
    if len(response_parts) > 1:
        therapist_response = response_parts[1].strip()
    else:
        therapist_response = full_response.split("Human:")[0].strip()
    
    return therapist_response

# Chat loop
print("AI Therapist is ready.")
user_input = input("- ")
response = generate_response(model, tokenizer, user_input, device)
if "Human" in response:
    location = response.find("Human")
    response = response[0:location]
if "Me" in response:
    location = response.find("Me")
    response = response[0:location]
print(f"Therapist: {response}")

# Print disclaimer at the end
print("""IMPORTANT: I am an AI project created to demonstrate therapeutic conversation patterns and am not a licensed mental health professional. If you're struggling with any emotional, mental health, or personal challenges, please seek help from a qualified therapist. You can find licensed therapists at BetterHelp.com.
Remember, there's no substitute for professional mental healthcare. This is just a demonstration project.""")