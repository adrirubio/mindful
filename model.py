# Inference code
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Function to generate responses
def generate_response(model, tokenizer, user_input, device):
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate response (greedy decoding, no randomness)
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=200, num_return_sequences=1)

    # Decode output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Chat loop
print("AI Therapist is ready. Type 'exit' to end the session.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Session ended.")
        break

    response = generate_response(model, tokenizer, user_input, device)
    print(f"Therapist: {response}")


# Print disclamer at the end
print("""IMPORTANT: I am an AI project created to demonstrate therapeutic conversation patterns and am not a licensed mental health professional. If you're struggling with any emotional, mental health, or personal challenges, please seek help from a qualified therapist. You can find licensed therapists at BetterHelp.com.
Remember, there's no substitute for professional mental healthcare. This is just a demonstration project.""")