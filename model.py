import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
tokenizer_path = "facebook/opt-2.7b"
model_path = "/home/adrian/Documents/model-weights/ai-therapist/transfer_learning_chatbot.pth"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Ensure pad token is set correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(tokenizer_path).to(device)

# Load fine-tuned weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

def generate_response(model, tokenizer, input_text, device):
    """Generate a response from the AI therapist with proper probability handling."""
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate response with controlled randomness
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            temperature=0.7,       # Lowered for stability
            top_p=0.9,             # Nucleus sampling
            repetition_penalty=1.2, # Penalize repetition
            do_sample=True
        )

        # Ensure output does not contain NaN or invalid values
        output = torch.clamp(output, min=0)  # Prevent negative values

    # Decode response
    response = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
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