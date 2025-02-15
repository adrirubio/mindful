import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path, tokenizer_path):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(tokenizer_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device

def generate_response(model, tokenizer, device, user_input, max_length=150):
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model_path = "/home/adrian/Documents/model-weights/ai-therapist/transfer_learning_therapist.pth"
    tokenizer_path = "facebook/opt-2.7b"
    
    model, tokenizer, device = load_model(model_path, tokenizer_path)
    
    print("AI Therapist is ready. Type 'exit' to end the session.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = generate_response(model, tokenizer, device, user_input)
        print(f"Therapist: {response}")