import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def detailed_tensor_debug(tensor, name="Tensor"):
    """Provide detailed information about a tensor."""
    print(f"\n{name} Debugging:")
    print(f"Tensor Type: {tensor.dtype}")
    print(f"Tensor Shape: {tensor.shape}")
    print(f"Tensor Min Value: {tensor.min()}")
    print(f"Tensor Max Value: {tensor.max()}")
    print(f"Unique Values: {torch.unique(tensor)}")

def advanced_label_decoding_debug(dataset, tokenizer, max_length=512):
    """Advanced debugging for label decoding."""
    print("=" * 50)
    print("ADVANCED LABEL DECODING DEBUGGING")
    print("=" * 50)
    
    # Tokenizer information
    print("\nTokenizer Information:")
    print(f"Pad Token ID: {tokenizer.pad_token_id}")
    print(f"EOS Token ID: {tokenizer.eos_token_id}")
    print(f"Vocab Size: {len(tokenizer.vocab)}")
    
    # Prepare dataset
    dataset_dict = dataset.to_dict()
    
    # Select a sample response
    sample_response = dataset_dict['Response'][0]
    
    print("\nSample Response:")
    print(sample_response)
    
    # Tokenize the response with verbose settings
    print("\nTokenization Debugging:")
    encodings = tokenizer(
        sample_response, 
        truncation=True, 
        max_length=max_length, 
        padding='max_length', 
        return_tensors='pt'
    )
    
    # Detailed tensor debugging
    detailed_tensor_debug(encodings['input_ids'], "Input IDs")
    detailed_tensor_debug(encodings['attention_mask'], "Attention Mask")
    
    # Safe decoding attempts
    print("\nDecoding Attempts:")
    try:
        # Method 1: Direct decoding
        print("Method 1: Direct Decoding")
        decoded_1 = tokenizer.decode(encodings['input_ids'][0].tolist(), skip_special_tokens=True)
        print(decoded_1)
    except Exception as e:
        print(f"Method 1 Failed: {e}")
    
    try:
        # Method 2: Numpy conversion
        print("\nMethod 2: Numpy Conversion")
        numpy_ids = encodings['input_ids'][0].numpy()
        decoded_2 = tokenizer.decode(numpy_ids.tolist(), skip_special_tokens=True)
        print(decoded_2)
    except Exception as e:
        print(f"Method 2 Failed: {e}")
    
    try:
        # Method 3: Filtered decoding
        print("\nMethod 3: Filtered Decoding")
        filtered_ids = [
            token_id for token_id in encodings['input_ids'][0].tolist() 
            if token_id not in [tokenizer.pad_token_id, -100]
        ]
        decoded_3 = tokenizer.decode(filtered_ids, skip_special_tokens=True)
        print(decoded_3)
    except Exception as e:
        print(f"Method 3 Failed: {e}")

    # Custom range check
    print("\nToken ID Range Check:")
    input_ids = encodings['input_ids'][0]
    valid_mask = (input_ids >= 0) & (input_ids < len(tokenizer.vocab))
    print(f"Percentage of Valid Token IDs: {valid_mask.float().mean().item() * 100:.2f}%")

# Main execution
dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

advanced_label_decoding_debug(dataset, tokenizer)
