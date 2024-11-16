import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load chatbot tokenizer and model
# chatbot_tokenizer_path = "facebook/opt-350m"
# chatbot_model_path = "/home/adrian/Documents/Perceptron/model_weights/transfer_learning_chatbot.pth"
# chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_tokenizer_path)
# chatbot_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# chatbot_model.load_state_dict(torch.load(chatbot_model_path)
chatbot_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")


# Prepare model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# chatbot_model.to(device)
# chatbot_model.train() # Set to training mode

# Load the mental health dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]

class TherapyDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        dataset = dataset.to_dict()
        split = int(len(dataset["Context"]) * 0.9)

        if train:
            self.dataset = {
                'Context': dataset["Context"][:split],
                'Response': dataset["Response"][:split]
            }
        else:
            self.dataset = {
                'Context': dataset["Context"][split:],
                'Response': dataset["Response"][split:]
            }

    def __len__(self):
        return len(self.dataset['Context'])

    def __getitem__(self, idx):
        # Format the conversation as requested
        conversation = (f"Patient: {self.dataset['Context'][idx]}\n"
                      f"Therapist: {self.dataset['Response'][idx]}")

        # Tokenize the entire conversation as a single piece of text
        encodings = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in encodings.items()}

# Create train and test datasets using the class
train_dataset = TherapyDataset(dataset, chatbot_tokenizer, train=True)
test_dataset = TherapyDataset(dataset, chatbot_tokenizer, train=False)

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset


# Print some examples
print("Context: " + train_dataset['Context'][0])
print("Response: " + train_dataset['Response'][0])

print("Context: " + test_dataset['Context'][14])
print("Response: " + test_dataset['Response'][14])
