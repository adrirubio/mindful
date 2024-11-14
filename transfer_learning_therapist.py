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
chatbot_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


# Prepare model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# chatbot_model.to(device)
# chatbot_model.train() # Set to training mode

# Load the mental health dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]
print(dataset[0])  # Print the first item to understand its structure


class TherapyDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, train=True):  # Added train parameter
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Divide the dataset into train and test
        split = int(len(dataset) * 0.9)
        if train:
            self.dataset = dataset[:split]
        else:
            self.dataset = dataset[split:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Format the conversation
        # Print to check the structure of self.dataset[idx]
        print(type(self.dataset[idx]), self.dataset[idx])  # Temporary debug line
    
        # Assuming `self.dataset[idx]` is a dictionary with "Context" and "Response"
        try:
            conversation = (f"Patient: {self.dataset[idx]['Context']}\n"
                        f"Therapist: {self.dataset[idx]['Response']}")
        except TypeError:
            raise TypeError("Each item in dataset is not a dictionary. Check dataset structure.")

        # Tokenize - fixed method name from tokenize to tokenizer
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

# Print some examples
print(train_dataset[0])
print(test_dataset[0])


