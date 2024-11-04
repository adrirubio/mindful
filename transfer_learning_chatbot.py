import torch
torch.cuda.empty_cache()
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load DailyDialog dataset
dataset = load_dataset("daily_dialog")

# Load the pre-trained OpenAssistant model and tokenizer
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a sample user input
user_input = "Hello! How are you?"
input_ids = tokenizer.encode(user_input, return_tensors="pt")
output = model.generate(input_ids, max_length=40, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)

# Access the dialog sections
train_dialog = dataset["train"]["dialog"]
test_dialog = dataset["test"]["dialog"]

print(f"Train set size: {len(train_dialog)}")
print(f"Test set size: {len(test_dialog)}")

class DailyDialogDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return(len(self.dialogues))

    def __getitem__(self, idx):
        dialogue = " ".join(self.dialogues[idx])
        encoded = self.tokenizer(dialogue, truncation=True, padding="max_length", max_length=self.max_length, return_tensors='pt')
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        return input_ids, input_ids  # Returning input_ids as both input and target

# Create train and test datasets instances
train_dataset = DailyDialogDataset(train_dialog, tokenizer)
test_dataset = DailyDialogDataset(test_dialog, tokenizer)

# Create batches for improved computational efficiency
batch_size = 8
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Unfreeze some layers
for param in model.transformer.wte.parameters():
    param.requires_grad = True
for param in model.transformer.wpe.parameters():
    param.requires_grad = True
for layer in model.transformer.h[-4:]: # Unfreeze last 4 layers
    for param in layer.parameters():
        param.requires_grad = True

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)

# Training loop
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    for it in range(epoch):
        t0 = datetime.now()
        train_loss = []
        model.train()
        for batch in train_loader():
            # Get batch data
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            targets = lables.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            ouptuts = model(input_ids, targets)
            loss = output.loss

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utilis.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            train_loss.append(loss.item())

        # Get train loss
        train_loss = np.mean(train_loss) # a little misleading

        # Evaluation phase
        model.eval()

        with torch.no_grad():
            for batch in test_loader():
                # Get batch data
                input_ids, lables = batch
                input_ids = input_ids.to(device)
                targets = labels.to(device)

                # Forward pass
                outputs = model(input_ids, targets)
                loss = outputs.loss
                test_loss.append(loss.item())

            # Get test loss
            test_loss = np.mean(test_loss)
