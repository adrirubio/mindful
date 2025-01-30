import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

### Temporary changes

# Load tokenizer and model
# tokenizer_path = "facebook/opt-2.7b"
# model_path = "transfer_learning_chatbot.pth"
# tokenizer = AutoTokenizer.from_pretrained(chatbot_tokenizer_path)
# model = model.load_state_dict(torch.load(chatbot_model_path)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

# Prepare model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the mental health dataset (train section)
dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]

class TherapyDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, train=True):
        # Convert dataset to dictionary
        dataset_dict = dataset.to_dict()

        # Create a set of unique contexts to remove duplicates
        unique_contexts = list(set(dataset_dict["Context"]))
        unique_responses = []

        # Create a mapping of unique contexts to their first unique response
        context_to_response = {}
        for context, response in zip(dataset_dict["Context"], dataset_dict["Response"]):
            if context not in context_to_response:
                context_to_response[context] = response

        # Convert back to lists
        unique_contexts = list(context_to_response.keys())
        unique_responses = list(context_to_response.values())

        # Perform train/test split
        split = int(len(unique_contexts) * 0.9)

        if train:
            self.dataset = {
                'Context': unique_contexts[:split],
                'Response': unique_responses[:split]
            }
        else:
            self.dataset = {
                'Context': unique_contexts[split:],
                'Response': unique_responses[split:]
            }

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Debug print
        print(f"{'Train' if train else 'Test'} dataset size:", len(self.dataset['Context']))

    def __len__(self):
        return len(self.dataset['Context'])

    def __getitem__(self, idx):
        # Ensure the context and response for this specific index are used
        patient_context = self.dataset['Context'][idx]
        therapist_response = self.dataset['Response'][idx]

        # Tokenization remains the same
        patient_encodings = self.tokenizer(
            patient_context,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )

        therapist_encodings = self.tokenizer(
            therapist_response,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )

        input_ids = torch.tensor(patient_encodings['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(patient_encodings['attention_mask'], dtype=torch.long)

        labels = torch.tensor(therapist_encodings['input_ids'], dtype=torch.long)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Create train and test datasets
train_dataset = TherapyDataset(dataset, tokenizer, train=True)
test_dataset = TherapyDataset(dataset, tokenizer, train=False)

# Print patient context
print("Patient Context:")
print(tokenizer.decode(train_dataset[0]['input_ids'].tolist(), skip_special_tokens=True))

# Print therapist response
print("Therapist Response:")
# Filter out -100 values before decoding
valid_labels = train_dataset[0]['labels']
valid_labels = valid_labels[valid_labels != -100]
print(tokenizer.decode(valid_labels.tolist(), skip_special_tokens=True))

# Load batches
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Unfreeze some layers
for param in model.parameters():
    param.requires_grad = False
for param in model.model.decoder.embed_tokens.parameters():
    param.requires_grad = True
for param in model.model.decoder.embed_positions.parameters():
    param.requires_grad = True

for layer in model.model.decoder.layers[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.lm_head.parameters():
    param.requires_grad = True

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)

# Training loop
def batch_gd(model, optimizer, train_loader, test_loader, epochs, device=device):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        model.train()
        for batch in train_loader:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Get train and test losses
        train_losses[it] = np.mean(train_loss)

        # Test loop
        test_loss = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                test_loss.append(loss.item())

            test_losses[it] = np.mean(test_loss)

        print(f"Epoch {it+1} - Train Loss: {train_losses[it]:.4f} - Test Loss: {test_losses[it]:.4f} - Time: {datetime.now() - t0}")

    return train_losses, test_losses

# Run training loop
train_losses, test_losses = batch_gd(model, optimizer, train_loader, test_loader, epochs=10)

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

# Save model and tokenizer
model_save_path = "transfer_learning_therapist.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Accuracy
n_correct = 0
n_total = 0
for batch in test_loader:
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)

    # Get predictions
    _, predictions = torch.max(outputs, 1)

    # Update counts
    n_correct += (predictions == labels).sum().item()
    n_total += labels.shape[0]

print(f"Accuracy: {n_correct / n_total}")

with torch.no_grad():
    for batch in test_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == labels).sum().item()
        n_total += labels.shape[0]

print(f'Accuracy: {n_correct / n_total}')