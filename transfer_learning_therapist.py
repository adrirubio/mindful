import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Load tokenizer and model
tokenizer_path = "facebook/opt-2.7b"
model_path = "/home/adrian/Documents/model-weights/ai-therapist/transfer_learning_chatbot.pth"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Create an instance of the model with gradient checkpointing
config = AutoConfig.from_pretrained(tokenizer_path)
config.use_cache = False  # Disable KV cache for training
config.gradient_checkpointing = True

model = AutoModelForCausalLM.from_pretrained(
    tokenizer_path,
    config=config
)

# Load previous weights
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Successfully loaded previous weights")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Starting with fresh model")

# Move model to device
model.to(device)

# Load the mental health dataset (train section)
dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]

class TherapyDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Convert dataset to dictionary
        dataset_dict = dataset.to_dict()
        
        # Create list of unique conversation pairs
        unique_pairs = {}
        for context, response in zip(dataset_dict["Context"], dataset_dict["Response"]):
            if context not in unique_pairs:  # Only take first response for each context
                unique_pairs[context] = response

        # Convert to lists and sort for deterministic behavior
        contexts = sorted(list(unique_pairs.keys()))
        responses = [unique_pairs[context] for context in contexts]

        # Perform train/test split
        split_idx = int(len(contexts) * 0.9)
        
        if train:
            self.contexts = contexts[:split_idx]
            self.responses = responses[:split_idx]
        else:
            self.contexts = contexts[split_idx:]
            self.responses = responses[split_idx:]

        print(f"{'Train' if train else 'Test'} dataset size: {len(self.contexts)}")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        # Format input with special tokens
        input_text = f"Patient: {self.contexts[idx]}\nTherapist:"
        target_text = self.responses[idx]

        # Tokenize input and target separately
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )

        target_encodings = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )

        # Convert to tensors
        input_ids = torch.tensor(input_encodings['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(input_encodings['attention_mask'], dtype=torch.long)
        
        # Create labels tensor and mask padding
        labels = torch.tensor(target_encodings['input_ids'], dtype=torch.long)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Create train and test datasets
train_dataset = TherapyDataset(dataset, tokenizer, train=True)
test_dataset = TherapyDataset(dataset, tokenizer, train=False)

# Create data loaders with proper worker initialization
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

batch_size = 4
grad_accumulation_steps = 4
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    worker_init_fn=worker_init_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    worker_init_fn=worker_init_fn
)

# Layer unfreezing with proper initialization
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layers
trainable_layers = [
    model.model.decoder.embed_tokens,
    model.model.decoder.embed_positions,
    *model.model.decoder.layers[-2:],
    model.lm_head
]

for layer in trainable_layers:
    for param in layer.parameters():
        param.requires_grad = True

# Optimizer with learning rate scheduling
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-6,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10,
    eta_min=1e-6
)

def train_epoch(model, optimizer, train_loader, device, grad_accumulation_steps):
    model.train()
    total_loss = 0
    acc_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(train_loader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Loss scaling
            loss = outputs.loss / grad_accumulation_steps
            
            # Check for NaN/inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected in loss at batch {i}")
                continue
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            
            # Gradient accumulation
            if (i + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            acc_loss += loss.item() * grad_accumulation_steps
            
            if (i + 1) % 100 == 0:
                print(f"Batch {i+1}, Current loss: {acc_loss/(i+1)}")
                
        except RuntimeError as e:
            print(f"Error in batch {i}: {e}")
            optimizer.zero_grad()
            continue
            
    return acc_loss / len(train_loader)

# Modified training loop
def train_model(model, optimizer, scheduler, train_loader, test_loader, epochs, device):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            device,
            grad_accumulation_steps
        )
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                test_loss += outputs.loss.item()
        
        test_loss /= len(test_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss
            }
            torch.save(
                checkpoint,
                f"/home/adrian/Documents/model-weights/ai-therapist/checkpoint_epoch_{epoch+1}.pth"
            )
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    return train_losses, test_losses

# Train the model
train_losses, test_losses = train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    epochs=10,
    device=device
)

# Plot training curves
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.legend()
plt.show()

# Save final model
torch.save(
    model.state_dict(),
    "/home/adrian/Documents/model-weights/ai-therapist/transfer_learning_therapist_final.pth"
)