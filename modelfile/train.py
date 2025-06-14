import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import transformers
import torch.utils.data as Data
import tokenizers
from transformers import AutoModel, BertTokenizerFast, get_linear_schedule_with_warmup

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Load the cleaned dataset
print("Loading cleaned dataset...")
df = pd.read_csv('database/cleaned_emotionaldatabase.csv', encoding='latin-1')

# Split the data
print("Splitting data into train and test sets...")
train_text, temp_text, train_label, temp_label = train_test_split(
    df['cleaned_text'].values,
    df['actions'].values,
    test_size=0.15,
    random_state=2018,
    stratify=df['actions']
)

val_text, test_text, val_label, test_label = train_test_split(
    temp_text,
    temp_label,
    test_size=0.5,
    random_state=2018,
    stratify=temp_label
)

# Initialize BERT tokenizer
print("Initializing BERT tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

text = ["this very large model for that"]
sent_id = tokenizer.batch_encode_plus(text, add_special_tokens=True)

max_len = 28

# Convert numpy arrays to lists of strings
train_texts = [str(text) for text in train_text]
val_texts = [str(text) for text in val_text]
test_texts = [str(text) for text in test_text]

# Increase batch size for faster training
batch_size = 64  # Increased from 32 to 64

tokens_train = tokenizer.batch_encode_plus(
    train_texts,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
    return_token_type_ids=False
)

token_val = tokenizer.batch_encode_plus(
    val_texts,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
    return_token_type_ids=False
)

token_test = tokenizer.batch_encode_plus(
    test_texts,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt',
    return_token_type_ids=False
)

# Fix tensor creation warnings
train_seq = tokens_train['input_ids'].clone().detach()
train_mask = tokens_train['attention_mask'].clone().detach()
train_labels = torch.tensor(train_label)
train_data = Data.TensorDataset(train_seq, train_mask, train_labels)
train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_seq = token_val['input_ids'].clone().detach()
val_mask = token_val['attention_mask'].clone().detach()
val_labels = torch.tensor(val_label)
val_data = Data.TensorDataset(val_seq, val_mask, val_labels)
val_loader = Data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

test_seq = token_test['input_ids'].clone().detach()
test_mask = token_test['attention_mask'].clone().detach()
test_labels = torch.tensor(test_label)
test_data = Data.TensorDataset(test_seq, test_mask, test_labels)
test_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Save test data for later evaluation
torch.save({
    'text': test_text,
    'label': test_label
}, 'test_data.pt')

batch_size = 32  # Increased from 16 to 32 for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable mixed precision training for faster training
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

train_data = TensorDataset(train_seq, train_mask, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_labels)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Initialize BERT model
print("Initializing BERT model...")
bert = AutoModel.from_pretrained('bert-base-uncased')

# Define training parameters
epochs = 5  # Reduced from 15 to 5 epochs
target_accuracy = 0.91

# Initialize optimizer with higher learning rate
optimizer = torch.optim.AdamW(bert.parameters(), lr=1e-4)  # Increased from 5e-5 to 1e-4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps * 0.05,  # Reduced warmup to 5%
    num_training_steps=total_steps
)

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)  # Increased dropout for regularization
        self.fc = nn.Linear(768, 2)  # Simplified architecture: direct to output
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[1]
        x = self.dropout(cls_hs)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
model = BERT_Arch(bert)
model.to(device)

# Fix class weight computation
classes = np.unique(train_label)  # Use train_label instead of train_labels
class_wts = compute_class_weight('balanced', classes=classes, y=train_label)
weights = torch.tensor(class_wts, dtype=torch.float).to(device)

def train(epoch):
    model.train()
    total_loss, total_accuracy = 0, 0
    criterion = nn.CrossEntropyLoss()  # Add proper loss function
    
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_acc = total_accuracy / len(train_dataloader)
            print(f"Epoch {epoch + 1} | Step {step} | Avg Train Loss: {avg_train_loss:.4f} | Avg Train Accuracy: {avg_train_acc:.4f}")
        
        sent_id, mask, labels = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        
        # Use mixed precision training if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(sent_id, mask)
                loss = criterion(outputs, labels)  # Use CrossEntropyLoss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(sent_id, mask)
            loss = criterion(outputs, labels)  # Use CrossEntropyLoss
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_accuracy += (preds == labels).sum().item() / len(preds)
    
    return total_loss / len(train_dataloader), total_accuracy / len(train_dataloader)

def evaluate():
    model.eval()
    total_loss, total_accuracy = 0, 0
    criterion = nn.CrossEntropyLoss()  # Add proper loss function
    
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            sent_id, mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(sent_id, mask)
            loss = criterion(outputs, labels)  # Use CrossEntropyLoss
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_accuracy += (preds == labels).sum().item() / len(preds)
    return total_loss / len(val_dataloader), total_accuracy / len(val_dataloader)

best_valid_loss = float('inf')
best_valid_acc = 0.0

print("Starting training...")
for epoch in range(epochs):
    train_loss, train_acc = train(epoch)
    valid_loss, valid_acc = evaluate()

    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f}")

    # Save model if validation accuracy improves
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Model saved with validation accuracy: {valid_acc:.4f}")

    # Early stopping if target accuracy is reached
    if valid_acc >= target_accuracy:
        print(f"\nTarget accuracy of {target_accuracy:.4f} reached! Stopping training.")
        print(f"Final model saved with validation accuracy: {valid_acc:.4f}")
        break

print("\nTraining completed!")
print(f"Best validation accuracy achieved: {best_valid_acc:.4f}")


    
    












