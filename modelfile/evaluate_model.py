import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the same model architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[1]
        x = self.dropout(cls_hs)
        x = self.fc(x)
        x = self.softmax(x)
        return x

def load_model(model_path):
    # Initialize BERT model and tokenizer
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Initialize our model
    model = BERT_Arch(bert)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, tokenizer

def predict_text(model, tokenizer, text, max_len=28):
    # Tokenize the text
    tokens = tokenizer.batch_encode_plus(
        [text],
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=False
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(tokens['input_ids'], tokens['attention_mask'])
        preds = torch.argmax(outputs, dim=1)
        probabilities = outputs[0]
    
    return preds.item(), probabilities.tolist()

def evaluate_model(model, tokenizer, test_file):
    # Read the test file
    print("Loading test data...")
    df = pd.read_csv(test_file, encoding='latin-1')
    
    # Initialize lists for predictions and true labels
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    # Process each text
    print("Making predictions...")
    for idx, row in df.iterrows():
        text = str(row['cleaned_text'])
        true_label = row['actions']
        
        # Get prediction
        pred, probs = predict_text(model, tokenizer, text)
        
        # Store results
        all_predictions.append(pred)
        all_true_labels.append(true_label)
        all_confidences.append(max(probs))
        
        # Print progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} samples...")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions)
    recall = recall_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions)
    
    # Print detailed metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_true_labels, all_predictions, target_names=['Not Spam', 'Spam']))
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Calculate average confidence for correct and incorrect predictions
    correct_confidences = [conf for pred, true, conf in zip(all_predictions, all_true_labels, all_confidences) if pred == true]
    incorrect_confidences = [conf for pred, true, conf in zip(all_predictions, all_true_labels, all_confidences) if pred != true]
    
    print("\nConfidence Analysis:")
    print(f"Average confidence for correct predictions: {np.mean(correct_confidences):.4f}")
    print(f"Average confidence for incorrect predictions: {np.mean(incorrect_confidences):.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'text': df['cleaned_text'],
        'true_label': all_true_labels,
        'predicted_label': all_predictions,
        'confidence': all_confidences,
        'correct': [pred == true for pred, true in zip(all_predictions, all_true_labels)]
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nDetailed results saved to 'evaluation_results.csv'")

if __name__ == "__main__":
    # Load the trained model
    print("Loading model...")
    model, tokenizer = load_model('best_model.pt')
    
    # Get the test file path from user
    test_file = input("Enter the path to your test file (CSV format with 'cleaned_text' and 'actions' columns): ")
    
    # Evaluate the model
    evaluate_model(model, tokenizer, test_file) 