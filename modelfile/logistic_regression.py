import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('database/cleaned_emotionaldatabase.csv', encoding='latin-1')

# Clean the data
print("Cleaning data...")
df['cleaned_text'] = df['cleaned_text'].fillna('')  # Replace NaN with empty string
df = df[df['cleaned_text'].str.strip() != '']  # Remove empty strings

# Split the data
print("Splitting data...")
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

# Convert text to TF-IDF features
print("Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
X_train = vectorizer.fit_transform(train_text)
X_val = vectorizer.transform(val_text)
X_test = vectorizer.transform(test_text)

# Train the model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
model.fit(X_train, train_label)

# Evaluate on validation set
print("\nEvaluating on validation set...")
val_pred = model.predict(X_val)
val_accuracy = accuracy_score(val_label, val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(test_label, test_pred)
precision = precision_score(test_label, test_pred)
recall = recall_score(test_label, test_pred)
f1 = f1_score(test_label, test_pred)

# Print detailed metrics
print("\nTest Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(test_label, test_pred, target_names=['Not Spam', 'Spam']))

# Create confusion matrix
cm = confusion_matrix(test_label, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('logistic_regression_confusion_matrix.png')
print("\nConfusion matrix saved as 'logistic_regression_confusion_matrix.png'")

# Save detailed results
results_df = pd.DataFrame({
    'text': test_text,
    'true_label': test_label,
    'predicted_label': test_pred,
    'confidence': np.max(test_proba, axis=1),
    'correct': test_pred == test_label
})
results_df.to_csv('logistic_regression_results.csv', index=False)
print("\nDetailed results saved to 'logistic_regression_results.csv'")

# Save the model and vectorizer
joblib.dump(model, 'logicregretion.pkl')
joblib.dump(vectorizer, 'logicregretion_vectorizer.pkl')
print("\nModel and vectorizer saved as 'logicregretion.pkl' and 'logicregretion_vectorizer.pkl'") 