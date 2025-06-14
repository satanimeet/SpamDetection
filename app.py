import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn

# Set page config
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .spam {
        background-color: #ffebee;
        border: 1px solid #ffcdd2;
    }
    .not-spam {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
    }
    .model-selector {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Define BERT model architecture
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

# Load BERT model
@st.cache_resource
def load_bert_model():
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load('/Users/meetsatani/spamnlp/model/best_model.pt'))
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Load Logistic Regression model
@st.cache_resource
def load_lr_model():
    model = joblib.load('/Users/meetsatani/spamnlp/model/logicregretion.pkl')
    vectorizer = joblib.load('/Users/meetsatani/spamnlp/model/logicregretion_vectorizer.pkl')
    return model, vectorizer

# Predict using BERT
def predict_bert(text, model, tokenizer):
    tokens = tokenizer.batch_encode_plus(
        [text],
        max_length=28,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=False
    )
    
    with torch.no_grad():
        outputs = model(tokens['input_ids'], tokens['attention_mask'])
        preds = torch.argmax(outputs, dim=1)
        probabilities = outputs[0]
    
    return preds.item(), probabilities.tolist()

# Predict using Logistic Regression
def predict_lr(text, model, vectorizer):
    text_features = vectorizer.transform([text])
    prediction = model.predict(text_features)[0]
    probability = model.predict_proba(text_features)[0]
    return prediction, probability.tolist()

# Main UI
st.title('🛡️ Spam Detection System')
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Select a model and enter text to check if it's spam or not.
        </p>
    </div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    st.subheader("Model Selection")
    model_type = st.radio(
        "Choose your model:",
        ["BERT", "Logistic Regression"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model description
    if model_type == "BERT":
        st.info("""
        🤖 **BERT Model**
        - Deep learning based
        - Better at understanding context
        - Slower but more accurate
        """)
    else:
        st.info("""
        📊 **Logistic Regression**
        - Traditional ML approach
        - Faster predictions
        - Good for simple patterns
        """)

with col2:
    st.subheader("Text Input")
    text_input = st.text_area(
        "Enter your text here:",
        height=150,
        placeholder="Type or paste your text here..."
    )

# Load the selected model
if model_type == "BERT":
    model, tokenizer = load_bert_model()
else:
    model, vectorizer = load_lr_model()

# Prediction button
if st.button('🔍 Analyze Text', use_container_width=True):
    if text_input:
        # Make prediction
        if model_type == "BERT":
            prediction, probabilities = predict_bert(text_input, model, tokenizer)
        else:
            prediction, probabilities = predict_lr(text_input, model, vectorizer)
        
        # Display results
        result = "Spam" if prediction == 1 else "Not Spam"
        confidence = max(probabilities) * 100
        
        # Create result box with appropriate styling
        result_class = "spam" if prediction == 1 else "not-spam"
        st.markdown(f"""
            <div class='result-box {result_class}'>
                <h2 style='text-align: center; margin-bottom: 1rem;'>
                    {result}
                </h2>
                <p style='text-align: center; font-size: 1.2rem;'>
                    Confidence: {confidence:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show probability distribution
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Class': ['Not Spam', 'Spam'],
            'Probability': [probabilities[0] * 100, probabilities[1] * 100]
        })
        
        # Create a bar chart with custom colors
        chart = st.bar_chart(
            prob_df.set_index('Class'),
            use_container_width=True
        )
        
    else:
        st.warning("⚠️ Please enter some text to analyze.") 