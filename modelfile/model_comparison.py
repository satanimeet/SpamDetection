import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Model Performance Comparison",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the results
@st.cache_data
def load_results():
    # Load BERT results
    bert_results = pd.read_csv('modelcsv/test_results.csv')
    # Load Logistic Regression results
    lr_results = pd.read_csv('modelcsv/logistic_regression_results.csv')
    return bert_results, lr_results

# Calculate metrics
def calculate_metrics(results):
    accuracy = (results['correct'].sum() / len(results)) * 100
    precision = (results[results['predicted_label'] == 1]['correct'].sum() / 
                results[results['predicted_label'] == 1].shape[0]) * 100
    recall = (results[results['true_label'] == 1]['correct'].sum() / 
             results[results['true_label'] == 1].shape[0]) * 100
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1

# Create confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=['Not Spam', 'Spam'],
                    y=['Not Spam', 'Spam'],
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues')
    fig.update_layout(title=title)
    return fig

# Main app
st.title('📊 Model Performance Comparison')

# Load results
bert_results, lr_results = load_results()

# Calculate metrics for both models
bert_metrics = calculate_metrics(bert_results)
lr_metrics = calculate_metrics(lr_results)

# Create two columns for metrics
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("BERT Model Performance")
    st.markdown(f"""
        <div style='text-align: center;'>
            <div class='metric-value'>{bert_metrics[0]:.2f}%</div>
            <div class='metric-label'>Accuracy</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        | Metric | Value |
        |--------|--------|
        | Precision | {:.2f}% |
        | Recall | {:.2f}% |
        | F1 Score | {:.2f}% |
    """.format(bert_metrics[1], bert_metrics[2], bert_metrics[3]))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Logistic Regression Performance")
    st.markdown(f"""
        <div style='text-align: center;'>
            <div class='metric-value'>{lr_metrics[0]:.2f}%</div>
            <div class='metric-label'>Accuracy</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        | Metric | Value |
        |--------|--------|
        | Precision | {:.2f}% |
        | Recall | {:.2f}% |
        | F1 Score | {:.2f}% |
    """.format(lr_metrics[1], lr_metrics[2], lr_metrics[3]))
    st.markdown('</div>', unsafe_allow_html=True)

# Create comparison plots
st.subheader("Model Comparison")

# Create a bar chart comparing metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'] * 2,
    'Value': bert_metrics + lr_metrics,
    'Model': ['BERT'] * 4 + ['Logistic Regression'] * 4
})

fig = px.bar(metrics_df, x='Metric', y='Value', color='Model',
             barmode='group', title='Performance Metrics Comparison',
             color_discrete_sequence=['#1f77b4', '#ff7f0e'])
st.plotly_chart(fig, use_container_width=True)

# Create confusion matrices
col1, col2 = st.columns(2)

with col1:
    bert_cm = plot_confusion_matrix(bert_results['true_label'], 
                                  bert_results['predicted_label'],
                                  'BERT Confusion Matrix')
    st.plotly_chart(bert_cm, use_container_width=True)

with col2:
    lr_cm = plot_confusion_matrix(lr_results['true_label'], 
                                lr_results['predicted_label'],
                                'Logistic Regression Confusion Matrix')
    st.plotly_chart(lr_cm, use_container_width=True)

# Show detailed classification reports
st.subheader("Detailed Classification Reports")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### BERT Model")
    st.text(classification_report(bert_results['true_label'], 
                                bert_results['predicted_label'],
                                target_names=['Not Spam', 'Spam']))

with col2:
    st.markdown("### Logistic Regression")
    st.text(classification_report(lr_results['true_label'], 
                                lr_results['predicted_label'],
                                target_names=['Not Spam', 'Spam']))

# Show sample predictions
st.subheader("Sample Predictions")

# Create tabs for each model
tab1, tab2 = st.tabs(["BERT Predictions", "Logistic Regression Predictions"])

with tab1:
    st.dataframe(bert_results[['text', 'true_label', 'predicted_label', 'confidence']]
                .rename(columns={
                    'text': 'Text',
                    'true_label': 'True Label',
                    'predicted_label': 'Predicted Label',
                    'confidence': 'Confidence'
                })
                .head(10))

with tab2:
    st.dataframe(lr_results[['text', 'true_label', 'predicted_label', 'confidence']]
                .rename(columns={
                    'text': 'Text',
                    'true_label': 'True Label',
                    'predicted_label': 'Predicted Label',
                    'confidence': 'Confidence'
                })
                .head(10)) 