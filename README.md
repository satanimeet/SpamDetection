# Spam Detection System

A machine learning-based spam detection system that uses both BERT and Logistic Regression models to classify text messages as spam or not spam.

## Features

- Dual model approach using BERT and Logistic Regression
- Interactive Streamlit web interface for predictions
- Model performance comparison dashboard
- High accuracy and F1 score
- Easy-to-use API for predictions

## Project Structure

```
spamnlp/
├── app.py                    # Streamlit web interface
├── model_comparison.py       # Model performance visualization
├── model/                    # Directory for saved models
├── modelcsv/                 # Directory for model evaluation results
├── database/                 # Dataset directory
└── requirements.txt          # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spamnlp.git
cd spamnlp
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit web interface:
```bash
streamlit run app.py
```

2. View model performance comparison:
```bash
streamlit run model_comparison.py
```

## Models

### BERT Model
- Uses pre-trained BERT for text classification
- Achieves high accuracy in spam detection
- Handles complex language patterns

### Logistic Regression
- Fast and efficient model
- Good baseline performance
- Easy to interpret results

## Performance Metrics

Both models are evaluated on:
- Accuracy
- Precision
- Recall
- F1 Score

Detailed performance metrics can be viewed in the model comparison dashboard.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 