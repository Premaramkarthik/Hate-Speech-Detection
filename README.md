# Hate Speech Detection System

## Overview
This project implements a machine learning-based hate speech detection system for Twitter data. The system uses Natural Language Processing (NLP) techniques and a Logistic Regression model to classify tweets as either hate speech or non-hate speech.

## Features
- Text preprocessing pipeline including tokenization, lemmatization, and stopword removal
- TF-IDF vectorization with n-gram features
- Optimized Logistic Regression classifier
- Comprehensive evaluation metrics and visualizations
- Hyperparameter tuning using GridSearchCV

## Dataset
The project uses a labeled Twitter dataset (`hateDetection_train.csv`) containing tweets classified as hate speech (1) or non-hate speech (0).

## Project Structure
- `Hate speech detection on twitter data_live.ipynb`: Main notebook containing the complete pipeline
- `hateDetection_train.csv`: Training dataset (not included in repository)

## Implementation Details

### Data Preprocessing
- Lowercase conversion
- URL and special character removal
- Tokenization and stopword removal
- Lemmatization for text normalization

### Feature Engineering
- TF-IDF vectorization
- N-gram feature extraction (1-3 grams)

### Model Training
- Logistic Regression with hyperparameter tuning
- Cross-validation for optimal parameter selection

### Evaluation
- Accuracy, precision, recall, and F1-score metrics
- Confusion matrix visualization
- Classification reports

## Results
The optimized Logistic Regression model achieved high accuracy in classifying hate speech, with detailed performance metrics available in the notebook.

## Requirements
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- wordcloud

## How to Use
1. Clone this repository
2. Ensure you have the required dataset (`hateDetection_train.csv`)
3. Install required libraries: `pip install -r requirements.txt`
4. Run the Jupyter notebook: `Hate speech detection on twitter data_live.ipynb`

## Future Improvements
- Test additional machine learning algorithms (Random Forest, SVM, etc.)
- Implement deep learning approaches (LSTM, BERT, etc.)
- Add real-time prediction capability for streaming tweets
- Expand dataset for improved generalization

## License
[MIT License](LICENSE)


