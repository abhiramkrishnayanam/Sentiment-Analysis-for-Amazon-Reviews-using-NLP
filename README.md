# Sentiment Analysis of Amazon Reviews

This project focuses on performing sentiment analysis on Amazon product reviews. The goal is to predict whether the sentiment of a review is positive or negative. It leverages natural language processing (NLP) techniques, including text preprocessing, feature extraction, and machine learning to predict sentiment.

## Project Overview

The project involves the following steps:

1. Loading the Dataset
2. Data Preprocessing
3. Text Vectorization
4. Model Training
5. Prediction and Evaluation

## Steps Implemented

### 1. Loading the Dataset
The dataset used in this project contains Amazon product reviews, including the review text and the associated sentiment label (positive/negative). The data is loaded using the pandas library for easy manipulation and analysis.

### 2. Data Preprocessing
Data preprocessing is an essential step in NLP to prepare the raw text for analysis. In this project, several preprocessing techniques were applied:

- **Lowercasing**: All text was converted to lowercase to ensure uniformity and avoid case-sensitive discrepancies.
- **Removing Special Characters**: Non-alphabetical characters, such as punctuation marks and special symbols, were removed to reduce noise in the data.
- **Tokenization**: The text was split into individual words (tokens) for easier analysis and modeling.
- **Removing Stopwords**: Commonly used words (like "and", "the", etc.) that don't carry significant meaning were removed using a list from the NLTK library.
- **Stemming**: Words were reduced to their root form (e.g., "running" becomes "run") using the NLTK library to standardize the text.
- **Lemmatization**: Further improved by reducing words to their base form (e.g., "better" becomes "good"), preserving meaningful information while simplifying the dataset.

### 3. Text Vectorization
Text data needs to be converted into numerical form for machine learning models to process it. In this project, text vectorization was performed using the **TF-IDF (Term Frequency-Inverse Document Frequency)** method. This technique converts the text into a matrix of features, capturing the importance of words in the context of the entire dataset.

- **TF-IDF**: This method helps in converting each review into a vector where the importance of each word is based on its frequency in the review and its inverse frequency across all reviews. It helps in giving less weight to commonly occurring words and more weight to words that are unique to specific reviews.

### 4. Model Training
Once the data is preprocessed and vectorized, the next step is to train a machine learning model. The model used in this project was a **Logistic Regression** classifier, which is widely used for binary classification problems like sentiment analysis.

- The dataset was split into training and test sets to train the model on a portion of the data and evaluate its performance on unseen data.
- The model was trained to predict the sentiment of Amazon reviews (positive or negative) based on the processed text features.

### 5. Prediction and Evaluation
The trained model was used to predict the sentiment of reviews in the test set. The performance of the model was evaluated using various metrics, including:

- **Accuracy**: The percentage of correct predictions made by the model.
- **Confusion Matrix**: A matrix showing the number of true positive, true negative, false positive, and false negative predictions.
- **Precision, Recall, F1-Score**: These metrics help evaluate the model’s performance, especially when dealing with imbalanced classes.




