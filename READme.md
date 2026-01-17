# Twitter Sentiment Analysis using BiLSTM

[![Streamlit App](https://img.shields.io/badge/Streamlit-Try%20Live-brightgreen)](https://check-tweet-sentiment.streamlit.app/)

> 🚀 Anyone can use this app to check tweet sentiment instantly using a deep learning BiLSTM model.

---


## 📌 Project Overview
This project focuses on **sentiment analysis of Twitter data** using deep learning.  
The dataset contains **69,000+ tweets**, and the goal is to classify sentiments accurately by applying robust text preprocessing, model training, and experimentation techniques.

---

## 📊 Exploratory Data Analysis (EDA)
- Analyzed the **distribution of sentiment classes**
- Studied **tweet length patterns**
- Identified **noise, emojis, HTML tags, and text repetitions**
- Checked for **class imbalance**

---

## 🧹 Data Preprocessing Pipeline
The following preprocessing steps were applied sequentially:

1. **Emoji Conversion**
   - Converted emojis into their corresponding textual meaning

2. **HTML Cleaning**
   - Removed HTML tags using **BeautifulSoup**

3. **Text Normalization**
   - Expanded contractions (e.g., *can't → cannot*)
   - Converted text to **lowercase**
   - Reduced repeated characters (e.g., *soooo → soo*)
   - Removed less important special characters
   - Removed leading and trailing whitespaces

4. **Tokenization**
   - Tokenized text into words
   - Removed stopwords using **spaCy**
   - Applied **lemmatization** to reduce words to their base form

5. **Text to Numerical Conversion**
   - Converted tokens into numerical sequences
   - Applied **padding** to ensure equal sequence length

6. **Target Encoding**
   - Used **LabelEncoder** for encoding sentiment labels

---

## 🧠 Model Architecture
- Implemented a **Bi-directional LSTM (BiLSTM)** using **PyTorch**
- Captured both **past and future context** in tweets

---

## ⚙️ Model Training & Experimentation
- Used **MLflow** for:
  - Experiment tracking
  - Metric logging
- Used **Optuna** for:
  - Hyperparameter tuning
  - Optimizing model performance

---

## 📈 Model Performance (Test Dataset)
- **Accuracy:** 88%
- **Macro F1-Score:** 88%
- **AUC Score:** 0.98

These results indicate strong generalization and excellent classification performance.

---

## 🚀 Deployment
- Converted the trained model into a **Streamlit web application**
- Enables **real-time sentiment prediction** for user-input tweets

---

## 🛠️ Tech Stack
- Python
- PyTorch
- spaCy
- BeautifulSoup
- MLflow
- Optuna
- Streamlit

---

## 📌 Conclusion
This project demonstrates an end-to-end **NLP pipeline**, from data preprocessing and deep learning model training to experimentation, optimization, and deployment.

---

