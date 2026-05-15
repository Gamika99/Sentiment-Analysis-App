# 🎬 Sentiment Analysis App

A web application that analyzes movie reviews and predicts if they are **Positive** or **Negative** using Machine Learning.

## 📖 About The Project

This project is a complete end-to-end machine learning application that:
- Takes movie reviews as input
- Processes and cleans the text
- Predicts whether the review is Positive or Negative
- Returns confidence scores
- Provides a web interface and REST API

The model is trained on 50,000 IMDB movie reviews and achieves **92% accuracy**.

## ✨ Features

### Core Features
- **Real-time Sentiment Analysis** - Get instant predictions
- **High Accuracy** - 89% accuracy on test data
- **Confidence Scores** - See probability percentages
- **Beautiful Web Interface** - Modern, responsive design
- **REST API** - Programmatic access for developers
- **Batch Prediction** - Analyze multiple reviews at once

### Technical Features
- Text preprocessing (cleaning, stopword removal, stemming)
- TF-IDF vectorization with n-grams
- Logistic Regression classification
- JSON serialization support
- CORS enabled for cross-origin requests

## 🛠 Technologies Used

| Category | Technologies |
| **Backend** | Flask, Python 3.8+ |
| **Machine Learning** | Scikit-learn, Logistic Regression |
| **NLP** | NLTK, TF-IDF Vectorizer |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Joblib |
| **Frontend** | HTML5, CSS3, JavaScript |

## 📥 Installation Steps

### Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed
- pip package manager
- Git (optional)

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/gamika99/sentiment-analysis-app.git
cd sentiment-analysis-app

### Step 2: Clone or Download the Project
python -m venv venv

### Step 3: Activate Virtual Environment

```bash
venv\Scripts\activate

### Step 4: Install dependencies
pip install flask flask-cors joblib numpy scikit-learn pandas nltk matplotlib seaborn

### Step 5: Download NLTK Data
bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

### Step 6: Download Dataset
Go to: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### Step 7: Preprocess Data
bash
python src/preprocess.py

### Step 8: Train Model
bash
python src/train.py

### Step 9: Run App
bash
python app.py