"""
Data Preprocessing Pipeline for Sentiment Analysis
Handles text cleaning, normalization, and feature extraction
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Download required NLTK data (run once)
def download_nltk_data():
    """Download necessary NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    print("NLTK data ready!")

class TextPreprocessor:
    """Handles all text cleaning and normalization"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean and normalize text:
        - Lowercase
        - Remove HTML tags
        - Remove punctuation and special characters
        - Remove extra whitespace
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags (e.g., <br />, <p>, etc.)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove punctuation and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove common English stopwords"""
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        return ' '.join(words)
    
    def apply_stemming(self, text):
        """Apply Porter stemming to reduce words to root form"""
        words = text.split()
        words = [self.stemmer.stem(w) for w in words]
        return ' '.join(words)
    
    def preprocess(self, text, apply_stemming=True):
        """
        Complete preprocessing pipeline
        Args:
            text: Raw review text
            apply_stemming: Whether to apply stemming (True for training)
        Returns:
            Cleaned and processed text
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        if apply_stemming:
            text = self.apply_stemming(text)
        return text

def create_tfidf_vectorizer(max_features=5000):
    """
    Create TF-IDF vectorizer for converting text to numerical features
    Args:
        max_features: Maximum number of features to keep
    Returns:
        Configured TfidfVectorizer
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        sublinear_tf=True,     # Use 1+log(tf)
        min_df=2,             # Ignore terms that appear in less than 2 docs
        max_df=0.95           # Ignore terms that appear in >95% of docs
    )

def load_and_preprocess_data(filepath='data/cleaned_reviews.csv', 
                             test_size=0.2,
                             random_state=42):
    """
    Main function to load data, preprocess, and create train/test splits
    
    Args:
        filepath: Path to cleaned CSV file
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    print("=" * 50)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Load data
    print(f"\n1. Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df)} reviews")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess reviews
    print("\n2. Cleaning and preprocessing reviews...")
    df['cleaned_review'] = df['review'].apply(
        lambda x: preprocessor.preprocess(x, apply_stemming=True)
    )
    
    # Show sample
    print("\n   Sample preprocessing:")
    print(f"   Original: {df['review'].iloc[0][:100]}...")
    print(f"   Cleaned:  {df['cleaned_review'].iloc[0][:100]}...")
    
    # Convert sentiment to binary
    df['sentiment_binary'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Create TF-IDF features
    print("\n3. Creating TF-IDF features...")
    vectorizer = create_tfidf_vectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment_binary'].values
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Features created: {len(vectorizer.get_feature_names_out())}")
    
    # Split data
    print(f"\n4. Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")
    
    # Save preprocessor and vectorizer
    print("\n5. Saving preprocessor and vectorizer...")
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    print("\n✅ Preprocessing complete!")
    print(f"   Files saved to models")
    
    return X_train, X_test, y_train, y_test, vectorizer

if __name__ == "__main__":
    # Download NLTK data if needed
    download_nltk_data()
    
    # Run preprocessing
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    
    # Save processed data for later use
    import scipy.sparse
    scipy.sparse.save_npz('models/X_train.npz', X_train)
    scipy.sparse.save_npz('models/X_test.npz', X_test)
    joblib.dump(y_train, 'models/y_train.joblib')
    joblib.dump(y_test, 'models/y_test.joblib')
    
    print("\n📊 Data Summary:")
    print(f"   Training positives: {sum(y_train)}")
    print(f"   Training negatives: {len(y_train) - sum(y_train)}")
    print(f"   Test positives: {sum(y_test)}")
    print(f"   Test negatives: {len(y_test) - sum(y_test)}")