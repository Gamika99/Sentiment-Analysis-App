"""
Flask Web Application for Sentiment Analysis
Loads trained model and serves predictions via REST API
"""

import re
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Global variables for model and preprocessor
model = None
vectorizer = None
preprocessor = None

class TextPreprocessor:
    """Simple text preprocessor matching the training pipeline"""
    
    def __init__(self):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        return ' '.join(words)
    
    def apply_stemming(self, text):
        """Apply Porter Stemming"""
        words = text.split()
        words = [self.stemmer.stem(w) for w in words]
        return ' '.join(words)

def load_artifacts():
    """Load trained model, vectorizer, and preprocessor"""
    global model, vectorizer, preprocessor
    
    print("=" * 50)
    print("LOADING MODEL ARTIFACTS")
    print("=" * 50)
    
    try:
        # Load trained model
        model = joblib.load('model.pkl')
        print("✅ Loaded sentiment_model.pkl")
        
        # Load TF-IDF vectorizer
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        print("✅ Loaded tfidf_vectorizer.joblib")
        
        # Create preprocessor instance (don't load from file)
        preprocessor = TextPreprocessor()
        print("✅ Created TextPreprocessor instance")
        
        print("\n🚀 All artifacts loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading artifacts: {e}")
        print("   Please run src/train.py first to generate model files")
        raise

def clean_user_input(text):
    """
    Clean user input (same preprocessing as training)
    Args:
        text: Raw review text from user
    Returns:
        Cleaned text ready for vectorization
    """
    cleaned = preprocessor.clean_text(text)
    cleaned = preprocessor.remove_stopwords(cleaned)
    cleaned = preprocessor.apply_stemming(cleaned)
    return cleaned

def predict_sentiment(review_text):
    """
    Predict sentiment of a single review
    Args:
        review_text: Raw review string
    Returns:
        Dictionary with prediction results (JSON serializable)
    """
    # Clean the input
    cleaned_review = clean_user_input(review_text)
    
    # Transform to TF-IDF features
    features = vectorizer.transform([cleaned_review])
    
    # Get prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # Convert numpy types to Python native types for JSON serialization
    if isinstance(prediction, np.generic):
        prediction = prediction.item()
    else:
        prediction = int(prediction)
    
    # Get confidence score
    confidence = probability[1] if prediction == 1 else probability[0]
    
    # Format result with native Python types
    return {
        'sentiment': "Positive" if prediction == 1 else "Negative",
        'confidence': float(round(confidence * 100, 2)),
        'prediction_label': prediction,
        'probabilities': {
            'positive': float(round(probability[1] * 100, 2)),
            'negative': float(round(probability[0] * 100, 2))
        }
    }

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts JSON with 'review' field or form data
    Returns sentiment prediction with confidence
    """
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            review_text = data.get('review', '')
        else:
            review_text = request.form.get('review', '')
        
        # Validate input
        if not review_text or len(review_text.strip()) < 3:
            return jsonify({
                'error': 'Please enter a valid review (minimum 3 characters)'
            }), 400
        
        # Get prediction
        result = predict_sentiment(review_text)
        
        # Add input metadata (ensure all values are JSON serializable)
        result.update({
            'review_length': int(len(review_text)),
            'word_count': int(len(review_text.split())),
            'truncated_review': str(review_text[:200] + '...') if len(review_text) > 200 else str(review_text)
        })
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint (bonus feature)
    Accepts JSON with list of reviews
    """
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        
        if not reviews or len(reviews) > 100:
            return jsonify({
                'error': 'Please provide 1-100 reviews'
            }), 400
        
        results = []
        for review in reviews:
            if review and len(review.strip()) >= 3:
                pred = predict_sentiment(review)
                results.append({
                    'review': str(review[:100] + '...') if len(review) > 100 else str(review),
                    'sentiment': pred['sentiment'],
                    'confidence': pred['confidence']
                })
        
        # Calculate batch statistics
        positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
        
        return jsonify({
            'total': int(len(results)),
            'positive_count': int(positive_count),
            'negative_count': int(len(results) - positive_count),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load artifacts when app starts
print("\n🔧 Initializing Sentiment Analysis Web App...")
load_artifacts()
print("\n✨ App ready! Visit http://localhost:5000\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)