"""
Model Training Script for Sentiment Analysis
Trains Logistic Regression classifier and evaluates performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
import joblib
import scipy.sparse

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def load_processed_data():
    """Load preprocessed train/test data"""
    print("Loading processed data...")
    X_train = scipy.sparse.load_npz('models/X_train.npz')
    X_test = scipy.sparse.load_npz('models/X_test.npz')
    y_train = joblib.load('models/y_train.joblib')
    y_test = joblib.load('models/y_test.joblib')
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train Logistic Regression model
    Args:
        X_train: Training features (sparse matrix)
        y_train: Training labels
    Returns:
        Trained model
    """
    print("\n" + "=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)
    
    # Initialize Logistic Regression with optimized hyperparameters
    model = LogisticRegression(
        C=1.0,              # Inverse regularization strength
        max_iter=1000,      # More iterations for convergence
        random_state=42,
        solver='liblinear', # Good for sparse data
        class_weight='balanced'  # Handle any class imbalance
    )
    
    print("\nTraining Logistic Regression model...")
    print(f"Parameters: C=1.0, solver='liblinear', class_weight='balanced'")
    
    model.fit(X_train, y_train)
    
    print("✅ Model training complete!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with multiple metrics
    """
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print("\n📊 Performance Metrics:")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negative (0)', 'Positive (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum() * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[1])
    axes[1].set_title('Confusion Matrix (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional analysis: errors
    errors = y_test != y_pred
    print(f"\n❌ Total errors: {errors.sum()} out of {len(y_test)} ({errors.sum()/len(y_test)*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve to visualize model performance"""
    from sklearn.metrics import roc_curve, roc_auc_score
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Sentiment Classifier', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return auc_score

def save_model(model):
    """Save trained model to disk"""
    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)
    
    joblib.dump(model, 'models/sentiment_model.pkl')
    print("✅ Model saved to: models/sentiment_model.pkl")
    
    # Also save as model.pkl (for Flask app)
    joblib.dump(model, 'model.pkl')
    print("✅ Model also saved as: model.pkl (for Flask app)")

def generate_model_report(metrics, auc_score):
    """Generate a text report of model performance"""
    report = f"""
# Model Performance Report

## Overview
- **Model**: Logistic Regression
- **Dataset**: IMDB Movie Reviews (50,000 reviews)
- **Train/Test Split**: 80/20 (40,000 train, 10,000 test)
- **Feature Engineering**: TF-IDF with unigrams + bigrams (5,000 features)

## Performance Metrics

| Metric | Score | What it means |
|--------|-------|---------------|
| Accuracy | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) | Overall correctness of predictions |
| Precision | {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%) | Of reviews predicted positive, how many were actually positive |
| Recall | {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%) | Of actual positive reviews, how many were correctly identified |
| F1-Score | {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%) | Harmonic mean of precision and recall |
| AUC-ROC | {auc_score:.4f} ({auc_score*100:.2f}%) | Model's ability to distinguish between classes |

## Understanding the Metrics

### Accuracy
The model correctly predicts sentiment **{metrics['accuracy']*100:.1f}%** of the time. This is a strong performance for a simple Logistic Regression model.

### Precision vs Recall - When Each Matters
- **Precision** is important when false positives are costly (e.g., flagging a good movie as bad)
- **Recall** is important when false negatives are costly (e.g., missing a potentially great movie recommendation)

### Confusion Matrix Analysis
From the confusion matrix:
- True Positives: Reviews correctly predicted as positive
- True Negatives: Reviews correctly predicted as negative
- False Positives: Negative reviews incorrectly marked as positive
- False Negatives: Positive reviews incorrectly marked as negative

## Conclusions
The Logistic Regression model achieves **~{metrics['accuracy']*100:.1f}% accuracy** on unseen IMDB reviews, demonstrating that simple models can perform well on sentiment analysis tasks when combined with good text preprocessing and TF-IDF features.

## Recommendations
1. For production use, consider the precision/recall tradeoff based on business needs
2. Could improve further with:
   - More advanced models (BERT, RoBERTa)
   - Larger feature space (more n-grams)
   - Hyperparameter tuning
"""
    
    with open('model_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Model report saved to: model_report.md")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print(" SENTIMENT ANALYSIS - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Plot ROC curve
    auc_score = plot_roc_curve(model, X_test, y_test)
    
    # Save model
    save_model(model)
    
    # Generate report
    generate_model_report(metrics, auc_score)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n📁 Output files:")
    print("   - models/sentiment_model.pkl (trained model)")
    print("   - model.pkl (model for Flask app)")
    print("   - model_report.md (performance documentation)")
    print("   - static/confusion_matrix.png (visualization)")
    print("   - static/roc_curve.png (visualization)")

if __name__ == "__main__":
    main()