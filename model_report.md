
# Model Performance Report

## Overview
- **Model**: Logistic Regression
- **Dataset**: IMDB Movie Reviews (50,000 reviews)
- **Train/Test Split**: 80/20 (40,000 train, 10,000 test)
- **Feature Engineering**: TF-IDF with unigrams + bigrams (5,000 features)

## Performance Metrics

| Metric | Score | What it means |
|--------|-------|---------------|
| Accuracy | 0.8883 (88.83%) | Overall correctness of predictions |
| Precision | 0.8815 (88.15%) | Of reviews predicted positive, how many were actually positive |
| Recall | 0.8981 (89.81%) | Of actual positive reviews, how many were correctly identified |
| F1-Score | 0.8897 (88.97%) | Harmonic mean of precision and recall |
| AUC-ROC | 0.9585 (95.85%) | Model's ability to distinguish between classes |

## Understanding the Metrics

### Accuracy
The model correctly predicts sentiment **88.8%** of the time. This is a strong performance for a simple Logistic Regression model.

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
The Logistic Regression model achieves **~88.8% accuracy** on unseen IMDB reviews, demonstrating that simple models can perform well on sentiment analysis tasks when combined with good text preprocessing and TF-IDF features.

## Recommendations
1. For production use, consider the precision/recall tradeoff based on business needs
2. Could improve further with:
   - More advanced models (BERT, RoBERTa)
   - Larger feature space (more n-grams)
   - Hyperparameter tuning
