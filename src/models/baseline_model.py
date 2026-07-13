import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

def train_baseline_model(train_csv_path, val_csv_path):
    """
    Experiment 0 (Global Baseline): 
    Trains a standard Bag-of-Words (TF-IDF) + Logistic Regression classifier 
    on raw notes to establish floor predictive performance for 30-day readmission.
    """
    print("="*50)
    print(" Training Global Baseline (BoW + LR)")
    print("="*50)
    
    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        print("Error: Preprocessed data not found. Run preprocess.py first.")
        return
        
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    # We use 'source' which contains the raw unstructured clinical notes
    X_train = train_df['source'].fillna('')
    y_train = train_df['readmitted_30day']
    
    X_val = val_df['source'].fillna('')
    y_val = val_df['readmitted_30day']
    
    # Define the Bag-of-Words + Logistic Regression Pipeline
    # Using class_weight='balanced' to handle the 9.95% minority class imbalance
    baseline_pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=10000, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    print(f"Training on {len(X_train)} samples...")
    baseline_pipeline.fit(X_train, y_train)
    print("Training complete.")
    
    print("\nEvaluating on Validation Set...")
    # Predict probabilities for AUROC
    y_pred_proba = baseline_pipeline.predict_proba(X_val)[:, 1]
    y_pred = baseline_pipeline.predict(X_val)
    
    auroc = roc_auc_score(y_val, y_pred_proba)
    print(f"Baseline Validation AUROC: {auroc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return baseline_pipeline

if __name__ == "__main__":
    train_path = os.path.join("..", "..", "data", "processed", "train.csv")
    val_path = os.path.join("..", "..", "data", "processed", "val.csv")
    
    # Try local paths if running from root
    if not os.path.exists(train_path):
        train_path = os.path.join("data", "processed", "train.csv")
        val_path = os.path.join("data", "processed", "val.csv")
        
    train_baseline_model(train_path, val_path)
