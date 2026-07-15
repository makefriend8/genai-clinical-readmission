import os
import pandas as pd
import evaluate
from sklearn.metrics import roc_auc_score, classification_report

def evaluate_summarization(generated_summaries, reference_summaries):
    """
    Evaluates the quality of LongT5 generated summaries against clinician-authored BHCs.
    Metrics used: ROUGE-1, ROUGE-2, ROUGE-L
    """
    print("Evaluating Summarization Quality (ROUGE)...")
    
    # Load the official ROUGE metric from HuggingFace evaluate
    rouge = evaluate.load('rouge')
    
    # Compute scores
    results = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    
    print(f"ROUGE-1: {results.get('rouge1', 0.0):.4f}")
    print(f"ROUGE-2: {results.get('rouge2', 0.0):.4f}")
    print(f"ROUGE-L: {results.get('rougeL', 0.0):.4f}")
    
    return results

def evaluate_classification(predictions, ground_truth_labels):
    """
    Evaluates the downstream predictive performance of the summaries.
    Metric used: Area Under the Receiver Operating Characteristic Curve (AUROC)
    """
    print("Evaluating Downstream Classification (AUROC)...")
    
    auroc = roc_auc_score(ground_truth_labels, predictions)
    print(f"AUROC Score: {auroc:.4f}")
    
    # Also print standard classification metrics for completeness
    # threshold at 0.5 for binary classification report
    binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
    print("\nClassification Report:")
    print(classification_report(ground_truth_labels, binary_preds))
    
    return auroc

def main():
    print("="*50)
    print(" Evaluation Pipeline")
    print("="*50)
    
    test_file = os.path.join("data", "processed", "test.csv")
    if not os.path.exists(test_file):
        test_file = os.path.join("..", "data", "processed", "test.csv")
        
    if not os.path.exists(test_file):
        print(f"Error: test.csv not found. Ensure preprocess.py has been run.")
        return
        
    print(f"Loaded Test Set: {test_file}")
    
    # Running a dry-run with mock data to validate the evaluation pipeline execution.
    # During full evaluation, these arrays will be populated with actual model predictions.
    print("\n[Running Dry-Run Evaluation on Mock Data]")
    
    dummy_generated = ["The patient had a successful joint replacement and was discharged home."]
    dummy_reference = ["Patient admitted for joint arthroplasty, surgery went well, discharged home in stable condition."]
    
    evaluate_summarization(dummy_generated, dummy_reference)
    
    print("-" * 50)
    
    dummy_predictions = [0.1, 0.8, 0.35, 0.9]
    dummy_labels = [0, 1, 0, 1]
    
    evaluate_classification(dummy_predictions, dummy_labels)
    
    print("\n[SUCCESS] Evaluation framework is fully functional and ready for full-scale outputs.")

if __name__ == "__main__":
    main()
