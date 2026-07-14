import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def run_feasibility_test():
    print("="*50)
    print(" Preliminary Experiment 1: LongT5 Real Data Feasibility")
    print("="*50)
    
    # Load actual training data
    train_path = os.path.join("..", "data", "processed", "train.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join("data", "processed", "train.csv")
        
    if not os.path.exists(train_path):
        print("Error: train.csv not found. Please run preprocess.py first.")
        return
        
    print(f"Loading real data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Grab the very first clinical note
    sample_note = df['input'].iloc[0]
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on Device: {device}")
    
    model_name = "google/long-t5-tglobal-base"
    print(f"\nInitializing {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        print("\nTokenizing 1 real clinical note...")
        inputs = tokenizer(sample_note, max_length=4096, truncation=True, return_tensors="pt").to(device)
        
        print(f"Input Tensor Shape: {inputs['input_ids'].shape} (1 batch, {inputs['input_ids'].shape[1]} tokens)")
        
        print("\nRunning forward pass through LongT5...")
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=50)
            
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("[SUCCESS] Feasibility Test Passed! Model successfully processed real clinical data and generated output.")
        print(f"Sample generated text snippet: '{decoded_output}'")
        
    except Exception as e:
        print(f"[FAILED] Feasibility Test Failed: {str(e)}")

if __name__ == "__main__":
    run_feasibility_test()
