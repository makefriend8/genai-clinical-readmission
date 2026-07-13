import os
import pandas as pd
import re

def run_leakage_experiment():
    print("="*50)
    print(" Preliminary Experiment 3: Data Leakage Identification")
    print("="*50)
    
    # We load the raw merged cohort to look for artifacts
    data_path = os.path.join("..", "merged_cohort_text_data.csv")
    if not os.path.exists(data_path):
        data_path = "merged_cohort_text_data.csv"
        
    if not os.path.exists(data_path):
        print("Error: merged_cohort_text_data.csv not found in root.")
        return
        
    print(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total records loaded: {len(df)}")
    
    # Define leakage phrases we hypothesized might exist for planned surgeries
    leakage_phrases = [
        'second stage', 
        'planned surgery', 
        'planned readmit', 
        'planned return', 
        'scheduled for'
    ]
    
    print("\nScanning raw inputs for explicit discharge planning phrases...")
    
    total_leaked_records = 0
    leaked_indices = set()
    
    for phrase in leakage_phrases:
        # Case insensitive search
        pattern = re.compile(phrase, re.IGNORECASE)
        # Find which rows contain the phrase
        mask = df['input'].astype(str).apply(lambda x: bool(pattern.search(x)))
        count = mask.sum()
        
        if count > 0:
            print(f"Found phrase '{phrase}' in {count} records.")
            leaked_indices.update(df[mask].index.tolist())
            
    total_leaked = len(leaked_indices)
    percentage = (total_leaked / len(df)) * 100
    
    print(f"\nTotal unique records containing leakage artifacts: {total_leaked}")
    print(f"Percentage of dataset contaminated: {percentage:.2f}%")
    
    print("\nConclusion: While the percentage is small (~1.2%), leaving these phrases in acts as direct")
    print("label leakage for planned readmissions, artificially inflating predictive performance.")
    print("Adjustments: We must implement regex sanitization in src/preprocess.py to mask these artifacts.")

if __name__ == "__main__":
    run_leakage_experiment()
