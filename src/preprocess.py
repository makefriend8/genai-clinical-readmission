import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

def mask_leakage(text):
    """
    Mask explicit planning phrases to prevent label leakage.
    We replace them with [MASK] to keep the model from learning these artifacts.
    """
    if pd.isna(text):
        return ""
        
    leakage_keywords = [
        r'second stage', 
        r'planned surgery', 
        r'planned readmit', 
        r'planned return', 
        r'scheduled for'
    ]
    
    masked_text = str(text)
    # Perform case-insensitive regex replacement
    for kw in leakage_keywords:
        # We replace the keyword and surrounding spaces with [MASK]
        masked_text = re.sub(kw, '[MASK]', masked_text, flags=re.IGNORECASE)
        
    return masked_text

def main():
    print("="*50)
    print(" Data Preprocessing Pipeline")
    print("="*50)
    
    # Define paths
    input_path = 'merged_cohort_text_data.csv'
    output_dir = os.path.join('data', 'processed')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[1/4] Loading raw data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please ensure the file is in the root directory.")
        return

    print("\n[2/4] Masking label leakage artifacts in 'input' text...")
    if 'input' in df.columns:
        df['input_clean'] = df['input'].apply(mask_leakage)
        
        # Verify masking
        test_mask = df['input_clean'].str.contains(r'\[MASK\]', regex=True, na=False)
        print(f"Masked label leakage in {test_mask.sum()} records.")
    else:
        print("Error: 'input' column not found in data.")
        return
        
    # We will use 'input_clean' for model training, but let's rename it to 'input' and 'target' 
    # to match standard HuggingFace Seq2Seq formats.
    if 'target' not in df.columns:
        print("Error: 'target' column not found in data.")
        return
        
    # Keep only the essential columns for generative modeling
    processed_df = df[['note_id', 'subject_id', 'hadm_id', 'input_clean', 'target', 'readmitted_30day']].copy()
    processed_df.rename(columns={'input_clean': 'input', 'target': 'target'}, inplace=True)
    
    # Drop empty rows
    processed_df = processed_df.dropna(subset=['input', 'target'])
    print(f"Data shape after cleaning: {processed_df.shape}")

    print("\n[3/4] Splitting data into Train (80%), Val (10%), Test (10%)...")
    # We use stratified split to maintain the 9.95% readmission ratio across all sets
    train_val_df, test_df = train_test_split(
        processed_df, 
        test_size=0.10, 
        random_state=42, 
        stratify=processed_df['readmitted_30day']
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=1/9, # 1/9 of 90% is 10% of total
        random_state=42, 
        stratify=train_val_df['readmitted_30day']
    )
    
    print(f"Train set: {len(train_df)} records")
    print(f"Val set:   {len(val_df)} records")
    print(f"Test set:  {len(test_df)} records")
    
    print("\n[4/4] Saving processed datasets...")
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Datasets successfully saved to {output_dir}")
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()
