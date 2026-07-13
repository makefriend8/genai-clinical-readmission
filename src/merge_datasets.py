import os
import pandas as pd

def merge_raw_datasets():
    print("="*50)
    print(" Data Merging: Structured Cohort + Unstructured Text")
    print("="*50)
    
    cohort_path = "cohort_final_v2.csv"
    bhc_path = "mimic-iv-bhc.csv"
        
    if not os.path.exists(cohort_path) or not os.path.exists(bhc_path):
        print(f"Error: Could not find raw data files.")
        print(f"Ensure {cohort_path} and {bhc_path} exist in the current directory.")
        return
        
    print(f"Loading Structured Cohort ({cohort_path})...")
    cohort_df = pd.read_csv(cohort_path)
    print(f"Cohort Records: {len(cohort_df)}")
    
    print(f"Loading Unstructured BHC Text ({bhc_path})...")

    bhc_df = pd.read_csv(bhc_path, usecols=['note_id', 'input', 'target'])
    print(f"BHC Records: {len(bhc_df)}")
    
    print("\nMerging on 'note_id' (Inner Join)...")
    merged_df = pd.merge(cohort_df, bhc_df, on='note_id', how='inner')
    
    print(f"Final Merged Records: {len(merged_df)}")
    
    output_path = "merged_cohort_text_data.csv"
    merged_df.to_csv(output_path, index=False)
    
    # Export structure and a 5-row sample 
    sample_path = os.path.join("docs", "merged_dataset_sample.txt")
    if not os.path.exists("docs"):
        os.makedirs("docs")
        
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(" Merged Dataset Structure (Data Types)\n")
        f.write("="*50 + "\n")
        f.write(str(merged_df.dtypes) + "\n\n")
        
        f.write("="*50 + "\n")
        f.write(" First 5 Rows (Sample - REDACTED FOR DUA COMPLIANCE)\n")
        f.write("="*50 + "\n")
        
        # Create a redacted copy of the first 5 rows to strictly comply with MIMIC-IV Data Use Agreement
        sample_df = merged_df.head(5).copy()
        for col in ['input', 'target']:
            if col in sample_df.columns:
                sample_df[col] = sample_df[col].apply(
                    lambda x: f"[REDACTED PER MIMIC-IV DUA | Original Length: {len(str(x))} chars]"
                )
        
        # Use option_context to prevent pandas from truncating columns
        with pd.option_context('display.max_columns', None, 
                               'display.max_colwidth', None, 
                               'display.width', None):
            f.write(str(sample_df) + "\n")
        
    print(f"\n[SUCCESS] Merge complete! Output saved to: {output_path}")
    print(f"[SUCCESS] Data structure and 5-row sample exported to: {sample_path}")

if __name__ == "__main__":
    merge_raw_datasets()
