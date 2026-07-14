import os
import pandas as pd
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset
import sys

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.longt5_model import get_longt5_model, get_longt5_tokenizer

class ClinicalNotesDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_input_length=4096, max_target_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize input (Raw Notes)
        model_inputs = self.tokenizer(
            str(row['input']), 
            max_length=self.max_input_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target (BHC)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                str(row['target']), 
                max_length=self.max_target_length, 
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            )
            
        # Squeeze to remove batch dimension added by return_tensors="pt"
        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        labels_ids = labels["input_ids"].squeeze()
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids
        }

def main():
    print("="*50)
    print(" LongT5 Training Script")
    print("="*50)
    
    # Configuration
    model_name = "google/long-t5-tglobal-base"
    train_file = os.path.join("data", "processed", "train.csv")
    val_file = os.path.join("data", "processed", "val.csv")
    output_dir = os.path.join("models", "longt5-finetuned")
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Please run src/preprocess.py first.")
        return

    print("[1/3] Loading Tokenizer and Model...")
    tokenizer = get_longt5_tokenizer(model_name)
    model = get_longt5_model(model_name)
    
    print("[2/3] Preparing Datasets...")
    # For Week 11 dry-run, we might want to subset the data to prevent OOM on local machines
    train_dataset = ClinicalNotesDataset(train_file, tokenizer)
    val_dataset = ClinicalNotesDataset(val_file, tokenizer)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    print("[3/3] Setting up Trainer...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1, # Very small batch size for LongT5 due to memory constraints
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        # max_steps=5, # Uncomment for a quick 5-step dry run during development
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("Trainer setup complete! Ready to start training.")
    print("To execute training, uncomment trainer.train() in the script.")
    # trainer.train()

if __name__ == "__main__":
    main()
