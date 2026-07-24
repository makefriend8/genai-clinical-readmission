import os
import sys
import torch
import pandas as pd
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import load_config, save_generated_samples
from src.models.longt5_model import get_longt5_model, get_longt5_tokenizer
from scripts.train_longt5 import ClinicalNotesDataset

def train_model(config):
    """
    Executes the LongT5 training loop based on config parameters.
    """
    print("="*50)
    print(" [1/3] Initiating LongT5 Training Loop")
    print("="*50)
    
    model_name = config['model']['base_model']
    output_dir = os.path.join("models", "longt5-finetuned")
    
    tokenizer = get_longt5_tokenizer(model_name)
    model = get_longt5_model(model_name)
    
    train_file = config['data']['train_path']
    val_file = config['data']['val_path']
    
    # Check if we need to adjust path for execution
    if not os.path.exists(train_file) and os.path.exists(os.path.join("..", train_file)):
        train_file = os.path.join("..", train_file)
        val_file = os.path.join("..", val_file)

    train_dataset = ClinicalNotesDataset(train_file, tokenizer, 
                                         max_input_length=config['model']['max_input_length'],
                                         max_target_length=config['model']['max_output_length'])
    val_dataset = ClinicalNotesDataset(val_file, tokenizer,
                                       max_input_length=config['model']['max_input_length'],
                                       max_target_length=config['model']['max_output_length'])

    # Determine fp16 dynamically
    use_fp16 = config['training'].get('fp16', False) and torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=float(config['training']['learning_rate']),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=config['training']['num_train_epochs'],
        predict_with_generate=True,
        fp16=use_fp16,
        gradient_checkpointing=config['training']['gradient_checkpointing']
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    print(f"Starting training for {config['training']['num_train_epochs']} epoch(s)...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")
    return output_dir

def generate_preliminary_outputs(config, model_dir):
    """
    Generates BHC samples using the trained model on the test set.
    """
    print("\n"+"="*50)
    print(" [2/3] Generating Preliminary BHC Outputs")
    print("="*50)
    
    tokenizer = get_longt5_tokenizer(model_dir)
    model = get_longt5_model(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_file = config['data']['test_path']
    if not os.path.exists(test_file) and os.path.exists(os.path.join("..", test_file)):
        test_file = os.path.join("..", test_file)
        
    test_df = pd.read_csv(test_file)
    samples_to_generate = config['outputs']['samples_to_generate']
    
    # Take a subset
    subset = test_df.head(samples_to_generate)
    results = []
    
    print(f"Generating summaries for {samples_to_generate} patients...")
    for idx, row in subset.iterrows():
        input_text = str(row['input'])
        target_text = str(row['target'])
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=config['model']['max_input_length'], truncation=True).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=config['model']['max_output_length'],
                num_beams=4,
                early_stopping=True
            )
            
        generated_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({
            'input': input_text,
            'reference': target_text,
            'generated': generated_summary
        })
        print(f"  -> Generated {idx+1}/{samples_to_generate}")
        
    return results

def main():
    # 1. Load config
    config = load_config()
    
    # 2. Train Model (1 epoch)
    final_model_dir = train_model(config)
    
    # 3. Generate Outputs
    results = generate_preliminary_outputs(config, final_model_dir)
    
    # 4. Save Outputs
    print("\n"+"="*50)
    print(" [3/3] Saving Results")
    print("="*50)
    save_generated_samples(results, config['outputs']['output_file'])
    
    print("\nEnd-to-End Pipeline Complete. Check outputs/ generated files.")

if __name__ == "__main__":
    main()
