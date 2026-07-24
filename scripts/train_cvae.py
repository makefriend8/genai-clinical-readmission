import os
import sys
import torch
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cvae_model import CVAE, cvae_loss_function
from src.models.clinical_longformer_model import get_clinical_longformer_embeddings_model, get_clinical_longformer_tokenizer

def extract_real_embeddings(texts, model, tokenizer, device, batch_size=8):
    """
    Passes clinical notes through Clinical-Longformer to get 768-D [CLS] embeddings.
    """
    model.eval()
    all_embeddings = []
    
    print(f"Extracting embeddings for {len(texts)} samples using Clinical-Longformer...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Longformer handles up to 4096 tokens
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=4096, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Take the [CLS] token representation (the first token) as the sentence embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())
            
    return torch.cat(all_embeddings, dim=0)

def train_cvae_real_data_loop():
    print("="*50)
    print(" [CVAE Module] Initiating CVAE Integration")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_file = os.path.join("data", "processed", "train.csv")
        
    if not os.path.exists(train_file):
        print(f"Error: Could not find training data at {train_file}.")
        return
        
    print(f"-> Loading dataset from {train_file}")
    df = pd.read_csv(train_file)
    
    # Set USE_FULL_DATASET = True to execute full-scale training on the entire dataset.
    USE_FULL_DATASET = False
    
    if USE_FULL_DATASET:
        demo_subset = df.copy()
        print(f"-> Processing full dataset: {len(demo_subset)} samples")
    else:
        demo_subset = df.head(50).copy()
        print(f"-> Processing demo subset: {len(demo_subset)} samples")
        
    texts = demo_subset['input'].astype(str).tolist()
    
    # Ensure the exact label column exists from preprocessed data
    if 'readmitted_30day' not in demo_subset.columns:
        raise ValueError("Error: 'readmitted_30day' column not found in the dataset. Please check your data preprocessing pipeline.")
        
    labels_tensor = torch.tensor(demo_subset['readmitted_30day'].values)
        
    conditions = F.one_hot(labels_tensor, num_classes=2).float().to(device)

    # 2. Extract Embeddings using Clinical-Longformer
    print("-> Loading Clinical-Longformer...")
    lf_tokenizer = get_clinical_longformer_tokenizer()
    lf_model = get_clinical_longformer_embeddings_model()
    lf_model.to(device)
    
    # Get true embeddings
    real_embeddings = extract_real_embeddings(texts, lf_model, lf_tokenizer, device, batch_size=2)
    real_embeddings = real_embeddings.to(device)
    
    # Free up memory before training CVAE
    del lf_model
    torch.cuda.empty_cache()

    # 3. Train CVAE
    print("-> Initializing CVAE...")
    cvae = CVAE(input_dim=768, hidden_dim=256, latent_dim=128, num_classes=2).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)
    cvae.train()
    
    epochs = 5
    print(f"-> Training CVAE for {epochs} epochs on real embeddings...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = cvae(real_embeddings, conditions)
        
        # Calculate loss
        loss = cvae_loss_function(recon_batch, real_embeddings, mu, logvar, beta=1.0)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    print("-> CVAE Pipeline verified successfully.")
    
    # 4. Save Checkpoint
    output_dir = os.path.join("models", "cvae-checkpoint")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(cvae.state_dict(), os.path.join(output_dir, "cvae_weights_real.pt"))
    print(f"-> Real Model weights saved to {output_dir}/cvae_weights_real.pt")

if __name__ == "__main__":
    train_cvae_real_data_loop()
