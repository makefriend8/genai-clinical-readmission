import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for latent space data augmentation.
    Used to generate synthetic minority-class embeddings (e.g. readmission = 1).
    """
    def __init__(self, input_dim=768, hidden_dim=256, latent_dim=128, num_classes=2):
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        # Inputs: original embedding + one-hot encoded condition
        self.encoder_fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.encoder_fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        # Inputs: latent vector + one-hot encoded condition
        self.decoder_fc1 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        # Concatenate embedding and condition
        inputs = torch.cat([x, c], dim=1)
        h1 = F.relu(self.encoder_fc1(inputs))
        mu = self.encoder_fc2_mu(h1)
        logvar = self.encoder_fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # Concatenate latent vector and condition
        inputs = torch.cat([z, c], dim=1)
        h3 = F.relu(self.decoder_fc1(inputs))
        # No non-linear activation at the end because embeddings can have negative values
        return self.decoder_fc2(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z, c)
        return reconstructed_x, mu, logvar
        
    def generate(self, c, num_samples):
        """
        Generate synthetic samples for a given condition.
        """
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z, c)

def cvae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the VAE loss function: Reconstruction Loss + KL Divergence.
    """
    # MSE for reconstruction loss since we are dealing with continuous embeddings
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + beta * KLD

if __name__ == "__main__":
    # Quick sanity check
    print("Testing CVAE architecture...")
    model = CVAE()
    
    # Dummy data: batch of 4 embeddings of size 768
    dummy_x = torch.randn(4, 768)
    
    # Dummy labels: [1, 0, 1, 1] (one-hot encoded)
    dummy_labels = torch.tensor([1, 0, 1, 1])
    dummy_c = F.one_hot(dummy_labels, num_classes=2).float()
    
    recon_x, mu, logvar = model(dummy_x, dummy_c)
    loss = cvae_loss_function(recon_x, dummy_x, mu, logvar)
    
    print(f"Output shape: {recon_x.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("CVAE architecture successfully loaded!")
