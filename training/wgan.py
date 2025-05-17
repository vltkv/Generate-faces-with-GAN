import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def wasserstein_loss(y_pred, y_true):
    """
    Wasserstein loss - expectation of critic outputs
    """
    return -torch.mean(y_pred * y_true)

def train_wgan(dataloader, generator, critic, optimizer_G, optimizer_C,
               num_epochs, device, nz=100, fixed_noise=None, 
               save_interval=500, n_critic=5, clip_value=0.01):
    """
    WGAN training loop
    
    Args:
        dataloader: Training data loader
        generator: Generator network
        critic: Critic network (discriminator without sigmoid)
        optimizer_G: Generator optimizer
        optimizer_C: Critic optimizer
        num_epochs: Number of training epochs
        device: Device to train on
        nz: Dimension of latent space
        fixed_noise: Fixed noise for tracking progress
        save_interval: How often to save generator output
        n_critic: Number of critic updates per generator update
        clip_value: Weight clipping value
        
    Returns:
        G_losses, C_losses, img_list
    """
    logger.info("Starting WGAN Training Loop...")
    
    # Lists to track progress
    img_list = []
    G_losses = []
    C_losses = []
    iters = 0
    
    if fixed_noise is None:
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # Prepare real and fake labels
    real_label = torch.ones(1).to(device)
    fake_label = -torch.ones(1).to(device)
    
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            ############################
            # (1) Update critic
            ###########################
            # Train with real images
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            
            for _ in range(n_critic):
                optimizer_C.zero_grad()
                
                # Forward real batch through critic
                output_real = critic(real_cpu).view(-1)
                
                # Generate batch of fake images
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = generator(noise)
                
                # Forward fake batch through critic
                output_fake = critic(fake.detach()).view(-1)
                
                # Wasserstein loss - maximize E[critic(real)] - E[critic(fake)]
                critic_loss = -(torch.mean(output_real) - torch.mean(output_fake))
                
                # Update critic
                critic_loss.backward()
                optimizer_C.step()
                
                # Clip weights to enforce Lipschitz constraint
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)
                
                C_losses.append(critic_loss.item())
            
            ############################
            # (2) Update generator
            ###########################
            optimizer_G.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            
            # Forward fake batch through critic
            output = critic(fake).view(-1)
            
            # Generator loss - maximize E[critic(fake)]
            generator_loss = -torch.mean(output)
            
            generator_loss.backward()
            optimizer_G.step()
            
            G_losses.append(generator_loss.item())
            
            # Output training stats
            if i % 50 == 0:
                logger.info(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                      f"Loss_C: {critic_loss.item():.4f} Loss_G: {generator_loss.item():.4f} "
                      f"Mean C(x): {output_real.mean().item():.4f} "
                      f"Mean C(G(z)): {output_fake.mean().item():.4f}")
            
            # Save generator progress
            if (iters % save_interval == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                from torchvision.utils import make_grid
                img_list.append(make_grid(fake, padding=2, normalize=True))
            
            iters += 1
    
    return G_losses, C_losses, img_list