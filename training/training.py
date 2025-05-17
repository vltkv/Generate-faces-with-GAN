import torch
import time
from tqdm import tqdm
import logging
from training.loss_functions import discriminator_loss, generator_loss

# Configure logging
logger = logging.getLogger(__name__)

def train_gan(dataloader, netG, netD, optimizerG, optimizerD, criterion, 
              num_epochs, device, nz=100, fixed_noise=None, save_interval=500):
    """
    Main training loop for DCGAN
    
    Args:
        dataloader: DataLoader with training data
        netG: Generator network
        netD: Discriminator network
        optimizerG: Generator optimizer
        optimizerD: Discriminator optimizer
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on
        nz: Dimension of latent space
        fixed_noise: Fixed noise for tracking progress
        save_interval: How often to save generator output
        
    Returns:
        G_losses, D_losses, img_list
    """
    logger.info("Starting Training Loop...")
    
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    if fixed_noise is None:
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # (1) Update D network
            netD.zero_grad()
            
            # Train with real batch
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            
            # Generate fake image batch
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            
            # Calculate losses and update D
            d_loss, d_real_loss, d_fake_loss, d_x, d_g_z1 = discriminator_loss(
                netD, real_cpu, fake, criterion, device
            )
            d_loss.backward()
            optimizerD.step()

            # (2) Update G network
            netG.zero_grad()
            
            g_loss, d_g_z2 = generator_loss(
                netD, fake, criterion, device
            )
            g_loss.backward()
            optimizerG.step()
            
            # Save losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            
            # Output training stats periodically
            if i % 50 == 0:
                logger.info(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} "
                      f"D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}")
            
            # Check generator progress by saving output on fixed_noise
            if (iters % save_interval == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                from torchvision.utils import make_grid
                img_list.append(make_grid(fake, padding=2, normalize=True))
            
            iters += 1
            
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {time.time() - start_time:.2f}s")
    
    return G_losses, D_losses, img_list