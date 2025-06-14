import torch
import time
from tqdm import tqdm
import logging
from training.loss_functions import discriminator_loss, generator_loss
from training.mixup import mixup_data, mixup_criterion
from torchvision.utils import make_grid

# Configure logging
logger = logging.getLogger(__name__)

def train_gan(dataloader, netG, netD, optimizerG, optimizerD, criterion, 
              num_epochs, device, nz=100, fixed_noise=None, save_interval=500,
              use_mixup=True, mixup_alpha=0.2):
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
        use_mixup: Whether to use MixUp augmentation
        mixup_alpha: MixUp alpha parameter
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
            
            if use_mixup: # apply to real images
                real_cpu, lam = mixup_data(real_cpu, mixup_alpha)
            
            # Generate fake image batch
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            
            if use_mixup: # apply to fake images
                fake, fake_lam = mixup_data(fake, mixup_alpha)
            
            # Calculate losses and update D
            if use_mixup:
                # Create labels for mixup
                real_label = torch.full((batch_size,), 1., dtype=torch.float, device=device)
                fake_label = torch.full((batch_size,), 0., dtype=torch.float, device=device)
                
                # Get discriminator outputs
                real_output = netD(real_cpu).view(-1)
                fake_output = netD(fake.detach()).view(-1)
                
                # Calculate mixup losses
                d_real_loss = mixup_criterion(criterion, real_output, real_label, real_label, lam)
                d_fake_loss = mixup_criterion(criterion, fake_output, fake_label, fake_label, fake_lam)
                d_loss = d_real_loss + d_fake_loss
                
                real_score = real_output.mean().item()
                fake_score = fake_output.mean().item()
            else:
                d_loss, d_real_loss, d_fake_loss, real_score, fake_score = discriminator_loss(
                    netD, real_cpu, fake, criterion, device
                )
            
            d_loss.backward()
            optimizerD.step()

            # (2) Update G network
            netG.zero_grad()
            
            if use_mixup:
                # Generate new fake images for generator update
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                fake, fake_lam = mixup_data(fake, mixup_alpha)
                
                # Calculate generator loss with mixup
                fake_output = netD(fake).view(-1)
                g_loss = mixup_criterion(criterion, fake_output, real_label, real_label, fake_lam)
                fake_score = fake_output.mean().item()
            else:
                g_loss, fake_score = generator_loss(
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
                      f"D(x): {real_score:.4f} D(G(z)): {fake_score:.4f}")
            
            # Check generator progress by saving output on fixed_noise
            if (iters % save_interval == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(make_grid(fake, padding=2, normalize=True))
            
            iters += 1
            
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {time.time() - start_time:.2f}s")
    
    return G_losses, D_losses, img_list