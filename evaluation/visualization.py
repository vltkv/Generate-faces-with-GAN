import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def plot_losses(G_losses, D_losses, save_path=None):
    """Plot generator and discriminator loss curves"""
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator", color='#1f77b4')
    plt.plot(D_losses, label="Discriminator", color='#ff7f0e')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Loss plot saved to {save_path}")
    
    plt.close()
    return save_path

def compare_real_fake(real_batch, fake_batch, save_path=None, num_images=64):
    """Create detailed comparison between real and generated images and save individual images"""
    plt.figure(figsize=(15, 8))
    
    # Real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images", fontsize=14)
    if isinstance(real_batch, torch.Tensor) and real_batch.dim() == 4:
        real_grid = vutils.make_grid(real_batch, padding=2, normalize=True)
        plt.imshow(np.transpose(real_grid, (1, 2, 0)))
    else:
        plt.imshow(np.transpose(real_batch, (1, 2, 0)))
    
    # Fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Generated Images", fontsize=14)
    if isinstance(fake_batch, torch.Tensor) and fake_batch.dim() == 4:
        fake_grid = vutils.make_grid(fake_batch, padding=2, normalize=True)
        plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
    else:
        plt.imshow(np.transpose(fake_batch, (1, 2, 0)))
    
    # Add timestamp
    plt.figtext(0.5, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Comparison image saved to {save_path}")
    
    plt.close()
    
    # Save individual images
    if save_path:
        save_dir = os.path.dirname(save_path)
        
        real_dir = os.path.join(save_dir, 'individual_real')
        fake_dir = os.path.join(save_dir, 'individual_fake')
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        if isinstance(real_batch, torch.Tensor) and real_batch.dim() == 4:
            for i in range(min(num_images, real_batch.size(0))):
                img = real_batch[i].detach().cpu()
                vutils.save_image(img, os.path.join(real_dir, f'real_{i+1}.png'), normalize=True)
        
        if isinstance(fake_batch, torch.Tensor) and fake_batch.dim() == 4:
            for i in range(min(num_images, fake_batch.size(0))):
                img = fake_batch[i].detach().cpu()
                vutils.save_image(img, os.path.join(fake_dir, f'generated_{i+1}.png'), normalize=True)
        
        logger.info(f"Individual images saved to {real_dir} and {fake_dir}")
        
        print("\n" + "="*70)
        print(f"RESULTS SAVED SUCCESSFULLY")
        print("-"*70)
        print(f"✓ Comparison image: {save_path}")
        print(f"✓ Individual real images: {real_dir}")
        print(f"✓ Individual generated images: {fake_dir}")
        print("="*70 + "\n")
    
    return save_path