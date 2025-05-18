import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch
import os
import logging
import torch.nn.functional as F
from datetime import datetime
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms import Resize, CenterCrop

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

def compare_real_fake(real_batch, img_list, save_path=None, num_images=64):
    """Create detailed comparison between real and generated images and save individual images"""
    # Create comparison grid
    real_grid = vutils.make_grid(real_batch[0][:64], padding=5, normalize=True)
    fake_grid = img_list[-1]
    
    plt.figure(figsize=(15, 8))
    
    # Real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images", fontsize=14)
    plt.imshow(torch.permute(real_grid, (1, 2, 0)).cpu().numpy())
    
    # Fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Generated Images", fontsize=14)
    plt.imshow(torch.permute(fake_grid, (1, 2, 0)).cpu().numpy())
    
    # Add timestamp
    plt.figtext(0.5, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}/comparison.png", bbox_inches='tight')
        logger.info(f"Comparison image saved to {save_path}")
    
    plt.close()
    
    return save_path

def compute_fid_is(real_images, fake_images, device='cuda'):
    """Compute FID and IS metrics for a batch of real and generated images, return results as a string."""
    assert isinstance(real_images, torch.Tensor) and isinstance(fake_images, torch.Tensor)
    
    # Resize to 299x299 for InceptionV3
    real_resized = F.interpolate(real_images.to(device), size=(299, 299), mode='bilinear', align_corners=False)
    fake_resized = F.interpolate(fake_images.to(device), size=(299, 299), mode='bilinear', align_corners=False)

    # FID
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.update(real_resized, real=True)
    fid.update(fake_resized, real=False)
    fid_score = fid.compute().item()
    
    # TODO:  Metric `InceptionScore` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
    #        Obie metryki puścić na wybranych próbkach danych
    # IS
    inception = InceptionScore(normalize=True).to(device)
    inception.update(fake_resized)
    is_mean, is_std = inception.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()

    # Prepare result string
    results = (
        "FID & IS METRICS\n"
        + "--------------\n"
        + f"FID Score: {fid_score:.4f} (lower is better)\n"
        + f"Inception Score: {is_mean:.4f} ± {is_std:.4f} (higher is better)\n"
        + "\n"
    )

    return results

