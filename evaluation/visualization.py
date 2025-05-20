import random
import lpips
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
from pytorch_msssim import ms_ssim
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
        os.makedirs(save_path, exist_ok=True)
        comparison_path = os.path.join(save_path, "comparison.png")
        plt.savefig(comparison_path, bbox_inches='tight')
        logger.info(f"Comparison image saved to {comparison_path}")
    
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

    # IS
    inception = InceptionScore(normalize=True).to(device)
    inception.update(fake_resized)
    is_mean, is_std = inception.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()
    
    # Prepare images for MS-SSIM metric:
    # MS-SSIM expects images resized to 256x256 (or any fixed size) and values in [0,1]
    fake_for_ssim = F.interpolate(fake_images.to(device), size=(256, 256), mode='bilinear', align_corners=False)

    # Compute average MS-SSIM over a number of random pairs of generated images
    n_pairs = 100
    ms_ssim_scores = []
    for _ in range(n_pairs):
        # Randomly sample two different indices from the batch
        i, j = random.sample(range(fake_for_ssim.size(0)), 2)
        img1 = fake_for_ssim[i].unsqueeze(0)  # Add batch dimension
        img2 = fake_for_ssim[j].unsqueeze(0)
        score = ms_ssim(img1, img2, data_range=1.0).item()
        ms_ssim_scores.append(score)
    ms_ssim_avg = sum(ms_ssim_scores) / n_pairs  # Average MS-SSIM; lower means more diversity

    # Initialize LPIPS model to measure perceptual similarity/differences
    loss_fn = lpips.LPIPS(net='alex').to(device)
    # LPIPS expects images in [-1,1] range, so scale accordingly
    fake_for_lpips = (fake_for_ssim * 2) - 1

    # Compute average LPIPS over the same number of random pairs as MS-SSIM
    lpips_scores = []
    for _ in range(n_pairs):
        i, j = random.sample(range(fake_for_lpips.size(0)), 2)
        img1 = fake_for_lpips[i].unsqueeze(0)
        img2 = fake_for_lpips[j].unsqueeze(0)
        with torch.no_grad():
            dist = loss_fn(img1, img2).item()
        lpips_scores.append(dist)
    lpips_avg = sum(lpips_scores) / n_pairs  # Higher LPIPS indicates greater perceptual diversity

    # Prepare result string
    results = (
        "MODEL EVALUATION METRICS\n"
        "------------------------\n"
        f"FID Score: {fid_score:.4f} (lower is better)\n"
        f"Inception Score: {is_mean:.4f} ± {is_std:.4f} (higher is better)\n"
        f"MS-SSIM (avg over {n_pairs} pairs): {ms_ssim_avg:.4f} (lower = more diversity)\n"
        f"LPIPS (avg over {n_pairs} pairs): {lpips_avg:.4f} (higher = more diversity)\n\n"
    )

    return results

