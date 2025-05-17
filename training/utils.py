import torch
import os
import logging
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)

def weights_init(m):
    """
    Custom weights initialization called on generator and discriminator networks
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def save_results(G_losses, D_losses, real_batch, img_list, output_dir, netG, netD, args):
    logger.info("Saving training results...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "individual_real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "individual_fake"), exist_ok=True)
    
    # Save loss plot
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator", color='#1f77b4')
    plt.plot(D_losses, label="Discriminator", color='#ff7f0e')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.close()
    
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
    
    plt.savefig(f"{output_dir}/comparison.png", bbox_inches='tight')
    plt.close()
    
    # Save individual real images
    for i in range(min(64, real_batch[0].size(0))):
        vutils.save_image(
            real_batch[0][i], 
            f"{output_dir}/individual_real/real_{i+1}.png",
            normalize=True
        )
    
    # Generate and save individual fake images
    with torch.no_grad():
        # Get the images from the last batch
        fake_batch = netG(torch.randn(64, args.nz, 1, 1, device=next(netG.parameters()).device))
        
        # Save individual fake images
        for i in range(fake_batch.size(0)):
            vutils.save_image(
                fake_batch[i], 
                f"{output_dir}/individual_fake/generated_{i+1}.png",
                normalize=True
            )
    
    # Save trained models
    torch.save(netG.state_dict(), f"{output_dir}/generator.pth")
    torch.save(netD.state_dict(), f"{output_dir}/discriminator.pth")
    
    # Create a results summary text file
    with open(f"{output_dir}/results_summary.txt", 'w') as f:
        f.write("GAN TRAINING RESULTS SUMMARY\n")
        f.write("===========================\n\n")
        
        f.write("TRAINING DETAILS\n")
        f.write("--------------\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("FINAL LOSSES\n")
        f.write("-----------\n")
        f.write(f"Final Generator Loss: {G_losses[-1]:.4f}\n")
        f.write(f"Final Discriminator Loss: {D_losses[-1]:.4f}\n")
        f.write("\n")
        
        f.write("RESULTS LOCATION\n")
        f.write("---------------\n")
        f.write(f"Loss Plot: {os.path.join(output_dir, 'loss_plot.png')}\n")
        f.write(f"Comparison Grid: {os.path.join(output_dir, 'comparison.png')}\n")
        f.write(f"Individual Real Images: {os.path.join(output_dir, 'individual_real')} (64 images)\n")
        f.write(f"Individual Generated Images: {os.path.join(output_dir, 'individual_fake')} (64 images)\n")
        f.write(f"Generator Model: {os.path.join(output_dir, 'generator.pth')}\n")
        f.write(f"Discriminator Model: {os.path.join(output_dir, 'discriminator.pth')}\n")
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE! Results saved to {output_dir}")
    print("-"*70)
    print(f"✓ Loss plot: {os.path.join(output_dir, 'loss_plot.png')}")
    print(f"✓ Comparison of real vs generated: {os.path.join(output_dir, 'comparison.png')}")
    print(f"✓ Individual real images: {os.path.join(output_dir, 'individual_real')} (64 images)")
    print(f"✓ Individual generated images: {os.path.join(output_dir, 'individual_fake')} (64 images)")
    print(f"✓ Trained models: {os.path.join(output_dir, 'generator.pth')} and {os.path.join(output_dir, 'discriminator.pth')}")
    print(f"✓ Summary report: {os.path.join(output_dir, 'results_summary.txt')}")
    print("="*70 + "\n")