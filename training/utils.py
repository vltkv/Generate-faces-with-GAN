import torch
import os
import logging
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from datetime import datetime
from evaluation.visualization import plot_losses, compare_real_fake, compute_fid_is

logger = logging.getLogger(__name__)

def weights_init(m):
    """Custom weights initialization called on generator and discriminator networks"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def save_results(G_losses, D_losses, real_batch, img_list, output_dir, netG, netD, args):
    logger.info("Saving training results...")
    
    real_dir = os.path.join(output_dir, "individual_real")
    fake_dir = os.path.join(output_dir, "individual_fake")
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Save loss plot
    plot_losses(G_losses, D_losses, save_path=os.path.join(output_dir, "loss_plot.png"))
    
    compare_real_fake(real_batch, img_list, save_path=output_dir)
    
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
                
        f.write(
            compute_fid_is(
                real_batch[0].to(next(netG.parameters()).device),
                fake_batch,
                device=next(netG.parameters()).device
            )
        )
        
        f.write("RESULTS LOCATION\n")
        f.write("---------------\n")
        f.write(f"Loss Plot: {os.path.join(output_dir, 'loss_plot.png')}\n")
        f.write(f"Comparison Grid: {os.path.join(output_dir, 'comparison.png')}\n")
        f.write(f"Individual Real Images: {real_dir} (64 images)\n")
        f.write(f"Individual Generated Images: {fake_dir} (64 images)\n")
        f.write(f"Generator Model: {os.path.join(output_dir, 'generator.pth')}\n")
        f.write(f"Discriminator Model: {os.path.join(output_dir, 'discriminator.pth')}\n")
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE! Results saved to {output_dir}")
    print("-"*70)
    print(f"✓ Loss plot: {os.path.join(output_dir, 'loss_plot.png')}")
    print(f"✓ Comparison of real vs generated: {os.path.join(output_dir, 'comparison.png')}")
    print(f"✓ Individual real images: {real_dir} (64 images)")
    print(f"✓ Individual generated images: {fake_dir} (64 images)")
    print(f"✓ Trained models: {os.path.join(output_dir, 'generator.pth')} and {os.path.join(output_dir, 'discriminator.pth')}")
    print(f"✓ Summary report: {os.path.join(output_dir, 'results_summary.txt')}")
    print("="*70 + "\n")