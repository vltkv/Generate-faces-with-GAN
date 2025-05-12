import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torchvision.utils as vutils
import logging

logger = logging.getLogger(__name__)

def plot_losses(G_losses, D_losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        logger.info("Loss plot saved to {save_path}")
    plt.show()

def create_progress_animation(img_list): # generator's progress
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.close()
    return ani

def compare_real_fake(real_batch, fake_batch, save_path=None):
    plt.figure(figsize=(15, 7))
    
    # Real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(real_batch, (1, 2, 0)))
    
    # Fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_batch, (1, 2, 0)))
    
    if save_path:
        plt.savefig(save_path)
        logger.info("Comparison image saved to {save_path}")
    
    plt.show()