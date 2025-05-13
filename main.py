import torch
import random
import torch.nn as nn
import numpy as np
import argparse
import os
import logging

from data.preprocessing import get_dataset, create_dataloader
from models.generator import Generator, weights_init
from models.discriminator import Discriminator
from training.optimizers import get_optimizers
from training.loss_functions import get_criterion, discriminator_loss, generator_loss
from training.training import train_gan
from evaluation.visualization import plot_losses, compare_real_fake
import torchvision.utils as vutils

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_device_and_seed(opt):
    """
    Sets the random seed for reproducibility and selects the computation device (CPU or GPU).

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing configuration options.

    Returns:
        torch.device: The selected device for training (either 'cuda:0' or 'cpu').
    """
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        logger.warning(f"Deterministic algorithms not supported: {e}")
    
    # Decide which device to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
    logger.info(f"Using device: {device}")
    return device

def prepare_data(opt):
    """
    Loads the dataset and creates a DataLoader based on the provided options.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing dataset and loader configuration.

    Returns:
        tuple: A tuple containing (dataset, dataloader). Returns (None, None) if loading fails.
    """
    logger.info("Setting up dataset...")
    dataset = get_dataset(
        opt.dataroot, 
        opt.image_size, 
        use_kagglehub=opt.use_kagglehub,
        max_images=opt.max_images
    )
    if dataset is None:
        logger.error("Failed to load dataset.")
        return None, None

    dataloader = create_dataloader(dataset, opt.batch_size, opt.workers)
    if dataloader is None:
        logger.error("Failed to create dataloader.")
    return dataloader

def weights_init(m):
    """
    Initializes weights for model layers using DCGAN-style initialization:
    - Normal distribution for Conv layers
    - Normal distribution with mean 1 and std 0.02 for BatchNorm layers

    Args:
        m (nn.Module): A layer/module from the neural network.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def create_models(opt, device):
    """
    Instantiates and initializes the generator and discriminator models.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments with model configuration.
        device (torch.device): The device to move the models to (CPU or GPU).

    Returns:
        tuple: The initialized Generator and Discriminator models.
    """
    logger.info("Creating models...")
    netG = Generator(opt.nz, opt.ngf, opt.nc, opt.ngpu).to(device)
    netD = Discriminator(opt.nc, opt.ndf, opt.ngpu).to(device)
    
    # multi-GPU
    if (device.type == 'cuda') and (opt.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(opt.ngpu)))
        netD = nn.DataParallel(netD, list(range(opt.ngpu)))
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    logger.info(f"Generator:\n{netG}")
    logger.info(f"Discriminator:\n{netD}")
    return netG, netD

def save_outputs(G_losses, D_losses, real_images, fake_images, output_dir, netG, netD):
    """
    Saves training results: loss plots, image comparisons, and model checkpoints.

    Args:
        G_losses (list): List of generator loss values over training iterations.
        D_losses (list): List of discriminator loss values over training iterations.
        real_images (Tensor): A batch of real images from the dataset.
        fake_images (Tensor): A batch of generated images from the final training epoch.
        output_dir (str): Directory where results will be saved.
        netG (nn.Module): Trained generator model.
        netD (nn.Module): Trained discriminator model.
    """
    logger.info("Generating visualizations...")
    plot_losses(G_losses, D_losses, save_path=f"{output_dir}/loss_plot.png")
    compare_real_fake(real_images, fake_images, f"{output_dir}/comparison.png")
    
    logger.info("Saving models...")
    torch.save(netG.state_dict(), f"{output_dir}/generator.pth")
    torch.save(netD.state_dict(), f"{output_dir}/discriminator.pth")
    logger.info("Training complete! Outputs saved.") 
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data/celeba', help='path to dataset')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size during training')
    parser.add_argument('--image_size', type=int, default=64, help='spatial size of training images')
    parser.add_argument('--nc', type=int, default=3, help='number of channels in the training images')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='size of feature maps in discriminator')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 hyperparameter for Adam optimizers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs available')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='directory to save results')
    parser.add_argument('--use_kagglehub', action='store_true', help='download dataset from KaggleHub')
    parser.add_argument('--max_images', type=int, default=200000, help='maximum number of images to use (KaggleHub only)')
    opt = parser.parse_args()
    
    # output directory
    os.makedirs(opt.output_dir, exist_ok=True)
    device = setup_device_and_seed(opt)

    dataloader = prepare_data(opt)
    if dataloader is None:
        return
    
    netG, netD = create_models(opt, device)
    
    # Set up loss function and optimizers
    criterion = get_criterion()
    optimizerG, optimizerD = get_optimizers(netG, netD, opt.lr, opt.beta1)
    
    # Create fixed noise for visualization
    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)
    
    # Train the model
    logger.info("Starting training...")
    G_losses, D_losses, img_list = train_gan(
        dataloader, netG, netD, optimizerG, optimizerD, criterion,
        opt.num_epochs, device, opt.nz, fixed_noise
    )
    
    real_batch = next(iter(dataloader))
    real_images = vutils.make_grid(real_batch[0][:64], padding=5, normalize=True)
    fake_images = img_list[-1]

    save_outputs(G_losses, D_losses, real_images, fake_images, opt.output_dir, netG, netD)

    
if __name__ == "__main__":
    main()