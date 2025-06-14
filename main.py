import torch
import random
import argparse
import os
import logging

from data.preprocessing import get_dataset, create_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from training.utils import weights_init, save_results  # Import save_results from models.utils
from training.optimizers import get_optimizers
from training.loss_functions import get_criterion
from training.training import train_gan
from training.wgan import train_wgan
import torchvision.utils as vutils
from torch import nn


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data/celeba', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size during training')
    parser.add_argument('--image_size', type=int, default=64, help='spatial size of training images')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for optimizers')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='directory to save results')
    parser.add_argument('--use_kagglehub', action='store_true', help='download dataset from KaggleHub')
    parser.add_argument('--max_images', type=int, default=200000, help='maximum number of images to use (KaggleHub only)')
    parser.add_argument('--augment', action='store_false', help='apply data augmentation')
    parser.add_argument('--wgan', action='store_true', help='use Wasserstein GAN')
    parser.add_argument('--use_mixup', action='store_true', help='use MixUp augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='MixUp alpha parameter')
    opt = parser.parse_args()
    
    opt.ngpu = 1  # 1 GPU
    opt.nz = 100  # latent dimension
    opt.beta1 = 0.5  # beta1 for Adam optimizer
    opt.n_critic = 5  # critic iterations for WGAN
    opt.clip_value = 0.01  # weight clipping value for WGAN
    
    os.makedirs(opt.output_dir, exist_ok=True)
    
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        logger.warning("Unable to set deterministic algorithms - might affect reproducibility.")
    
    # Which device to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    logger.info("Setting up dataset...")
    dataset = get_dataset(
        opt.dataroot, 
        opt.image_size, 
        use_kagglehub=opt.use_kagglehub,
        max_images=opt.max_images,
        augment=opt.augment
    )
    
    if dataset is None:
        logger.error("Failed to load dataset. Exiting...")
        return
    
    dataloader = create_dataloader(dataset, opt.batch_size)
    
    if dataloader is None:
        return
    
    logger.info("Creating models...")
    netG = Generator(nz=opt.nz).to(device)
    netD = Discriminator().to(device)
    
    # Handle multi-GPU if available
    if (device.type == 'cuda') and (opt.ngpu > 1):
        netG = torch.nn.DataParallel(netG, list(range(opt.ngpu)))
        netD = torch.nn.DataParallel(netD, list(range(opt.ngpu)))
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    logger.info("Generator architecture:")
    logger.info(netG)
    logger.info("Discriminator architecture:")
    logger.info(netD)
    
    # Set up loss function and optimizers
    criterion = get_criterion()
    optimizerG, optimizerD = get_optimizers(netG, netD, opt.lr, opt.beta1)
    
    # Create fixed noise for visualization
    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)
    
    # Train the model
    logger.info("Starting training...")
    if opt.wgan:
        # Remove sigmoid from critic for WGAN
        if isinstance(netD.main[-1], nn.Sigmoid):
            netD.main = nn.Sequential(*list(netD.main.children())[:-1])
            logger.info("Removed sigmoid from critic for WGAN")
        
        # RMSprop optimizer for WGAN
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lr)
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lr)
        
        G_losses, D_losses, img_list = train_wgan(
            dataloader, netG, netD, optimizerG, optimizerD,
            opt.num_epochs, device, opt.nz, fixed_noise,
            n_critic=opt.n_critic, clip_value=opt.clip_value
        )
    else:
        # Standard GAN
        G_losses, D_losses, img_list = train_gan(
            dataloader, netG, netD, optimizerG, optimizerD, criterion,
            opt.num_epochs, device, opt.nz, fixed_noise,
            use_mixup=opt.use_mixup, mixup_alpha=opt.mixup_alpha
        )
    
    real_batch = next(iter(dataloader))
    
    save_results(G_losses, D_losses, real_batch, img_list, opt.output_dir, netG, netD, opt)
    
if __name__ == "__main__":
    main()