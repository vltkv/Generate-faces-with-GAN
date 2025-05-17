import torch
import random
import argparse
import os
import logging

from data.preprocessing import get_dataset, create_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from models.utils import weights_init
from training.optimizers import get_optimizers
from training.loss_functions import get_criterion, discriminator_loss, generator_loss
from training.training import train_gan
from evaluation.visualization import plot_losses, compare_real_fake
import torchvision.utils as vutils


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    parser.add_argument('--augment', action='store_true', help='apply data augmentation')
    opt = parser.parse_args()
    
    # output directory
    os.makedirs(opt.output_dir, exist_ok=True)
    
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        logger.warning("Unable to set deterministic algorithms - might affect reproducibility.")
    
    # Decide which device to run on
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
    
    dataloader = create_dataloader(dataset, opt.batch_size, opt.workers)
    
    if dataloader is None:
        logger.error("Failed to create dataloader. Exiting...")
        return
    
    logger.info("Creating models...")
    netG = Generator(opt.nz, opt.ngf, opt.nc).to(device)
    netD = Discriminator(opt.ndf, opt.nc).to(device)
    
    # multi-GPU
    if (device.type == 'cuda') and (opt.ngpu > 1):
        netG = torch.nn.DataParallel(netG, list(range(opt.ngpu)))
        netD = torch.nn.DataParallel(netD, list(range(opt.ngpu)))
    
    # custom weights
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
    G_losses, D_losses, img_list = train_gan(
        dataloader, netG, netD, optimizerG, optimizerD, criterion,
        opt.num_epochs, device, opt.nz, fixed_noise
    )
    
    # Visualize results
    logger.info("Generating visualizations...")
    plot_losses(G_losses, D_losses, save_path=f"{opt.output_dir}/loss_plot.png")
    
    # Get a batch of real images
    real_batch = next(iter(dataloader))
    real_images = vutils.make_grid(real_batch[0][:64], padding=5, normalize=True)
    
    # Compare with generated images
    compare_real_fake(real_images, img_list[-1], save_path=f"{opt.output_dir}/comparison.png")
    
    # Save the models
    logger.info("Saving models...")
    torch.save(netG.state_dict(), f"{opt.output_dir}/generator.pth")
    torch.save(netD.state_dict(), f"{opt.output_dir}/discriminator.pth")
    
    logger.info("Training complete! Models saved.")
    
if __name__ == "__main__":
    main()