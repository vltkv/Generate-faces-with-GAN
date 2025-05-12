import os
import logging
import shutil
import random
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        img.close()
        return True
    except Exception as e:
        logger.warning(f"Invalid image {file_path}: {str(e)}")
        return False

def setup_kagglehub_dataset(dataroot, max_images=None):
    """
    Returns:
        True if successful, False otherwise
    """
    try:
        import kagglehub
        logger.info("Downloading CelebA dataset from KaggleHub...")
        dataset_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        
        celeba_nested_folder = os.path.join(dataset_path, "img_align_celeba", "img_align_celeba")
        if not os.path.exists(celeba_nested_folder):
            logger.error(f"Expected folder structure not found at {celeba_nested_folder}")
            return False
            
        target_folder = os.path.join(dataroot, "img_align_celeba")
        os.makedirs(target_folder, exist_ok=True)
        
        jpg_files = [f for f in os.listdir(celeba_nested_folder) if f.endswith('.jpg')]
        logger.info(f"Found {len(jpg_files)} images in the KaggleHub dataset")
        
        if not jpg_files:
            logger.error("No jpg files found in the dataset")
            return False
            
        # Copy valid images to target folder
        copied_count = 0
        for file in jpg_files:
            if max_images is not None and copied_count >= max_images:
                break
                
            src_path = os.path.join(celeba_nested_folder, file)
            if is_valid_image(src_path):
                shutil.copy(src_path, os.path.join(target_folder, file))
                copied_count += 1
                if copied_count % 1000 == 0:
                    logger.info(f"Copied {copied_count} images...")
            else:
                logger.warning(f"Skipping invalid image: {file}")
                
        logger.info(f"Successfully copied {copied_count} images to {target_folder}")
        return True
        
    except ImportError:
        logger.error("kagglehub package not installed. Install with: pip install kagglehub")
        return False
    except Exception as e:
        logger.error(f"Error setting up KaggleHub dataset: {str(e)}")
        return False

def get_dataset(dataroot, image_size=64, use_kagglehub=False, max_images=None):
    if use_kagglehub:
        success = setup_kagglehub_dataset(dataroot, max_images)
        if not success:
            logger.error("Failed to set up KaggleHub dataset")
            return None
    
    if not os.path.exists(dataroot):
        logger.error(f"Dataset directory {dataroot} does not exist")
        return None
        
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    try:
        dataset = dset.ImageFolder(root=dataroot, transform=transform)
        logger.info(f"Successfully loaded dataset with {len(dataset)} images")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def create_dataloader(dataset, batch_size=128, workers=2):
    if dataset is None:
        logger.error("Cannot create dataloader: dataset is None")
        return None
        
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers
    )
    return dataloader