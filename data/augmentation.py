import torch
import torchvision.transforms as transforms

def get_training_augmentation(image_size=64, apply_prob=0.5):
    """
    Spatial: RandimHorizontalFlip, random rotation +- 10 deg, RandomAffine (przesuniecie)
    Resizing and cropping, Color and intensity transformations, Add random noise
    
    Args:
        image_size: Size of the output images
        apply_prob: Probability of applying each augmentation
        
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        # Spatial transformations
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=apply_prob),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=apply_prob),
        
        # Resize and crop
        transforms.Resize(int(image_size * 1.1)),
        transforms.RandomCrop(image_size),
        
        # Color and intensity transformations
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )], p=apply_prob),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        
        # Add random noise
        transforms.RandomApply([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
        ], p=apply_prob/2),
    ])