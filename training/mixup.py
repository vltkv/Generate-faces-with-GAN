import torch
import torch.nn.functional as F

def mixup_data(x, alpha=0.2):
    """
    Performs mixup augmentation on a batch of images.
    
    Args:
        x: Batch of images (B, C, H, W)
        alpha: Mixup parameter (default: 0.2)
        
    Returns:
        mixed_x: Mixed images
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Returns loss for mixup augmented data.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient

    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b) 