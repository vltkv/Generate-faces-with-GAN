import torch
import torch.nn as nn

def get_criterion():
    return nn.BCELoss()

def discriminator_loss(discriminator, real_images, fake_images, criterion, device):
    """
    Returns:
        total_loss, real_loss, fake_loss, real_score, fake_score
    """
    batch_size = real_images.size(0)
    real_label = torch.full((batch_size,), 1., dtype=torch.float, device=device)
    fake_label = torch.full((batch_size,), 0., dtype=torch.float, device=device)
    
    # Real images loss
    real_output = discriminator(real_images).view(-1)
    d_real_loss = criterion(real_output, real_label)
    real_score = real_output.mean().item()
    
    # Fake images loss
    fake_output = discriminator(fake_images.detach()).view(-1)
    d_fake_loss = criterion(fake_output, fake_label)
    fake_score = fake_output.mean().item()
    
    # Combined loss
    d_loss = d_real_loss + d_fake_loss
    
    return d_loss, d_real_loss, d_fake_loss, real_score, fake_score

def generator_loss(discriminator, fake_images, criterion, device):
    batch_size = fake_images.size(0)
    real_label = torch.full((batch_size,), 1., dtype=torch.float, device=device)
    
    # Generator wants discriminator to think its images are real
    fake_output = discriminator(fake_images).view(-1)
    g_loss = criterion(fake_output, real_label)
    fake_score = fake_output.mean().item()
    
    return g_loss, fake_score
