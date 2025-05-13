import torch.optim as optim

def get_optimizers(netG, netD, lr=0.0002, beta1=0.5):
    """    
    Args:
        netG: Generator network
        netD: Discriminator network
        lr: Learning rate
        beta1: Beta1 parameter for Adam optimizer
    """
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    return optimizerG, optimizerD