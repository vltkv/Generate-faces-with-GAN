import torch.optim as optim

class OptimizerFactory:
    def __init__(self, optimizer_type="adam", lr=0.0002, beta1=0.5, momentum=0.9):
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.beta1 = beta1
        self.momentum = momentum

    def create(self, model):
        if self.optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        elif self.optimizer_type == "sgd":
            return optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")


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