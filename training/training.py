import torch
import torch.nn as nn
from models import Generator
from models import Discriminator
from training import OptimizerFactory

nz = 100  # size of z latent vector (i.e. size of generator input)
ngf = 64  # size of feature maps in generator
ndf = 64  # size of feature maps in discriminator
nc = 3  # number of channels in the training images. For color images this is 3
num_epochs = 5
batch_size = 128
optimizer_type = "Adam"

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G_losses = []
D_losses = []
img_list = []
iters = 0

netG = Generator(nz, ngf, nc).to(device)
netG.apply(weights_init)
netD = Discriminator(ndf, nc).to(device)
netD.apply(weights_init)

opt_factory = OptimizerFactory(optimizer_type="adam", lr=0.0002, beta1=0.5)
optimizerD = opt_factory.create(netD)
optimizerG = opt_factory.create(netG)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # update Discriminator
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # real data
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, torch.ones_like(output))
        errD_real.backward()
        D_x = output.mean().item()
        
        # fake data
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, torch.zeros_like(output))
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()
                
        # update Generator
        netG.zero_grad()
        label.fill_(real_label) # generator wants fake data labelled as real
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        iters += 1
