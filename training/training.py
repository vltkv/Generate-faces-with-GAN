G_losses = []
D_losses = []
iters = 0

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
        output_fake = netD(fake.detach()).view(-1)
        errD_fake = criterion(output_fake, torch.zeros_like(output_fake))
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()
                
        # update Generator
        netG.zero_grad()
        label.fill_(real_label) # generator wants fake data labelled as real
        output = netD(fake).view(-1)
        errG = criterion(output, torch.ones_like(output))
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        iters += 1
