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
        output_real = netD(real_cpu).view(-1)
        errD_real = criterion(output_real, torch.ones_like(output_real))
        errD_real.backward()
        
        # fake data
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output_fake = netD(fake.detach()).view(-1)
        errD_fake = criterion(output_fake, torch.zeros_like(output_fake))
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()
                
        # update Generator
        netG.zero_grad()
        output = netD(fake).view(-1)
        errG = criterion(output, torch.ones_like(output))
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}')

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        iters += 1
