import torch
from tqdm import tqdm

def train_step(
        generator_src_trg,
        generator_trg_src,
        disc_src,
        disc_trg,
        train_loader,
        gen_optimizer,
        disc_optimizer,
        g_scaler,
        d_scaler,
        config
):
    loop = tqdm(train_loader, leave=True)
    # trainloader need to be tqdm style
    for idx, (real_A, real_B) in enumerate(loop):

        # real_A = real_A.squeeze(1)
        # real_B = real_B.squeeze(1)
        
        real_A = real_A.to(config.DEVICE)
        real_B = real_B.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            fake_B = generator_src_trg(real_A)
            cycle_A = generator_trg_src(fake_B)

            fake_A = generator_trg_src(real_B)
            cycle_B = generator_src_trg(fake_A)
            identity_A = generator_trg_src(real_A)
            identity_B = generator_src_trg(real_B)

            d_fake_A = disc_src(fake_A)
            d_fake_B = disc_trg(fake_B)
            # d_fake_cycle_A = disc_src(cycle_A)
            # d_fake_cycle_B = disc_trg(cycle_B)
            cycleLoss = torch.mean(
                torch.abs(real_A - cycle_A) + torch.mean(torch.abs(real_B - cycle_B))
            )

            identity_loss = torch.mean(
                torch.abs(real_A - identity_A) + torch.mean(torch.abs(real_B - identity_B))
            )

            generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
            generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

            generator_loss = generator_loss_A2B + generator_loss_B2A + config.LAMBDA_CYCLE * cycleLoss + \
                                config.LAMBDA_IDENTITY * identity_loss
        # add generator loss
        gen_optimizer.zero_grad()
        g_scaler.scale(generator_loss).backward()
        g_scaler.step(gen_optimizer)
        g_scaler.update()
        
        # discriminator train
        with torch.cuda.amp.autocast():
            d_real_A = disc_src(real_A)
            d_real_B = disc_trg(real_B)

            generated_A = generator_trg_src(real_B)
            d_fake_A = disc_src(generated_A)

            cycled_B = generator_src_trg(generated_A)
            d_cycled_B = disc_trg(cycled_B)

            generated_B = generator_src_trg(real_A)
            d_fake_B = disc_trg(generated_B)

            cycled_A = generator_trg_src(generated_B)
            d_cycled_A = disc_src(cycled_A)

            d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
            d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
            d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

            d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
            d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
            d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

            d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
            d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)

            d_loss_A_2nd = (d_loss_A_real + d_loss_A_cycled) / 2.0
            d_loss_B_2nd = (d_loss_B_real + d_loss_B_cycled) / 2.0

            d_loss = (d_loss_A + d_loss_B) / 2.0 + (d_loss_A_2nd + d_loss_B_2nd) / 2.0
            # add to store d_loss
        disc_optimizer.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(disc_optimizer)
        d_scaler.update()
        loop.set_postfix(
                            Gen_loss=generator_loss.item(),
                            Disc_loss = d_loss.item(),
                            Gen_loss_a2b = generator_loss_A2B.item(),
                            Gen_loss_b2a = generator_loss_B2A.item(),
                            Identity_loss = identity_loss.item(),
                            Cycle_loss = cycleLoss.item(),
                            Disc_loss_A = d_loss_A.item(),
                            Disc_loss_B = d_loss_B.item()
                        )