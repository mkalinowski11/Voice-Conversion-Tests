import torch
from tqdm import tqdm

def train_fn(disc_target, disc_source, gen_source_target, gen_target_source, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, config):
    TARGET_REALS = 0
    TARGET_FAKES = 0
    loop = tqdm(loader, leave=True)

    for idx, (source_voice, target_voice) in enumerate(loop):
        source_voice = source_voice.to(config.DEVICE)
        target_voice = target_voice.to(config.DEVICE)

        # Train Discriminators
        with torch.cuda.amp.autocast():
            fake_target_voice = gen_source_target(source_voice)

            d_target_voice_real = disc_target(target_voice)
            d_target_voice_fake = disc_target(fake_target_voice.detach())

            TARGET_REALS += d_target_voice_real.mean().item()
            TARGET_FAKES += d_target_voice_fake.mean().item()

            d_target_real_loss = mse(d_target_voice_real, torch.ones_like(d_target_voice_real))
            d_target_fake_loss = mse(d_target_voice_fake, torch.zeros_like(d_target_voice_fake))
            d_target_loss = d_target_real_loss + d_target_fake_loss

            fake_source_voice = gen_target_source(target_voice)

            d_src_real = disc_source(source_voice)
            d_src_fake = disc_source(fake_source_voice.detach())

            d_source_real_loss = mse(d_src_real, torch.ones_like(d_src_real))
            d_source_fake_loss = mse(d_src_fake, torch.zeros_like(d_src_fake))
            d_source_loss = d_source_real_loss + d_source_fake_loss

            D_loss = (d_target_loss + d_source_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # Train Generators
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            d_target_fake = disc_target(fake_target_voice)
            d_source_fake = disc_source(fake_source_voice)

            loss_g_target = mse(d_target_fake, torch.ones_like(d_target_fake))
            loss_g_source = mse(d_source_fake, torch.ones_like(d_source_fake))

            # cycle loss
            cycle_source = gen_target_source(fake_target_voice)
            cycle_target = gen_source_target(fake_source_voice)
            cycle_source_loss = l1(source_voice, cycle_source)
            cycle_target_loss = l1(target_voice, cycle_target)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_source = gen_target_source(source_voice)
            identity_target = gen_source_target(target_voice)
            identity_source_loss = l1(source_voice, identity_source)
            identity_target_loss = l1(target_voice, identity_target)

            G_loss = (
                loss_g_target
                + loss_g_source
                + cycle_source_loss * config.LAMBDA_CYCLE
                + cycle_target_loss * config.LAMBDA_CYCLE
                + identity_source_loss * config.LAMBDA_IDENTITY
                + identity_target_loss * config.LAMBDA_IDENTITY
            )
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        loop.set_postfix(H_real=TARGET_REALS/(idx+1), H_fake=TARGET_FAKES/(idx+1))