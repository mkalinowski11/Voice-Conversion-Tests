import torch

def train_step(model, optimizer, data, criterion, lambda_kl, config):
    mu, log_sigma, emb, dec = model(data)
    loss_rec = criterion(dec, data)
    loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
    loss = config['lambda']['lambda_rec'] * loss_rec + lambda_kl * loss_kl
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 
                max_norm=config['optimizer']['grad_norm'])
    optimizer.step()
    meta = {'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item(),
                'grad_norm': grad_norm}
    return meta