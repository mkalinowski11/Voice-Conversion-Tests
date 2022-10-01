import torch

def save_checkpoint(model, optimizer, config, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "current_epoch": config.CURRENT_EPOCH
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, config, filename="my_checkpoint.pth"):
  checkpoint = torch.load(filename, map_location=config.DEVICE)
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  config.CURRENT_EPOCH = checkpoint["current_epoch"]
  for param_group in optimizer.param_groups:
        param_group["lr"] = config.LEARNING_RATE