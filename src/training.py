import torch
import torch.nn.functional as F

from src.diffusion import q_sample


def train_step(model, batch, alpha_bar, optimizer, T, device):
    model.train()
    coords0 = batch["coords"].to(device)
    B = coords0.shape[0]

    t = torch.randint(0, T, (B,), dtype=torch.long, device=device)
    xt, eps = q_sample(coords0, t, alpha_bar)

    eps_hat = model(xt, t)
    loss = F.mse_loss(eps_hat, eps)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def train_step_tf(model, batch, alpha_bar, optimizer, T, device):
    model.train()
    coords0 = batch["coords"].to(device)
    B = coords0.shape[0]

    t = torch.randint(0, T, (B,), dtype=torch.long, device=device)
    xt, eps = q_sample(coords0, t, alpha_bar)

    eps_hat = model(xt, t)
    loss = F.mse_loss(eps_hat, eps)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()
