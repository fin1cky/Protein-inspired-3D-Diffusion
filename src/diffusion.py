import math
import torch


def cosine_schedule(T, s=0.008, device="cpu"):
    steps = torch.arange(T + 1, dtype=torch.float32, device=device)
    x = (steps / T + s) / (1 + s)
    f = torch.cos(x * math.pi / 2) ** 2
    alpha_bar_u = f / f[0]

    betas = 1 - (alpha_bar_u[1:] / alpha_bar_u[:-1])
    betas = betas.clamp(1e-8, 0.999)

    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bar


def compute_posterior_variance(betas, alpha_bar):
    alpha_bar_prev = torch.cat(
        [torch.ones(1, dtype=alpha_bar.dtype, device=alpha_bar.device), alpha_bar[:-1]],
        dim=0,
    )
    posterior_var = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
    posterior_var = posterior_var.clamp(min=1e-20)
    return posterior_var


def extract(a, t, x_shape):
    out = a[t]
    while out.ndim < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def q_sample(coords0, t, alpha_bar, eps=None):
    if eps is None:
        eps = torch.randn_like(coords0)

    alpha_bar_t = extract(alpha_bar, t, coords0.shape)
    xt = torch.sqrt(alpha_bar_t) * coords0 + torch.sqrt(1 - alpha_bar_t) * eps

    return xt, eps
