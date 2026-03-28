import torch

from src.diffusion import extract

def predict_x0_from_eps(xt, eps_hat, t, alpha_bar):
    alpha_bar_t = extract(alpha_bar, t, xt.shape)
    return (xt - torch.sqrt(1.0 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)

@torch.no_grad()
def p_sample(model, xt, t_scalar, betas, alphas, alpha_bar, posterior_var):
    B = xt.shape[0]
    device = xt.device

    t = torch.full((B,), t_scalar, dtype=torch.long, device=device)
    eps_hat = model(xt, t)

    beta_t = extract(betas, t, xt.shape)
    alpha_t = extract(alphas, t, xt.shape)
    alpha_bar_t = extract(alpha_bar, t, xt.shape)
    posterior_var_t = extract(posterior_var, t, xt.shape)

    mu = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / (torch.sqrt(1 - alpha_bar_t)) * eps_hat))

    if t_scalar == 0:
        return mu

    z = torch.randn_like(xt)
    x_prev = mu + torch.sqrt(posterior_var_t) * z

    return x_prev


@torch.no_grad()
def ddpm_sample(model, n_samples, chain_len, betas, alphas, alpha_bar, posterior_var, device):
    model.eval()
    xt = torch.randn(n_samples, chain_len, 3, device=device)

    for t in reversed(range(len(betas))):
        xt = p_sample(
            model=model,
            xt=xt,
            t_scalar=t,
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            posterior_var=posterior_var,
        )

    return xt

@torch.no_grad()
def p_sample_stable(model, xt, t_scalar, betas, alphas, alpha_bar, posterior_var, x0_clip=3.0):
    B = xt.shape[0]
    device = xt.device

    t = torch.full((B,), t_scalar, dtype=torch.long, device=device)

    beta_t = extract(betas, t, xt.shape)
    alpha_t = extract(alphas, t, xt.shape)
    alpha_bar_t = extract(alpha_bar, t, xt.shape)
    posterior_var_t = extract(posterior_var, t, xt.shape)

    alpha_bar_prev = torch.cat(
        [torch.ones(1, dtype=alpha_bar.dtype, device=alpha_bar.device), alpha_bar[:-1]], dim=0
    )
    alpha_bar_prev_t = extract(alpha_bar_prev, t, xt.shape)

    eps_hat = model(xt, t)

    x0_hat = predict_x0_from_eps(xt, eps_hat, t, alpha_bar)
    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)

    coef1 = (torch.sqrt(alpha_bar_prev_t) * beta_t) / (1 - alpha_bar_t)
    coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev_t)) / (1 - alpha_bar_t)
    mu = coef1 * x0_hat + coef2 * xt

    if t_scalar > 0:
        z = torch.randn_like(xt)
        x_prev = mu + torch.sqrt(posterior_var_t) * z
    else:
        x_prev = mu

    return x_prev


@torch.no_grad()
def ddpm_sample_stable(model, n_samples, chain_len, betas, alphas, alpha_bar, posterior_var, device, x0_clip=3.0):
    model.eval()
    xt = torch.randn(n_samples, chain_len, 3, device=device)

    for t in reversed(range(len(betas))):
        xt = p_sample_stable(
            model=model,
            xt=xt,
            t_scalar=t,
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            posterior_var=posterior_var,
            x0_clip=x0_clip,
        )

    return xt

@torch.no_grad()
def ddim_step(model, xt, t_scalar, s_scalar, alpha_bar, x0_clip=3.0):
    B = xt.shape[0]
    device = xt.device

    t = torch.full((B,), t_scalar, dtype=torch.long, device=device)
    s = torch.full((B,), s_scalar, dtype=torch.long, device=device)

    alpha_bar_t = extract(alpha_bar, t, xt.shape)
    alpha_bar_s = extract(alpha_bar, s, xt.shape)

    eps_hat = model(xt, t)

    x0_hat = (xt - torch.sqrt(1 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)

    xs = torch.sqrt(alpha_bar_s) * x0_hat + torch.sqrt(1 - alpha_bar_s) * eps_hat

    return xs


@torch.no_grad()
def ddim_sample(model, n_samples, chain_len, alpha_bar, device, num_steps=1000, x0_clip=3.0):
    model.eval()

    T = len(alpha_bar)
    xt = torch.randn(n_samples, chain_len, 3, device=device)

    ts = torch.linspace(T - 1, 0, steps=num_steps, device=device).long()

    for i in range(len(ts) - 1):
        t_scalar = ts[i].item()
        s_scalar = ts[i + 1].item()
        xt = ddim_step(
            model=model,
            xt=xt,
            t_scalar=t_scalar,
            s_scalar=s_scalar,
            alpha_bar=alpha_bar,
            x0_clip=x0_clip,
        )

    t_final = torch.full((n_samples,), ts[-1].item(), dtype=torch.long, device=device)
    eps_hat = model(xt, t_final)
    alpha_bar_final = extract(alpha_bar, t_final, xt.shape)
    x0_hat = (xt - torch.sqrt(1 - alpha_bar_final) * eps_hat) / torch.sqrt(alpha_bar_final)
    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)

    return x0_hat

def ddpm_reverse_mean_var(model, xt, t_int, betas, alphas, alpha_bar, posterior_var, x0_clip=3.0):
    B = xt.shape[0]
    device = xt.device

    t = torch.full((B,), t_int, device=device, dtype=torch.long)
    eps_hat = model(xt, t)

    x0_hat = predict_x0_from_eps(xt, eps_hat, t, alpha_bar)

    if x0_clip is not None:
        x0_hat = x0_hat.clamp(-x0_clip, x0_clip)

    alpha_bar_prev = torch.cat(
        [torch.ones(1, device=alpha_bar.device, dtype=alpha_bar.dtype), alpha_bar[:-1]], dim=0
    )

    beta_t = extract(betas, t, xt.shape)
    alpha_t = extract(alphas, t, xt.shape)
    alpha_bar_t = extract(alpha_bar, t, xt.shape)
    alpha_bar_prev_t = extract(alpha_bar_prev, t, xt.shape)
    posterior_var_t = extract(posterior_var, t, xt.shape)

    coef1 = beta_t * torch.sqrt(alpha_bar_prev_t) / (1.0 - alpha_bar_t)
    coef2 = (1.0 - alpha_bar_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)

    mean_t = coef1 * x0_hat + coef2 * xt

    return mean_t, posterior_var_t, x0_hat, eps_hat
