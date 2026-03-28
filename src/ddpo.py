import math
import torch
import torch.nn.functional as F

from src.diffusion import extract, q_sample
from src.sampling import ddpm_reverse_mean_var


def gaussian_log_prob(x, mean, var, eps=1e-12):
    var = var.clamp_min(eps)
    log_two_pi = math.log(2.0 * math.pi)

    log_per_elem = -0.5 * (((x - mean) ** 2) / var + torch.log(var) + log_two_pi)
    log_per_chain = log_per_elem.reshape(x.shape[0], -1).sum(dim=1)

    return log_per_chain


def normalizer(x, eps=1e-8):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


@torch.no_grad()
def collect_ddpm_rollout_batch(
    model,
    n_samples,
    chain_len,
    betas,
    alphas,
    alpha_bar,
    posterior_var,
    reward_fn,
    reward_kwargs=None,
    projection_fn=None,
    device="cpu",
    x0_clip=3.0,
):
    if reward_kwargs is None:
        reward_kwargs = {}
    else:
        reward_kwargs = dict(reward_kwargs)

    T = len(betas)
    xt = torch.randn(n_samples, chain_len, 3, device=device)

    traj = []

    for t_int in reversed(range(T)):
        mean_t, var_t, x0_hat, eps_hat = ddpm_reverse_mean_var(
            model=model,
            xt=xt,
            t_int=t_int,
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            posterior_var=posterior_var,
            x0_clip=x0_clip,
        )

        if t_int > 0:
            noise = torch.randn_like(xt)
            x_prev = mean_t + torch.sqrt(var_t) * noise
        else:
            x_prev = mean_t

        traj.append({
            "t_int": t_int,
            "xt": xt.detach(),
            "x_prev": x_prev.detach(),
        })

        xt = x_prev

    x0_raw = xt
    x0_eval = x0_raw
    if projection_fn is not None:
        x0_eval = projection_fn(x0_raw)
    reward = reward_fn(x0_eval, **reward_kwargs).detach()

    return {
        "traj": traj,
        "x0_raw": x0_raw.detach(),
        "x0_eval": x0_eval.detach(),
        "reward": reward,
    }

def trajectory_log_prob_under_model(model, traj, betas, alphas, alpha_bar, posterior_var, x0_clip=3.0):
    total_log_p = None
    for step in traj:
        t_int = step["t_int"]
        xt = step["xt"]
        x_prev = step["x_prev"]

        if t_int == 0:
            continue

        mean_t, var_t, x0_hat, eps_hat = ddpm_reverse_mean_var(
            model=model,
            xt=xt,
            t_int=t_int,
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            posterior_var=posterior_var,
            x0_clip=x0_clip,
        )

        log_p_t = gaussian_log_prob(x_prev, mean_t, var_t)

        if total_log_p is None:
            total_log_p = log_p_t
        else:
            total_log_p = total_log_p + log_p_t

    return total_log_p

def ddpo_sf_update_step(
    model,
    optimizer,
    rollout_batch,
    betas,
    alphas,
    alpha_bar,
    posterior_var,
    x0_clip,
    grad_clip=1.0,
):
    rewards = rollout_batch["reward"]
    advantages = normalizer(rewards).detach()

    traj_log_p = trajectory_log_prob_under_model(
        model=model,
        traj=rollout_batch["traj"],
        betas=betas,
        alphas=alphas,
        alpha_bar=alpha_bar,
        posterior_var=posterior_var,
        x0_clip=x0_clip,
    )

    n_stochastic_steps = len(betas) - 1

    loss_pg = -(advantages * traj_log_p).mean() / math.sqrt(n_stochastic_steps)

    optimizer.zero_grad(set_to_none=True)
    loss_pg.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return {
        "loss_pg": float(loss_pg.item()),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "reward_min": float(rewards.min().item()),
        "reward_max": float(rewards.max().item()),
        "adv_mean": float(advantages.mean().item()),
        "adv_std": float(advantages.std(unbiased=False).item()),
        "traj_log_p_mean": float(traj_log_p.mean().item()),
        "traj_log_p_std": float(traj_log_p.std(unbiased=False).item()),
        "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
    }


def diffusion_anchor_loss(model, real_coords, alpha_bar):
    B = real_coords.shape[0]
    t = torch.randint(0, len(alpha_bar), (B,), device=real_coords.device, dtype=torch.long)
    xt, eps = q_sample(real_coords, t, alpha_bar)
    eps_hat = model(xt, t)
    return F.mse_loss(eps_hat, eps)


def ddpo_sf_update_step_anchor(
    model,
    optimizer,
    rollout_batch,
    betas,
    alphas,
    alpha_bar,
    posterior_var,
    anchor_batch=None,
    anchor_coef=0.0,
    x0_clip=3.0,
    grad_clip=1.0,
):
    rewards = rollout_batch["reward"]
    advantages = normalizer(rewards).detach()

    traj_log_p = trajectory_log_prob_under_model(
        model=model,
        traj=rollout_batch["traj"],
        betas=betas,
        alphas=alphas,
        alpha_bar=alpha_bar,
        posterior_var=posterior_var,
        x0_clip=x0_clip,
    )

    n_stochastic_steps = len(betas) - 1

    loss_pg = -(advantages * traj_log_p).mean() / math.sqrt(n_stochastic_steps)

    loss = loss_pg
    loss_anchor = torch.tensor(0.0, device=traj_log_p.device)
    if anchor_batch is not None and anchor_coef > 0.0:
        loss_anchor = diffusion_anchor_loss(model, anchor_batch, alpha_bar)
        loss = loss + anchor_coef * loss_anchor

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return {
        "loss_total": float(loss.item()),
        "loss_pg": float(loss_pg.item()),
        "loss_anchor": float(loss_anchor.item()),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "reward_min": float(rewards.min().item()),
        "reward_max": float(rewards.max().item()),
        "adv_mean": float(advantages.mean().item()),
        "adv_std": float(advantages.std(unbiased=False).item()),
        "traj_log_p_mean": float(traj_log_p.mean().item()),
        "traj_log_p_std": float(traj_log_p.std(unbiased=False).item()),
        "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
    }
