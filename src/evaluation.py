import random
import torch

from src.utils import set_seed
from src.ddpo import collect_ddpm_rollout_batch


@torch.no_grad()
def evaluate_model_reward(
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
    out = collect_ddpm_rollout_batch(
        model=model,
        n_samples=n_samples,
        chain_len=chain_len,
        betas=betas,
        alphas=alphas,
        alpha_bar=alpha_bar,
        posterior_var=posterior_var,
        reward_fn=reward_fn,
        reward_kwargs=reward_kwargs,
        projection_fn=projection_fn,
        device=device,
        x0_clip=x0_clip,
    )

    r = out["reward"]

    return {
        "reward_mean": float(r.mean().item()),
        "reward_std": float(r.std(unbiased=False).item()),
        "reward_min": float(r.min().item()),
        "reward_max": float(r.max().item()),
        "samples": out["x0_eval"],
        "rewards": r,
    }


@torch.no_grad()
def evaluate_model_reward_full(
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
    seed=0,
):
    set_seed(seed)

    out = collect_ddpm_rollout_batch(
        model=model,
        n_samples=n_samples,
        chain_len=chain_len,
        betas=betas,
        alphas=alphas,
        alpha_bar=alpha_bar,
        posterior_var=posterior_var,
        reward_fn=reward_fn,
        reward_kwargs=reward_kwargs,
        projection_fn=projection_fn,
        device=device,
        x0_clip=x0_clip,
    )

    r = out["reward"]

    return {
        "reward_mean": float(r.mean().item()),
        "reward_std": float(r.std(unbiased=False).item()),
        "reward_min": float(r.min().item()),
        "reward_max": float(r.max().item()),
        "samples": out["x0_eval"],
        "rewards": r,
    }


def get_sample_indices(rewards, k=3, seed=0):
    set_seed(seed)
    idx_sorted = torch.argsort(rewards, descending=True)

    top_idx = idx_sorted[:k].tolist()
    bottom_idx = idx_sorted[-k:].tolist()

    all_idx = list(range(len(rewards)))
    rand_idx = random.sample(all_idx, k)

    return {
        "top": top_idx,
        "random": rand_idx,
        "bottom": bottom_idx,
    }
