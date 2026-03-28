import io
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.chains import (
    sample_random_specs,
    stitch_segments,
    center_coords,
    ID_TO_MOTIF,
)
from src.diffusion import extract
from src.sampling import predict_x0_from_eps
from src.geometry import project_bond_lengths

def plot_chain_3d(coords, title="chain", ax=None):
    c = coords.detach().cpu()
    x, y, z = c[:, 0], c[:, 1], c[:, 2]

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=2, marker="o", markersize=3)
    ax.scatter(x[0], y[0], z[0], s=60, marker="o", label="start")
    ax.scatter(x[-1], y[-1], z[-1], s=60, marker="^", label="end")
    ax.set_title(title)
    ax.legend()

    eps = 1e-3
    xspan = max((x.max() - x.min()).item(), eps)
    yspan = max((y.max() - y.min()).item(), eps)
    zspan = max((z.max() - z.min()).item(), eps)
    ax.set_box_aspect((xspan, yspan, zspan))

    ax.view_init(elev=20, azim=-60)
    return ax


def plot_full_3d_chain(coords, labels, title="composed chain", ax=None):
    coords = coords.detach().cpu()
    labels = labels.detach().cpu()

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, color="black", linewidth=1.0, alpha=0.7)

    colors = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:green",
    }

    for motif_id, motif_name in ID_TO_MOTIF.items():
        mask = labels == motif_id
        if mask.any():
            ax.scatter(
                x[mask], y[mask], z[mask],
                s=14, color=colors[motif_id], label=motif_name,
            )

    ax.scatter(x[0], y[0], z[0], s=60, marker="o", color="red", label="start")
    ax.scatter(x[-1], y[-1], z[-1], s=60, marker="^", color="purple", label="end")

    ax.set_title(title)
    ax.legend()

    eps = 1e-3
    xspan = max((x.max() - x.min()).item(), eps)
    yspan = max((y.max() - y.min()).item(), eps)
    zspan = max((z.max() - z.min()).item(), eps)
    ax.set_box_aspect((xspan, yspan, zspan))
    ax.view_init(elev=20, azim=-60)

    return ax


def recenter_np(coords):
    coords = np.asarray(coords)
    return coords - coords.mean(axis=0, keepdims=True)


def set_nice_3d_limits(ax, coords, pad=0.15, min_span=1.0):
    coords = np.asarray(coords)
    center = coords.mean(axis=0)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, min_span)
    r = 0.5 * np.max(spans) * (1.0 + pad)

    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.set_box_aspect((1, 1, 1))


def draw_chain_pretty(
    ax,
    coords,
    title="",
    elev=20,
    azim=-60,
    linewidth=2.2,
    markersize=2.5,
    recenter=True,
    show_legend=True,
):
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()

    coords = np.asarray(coords)
    if recenter:
        coords = recenter_np(coords)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    ax.plot(x, y, z, linewidth=linewidth, marker="o", markersize=markersize)
    ax.scatter(x[0], y[0], z[0], s=55, marker="o", label="start")
    ax.scatter(x[-1], y[-1], z[-1], s=55, marker="^", label="end")

    ax.set_title(title, fontsize=14)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)

    set_nice_3d_limits(ax, coords, pad=0.15, min_span=1.0)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

@torch.no_grad()
def sample_clean_chain_for_movie(
    total_len=64,
    bond_len=1.0,
    min_segments=3,
    max_segments=5,
    min_len=12,
    max_len=24,
    random_orient=True,
    center=True,
    device="cpu",
):
    specs = sample_random_specs(
        total_len=total_len,
        min_segments=min_segments,
        max_segments=max_segments,
        min_len=min_len,
        max_len=max_len,
    )

    x0, labels, seg_ids = stitch_segments(
        specs,
        bond_len=bond_len,
        random_orient=random_orient,
        device=device,
    )

    if center:
        x0 = center_coords(x0)

    return x0, labels, seg_ids, specs


@torch.no_grad()
def make_forward_states_closed_form_3d(x0, alpha_bar, capture_ts):
    device = x0.device
    L = x0.shape[0]
    eps = torch.randn_like(x0)

    frames = []
    for t_int in capture_ts:
        t = torch.tensor([t_int], dtype=torch.long, device=device)
        a_bar_t = extract(alpha_bar, t, (1, L, 3))[0]
        xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps
        frames.append(xt.detach().cpu().numpy())

    return frames


@torch.no_grad()
def _p_sample_stable_movie(model, x_t, t_scalar, betas, alphas, alpha_bar, posterior_var, x0_clip=3.0):
    B = x_t.shape[0]
    device = x_t.device

    t = torch.full((B,), t_scalar, dtype=torch.long, device=device)

    beta_t = extract(betas, t, x_t.shape)
    alpha_t = extract(alphas, t, x_t.shape)
    alpha_bar_t = extract(alpha_bar, t, x_t.shape)
    post_var_t = extract(posterior_var, t, x_t.shape)

    alpha_bar_prev = torch.cat(
        [torch.ones(1, dtype=alpha_bar.dtype, device=alpha_bar.device), alpha_bar[:-1]], dim=0
    )
    alpha_bar_prev_t = extract(alpha_bar_prev, t, x_t.shape)

    eps_hat = model(x_t, t)
    x0_hat = predict_x0_from_eps(x_t, eps_hat, t, alpha_bar)
    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)

    coef1 = (torch.sqrt(alpha_bar_prev_t) * beta_t) / (1.0 - alpha_bar_t)
    coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t)) / (1.0 - alpha_bar_t)
    mu = coef1 * x0_hat + coef2 * x_t

    if t_scalar > 0:
        z = torch.randn_like(x_t)
        x_prev = mu + torch.sqrt(post_var_t) * z
    else:
        x_prev = mu

    return x_prev


@torch.no_grad()
def make_reverse_states_ddpm_3d(
    model,
    chain_len,
    betas,
    alphas,
    alpha_bar,
    posterior_var,
    capture_ts_rev,
    x0_clip=3.0,
    project_bonds=False,
    bond_len=1.0,
    device="cpu",
):
    model.eval()

    x = torch.randn(1, chain_len, 3, device=device)
    capture_set = set(capture_ts_rev)
    saved = {}

    for t_int in reversed(range(len(betas))):
        if t_int in capture_set:
            x_save = x[0].detach().cpu()
            if project_bonds:
                x_save = project_bond_lengths(x_save, bond_len=bond_len)
            saved[t_int] = x_save.numpy()

        x = _p_sample_stable_movie(
            model=model,
            x_t=x,
            t_scalar=t_int,
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            posterior_var=posterior_var,
            x0_clip=x0_clip,
        )

    return [saved[t] for t in capture_ts_rev]


@torch.no_grad()
def _ddim_step_movie(model, x_t, t_scalar, s_scalar, alpha_bar, x0_clip=3.0):
    B = x_t.shape[0]
    device = x_t.device

    t = torch.full((B,), t_scalar, dtype=torch.long, device=device)
    s = torch.full((B,), s_scalar, dtype=torch.long, device=device)

    alpha_bar_t = extract(alpha_bar, t, x_t.shape)
    alpha_bar_s = extract(alpha_bar, s, x_t.shape)

    eps_hat = model(x_t, t)
    x0_hat = predict_x0_from_eps(x_t, eps_hat, t, alpha_bar)
    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)

    x_s = torch.sqrt(alpha_bar_s) * x0_hat + torch.sqrt(1.0 - alpha_bar_s) * eps_hat
    return x_s


@torch.no_grad()
def make_reverse_states_ddim_3d(
    model,
    chain_len,
    alpha_bar,
    num_steps=250,
    x0_clip=3.0,
    project_bonds=False,
    bond_len=1.0,
    device="cpu",
):
    model.eval()

    T = len(alpha_bar)
    x = torch.randn(1, chain_len, 3, device=device)

    ts = torch.linspace(T - 1, 0, steps=num_steps, device=device)
    ts = torch.round(ts).long()
    ts[-1] = 0
    ts = torch.unique_consecutive(ts)

    saved = []

    for i in range(len(ts) - 1):
        x_save = x[0].detach().cpu()
        if project_bonds:
            x_save = project_bond_lengths(x_save, bond_len=bond_len)
        saved.append(x_save.numpy())

        t_scalar = ts[i].item()
        s_scalar = ts[i + 1].item()

        x = _ddim_step_movie(
            model=model,
            x_t=x,
            t_scalar=t_scalar,
            s_scalar=s_scalar,
            alpha_bar=alpha_bar,
            x0_clip=x0_clip,
        )

    t_final = torch.zeros((1,), dtype=torch.long, device=device)
    eps_hat = model(x, t_final)
    x0_hat = predict_x0_from_eps(x, eps_hat, t_final, alpha_bar)
    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)

    x_save = x0_hat[0].detach().cpu()
    if project_bonds:
        x_save = project_bond_lengths(x_save, bond_len=bond_len)
    saved.append(x_save.numpy())

    return saved


@torch.no_grad()
def make_best_chain_movie(
    model,
    alpha_bar,
    betas=None,
    alphas=None,
    posterior_var=None,
    out_path="best_chain_movie.gif",
    total_len=64,
    bond_len=1.0,
    stride=10,
    fps=10,
    random_orient=True,
    center=True,
    reverse_mode="ddim",
    num_ddim_steps=250,
    x0_clip=3.0,
    project_bonds_reverse=True,
    device="cpu",
):
    import imageio.v2 as imageio

    x0, labels, seg_ids, specs = sample_clean_chain_for_movie(
        total_len=total_len,
        bond_len=bond_len,
        random_orient=random_orient,
        center=center,
        device=device,
    )
    x0_np = x0.detach().cpu().numpy()

    T = len(alpha_bar)
    capture_ts_fwd = list(range(0, T, stride))
    if capture_ts_fwd[-1] != T - 1:
        capture_ts_fwd.append(T - 1)

    fwd_states = make_forward_states_closed_form_3d(
        x0=x0,
        alpha_bar=alpha_bar,
        capture_ts=capture_ts_fwd,
    )

    if reverse_mode == "ddpm":
        if betas is None or alphas is None or posterior_var is None:
            raise ValueError("betas, alphas, and posterior_var are required for reverse_mode='ddpm'")

        capture_ts_rev = list(range(T - 1, -1, -stride))
        if capture_ts_rev[-1] != 0:
            capture_ts_rev.append(0)

        K = min(len(capture_ts_fwd), len(capture_ts_rev))
        capture_ts_fwd = capture_ts_fwd[:K]
        fwd_states = fwd_states[:K]
        capture_ts_rev = capture_ts_rev[:K]

        rev_states = make_reverse_states_ddpm_3d(
            model=model,
            chain_len=total_len,
            betas=betas,
            alphas=alphas,
            alpha_bar=alpha_bar,
            posterior_var=posterior_var,
            capture_ts_rev=capture_ts_rev,
            x0_clip=x0_clip,
            project_bonds=project_bonds_reverse,
            bond_len=bond_len,
            device=device,
        )
        reverse_titles = [f"Reverse DDPM: t={t}" for t in capture_ts_rev]

    elif reverse_mode == "ddim":
        rev_states_all = make_reverse_states_ddim_3d(
            model=model,
            chain_len=total_len,
            alpha_bar=alpha_bar,
            num_steps=num_ddim_steps,
            x0_clip=x0_clip,
            project_bonds=project_bonds_reverse,
            bond_len=bond_len,
            device=device,
        )

        import numpy as _np
        idxs = _np.linspace(0, len(rev_states_all) - 1, num=len(capture_ts_fwd))
        idxs = _np.round(idxs).astype(int)

        rev_states = [rev_states_all[i] for i in idxs]
        reverse_titles = [f"Reverse DDIM: frame {i}" for i in idxs]

    else:
        raise ValueError("reverse_mode must be 'ddpm' or 'ddim'")

    K = min(len(fwd_states), len(rev_states), len(capture_ts_fwd))
    fwd_states = fwd_states[:K]
    rev_states = rev_states[:K]
    capture_ts_fwd = capture_ts_fwd[:K]
    reverse_titles = reverse_titles[:K]

    frames = []

    for i in range(K):
        fig = plt.figure(figsize=(14, 4.5))

        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        draw_chain_pretty(ax1, x0_np, title="Clean reference", recenter=True, show_legend=True)

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        draw_chain_pretty(ax2, fwd_states[i], title=f"Forward: t={capture_ts_fwd[i]}", recenter=True, show_legend=True)

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        draw_chain_pretty(ax3, rev_states[i], title=reverse_titles[i], recenter=True, show_legend=True)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)
        buf.close()

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved GIF to: {out_path}")

    return {
        "x0": x0.detach().cpu(),
        "labels": labels.detach().cpu(),
        "seg_ids": seg_ids.detach().cpu(),
        "specs": specs,
    }
