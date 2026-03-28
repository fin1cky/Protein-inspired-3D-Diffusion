import torch

from src.geometry import (
    _as_batch,
    local_adj_anti_align_batched,
    local_gap2_align_batched,
    local_extension_ratio_batched,
)


def strandness_reward(coords, window=7, w_adj=0.80, w_gap2=0.20, return_details=False, eps=1e-8):
    coords_b, squeezed = _as_batch(coords)

    adj_anti = local_adj_anti_align_batched(coords_b, window=window, eps=eps)
    gap2_aln = local_gap2_align_batched(coords_b, window=window, eps=eps)
    ext = local_extension_ratio_batched(coords_b, window=window, eps=eps)

    window_score = w_adj * adj_anti + w_gap2 * gap2_aln
    reward = window_score.mean(dim=1)

    if not return_details:
        if squeezed:
            return reward[0]
        return reward

    out = {
        "reward": reward,
        "window_score": window_score,
        "adj_anti_align": adj_anti,
        "gap2_align": gap2_aln,
        "local_extension": ext,
        "best_window_idx": window_score.argmax(dim=1),
    }

    if squeezed:
        out = {
            k: (v[0] if torch.is_tensor(v) and v.ndim > 0 else v)
            for k, v in out.items()
        }

    return out
