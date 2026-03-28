import torch

from src.utils import normalize

def bond_vectors(coords):
    return coords[1:] - coords[:-1]


def unit_tangents(coords, eps=1e-8):
    b = bond_vectors(coords)
    return b / b.norm(dim=-1, keepdim=True).clamp_min(eps)


def contour_length(coords):
    return (coords[1:] - coords[:-1]).norm(dim=-1).sum()


def end_to_end_distance(coords):
    return (coords[-1] - coords[0]).norm()


def extension_ratio(coords, eps=1e-8):
    return end_to_end_distance(coords) / contour_length(coords).clamp_min(eps)


def rescale_steps_to_bond_length(coords, bond_len=1.0):
    steps = coords[1:] - coords[:-1]
    steps = normalize(steps) * bond_len
    new_coords = [torch.zeros(3, dtype=coords.dtype, device=coords.device)]
    for s in steps:
        new_coords.append(new_coords[-1] + s)
    return torch.stack(new_coords, dim=0)


def project_bond_lengths(coords, bond_len=1.0, eps=1e-8):
    steps = coords[1:] - coords[:-1]
    steps = steps / steps.norm(dim=-1, keepdim=True).clamp_min(eps)
    steps = steps * bond_len

    new_coords = [coords[0]]
    for s in steps:
        new_coords.append(new_coords[-1] + s)
    return torch.stack(new_coords, dim=0)

def _as_batch(coords):
    if coords.ndim == 2:
        return coords.unsqueeze(0), True
    elif coords.ndim == 3:
        return coords, False
    else:
        raise ValueError(f"Expected shape (L,3) or (B,L,3), got {tuple(coords.shape)}")


def bond_vectors_batched(coords_b):
    return coords_b[:, 1:] - coords_b[:, :-1]


def unit_tangents_batched(coords_b, eps=1e-8):
    b = bond_vectors_batched(coords_b)
    return b / b.norm(dim=-1, keepdim=True).clamp_min(eps)


def local_extension_ratio_batched(coords, window=7, eps=1e-8):
    coords_b, squeezed = _as_batch(coords)
    B, L, _ = coords_b.shape

    if window < 3 or window > L:
        raise ValueError(f"window must be in [3, L], got window={window}, L={L}")

    step_len = bond_vectors_batched(coords_b).norm(dim=-1)
    contour = step_len.unfold(dimension=1, size=window - 1, step=1).sum(dim=-1)

    chord = (coords_b[:, window - 1:] - coords_b[:, :L - window + 1]).norm(dim=-1)

    ext = chord / contour.clamp_min(eps)
    ext = ext.clamp(0.0, 1.0)

    if squeezed:
        return ext[0]
    return ext


def local_adj_anti_align_batched(coords, window=7, eps=1e-8):
    coords_b, squeezed = _as_batch(coords)
    B, L, _ = coords_b.shape

    if window < 4 or window > L:
        raise ValueError(f"window must be in [4, L], got window={window}, L={L}")

    u = unit_tangents_batched(coords_b, eps=eps)
    adj = (u[:, :-1] * u[:, 1:]).sum(dim=-1).clamp(-1.0, 1.0)

    anti = (1 - adj) / 2.0
    anti = anti.unfold(dimension=1, size=window - 2, step=1).mean(dim=-1)

    if squeezed:
        return anti[0]
    return anti


def local_gap2_align_batched(coords, window=7, eps=1e-8):
    coords_b, squeezed = _as_batch(coords)
    B, L, _ = coords_b.shape

    if window < 4 or window > L:
        raise ValueError(f"Window must be in [4, L], got window={window}, L={L}")

    u = unit_tangents_batched(coords_b, eps=eps)

    gap2 = (u[:, :-2] * u[:, 2:]).sum(dim=-1).clamp(-1.0, 1.0)
    gap2 = (1.0 + gap2) / 2.0
    gap2 = gap2.unfold(dimension=1, size=window - 3, step=1).mean(dim=-1)

    if squeezed:
        return gap2[0]
    return gap2
