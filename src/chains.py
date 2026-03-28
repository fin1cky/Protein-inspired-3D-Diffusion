import math
import random
import torch

from src.utils import normalize
from src.geometry import rescale_steps_to_bond_length


MOTIF_TO_ID = {"helix": 0, "strand": 1, "coil": 2}
ID_TO_MOTIF = {0: "helix", 1: "strand", 2: "coil"}


def random_rotation_matrix(dtype=torch.float32, device="cpu"):
    # QR decomposition on CPU (not supported on MPS), then move to target device
    A = torch.randn(3, 3, dtype=dtype, device="cpu")
    Q, R = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(device)


def apply_random_se3(coords, rotate=True, translate=True):
    out = coords.clone()
    if rotate:
        R = random_rotation_matrix(dtype=out.dtype, device=out.device)
        out = out @ R.T
    if translate:
        t = torch.randn(1, 3, dtype=out.dtype, device=out.device)
        out = out + t
    return out


def center_coords(coords):
    return coords - coords.mean(dim=0, keepdim=True)


def make_helix(L, radius=1.0, rise_per_res=0.6, residues_per_turn=3.6, bond_len=1.0, device="cpu"):
    i = torch.arange(L, dtype=torch.float32, device=device)
    theta = 2 * math.pi * i / residues_per_turn

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = rise_per_res * i

    coords = torch.stack([x, y, z], dim=-1)
    coords = rescale_steps_to_bond_length(coords, bond_len=bond_len)
    return coords


def make_strand(L, zigzag_const=1.0, z_const=0.5, forward_step=0.5, bond_len=1.0, device="cpu"):
    i = torch.arange(L, dtype=torch.float32, device=device)

    x = forward_step * i
    y = zigzag_const * ((-1.0) ** i)
    z = z_const * ((-1.0) ** (i + 1))

    coords = torch.stack([x, y, z], dim=-1)
    coords = rescale_steps_to_bond_length(coords, bond_len=bond_len)
    return coords


def make_coil(L, turn_strength=0.35, bond_len=1.0, device="cpu"):
    coords = [torch.zeros(3, dtype=torch.float32, device=device)]
    direction = torch.randn(3, device=device)
    direction = direction / direction.norm()

    for _ in range(L - 1):
        noise = turn_strength * torch.randn(3, device=device)
        direction = direction + noise
        direction = direction / direction.norm()
        next_pt = coords[-1] + bond_len * direction
        coords.append(next_pt)

    return torch.stack(coords, dim=0)


def generate_motif(kind, L, bond_len=1.0, device="cpu", **kwargs):
    if kind == "helix":
        return make_helix(L=L, bond_len=bond_len, device=device, **kwargs)
    elif kind == "strand":
        return make_strand(L=L, bond_len=bond_len, device=device, **kwargs)
    elif kind == "coil":
        return make_coil(L=L, bond_len=bond_len, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown motif kind: {kind}")

def stitch_segments(specs, bond_len=1.0, random_orient=True, device="cpu"):
    coords_parts = []
    label_parts = []
    segid_parts = []

    prev_end = None

    for seg_id, spec in enumerate(specs):
        kind = spec["kind"]
        contrib_len = spec["length"]
        kwargs = spec.get("kwargs", {})

        raw_len = contrib_len if seg_id == 0 else contrib_len + 1

        seg = generate_motif(
            kind=kind,
            L=raw_len,
            bond_len=bond_len,
            device=device,
            **kwargs,
        )

        if random_orient:
            R = random_rotation_matrix(dtype=seg.dtype, device=seg.device)
            seg = seg @ R.T

        seg = seg - seg[0:1]

        if seg_id == 0:
            placed = seg
        else:
            seg = seg + prev_end.unsqueeze(0)
            placed = seg[1:]

        coords_parts.append(placed)
        label_parts.append(
            torch.full((placed.shape[0],), MOTIF_TO_ID[kind], dtype=torch.long, device=device)
        )
        segid_parts.append(
            torch.full((placed.shape[0],), seg_id, dtype=torch.long, device=device)
        )

        prev_end = placed[-1]

    coords = torch.cat(coords_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    seg_ids = torch.cat(segid_parts, dim=0)

    return coords, labels, seg_ids

def sample_segment_lengths(total_len, n_segments, min_len=12, max_len=24):
    if total_len < n_segments * min_len:
        raise ValueError("total_len too small for chosen n_segments and min_len")
    if total_len > n_segments * max_len:
        raise ValueError("total_len too large for chosen n_segments and max_len")

    lengths = [min_len] * n_segments
    remaining = total_len - (n_segments * min_len)
    capacities = [max_len - min_len] * n_segments

    while remaining > 0:
        valid_idxs = [i for i in range(n_segments) if capacities[i] > 0]
        j = random.choice(valid_idxs)
        lengths[j] += 1
        capacities[j] -= 1
        remaining -= 1

    return lengths


def sample_motif_kinds(n_segments, motif_names=("helix", "strand", "coil")):
    kinds = []
    prev = None

    for _ in range(n_segments):
        choices = [m for m in motif_names if m != prev]
        kind = random.choice(choices)
        kinds.append(kind)
        prev = kind

    return kinds


def sample_motif_kwargs(kind):
    if kind == "helix":
        return {
            "radius": random.uniform(1.5, 2.2),
            "rise_per_res": random.uniform(0.20, 0.35),
            "residues_per_turn": random.uniform(8.0, 12.0),
        }
    elif kind == "strand":
        return {
            "zigzag_const": random.uniform(0.8, 1.2),
            "z_const": random.uniform(0.3, 0.7),
            "forward_step": random.uniform(0.4, 0.7),
        }
    elif kind == "coil":
        return {
            "turn_strength": random.uniform(0.20, 0.45),
        }
    else:
        raise ValueError(f"Unknown motif kind: {kind}")


def sample_random_specs(total_len=64, min_segments=3, max_segments=5, min_len=12, max_len=24):
    feasible_segment_counts = []
    for n in range(min_segments, max_segments + 1):
        if n * min_len <= total_len <= n * max_len:
            feasible_segment_counts.append(n)

    if not feasible_segment_counts:
        raise ValueError("No feasible number of segments for these constraints")

    n_segments = random.choice(feasible_segment_counts)
    lengths = sample_segment_lengths(
        total_len=total_len,
        n_segments=n_segments,
        min_len=min_len,
        max_len=max_len,
    )
    kinds = sample_motif_kinds(n_segments)

    specs = []
    for kind, length in zip(kinds, lengths):
        specs.append({
            "kind": kind,
            "length": length,
            "kwargs": sample_motif_kwargs(kind),
        })

    return specs
