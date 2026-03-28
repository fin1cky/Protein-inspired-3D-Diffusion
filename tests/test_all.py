import math
import torch
import pytest

from src.utils import set_seed, normalize, get_device
from src.geometry import (
    bond_vectors, unit_tangents, contour_length, end_to_end_distance,
    extension_ratio, rescale_steps_to_bond_length, project_bond_lengths,
    _as_batch, bond_vectors_batched, unit_tangents_batched,
    local_extension_ratio_batched, local_adj_anti_align_batched,
    local_gap2_align_batched,
)
from src.chains import (
    MOTIF_TO_ID, ID_TO_MOTIF,
    random_rotation_matrix, apply_random_se3, center_coords,
    make_helix, make_strand, make_coil, generate_motif,
    stitch_segments, sample_random_specs,
    sample_segment_lengths, sample_motif_kinds, sample_motif_kwargs,
)
from src.dataset import SyntheticChainDataset, create_dataloader
from src.diffusion import cosine_schedule, compute_posterior_variance, extract, q_sample
from src.models import time_embedding, EpsChainMLP, EpsChainTransformer
from src.training import train_step, train_step_tf
from src.sampling import (
    predict_x0_from_eps, p_sample, ddpm_sample,
    p_sample_stable, ddpm_sample_stable, ddim_step, ddim_sample,
    ddpm_reverse_mean_var,
)
from src.rewards import strandness_reward
from src.ddpo import (
    gaussian_log_prob, normalizer,
    collect_ddpm_rollout_batch, trajectory_log_prob_under_model,
    ddpo_sf_update_step, ddpo_sf_update_step_anchor, diffusion_anchor_loss,
)
from src.evaluation import evaluate_model_reward, evaluate_model_reward_full, get_sample_indices
from src.visualization import plot_chain_3d, plot_full_3d_chain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHAIN_LEN = 64
T_SMALL = 50  # small T for fast tests


@pytest.fixture(autouse=True)
def seed():
    set_seed(42)


@pytest.fixture
def schedule():
    betas, alphas, alpha_bar = cosine_schedule(T_SMALL, device="cpu")
    posterior_var = compute_posterior_variance(betas, alpha_bar)
    return betas, alphas, alpha_bar, posterior_var


@pytest.fixture
def transformer():
    return EpsChainTransformer(chain_len=CHAIN_LEN).to("cpu")


@pytest.fixture
def chain():
    specs = sample_random_specs(total_len=CHAIN_LEN)
    coords, labels, seg_ids = stitch_segments(specs, device="cpu")
    return center_coords(coords), labels, seg_ids


# ===========================================================================
# Utils
# ===========================================================================

class TestUtils:
    def test_set_seed_deterministic(self):
        set_seed(0)
        a = torch.randn(3)
        set_seed(0)
        b = torch.randn(3)
        assert torch.equal(a, b)

    def test_normalize(self):
        v = torch.tensor([[3.0, 0.0, 4.0]])
        n = normalize(v)
        assert torch.allclose(n.norm(dim=-1), torch.ones(1), atol=1e-6)

    def test_get_device(self):
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


# ===========================================================================
# Geometry
# ===========================================================================

class TestGeometry:
    def test_bond_vectors_shape(self, chain):
        coords, _, _ = chain
        bv = bond_vectors(coords)
        assert bv.shape == (CHAIN_LEN - 1, 3)

    def test_unit_tangents_norm(self, chain):
        coords, _, _ = chain
        ut = unit_tangents(coords)
        norms = ut.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_contour_length_positive(self, chain):
        coords, _, _ = chain
        cl = contour_length(coords)
        assert cl.item() > 0

    def test_extension_ratio_range(self, chain):
        coords, _, _ = chain
        er = extension_ratio(coords)
        assert 0 <= er.item() <= 1.0 + 1e-5

    def test_rescale_bond_length(self):
        coords = torch.randn(20, 3)
        rescaled = rescale_steps_to_bond_length(coords, bond_len=1.5)
        bl = (rescaled[1:] - rescaled[:-1]).norm(dim=-1)
        assert torch.allclose(bl, torch.full_like(bl, 1.5), atol=1e-5)

    def test_project_bond_lengths(self, chain):
        coords, _, _ = chain
        proj = project_bond_lengths(coords, bond_len=1.0)
        bl = (proj[1:] - proj[:-1]).norm(dim=-1)
        assert torch.allclose(bl, torch.ones_like(bl), atol=1e-5)

    def test_as_batch_2d(self):
        coords = torch.randn(10, 3)
        batched, squeezed = _as_batch(coords)
        assert batched.shape == (1, 10, 3)
        assert squeezed is True

    def test_as_batch_3d(self):
        coords = torch.randn(4, 10, 3)
        batched, squeezed = _as_batch(coords)
        assert batched.shape == (4, 10, 3)
        assert squeezed is False

    def test_batched_geometry_shapes(self):
        coords = torch.randn(2, 20, 3)
        assert bond_vectors_batched(coords).shape == (2, 19, 3)
        assert unit_tangents_batched(coords).shape == (2, 19, 3)
        ext = local_extension_ratio_batched(coords, window=7)
        assert ext.shape[0] == 2
        anti = local_adj_anti_align_batched(coords, window=7)
        assert anti.shape[0] == 2
        gap2 = local_gap2_align_batched(coords, window=7)
        assert gap2.shape[0] == 2

    def test_local_extension_ratio_range(self):
        coords = torch.randn(2, 20, 3)
        ext = local_extension_ratio_batched(coords, window=7)
        assert (ext >= 0).all() and (ext <= 1.0 + 1e-5).all()


# ===========================================================================
# Chains
# ===========================================================================

class TestChains:
    def test_constants(self):
        assert MOTIF_TO_ID["helix"] == 0
        assert ID_TO_MOTIF[1] == "strand"

    def test_random_rotation_matrix_orthogonal(self):
        R = random_rotation_matrix()
        assert R.shape == (3, 3)
        eye = R @ R.T
        assert torch.allclose(eye, torch.eye(3), atol=1e-5)
        assert torch.det(R).item() > 0  # proper rotation

    def test_make_helix_shape(self):
        h = make_helix(20)
        assert h.shape == (20, 3)

    def test_make_strand_shape(self):
        s = make_strand(20)
        assert s.shape == (20, 3)

    def test_make_coil_shape(self):
        c = make_coil(20)
        assert c.shape == (20, 3)

    def test_generate_motif_dispatch(self):
        for kind in ("helix", "strand", "coil"):
            m = generate_motif(kind, 15)
            assert m.shape == (15, 3)

    def test_generate_motif_invalid(self):
        with pytest.raises(ValueError):
            generate_motif("invalid", 10)

    def test_center_coords(self):
        coords = torch.randn(10, 3) + 100
        centered = center_coords(coords)
        assert torch.allclose(centered.mean(dim=0), torch.zeros(3), atol=1e-5)

    def test_apply_random_se3(self):
        coords = torch.randn(10, 3)
        transformed = apply_random_se3(coords)
        assert transformed.shape == coords.shape
        assert not torch.equal(transformed, coords)

    def test_stitch_segments_total_length(self):
        specs = sample_random_specs(total_len=CHAIN_LEN)
        coords, labels, seg_ids = stitch_segments(specs)
        assert coords.shape == (CHAIN_LEN, 3)
        assert labels.shape == (CHAIN_LEN,)
        assert seg_ids.shape == (CHAIN_LEN,)

    def test_sample_segment_lengths_sum(self):
        lengths = sample_segment_lengths(64, 4, min_len=12, max_len=24)
        assert sum(lengths) == 64
        assert all(12 <= l <= 24 for l in lengths)

    def test_sample_motif_kinds_no_consecutive_duplicates(self):
        kinds = sample_motif_kinds(10)
        for a, b in zip(kinds, kinds[1:]):
            assert a != b

    def test_sample_motif_kwargs_keys(self):
        assert "radius" in sample_motif_kwargs("helix")
        assert "zigzag_const" in sample_motif_kwargs("strand")
        assert "turn_strength" in sample_motif_kwargs("coil")


# ===========================================================================
# Dataset
# ===========================================================================

class TestDataset:
    def test_dataset_len(self):
        ds = SyntheticChainDataset(num_samples=10, total_len=CHAIN_LEN)
        assert len(ds) == 10

    def test_dataset_getitem(self):
        ds = SyntheticChainDataset(num_samples=5, total_len=CHAIN_LEN)
        item = ds[0]
        assert item["coords"].shape == (CHAIN_LEN, 3)
        assert item["labels"].shape == (CHAIN_LEN,)
        assert item["seg_ids"].shape == (CHAIN_LEN,)

    def test_create_dataloader(self):
        dl = create_dataloader(num_samples=8, batch_size=4, total_len=CHAIN_LEN)
        batch = next(iter(dl))
        assert batch["coords"].shape == (4, CHAIN_LEN, 3)


# ===========================================================================
# Diffusion
# ===========================================================================

class TestDiffusion:
    def test_cosine_schedule_shapes(self):
        betas, alphas, alpha_bar = cosine_schedule(100)
        assert betas.shape == (100,)
        assert alphas.shape == (100,)
        assert alpha_bar.shape == (100,)

    def test_cosine_schedule_monotonic(self):
        _, _, alpha_bar = cosine_schedule(100)
        diffs = alpha_bar[1:] - alpha_bar[:-1]
        assert (diffs <= 0).all()  # alpha_bar is decreasing

    def test_compute_posterior_variance(self, schedule):
        _, _, _, posterior_var = schedule
        assert posterior_var.shape == (T_SMALL,)
        assert (posterior_var > 0).all()

    def test_extract_shape(self):
        a = torch.randn(100)
        t = torch.tensor([5, 10])
        out = extract(a, t, (2, 64, 3))
        assert out.shape == (2, 1, 1)

    def test_q_sample_shape(self, schedule):
        _, _, alpha_bar, _ = schedule
        coords0 = torch.randn(4, CHAIN_LEN, 3)
        t = torch.randint(0, T_SMALL, (4,))
        xt, eps = q_sample(coords0, t, alpha_bar)
        assert xt.shape == (4, CHAIN_LEN, 3)
        assert eps.shape == (4, CHAIN_LEN, 3)

    def test_q_sample_t0_close_to_original(self):
        # Use full T=1000 schedule so alpha_bar[0] ≈ 1.0
        _, _, alpha_bar_full = cosine_schedule(1000)
        coords0 = torch.randn(1, CHAIN_LEN, 3)
        t = torch.zeros(1, dtype=torch.long)
        xt, _ = q_sample(coords0, t, alpha_bar_full)
        assert torch.allclose(xt, coords0, atol=0.1)


# ===========================================================================
# Models
# ===========================================================================

class TestModels:
    def test_time_embedding_shape(self):
        t = torch.tensor([0, 50, 99])
        emb = time_embedding(t, dim=128)
        assert emb.shape == (3, 128)

    def test_eps_chain_mlp_forward(self):
        model = EpsChainMLP(chain_len=CHAIN_LEN)
        xt = torch.randn(2, CHAIN_LEN, 3)
        t = torch.tensor([10, 20])
        out = model(xt, t)
        assert out.shape == (2, CHAIN_LEN, 3)

    def test_eps_chain_transformer_forward(self, transformer):
        xt = torch.randn(2, CHAIN_LEN, 3)
        t = torch.tensor([10, 20])
        out = transformer(xt, t)
        assert out.shape == (2, CHAIN_LEN, 3)

    def test_transformer_grad_flows(self, transformer):
        xt = torch.randn(2, CHAIN_LEN, 3)
        t = torch.tensor([10, 20])
        out = transformer(xt, t)
        loss = out.sum()
        loss.backward()
        assert transformer.coord_proj.weight.grad is not None


# ===========================================================================
# Training
# ===========================================================================

class TestTraining:
    def test_train_step_mlp(self, schedule):
        _, _, alpha_bar, _ = schedule
        model = EpsChainMLP(chain_len=CHAIN_LEN)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch = {"coords": torch.randn(4, CHAIN_LEN, 3)}
        loss = train_step(model, batch, alpha_bar, optimizer, T_SMALL, device="cpu")
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_tf(self, schedule, transformer):
        _, _, alpha_bar, _ = schedule
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
        batch = {"coords": torch.randn(4, CHAIN_LEN, 3)}
        loss = train_step_tf(transformer, batch, alpha_bar, optimizer, T_SMALL, device="cpu")
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_reduces_loss(self, schedule):
        _, _, alpha_bar, _ = schedule
        model = EpsChainMLP(chain_len=CHAIN_LEN)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = {"coords": torch.randn(8, CHAIN_LEN, 3)}
        loss1 = train_step(model, batch, alpha_bar, optimizer, T_SMALL, device="cpu")
        for _ in range(20):
            train_step(model, batch, alpha_bar, optimizer, T_SMALL, device="cpu")
        loss2 = train_step(model, batch, alpha_bar, optimizer, T_SMALL, device="cpu")
        assert loss2 < loss1


# ===========================================================================
# Sampling
# ===========================================================================

class TestSampling:
    def test_predict_x0_from_eps(self, schedule):
        _, _, alpha_bar, _ = schedule
        xt = torch.randn(2, CHAIN_LEN, 3)
        eps_hat = torch.randn(2, CHAIN_LEN, 3)
        t = torch.tensor([5, 10])
        x0 = predict_x0_from_eps(xt, eps_hat, t, alpha_bar)
        assert x0.shape == (2, CHAIN_LEN, 3)

    def test_ddpm_sample_shape(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        samples = ddpm_sample(transformer, 2, CHAIN_LEN, betas, alphas, alpha_bar, posterior_var, device="cpu")
        assert samples.shape == (2, CHAIN_LEN, 3)

    def test_ddpm_sample_stable_shape(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        samples = ddpm_sample_stable(transformer, 2, CHAIN_LEN, betas, alphas, alpha_bar, posterior_var, device="cpu")
        assert samples.shape == (2, CHAIN_LEN, 3)

    def test_ddim_sample_shape(self, schedule, transformer):
        _, _, alpha_bar, _ = schedule
        samples = ddim_sample(transformer, 2, CHAIN_LEN, alpha_bar, device="cpu", num_steps=T_SMALL)
        assert samples.shape == (2, CHAIN_LEN, 3)

    def test_ddpm_reverse_mean_var(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        xt = torch.randn(2, CHAIN_LEN, 3)
        mean, var, x0_hat, eps_hat = ddpm_reverse_mean_var(
            transformer, xt, 10, betas, alphas, alpha_bar, posterior_var,
        )
        assert mean.shape == (2, CHAIN_LEN, 3)
        assert var.shape == (2, 1, 1)
        assert x0_hat.shape == (2, CHAIN_LEN, 3)


# ===========================================================================
# Rewards
# ===========================================================================

class TestRewards:
    def test_strandness_reward_single(self, chain):
        coords, _, _ = chain
        r = strandness_reward(coords)
        assert r.ndim == 0  # scalar
        assert 0 <= r.item() <= 1.0

    def test_strandness_reward_batch(self):
        coords = torch.randn(4, 20, 3)
        r = strandness_reward(coords)
        assert r.shape == (4,)

    def test_strandness_reward_details(self, chain):
        coords, _, _ = chain
        out = strandness_reward(coords, return_details=True)
        assert "reward" in out
        assert "adj_anti_align" in out
        assert "gap2_align" in out
        assert "local_extension" in out

    def test_strand_higher_reward_than_helix(self):
        s = make_strand(64)
        h = make_helix(64)
        rs = strandness_reward(s)
        rh = strandness_reward(h)
        assert rs > rh


# ===========================================================================
# DDPO
# ===========================================================================

class TestDDPO:
    def test_gaussian_log_prob_shape(self):
        x = torch.randn(4, CHAIN_LEN, 3)
        mean = torch.randn(4, CHAIN_LEN, 3)
        var = torch.ones(4, 1, 1) * 0.1
        lp = gaussian_log_prob(x, mean, var)
        assert lp.shape == (4,)

    def test_normalizer(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        n = normalizer(x)
        assert torch.allclose(n.mean(), torch.tensor(0.0), atol=1e-5)

    def test_collect_rollout(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        rollout = collect_ddpm_rollout_batch(
            model=transformer, n_samples=2, chain_len=CHAIN_LEN,
            betas=betas, alphas=alphas, alpha_bar=alpha_bar, posterior_var=posterior_var,
            reward_fn=strandness_reward, device="cpu",
        )
        assert rollout["reward"].shape == (2,)
        assert rollout["x0_raw"].shape == (2, CHAIN_LEN, 3)
        assert len(rollout["traj"]) == T_SMALL

    def test_trajectory_log_prob(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        rollout = collect_ddpm_rollout_batch(
            model=transformer, n_samples=2, chain_len=CHAIN_LEN,
            betas=betas, alphas=alphas, alpha_bar=alpha_bar, posterior_var=posterior_var,
            reward_fn=strandness_reward, device="cpu",
        )
        log_p = trajectory_log_prob_under_model(
            transformer, rollout["traj"], betas, alphas, alpha_bar, posterior_var,
        )
        assert log_p.shape == (2,)
        assert torch.isfinite(log_p).all()

    def test_ddpo_sf_update_step(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-5)
        rollout = collect_ddpm_rollout_batch(
            model=transformer, n_samples=2, chain_len=CHAIN_LEN,
            betas=betas, alphas=alphas, alpha_bar=alpha_bar, posterior_var=posterior_var,
            reward_fn=strandness_reward, device="cpu",
        )
        stats = ddpo_sf_update_step(
            transformer, optimizer, rollout,
            betas, alphas, alpha_bar, posterior_var, x0_clip=3.0,
        )
        assert "loss_pg" in stats
        assert "reward_mean" in stats
        assert "grad_norm" in stats

    def test_ddpo_anchor_update(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-5)
        rollout = collect_ddpm_rollout_batch(
            model=transformer, n_samples=2, chain_len=CHAIN_LEN,
            betas=betas, alphas=alphas, alpha_bar=alpha_bar, posterior_var=posterior_var,
            reward_fn=strandness_reward, device="cpu",
        )
        anchor = torch.randn(2, CHAIN_LEN, 3)
        stats = ddpo_sf_update_step_anchor(
            transformer, optimizer, rollout,
            betas, alphas, alpha_bar, posterior_var,
            anchor_batch=anchor, anchor_coef=0.02,
        )
        assert "loss_total" in stats
        assert "loss_anchor" in stats
        assert stats["loss_anchor"] > 0

    def test_diffusion_anchor_loss(self, schedule, transformer):
        _, _, alpha_bar, _ = schedule
        coords = torch.randn(4, CHAIN_LEN, 3)
        loss = diffusion_anchor_loss(transformer, coords, alpha_bar)
        assert loss.ndim == 0
        assert loss.item() > 0


# ===========================================================================
# Evaluation
# ===========================================================================

class TestEvaluation:
    def test_evaluate_model_reward(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        out = evaluate_model_reward(
            model=transformer, n_samples=4, chain_len=CHAIN_LEN,
            betas=betas, alphas=alphas, alpha_bar=alpha_bar, posterior_var=posterior_var,
            reward_fn=strandness_reward, device="cpu",
        )
        assert "reward_mean" in out
        assert out["samples"].shape == (4, CHAIN_LEN, 3)
        assert out["rewards"].shape == (4,)

    def test_evaluate_model_reward_full_deterministic(self, schedule, transformer):
        betas, alphas, alpha_bar, posterior_var = schedule
        kwargs = dict(
            model=transformer, n_samples=4, chain_len=CHAIN_LEN,
            betas=betas, alphas=alphas, alpha_bar=alpha_bar, posterior_var=posterior_var,
            reward_fn=strandness_reward, device="cpu", seed=123,
        )
        out1 = evaluate_model_reward_full(**kwargs)
        out2 = evaluate_model_reward_full(**kwargs)
        assert out1["reward_mean"] == out2["reward_mean"]

    def test_get_sample_indices(self):
        rewards = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7])
        idx = get_sample_indices(rewards, k=2)
        assert idx["top"] == [1, 4]
        assert idx["bottom"] == [3, 0]
        assert len(idx["random"]) == 2


# ===========================================================================
# Visualization
# ===========================================================================

class TestVisualization:
    def test_plot_chain_3d(self, chain):
        import matplotlib
        matplotlib.use("Agg")
        coords, _, _ = chain
        ax = plot_chain_3d(coords, title="test")
        assert ax is not None

    def test_plot_full_3d_chain(self, chain):
        import matplotlib
        matplotlib.use("Agg")
        coords, labels, _ = chain
        ax = plot_full_3d_chain(coords, labels, title="test")
        assert ax is not None


# ===========================================================================
# Weight Loading
# ===========================================================================

class TestWeightLoading:
    @pytest.fixture
    def baseline_path(self):
        return "weights/baseline/chain_diffusion_transformer.pt"

    @pytest.fixture
    def ddpo_path(self):
        return "weights/DDPO-SF/best_ddpo_diffusion.pt"

    def test_load_baseline(self, baseline_path):
        ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
        assert "model" in ckpt
        model = EpsChainTransformer(chain_len=CHAIN_LEN)
        model.load_state_dict(ckpt["model"])

    def test_load_ddpo(self, ddpo_path):
        ckpt = torch.load(ddpo_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        model = EpsChainTransformer(chain_len=CHAIN_LEN)
        model.load_state_dict(ckpt["model_state_dict"])

    def test_baseline_generates_valid_samples(self, baseline_path, schedule):
        _, _, alpha_bar, _ = schedule
        ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
        model = EpsChainTransformer(chain_len=CHAIN_LEN)
        model.load_state_dict(ckpt["model"])
        with torch.no_grad():
            samples = ddim_sample(model, 2, CHAIN_LEN, alpha_bar, device="cpu", num_steps=T_SMALL)
        assert samples.shape == (2, CHAIN_LEN, 3)
        assert torch.isfinite(samples).all()

    def test_ddpo_generates_valid_samples(self, ddpo_path, schedule):
        _, _, alpha_bar, _ = schedule
        ckpt = torch.load(ddpo_path, map_location="cpu", weights_only=False)
        model = EpsChainTransformer(chain_len=CHAIN_LEN)
        model.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            samples = ddim_sample(model, 2, CHAIN_LEN, alpha_bar, device="cpu", num_steps=T_SMALL)
        assert samples.shape == (2, CHAIN_LEN, 3)
        assert torch.isfinite(samples).all()


# ===========================================================================
# MPS Device (skip if not available)
# ===========================================================================

MPS_AVAILABLE = torch.backends.mps.is_available()


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
class TestMPS:
    def test_schedule_on_mps(self):
        betas, alphas, alpha_bar = cosine_schedule(T_SMALL, device="mps")
        assert betas.device.type == "mps"

    def test_model_forward_on_mps(self):
        model = EpsChainTransformer(chain_len=CHAIN_LEN).to("mps")
        xt = torch.randn(2, CHAIN_LEN, 3, device="mps")
        t = torch.randint(0, T_SMALL, (2,), device="mps")
        with torch.no_grad():
            out = model(xt, t)
        assert out.device.type == "mps"
        assert out.shape == (2, CHAIN_LEN, 3)

    def test_ddim_sample_on_mps(self):
        _, _, alpha_bar = cosine_schedule(T_SMALL, device="mps")
        model = EpsChainTransformer(chain_len=CHAIN_LEN).to("mps")
        with torch.no_grad():
            samples = ddim_sample(model, 2, CHAIN_LEN, alpha_bar, device="mps", num_steps=T_SMALL)
        assert samples.device.type == "mps"

    def test_training_on_mps(self):
        betas, alphas, alpha_bar = cosine_schedule(T_SMALL, device="mps")
        model = EpsChainTransformer(chain_len=CHAIN_LEN).to("mps")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        ds = SyntheticChainDataset(num_samples=4, total_len=CHAIN_LEN, device="mps")
        from torch.utils.data import DataLoader
        batch = next(iter(DataLoader(ds, batch_size=2)))
        loss = train_step_tf(model, batch, alpha_bar, optimizer, T_SMALL, device="mps")
        assert isinstance(loss, float)

    def test_chain_generation_on_mps(self):
        h = make_helix(20, device="mps")
        assert h.device.type == "mps"
        s = make_strand(20, device="mps")
        assert s.device.type == "mps"
