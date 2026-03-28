from src.utils import set_seed, normalize, get_device

from src.geometry import (
    bond_vectors,
    unit_tangents,
    contour_length,
    end_to_end_distance,
    extension_ratio,
    rescale_steps_to_bond_length,
    project_bond_lengths,
    local_extension_ratio_batched,
    local_adj_anti_align_batched,
    local_gap2_align_batched,
)

from src.chains import (
    MOTIF_TO_ID,
    ID_TO_MOTIF,
    make_helix,
    make_strand,
    make_coil,
    generate_motif,
    stitch_segments,
    sample_random_specs,
    center_coords,
)

from src.dataset import SyntheticChainDataset, create_dataloader

from src.diffusion import (
    cosine_schedule,
    compute_posterior_variance,
    extract,
    q_sample,
)

from src.models import (
    time_embedding,
    EpsChainMLP,
    EpsChainTransformer,
)

from src.training import train_step, train_step_tf

from src.sampling import (
    predict_x0_from_eps,
    p_sample,
    ddpm_sample,
    p_sample_stable,
    ddpm_sample_stable,
    ddim_step,
    ddim_sample,
    ddpm_reverse_mean_var,
)

from src.rewards import strandness_reward

from src.ddpo import (
    gaussian_log_prob,
    normalizer,
    collect_ddpm_rollout_batch,
    trajectory_log_prob_under_model,
    ddpo_sf_update_step,
    ddpo_sf_update_step_anchor,
    diffusion_anchor_loss,
)

from src.evaluation import (
    evaluate_model_reward,
    evaluate_model_reward_full,
    get_sample_indices,
)

from src.visualization import (
    plot_chain_3d,
    plot_full_3d_chain,
    make_best_chain_movie,
)
