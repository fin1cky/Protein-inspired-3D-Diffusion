import math
import torch
import torch.nn as nn


def time_embedding(t: torch.Tensor, dim=128, max_period=10000):
    t = t.view(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * (torch.arange(0, half, device=t.device).float() / max(half - 1, 1))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    embs = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2 == 1:
        embs = torch.cat([embs, torch.zeros(embs.shape[0], 1, device=t.device)], dim=1)

    return embs


class EpsChainMLP(nn.Module):
    def __init__(self, chain_len, time_dim=128, hidden_dim=512):
        super().__init__()
        self.chain_len = chain_len
        self.time_dim = time_dim
        self.input_dim = chain_len * 3

        self.net = nn.Sequential(
            nn.Linear(self.input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def forward(self, xt, t):
        B, L, C = xt.shape
        assert L == self.chain_len
        assert C == 3

        x_flat = xt.reshape(B, L * C)
        t_emb = time_embedding(t, self.time_dim)

        h = torch.cat([x_flat, t_emb], dim=-1)
        output = self.net(h)
        eps_hat = output.reshape(B, L, C)

        return eps_hat


class EpsChainTransformer(nn.Module):
    def __init__(
        self,
        chain_len,
        dim_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        time_dim=128,
        dropout=0.0,
    ):
        super().__init__()
        self.chain_len = chain_len
        self.dim_model = dim_model
        self.time_dim = time_dim

        self.coord_proj = nn.Linear(3, dim_model)

        self.pos_emb = nn.Parameter(torch.randn(1, chain_len, dim_model) * 0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_norm = nn.LayerNorm(dim_model)
        self.out_proj = nn.Linear(dim_model, 3)

    def forward(self, xt, t):
        B, L, C = xt.shape
        assert L == self.chain_len
        assert C == 3

        h = self.coord_proj(xt)
        h = h + self.pos_emb[:, :L, :]

        t_emb = time_embedding(t, self.time_dim)
        t_tok = self.time_mlp(t_emb).unsqueeze(1)
        h = h + t_tok

        h = self.encoder(h)
        h = self.out_norm(h)
        eps_hat = self.out_proj(h)

        return eps_hat
