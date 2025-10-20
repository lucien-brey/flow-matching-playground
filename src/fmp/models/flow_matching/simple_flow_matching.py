import math

import torch
from torch import nn

"""
Basic flow matching model taken from
https://github.com/dome272/Flow-Matching
"""


class Block(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))


class SimpleFlowMatching(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.channels_data = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, xt, t):
        x = torch.cat([xt, t[:, None]], dim=-1)
        return self.net(x)


class MLP_2(nn.Module):
    def __init__(self, channels_data: int = 2, layers: int = 5, channels: int = 512, channels_t: int = 512):
        super().__init__()
        self.channels_t = channels_t
        self.channels_data = channels_data
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t: torch.Tensor, max_positions: int = 10000) -> torch.Tensor:
        t = t * max_positions
        half_dim = self.channels_t // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.channels_t % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.in_projection(xt)
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x

    @torch.no_grad()
    def sample(self, n_samples: int = 1000, steps: int = 1000, device: str = "cpu"):
        # sample data from source distribution (normal)
        traj = []
        xt = torch.randn(n_samples, self.channels_data).to(device)
        for t in torch.linspace(0, 1, steps).to(device):
            # predict vector field
            pred = self(xt, t.expand(xt.size(0)))
            # update xt
            xt = xt + (1 / steps) * pred
            traj.append(xt)
        # decode xt to smiles
        return traj
