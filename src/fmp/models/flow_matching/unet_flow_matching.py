import torch
from torch import nn

from fmp.models.unet import UNetModel


class UNetFlowMatching(nn.Module):
    def __init__(self, dim: tuple[int, int, int], num_channels: int, num_res_blocks: int):
        super().__init__()
        self.dim = dim
        self.unet_model = UNetModel(dim=dim, num_channels=num_channels, num_res_blocks=num_res_blocks)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet_model(t=t, x=xt)
