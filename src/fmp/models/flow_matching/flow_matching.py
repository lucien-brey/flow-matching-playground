import torch
from models.generative_models.flow_matching import SimpleFlowMatching, TransformerFlowMatching, UNetFlowMatching
from models.generative_models.flow_matching.conditional_flow_matcher import ExactOptimalTransportConditionalFlowMatcher
from models.smi_ted.wrapper_smited import SMITED
from torch import nn


class FlowMatching(nn.Module):
    def __init__(
        self, model: SimpleFlowMatching | UNetFlowMatching | TransformerFlowMatching, smi_ted: SMITED, sigma: float
    ):
        super().__init__()
        self.model = model
        self.smi_ted = smi_ted
        self.flow_matching = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(xt=xt, t=t)

    @torch.no_grad()
    def sample(self, shape: float, n_samples: int = 1000, steps: int = 1000, device: str = "cpu"):
        self.model.to(device=device)
        # sample data from source distribution (normal)
        traj = []
        xt = torch.randn(n_samples, shape).to(device)
        for t in torch.linspace(0, 1, steps):
            # predict vector field
            pred = self.model(xt, t.expand(xt.size(0)).to(device))
            # update xt
            xt = xt + (1 / steps) * pred
            traj.append(xt)
        # decode xt to smiles
        return traj[-1]  # last step
