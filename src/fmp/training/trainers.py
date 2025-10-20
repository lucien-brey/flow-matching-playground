from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torchcfm.optimal_transport import OTPlanSampler

from fmp.models.flow_matching.flow_matching import FlowMatching
from fmp.training.logger import writer

VAE_BETA = 0.0001


class BaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, device: str):
        self.device = device
        self.model = model.to(self.device)

    def load_checkpoint(self, model_ckpt):
        self.model.load_state_dict(torch.load(model_ckpt))

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def step(self):
        pass


class TrainerVAE(BaseTrainer):
    def __init__(self, model, vae_beta: float = VAE_BETA, device: str = "cpu"):
        super().__init__(model=model, device=device)
        self.vae_beta = vae_beta

    def loss(
        self,
        z: torch.Tensor,
        z_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ):
        """Overall loss function for the VAE"""
        _vae_loss = self.vae_loss(z, z_hat, mu, logvar)
        return _vae_loss

    def vae_loss(self, z: torch.Tensor, z_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """ELBO loss for the VAE"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # latent_reconstruction_loss = F.binary_cross_entropy(z_hat, z, reduction="sum")
        latent_reconstruction_loss = F.mse_loss(z, z_hat, reduction="sum")
        writer.add_scalar("latent_reconstruction_loss", latent_reconstruction_loss.item())
        writer.add_scalar("kl_loss", kl_loss.item())
        return latent_reconstruction_loss + self.vae_beta * kl_loss

    def step(self, embeddings: torch.Tensor):
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        z, z_hat, mu, logvar = self.model.forward(embeddings)

        return self.loss(
            z=z,
            z_hat=z_hat,
            mu=mu,
            logvar=logvar,
        )


class TrainerSimpleFlowMatching(BaseTrainer):
    def __init__(self, model: FlowMatching, device: str = "cpu"):
        super().__init__(model=model, device=device)
        self.ot_sampler = OTPlanSampler(method="exact")

    def loss(self, pred: torch.Tensor, target: torch.Tensor):
        return ((target - pred) ** 2).mean()

    def step(self, x1: torch.Tensor):
        t = torch.rand(x1.size(0)).to(x1.device)
        # source distribution (normal distribution)
        x0 = torch.randn_like(x1).to(x1.device)
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        target = x1 - x0
        xt = (1 - t.reshape(-1, 1)) * x0 + t.reshape(-1, 1) * x1
        pred = self.model(xt=xt, t=t)
        return self.loss(pred, target)


class TrainerUNetFlowMatching(BaseTrainer):
    def __init__(self, model: FlowMatching, device: str = "cpu"):
        super().__init__(model=model, device=device)
        self.ot_sampler = OTPlanSampler(method="exact")

    def loss(self, vt: torch.Tensor, ut: torch.Tensor, lambda_t: torch.Tensor, st: torch.Tensor, eps: torch.Tensor):
        flow_loss = torch.mean((vt - ut) ** 2)
        score_loss = torch.mean((lambda_t * st + eps) ** 2)
        return flow_loss + score_loss

    def step(self, x1: torch.Tensor):
        # reshape x1 to [batch_size,  1, 28, 28]
        x1 = x1.view(x1.size(0), 1, 28, 28)
        x0 = torch.randn_like(x1)
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        t, xt, ut, eps = self.model.flow_matching.sample_location_and_conditional_flow(x0, x1, return_noise=True)
        lambda_t = self.model.flow_matching.compute_lambda(t)
        vt = self.model(t=t, xt=xt)
        st = self.model(t=t, xt=xt)
        return self.loss(vt, ut, lambda_t, st, eps)


class TrainerTransformerFlowMatching(BaseTrainer):
    def __init__(self, model: FlowMatching, device: str = "cpu"):
        super().__init__(model=model, device=device)
        self.ot_sampler = OTPlanSampler(method="exact")

    def loss(self, vt: torch.Tensor, ut: torch.Tensor, lambda_t: torch.Tensor, st: torch.Tensor, eps: torch.Tensor):
        flow_loss = torch.mean((vt - ut) ** 2)
        score_loss = torch.mean((lambda_t * st + eps) ** 2)
        return flow_loss + score_loss

    def step(self, x1: torch.Tensor):
        x0 = torch.randn_like(x1)
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        t, xt, ut, eps = self.model.flow_matching.sample_location_and_conditional_flow(x0, x1, return_noise=True)
        lambda_t = self.model.flow_matching.compute_lambda(t)
        vt = self.model(t=t, xt=xt)
        st = self.model(t=t, xt=xt)
        return self.loss(vt, ut, lambda_t, st, eps)
