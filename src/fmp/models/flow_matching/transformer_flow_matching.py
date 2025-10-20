import torch
from torch import nn
from torch.nn import functional as F


class SelfAttentionBlock(nn.Module):
    def __init__(self, emb_size: torch.Tensor):
        super().__init__()
        self.emb_size = emb_size
        self.query = nn.Linear(self.emb_size, self.emb_size)
        self.keys = nn.Linear(self.emb_size, self.emb_size)
        self.values = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, X: torch.Tensor):
        Q = self.query(X)
        K = self.keys(X)
        V = self.values(X)

        scores = Q @ K.T

        # need to reshape scores for softmax expecting
        weights = F.softmax(scores, dim=-1)
        return weights @ V


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


class TransformerBlock(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.attention = SelfAttentionBlock(**params["attention"])
        self.mlp = MLP(**params["mlp"])
        self.batch_norm = nn.BatchNorm1d(num_features=params["attention"]["emb_size"])

    def forward(self, X: torch.Tensor):
        attention_weights = self.attention(X)
        normalized_attention_weights = self.batch_norm(attention_weights)
        residual_connections = X + normalized_attention_weights
        return self.mlp(residual_connections)


class TransformerFlowMatching(nn.Module):
    def __init__(self, params: dict, num_layers: int):
        super().__init__()
        self.layers = nn.Sequential(*[TransformerBlock(params=params) for _ in range(num_layers)])
        self.fn_X = nn.Linear(params["attention"]["emb_size"], params["attention"]["emb_size"] - 1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor):
        X = torch.cat([xt, t[:, None]], dim=-1)
        X = self.layers(X)
        return self.fn_X(X)
