import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from fmp.data import load_dataset
from fmp.enums import FlowMatchingTypes, GenerativeModels
from fmp.models.flow_matching import SimpleFlowMatching, TransformerFlowMatching, UNetFlowMatching
from fmp.models.flow_matching.flow_matching import FlowMatching
from fmp.training.trainers import (
    TrainerSimpleFlowMatching,
    TrainerTransformerFlowMatching,
    TrainerUNetFlowMatching,
)

HIDDEN_DIM = 768
INPUT_DIM = 768


def get_dataloader(dataset_name: str, batch_size: int = 32):
    dataset = load_dataset(name=dataset_name)
    if isinstance(dataset, IterableDataset):
        return DataLoader(dataset, batch_size=batch_size)
    elif isinstance(dataset, Dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not a valid dataset")


def load_flow_matching(model_name: str, model_kwargs: dict, model_ckpt: str = None, device: str = "cpu"):
    if model_name == FlowMatchingTypes.SIMPLE.value:
        # model = MLP_2(
        #     channels_data=model_kwargs.get("channels_data"),  # input x dim
        #     layers=model_kwargs.get("layers"),  # number of layers
        #     channels=model_kwargs.get("channels"),  # hidden dim
        #     channels_t=model_kwargs.get("channels_t"),  # t dim
        # )
        model = SimpleFlowMatching(
            dim=model_kwargs.get("channels_data"),
            w=model_kwargs.get("channels"),
            time_varying=True,
        )
        model = FlowMatching(model=model, sigma=model_kwargs.get("sigma"))
        trainer = TrainerSimpleFlowMatching(model=model, device=device)
    elif model_name == FlowMatchingTypes.UNET.value:
        model = UNetFlowMatching(
            dim=tuple(model_kwargs[model_name].get("dim")),
            num_channels=model_kwargs[model_name].get("num_channels"),
            num_res_blocks=model_kwargs[model_name].get("num_res_blocks"),
        )
        model = FlowMatching(model=model, sigma=model_kwargs.get("sigma"))
        trainer = TrainerUNetFlowMatching(model=model, device=device)
    elif model_name == FlowMatchingTypes.TRANSFORMER.value:
        model = TransformerFlowMatching(
            params=model_kwargs[model_name].get("params"), num_layers=model_kwargs[model_name].get("num_layers")
        )
        model = FlowMatching(model=model, sigma=model_kwargs.get("sigma"))
        trainer = TrainerTransformerFlowMatching(model=model, device=device)

    if model_ckpt is not None:
        trainer.load_checkpoint(model_ckpt=model_ckpt)
    return trainer


def load_trainer(
    model_type: str,
    model_name: str,
    model_kwargs: dict,
    trainer_kwargs,
    model_ckpt: str = None,
    device: str = "cpu",
):
    if model_type == GenerativeModels.FLOW_MATCHING.value:
        return load_flow_matching(
            model_name=model_name, model_kwargs=model_kwargs[model_type], model_ckpt=model_ckpt, device=device
        )


def get_optimizer(model: torch.nn.Module, optimizer_kwargs: dict):
    return torch.optim.Adam(model.parameters(), **optimizer_kwargs)


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_kwargs: dict):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)


def save_checkpoint(path: str | None, model: torch.nn.Module, epoch: int, is_final: bool = False):
    if path is None:
        return
    if is_final:
        path = os.path.join(path, "trained_model.pth")
    else:
        path = os.path.join(path, f"model_{epoch}.pth")
    torch.save(model.state_dict(), path)


def plot_latent_space(model, scale=5.0, n=25, digit_size=28, figsize=15, device: str = "cpu"):
    # display a n*n 2D manifold of digits
    figure = torch.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decoder(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title("VAE Latent Space Visualization")
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    # return figure
    return figure
