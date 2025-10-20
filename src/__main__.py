import os
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from fmp.helpers import set_seed
from fmp.training import train


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(config):
    set_seed(config.experiment.seed)
    if config.experiment.mode == "training":
        if config.experiment.save_checkpoints:
            checkpoint_path = os.path.join(
                config.experiment.checkpoint_path,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            OmegaConf.save(config, os.path.join(checkpoint_path, "config.yaml"))
        else:
            checkpoint_path = None
        train(
            checkpoint_path=checkpoint_path,
            model_type=config.experiment.model_type,
            model_name=config.experiment.model_name,
            model_kwargs=config.models.generative_models,
            trainer_kwargs=config.trainer,
            optimizer_kwargs=config.optimizer,
            model_ckpt=config.experiment.model_ckpt,
            epoch_save_interval=config.experiment.epoch_save_interval,
            method_embeddings=config.experiment.transform_method,
            dataset=config.experiment.dataset,
            device=config.models.device,
        )
