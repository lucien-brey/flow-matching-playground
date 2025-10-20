from tqdm.auto import tqdm

from fmp.training.logger import writer
from fmp.training.utils import get_dataloader, get_optimizer, get_scheduler, load_trainer, save_checkpoint

MAX_EPOCHS = 100000


def train(
    checkpoint_path: str,
    model_type: str,
    model_name: str,
    model_kwargs: dict,
    trainer_kwargs: dict,
    optimizer_kwargs: dict,
    model_ckpt: str = None,
    epoch_save_interval: int = 100,
    dataset: str = "mnist",
    device: str = "cpu",
):
    train_loader = get_dataloader(batch_size=trainer_kwargs.get("batch_size", 32), dataset_name=dataset)
    trainer = load_trainer(
        path="./src/models/smi_ted/inference/smi_ted_light",
        model_type=model_type,
        model_name=model_name,
        model_kwargs=model_kwargs,
        trainer_kwargs=trainer_kwargs,
        model_ckpt=model_ckpt,
        device=device,
    )

    optimizer = get_optimizer(
        model=trainer.model,
        optimizer_kwargs=optimizer_kwargs,
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_kwargs={
            "T_max": 100,
            "eta_min": 1e-5,
        },
    )

    if dataset == "mnist":
        for epoch in range(1000):
            for batch_idx, (images, _) in enumerate(tqdm(train_loader)):
                writer.update_global_step(epoch * len(train_loader) + batch_idx)
                optimizer.zero_grad()

                # flatten images
                images = images.view(images.size(0), -1).to(device=device)

                # append 4 black lines and row
                loss = trainer.step(images)

                loss.backward()
                optimizer.step()
                scheduler.step()

                writer.add_scalar("loss", loss.item())
                writer.add_scalar("lr", scheduler.get_last_lr()[0])

            samples = trainer.model.sample(n_samples=10, shape=images.shape[-1], device=device)
            # reshape samples to [10, 28, 28]
            samples = samples.view(10, 28, 28)
            for i, sample in enumerate(samples):
                sample = (sample - sample.min()) / (sample.max() - sample.min())
                sample = sample.unsqueeze(0)
                writer.add_image(f"samples_{i}", sample.detach().cpu())
            writer.add_image("mean_pp", samples.mean(dim=0).view(1, 28, 28).detach().cpu())
            writer.add_image("std_pp", samples.std(dim=0).view(1, 28, 28).detach().cpu())
            if epoch % epoch_save_interval == 0:
                save_checkpoint(path=checkpoint_path, model=trainer.model, epoch=epoch)
            # writer.add_image("latent_space", plot_latent_space(trainer.model, device=device).unsqueeze(0))

    save_checkpoint(path=checkpoint_path, model=trainer.model, epoch=epoch, is_final=True)
