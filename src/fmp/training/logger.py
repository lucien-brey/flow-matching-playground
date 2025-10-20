from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class CustomSummaryWriter:
    """Class that holds the state of the writer"""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.global_step = 0

    def add_scalar(self, tag: str, scalar_value: float):
        self.writer.add_scalar(tag, scalar_value, self.global_step)

    def add_histogram(self, tag: str, values: torch.Tensor):
        self.writer.add_histogram(tag, values, self.global_step)

    def add_image(self, tag: str, image: torch.Tensor):
        self.writer.add_image(tag, image, self.global_step)

    def add_text(self, tag: str, text: str):
        self.writer.add_text(tag, text, self.global_step)

    def update_global_step(self, global_step: int):
        self.global_step = global_step


writer = CustomSummaryWriter(writer=SummaryWriter(log_dir=f'runs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
