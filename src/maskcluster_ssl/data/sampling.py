"""Epoch-indexed training order independent of model and worker RNG."""

import torch
from torch.utils.data import Sampler

from .transforms import stream_seed


class EpochSampler(Sampler):
    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(stream_seed(self.seed, self.epoch, 0, "order"))
        return iter((self.epoch, int(index)) for index in torch.randperm(len(self.dataset), generator=generator))
