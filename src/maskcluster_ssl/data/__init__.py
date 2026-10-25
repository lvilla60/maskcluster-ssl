"""Data contracts for training and evaluation."""

from .hdf5 import ReIDH5Dataset, TrainingH5Dataset, collect_metadata

__all__ = ["ReIDH5Dataset", "TrainingH5Dataset", "collect_metadata"]
