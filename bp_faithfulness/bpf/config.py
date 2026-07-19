"""Config loading + global seeding. The YAML is the single source of truth."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Config:
    raw: dict

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def __getitem__(self, key):
        return self.raw[key]


def load_config(path: str | Path = None) -> Config:
    path = Path(path or Path(__file__).resolve().parent.parent / "config.yaml")
    with open(path) as f:
        return Config(yaml.safe_load(f))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
