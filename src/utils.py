from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


@contextmanager
def timer(name: str = "block"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{name}] {elapsed:.2f}s")
