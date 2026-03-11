# utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import numpy as np
import torch


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def exact_match_acc(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    pred: (N,4) long
    gt:   (N,4) long
    """
    return (pred.eq(gt).all(dim=1).float().mean().item())


@torch.no_grad()
def per_digit_acc(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return pred.eq(gt).float().mean().item()
