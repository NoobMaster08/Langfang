# model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN4(nn.Module):
    """
    Предсказывает 4 цифры (0..9) напрямую.
    Вход:  (N,1,H,W) где H=18, W=50 (по умолчанию)
    Выход: logits (N,4,10)
    """
    def __init__(self, in_ch: int = 1, num_classes: int = 10, positions: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.positions = positions

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 18x50 -> 9x25

            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 9x25 -> 4x12

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x12 -> 2x12

            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 2x12 -> 1x12

            nn.Dropout(0.10),
        )

        # делаем ровно 4 "позиции" по ширине
        self.pool = nn.AdaptiveAvgPool2d((1, positions))  # -> (N,128,1,4)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)          # (N,128,1,W')
        feat = self.pool(feat)           # (N,128,1,4)
        feat = feat.squeeze(2).permute(0, 2, 1)  # (N,4,128)
        logits = self.head(feat)         # (N,4,10)
        return logits
