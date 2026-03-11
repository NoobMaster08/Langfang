# dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _find_label_from_name(p: Path, digits: int = 4) -> Optional[str]:
    """
    Ищем первые digits цифр в имени файла.
    Поддерживает: 8766.png, 8766_anything.png
    """
    m = re.search(r"\d{" + str(digits) + r"}", p.stem)
    if not m:
        return None
    return m.group(0)


def _pil_to_tensor_gray(img: Image.Image) -> torch.Tensor:
    """
    PIL 'L' -> torch float32 (1,H,W) in [0..1]
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[None, ...]  # (1,H,W)
    return torch.from_numpy(arr)


class CaptchaDigitsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        img_h: int = 18,
        img_w: int = 50,
        digits: int = 4,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.digits = int(digits)
        self.augment = bool(augment)

        paths = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(p)
        paths.sort()

        # отфильтруем файлы без корректной метки
        good = []
        skipped = 0
        for p in paths:
            lab = _find_label_from_name(p, digits=self.digits)
            if lab is None:
                skipped += 1
                continue
            good.append(p)

        if not good:
            raise RuntimeError(f"No images with {IMG_EXTS} and {digits}-digit labels in: {self.root}")

        self.paths = good
        if skipped:
            print(f"[dataset] skipped {skipped} files with empty/invalid labels")

    def __len__(self) -> int:
        return len(self.paths)

    def _augment(self, img: Image.Image) -> Image.Image:
        # img is grayscale 'L'
        # очень мягкие аугментации — картинки маленькие
        if random.random() < 0.35:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

        # лёгкий jitter контраста/яркости (через autocontrast иногда)
        if random.random() < 0.20:
            img = ImageOps.autocontrast(img, cutoff=random.uniform(0.0, 1.5))

        # лёгкий шум
        if random.random() < 0.35:
            arr = np.array(img, dtype=np.float32)
            sigma = random.uniform(3.0, 10.0)  # в пикселях 0..255
            noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")

        # маленький сдвиг (±1..2 пикс)
        if random.random() < 0.35:
            dx = random.randint(-2, 2)
            dy = random.randint(-1, 1)
            bg = 255  # фон белый
            canvas = Image.new("L", img.size, color=bg)
            canvas.paste(img, (dx, dy))
            img = canvas

        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        label = _find_label_from_name(p, digits=self.digits)
        assert label is not None and len(label) == self.digits

        img = Image.open(p).convert("L")
        if img.size != (self.img_w, self.img_h):
            img = img.resize((self.img_w, self.img_h), resample=Image.BICUBIC)

        if self.augment:
            img = self._augment(img)

        x = _pil_to_tensor_gray(img)  # (1,H,W) in [0..1]
        # нормализация в [-1, 1]
        x = (x - 0.5) / 0.5

        y = torch.tensor([int(c) for c in label], dtype=torch.long)  # (digits,)
        return x, y


def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # (N,1,H,W)
    y = torch.stack(ys, dim=0)  # (N,digits)
    return x, y
