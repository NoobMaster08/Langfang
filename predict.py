# predict.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from model import DigitCNN4


def _load_img(path: Path, img_h: int, img_w: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if img.size != (img_w, img_h):
        img = img.resize((img_w, img_h), resample=Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[None, ...]  # (1,H,W)
    x = torch.from_numpy(arr)
    x = (x - 0.5) / 0.5
    return x.unsqueeze(0)  # (1,1,H,W)


def _gt_from_name(p: Path, digits: int = 4):
    m = re.search(r"\d{" + str(digits) + r"}", p.stem)
    return m.group(0) if m else None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--input", type=str, required=True, help="file or dir")
    ap.add_argument("--img-h", type=int, default=None)
    ap.add_argument("--img-w", type=int, default=None)
    ap.add_argument("--digits", type=int, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    img_h = args.img_h if args.img_h is not None else int(ckpt.get("img_h", 18))
    img_w = args.img_w if args.img_w is not None else int(ckpt.get("img_w", 50))
    digits = args.digits if args.digits is not None else int(ckpt.get("digits", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitCNN4(in_ch=1, num_classes=10, positions=digits).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    inp = Path(args.input)
    files = []
    if inp.is_dir():
        for p in sorted(inp.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                files.append(p)
    else:
        files = [inp]

    ok = 0
    total = 0
    for p in files:
        x = _load_img(p, img_h, img_w).to(device)
        logits = model(x)              # (1,4,10)
        pred = logits.argmax(dim=2)[0].tolist()
        pred_txt = "".join(str(d) for d in pred)

        gt = _gt_from_name(p, digits=digits)
        if gt is not None:
            total += 1
            good = (pred_txt == gt)
            ok += int(good)
            print(f"{p.name} -> {pred_txt}  (gt={gt}) {'OK' if good else 'BAD'}")
        else:
            print(f"{p.name} -> {pred_txt}")

    if total:
        print(f"Accuracy: {ok}/{total} = {ok/total:.3f}")


if __name__ == "__main__":
    main()
