# train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CaptchaDigitsDataset, collate_batch
from model import DigitCNN4
from utils import set_seeds, exact_match_acc, per_digit_acc


def save_ckpt(path: Path, model: torch.nn.Module, args, best_metric: float):
    ckpt = {
        "state_dict": model.state_dict(),
        "img_h": args.img_h,
        "img_w": args.img_w,
        "digits": args.digits,
        "best_em": best_metric,
    }
    torch.save(ckpt, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, required=True)
    ap.add_argument("--val-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default="checkpoints_digits4")

    ap.add_argument("--img-h", type=int, default=18)
    ap.add_argument("--img-w", type=int, default=50)
    ap.add_argument("--digits", type=int, default=4)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-delta", type=float, default=1e-4)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-aug", action="store_true")

    # fine-tune
    ap.add_argument("--init-from", type=str, default=None, help="path to checkpoint to init from")
    ap.add_argument("--freeze-features-epochs", type=int, default=0, help="freeze CNN blocks for first N epochs")

    args = ap.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hist_csv = out / "history.csv"
    with open(hist_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_em", "val_digit_acc", "lr"])

    # datasets
    train_ds = CaptchaDigitsDataset(
        args.train_dir, img_h=args.img_h, img_w=args.img_w, digits=args.digits, augment=(not args.no_aug)
    )
    if args.val_dir:
        val_ds = CaptchaDigitsDataset(
            args.val_dir, img_h=args.img_h, img_w=args.img_w, digits=args.digits, augment=False
        )
    else:
        n_val = max(1, int(len(train_ds) * args.val_split))
        n_tr = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed)
        )

    tr_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch
    )
    va_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch
    )

    # model
    model = DigitCNN4(in_ch=1, num_classes=10, positions=args.digits).to(device)

    # init-from
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[init] loaded weights from: {args.init_from}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)  # max по EM

    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_em = -1.0
    wait = 0
    best_path = out / "digits4_best.pt"

    # sanity
    x0, y0 = next(iter(tr_dl))
    with torch.no_grad():
        logits0 = model(x0.to(device))
    assert logits0.shape[1] == args.digits and logits0.shape[2] == 10
    print(f"[sanity] logits shape: {tuple(logits0.shape)} -> predicts {args.digits} digits")

    for epoch in range(1, args.epochs + 1):
        # freeze/unfreeze
        if args.freeze_features_epochs > 0:
            freeze = epoch <= args.freeze_features_epochs
            for p in model.features.parameters():
                p.requires_grad = (not freeze)
            if epoch == 1:
                print(f"[freeze] features frozen for first {args.freeze_features_epochs} epoch(s)")

        # ---- train ----
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0

        for x, y in tqdm(tr_dl, desc=f"epoch {epoch} train"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # (N,4)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)  # (N,4,10)
                # CE по каждой позиции
                loss = 0.0
                for i in range(args.digits):
                    loss = loss + F.cross_entropy(logits[:, i, :], y[:, i])
                loss = loss / args.digits

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            tr_loss_sum += loss.item() * bs
            tr_n += bs

        tr_loss = tr_loss_sum / max(1, tr_n)

        # ---- val ----
        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for x, y in tqdm(va_dl, desc=f"epoch {epoch} val"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)

                loss = 0.0
                for i in range(args.digits):
                    loss = loss + F.cross_entropy(logits[:, i, :], y[:, i])
                loss = loss / args.digits

                pred = logits.argmax(dim=2)  # (N,4)
                all_pred.append(pred.cpu())
                all_gt.append(y.cpu())

                bs = x.size(0)
                va_loss_sum += loss.item() * bs
                va_n += bs

        va_loss = va_loss_sum / max(1, va_n)
        pred = torch.cat(all_pred, dim=0)
        gt = torch.cat(all_gt, dim=0)

        em = exact_match_acc(pred, gt)
        dig_acc = per_digit_acc(pred, gt)

        sched.step(em)
        lr = opt.param_groups[0]["lr"]

        print(f"epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | EM {em:.3f} | digit_acc {dig_acc:.3f} | lr {lr:.2e}")

        with open(hist_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{va_loss:.6f}", f"{em:.6f}", f"{dig_acc:.6f}", f"{lr:.6e}"])

        # early stopping по EM (главная метрика)
        if em > best_em + args.min_delta:
            best_em = em
            wait = 0
            save_ckpt(best_path, model, args, best_em)
            print(f" ✓ saved best -> {best_path}")
        else:
            wait += 1
            print(f" no improvement ({wait}/{args.patience})")
            if wait >= args.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
