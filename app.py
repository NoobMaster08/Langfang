# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch

from model import DigitCNN4


def get_env_path(var_name: str, default: str) -> str:
    val = os.getenv(var_name, "").strip()
    return val if val else default


@st.cache_resource
def load_model_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    img_h = int(ckpt.get("img_h", 18))
    img_w = int(ckpt.get("img_w", 50))
    digits = int(ckpt.get("digits", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN4(in_ch=1, num_classes=10, positions=digits).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    return model, device, img_h, img_w, digits


def preprocess(pil_img: Image.Image, img_h: int, img_w: int) -> torch.Tensor:
    img = pil_img.convert("L")
    if img.size != (img_w, img_h):
        img = img.resize((img_w, img_h), resample=Image.BICUBIC)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[None, ...]  # (1,H,W)
    x = torch.from_numpy(arr)
    x = (x - 0.5) / 0.5   # [-1..1]
    return x.unsqueeze(0)  # (1,1,H,W)


@torch.no_grad()
def predict_digits(model, device, x: torch.Tensor):
    x = x.to(device)
    logits = model(x)  # (1,digits,10)
    pred = logits.argmax(dim=2)[0].tolist()
    pred_txt = "".join(str(d) for d in pred)

    probs = torch.softmax(logits, dim=2)
    conf = float(probs.max(dim=2).values.mean().item())
    return pred_txt, conf


def main():
    st.set_page_config(page_title="Digits CAPTCHA OCR", page_icon="🔢", layout="centered")
    st.title("🔢 Digits CAPTCHA OCR")

    # 1) путь берём из переменной окружения
    default_ckpt = "checkpoints_digits4/digits4_best.pt"
    ckpt_path = get_env_path("MODEL_PATH", default_ckpt)

    # 2) но даём возможность переопределить в UI (опционально)
    with st.sidebar:
        st.header("Settings")
        ckpt_path = st.text_input("MODEL_PATH", value=ckpt_path)
        st.caption("Можно задать через переменную окружения MODEL_PATH")

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        st.error(f"Checkpoint не найден: {ckpt_file.resolve()}")
        st.stop()

    try:
        model, device, img_h, img_w, digits = load_model_from_ckpt(str(ckpt_file))
    except Exception as e:
        st.error(f"Не удалось загрузить модель: {e}")
        st.stop()

    st.caption(f"Device: {device} | expected size: {img_w}×{img_h} | digits: {digits}")

    uploaded = st.file_uploader("Upload captcha image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if uploaded is None:
        st.stop()

    raw = uploaded.read()
    pil = Image.open(io.BytesIO(raw))

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil, caption="Input", use_container_width=True)

    x = preprocess(pil, img_h, img_w)

    if st.button("Predict", type="primary"):
        pred_txt, conf = predict_digits(model, device, x)
        with col2:
            st.subheader("Result")
            st.code(pred_txt)
            st.write(f"Confidence: **{conf:.3f}**")

        with st.expander("Preprocessed (what model sees)"):
            vis = pil.convert("L").resize((img_w, img_h), resample=Image.BICUBIC)
            st.image(vis, caption=f"{img_w}×{img_h}", use_container_width=False)


if __name__ == "__main__":
    main()
