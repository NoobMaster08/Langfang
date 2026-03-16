import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from model import CRNN   # твой model.py
from codec import TextCodec, DIGITS_ALPHABET  # или как у тебя называется алфавит/кодек


# --- config ---
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints_digits4/crnn_ctc_digits4_best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_H = int(os.getenv("IMG_H", "18"))
IMG_W = int(os.getenv("IMG_W", "50"))

app = FastAPI(title="Digits CAPTCHA OCR API")

_model: Optional[torch.nn.Module] = None
_codec: Optional[TextCodec] = None


def load_model():
    global _model, _codec
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    alphabet = ckpt.get("alphabet", "0123456789")
    _codec = TextCodec(alphabet)

    _model = CRNN(num_classes=len(_codec.alphabet), in_ch=1)
    _model.load_state_dict(ckpt["state_dict"], strict=True)
    _model.eval().to(DEVICE)


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    # grayscale
    img = pil_img.convert("L")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0
    # (1, H, W)
    arr = arr[None, :, :]
    # (1, 1, H, W)
    x = torch.from_numpy(arr).unsqueeze(0)
    return x


@torch.inference_mode()
def predict_tensor(x: torch.Tensor):
    x = x.to(DEVICE)
    logits = _model(x)                # (T,N,C)
    logp = F.log_softmax(logits, dim=2)

    # greedy decode
    pred = logp.argmax(dim=2).transpose(0, 1)[0].tolist()  # first sample
    text = _codec.decode_greedy(pred)

    probs = torch.softmax(logits, dim=2)
    conf = float(probs.max(dim=2).values.mean().item())
    return text, conf


@app.on_event("startup")
def _startup():
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_path": MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image")

    x = preprocess(img)
    text, conf = predict_tensor(x)
    return JSONResponse({"text": text, "confidence": conf})
