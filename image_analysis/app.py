"""Flask app: upload a card's front and back, get an estimated PSA grade."""
import io
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image
import torch

from image_analysis.modeling.train import EVAL_TF, MODELS, TwoViewNet

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = MODELS / "resnet18_full_weighted.pt"

_model = None


def get_model():
    global _model
    if _model is None:
        if not CKPT.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {CKPT}. Train it first with train.py."
            )
        m = TwoViewNet().to(DEVICE)
        m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        m.eval()
        _model = m
    return _model


def auto_crop(pil_img, margin_frac=0.02, min_area_frac=0.15, max_area_frac=0.90):
    """Crop to the card/slab's rectangular boundary if one is confidently found.

    Uses RETR_TREE so a card boundary nested inside the outer photo frame is
    still considered, and rejects near-full-image contours (the photo's own
    border) as well as anything too small or off-aspect to be a card/slab.
    Falls back to the original image untouched if nothing qualifies.
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 20, 80)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img

    h, w = gray.shape
    img_area = h * w
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        area_frac = area / img_area
        if not (min_area_frac <= area_frac <= max_area_frac):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(cw, 1)
        if not (1.1 <= aspect <= 2.0):
            continue
        if area > best_area:
            best_area = area
            best = (x, y, cw, ch)

    if best is None:
        return pil_img

    x, y, cw, ch = best
    mx, my = int(cw * margin_frac), int(ch * margin_frac)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + cw + mx), min(h, y + ch + my)
    if (x1 - x0) < w * 0.3 or (y1 - y0) < h * 0.3:
        return pil_img

    return pil_img.crop((x0, y0, x1, y1))


def load_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    if img.width > img.height:
        img = img.rotate(90, expand=True)
    img = auto_crop(img)
    return EVAL_TF(img).unsqueeze(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "front" not in request.files or "back" not in request.files:
        return jsonify({"error": "Both front and back images are required."}), 400

    try:
        front = load_image(request.files["front"]).to(DEVICE)
        back = load_image(request.files["back"]).to(DEVICE)
    except Exception:
        return jsonify({"error": "Could not read one of the images. Try a JPG or PNG."}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    with torch.no_grad():
        pred = model(front, back).clamp(1, 10).item()

    return jsonify({"grade_raw": pred, "grade_rounded": round(pred)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
