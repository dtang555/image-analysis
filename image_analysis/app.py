"""Flask app: upload a card's front and back, get an estimated PSA grade."""
import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request
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


def load_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    if img.width > img.height:
        img = img.rotate(90, expand=True)
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
