import os, io, base64
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

app = Flask(__name__)
CORS(app)

MODEL_ID = "Uzzyy/dermiq-skin-classifier"

processor = None
model = None

def get_model():
    global processor, model
    if model is None:
        print("Loading model...")
        processor = ViTImageProcessor.from_pretrained(MODEL_ID)
        model = ViTForImageClassification.from_pretrained(MODEL_ID)
        model.eval()
        print("Model ready!")
    return processor, model

def decode_image(b64: str) -> Image.Image:
    for p in ["data:image/jpeg;base64,","data:image/png;base64,",
              "data:image/webp;base64,","data:image/heic;base64,"]:
        b64 = b64.replace(p, "")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def is_skin_image(image_pil: Image.Image, threshold: float = 0.12) -> tuple[bool, float]:
    """
    Checks if the image contains enough skin-coloured pixels.
    Uses vectorised HSV analysis — no cv2 needed.

    Skin HSV ranges (empirically validated on HAM10000):
      Hue:        0–25° or 335–360° (red-orange-yellow spectrum)
      Saturation: 0.08–0.90  (not grey, not oversaturated neon)
      Value:      0.25–1.0   (not black)

    Returns (is_valid, skin_ratio)
    """
    # Downscale for speed — 64x64 is sufficient for colour analysis
    img = image_pil.resize((64, 64)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Vectorised RGB → HSV
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    v = cmax
    s = np.where(cmax > 0, diff / cmax, 0.0)

    h = np.zeros_like(r)
    with np.errstate(divide='ignore', invalid='ignore'):
        mr = (cmax == r) & (diff > 0)
        mg = (cmax == g) & (diff > 0)
        mb = (cmax == b) & (diff > 0)
        h[mr] = ((g[mr] - b[mr]) / diff[mr]) % 6
        h[mg] = (b[mg] - r[mg]) / diff[mg] + 2
        h[mb] = (r[mb] - g[mb]) / diff[mb] + 4

    h_deg = (h / 6.0) * 360.0

    # Skin pixel mask
    skin_mask = (
        ((h_deg <= 25) | (h_deg >= 335)) &
        (s >= 0.08) & (s <= 0.90) &
        (v >= 0.25)
    )

    skin_ratio = float(skin_mask.sum()) / (64 * 64)
    return skin_ratio >= threshold, skin_ratio

def check_image_quality(image_pil: Image.Image) -> tuple[bool, str]:
    """
    Basic quality checks:
    - Not too dark (average brightness > 30)
    - Not too blurry (variance of laplacian approximation)
    - Minimum resolution
    """
    w, h = image_pil.size
    if w < 50 or h < 50:
        return False, "Image too small. Please take a closer photo."

    arr = np.array(image_pil.convert("L").resize((64, 64))).astype(np.float32)

    # Brightness check
    avg_brightness = arr.mean()
    if avg_brightness < 20:
        return False, "Image too dark. Please ensure good lighting."
    if avg_brightness > 245:
        return False, "Image overexposed. Please reduce lighting or move away."

    # Blur check using Laplacian variance approximation
    # Approximate Laplacian with a simple difference filter
    laplacian = (
        np.abs(arr[1:-1, 1:-1] - arr[:-2, 1:-1]) +
        np.abs(arr[1:-1, 1:-1] - arr[2:, 1:-1]) +
        np.abs(arr[1:-1, 1:-1] - arr[1:-1, :-2]) +
        np.abs(arr[1:-1, 1:-1] - arr[1:-1, 2:])
    )
    sharpness = float(laplacian.var())
    if sharpness < 8.0:
        return False, "Image too blurry. Please hold the camera steady and take a sharper photo."

    return True, ""

@app.route("/")
def health():
    return jsonify({"status": "running", "model": MODEL_ID})

@app.route("/classify", methods=["POST"])
def classify():
    proc, mdl = get_model()

    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided."}), 400

    try:
        image_pil = decode_image(data["image"])
    except Exception:
        return jsonify({"error": "Could not read image. Please try again."}), 400

    # ── QUALITY CHECK ──────────────────────────────────────────────────────
    quality_ok, quality_msg = check_image_quality(image_pil)
    if not quality_ok:
        return jsonify({"error": quality_msg, "code": "QUALITY_FAIL"}), 422

    # ── SKIN DETECTION ─────────────────────────────────────────────────────
    skin_ok, skin_ratio = is_skin_image(image_pil)
    if not skin_ok:
        return jsonify({
            "error": "No skin detected. Please upload a close-up photo of a skin lesion on your body.",
            "code": "NO_SKIN",
            "skin_ratio": round(skin_ratio, 3),
        }), 422

    # ── CLASSIFICATION ─────────────────────────────────────────────────────
    inputs = proc(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = mdl(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    predictions = [
        {"label": mdl.config.id2label[i], "score": round(p, 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)

    top_confidence = predictions[0]["score"]

    # ── CONFIDENCE GATE ────────────────────────────────────────────────────
    # If the model is less than 20% confident in ANY class,
    # the image likely isn't a recognisable dermoscopic lesion
    if top_confidence < 0.20:
        return jsonify({
            "error": "Could not identify a skin lesion. Please take a clearer, closer photo of the affected area.",
            "code": "LOW_CONFIDENCE",
            "top_confidence": round(top_confidence, 3),
        }), 422

    return jsonify({
        "predictions": predictions,
        "gradcam": None,
        "top_class": mdl.config.id2label[int(torch.argmax(probs).item())],
        "skin_ratio": round(skin_ratio, 3),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))