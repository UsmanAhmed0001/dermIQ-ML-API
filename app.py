import os, io, base64, threading
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
model_ready = False

def load_model_background():
    global processor, model, model_ready
    print("Loading model...")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    model_ready = True
    print("Model ready!")

threading.Thread(target=load_model_background, daemon=True).start()

def decode_image(b64: str) -> Image.Image:
    for p in ["data:image/jpeg;base64,","data:image/png;base64,",
              "data:image/webp;base64,","data:image/heic;base64,"]:
        b64 = b64.replace(p, "")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def is_skin(image_pil: Image.Image, threshold: float = 0.07) -> tuple:
    """
    Skin detection using two published academic rule sets combined:

    1. Kovac et al. (2002) RGB rules — works well for light/medium skin:
       R>95, G>40, B>20, max-min>15, |R-G|>15, R>G, R>B

    2. YCbCr rules — works well for darker skin tones:
       77 <= Cb <= 127  and  133 <= Cr <= 173

    A pixel is "skin" if it passes EITHER rule set.
    Threshold: only 7% of pixels need to be skin-colored.
    This handles close-up lesion shots where the lesion fills most of the frame.
    """
    img = image_pil.resize((128, 128)).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    # ── Rule Set 1: Kovac RGB rules ──────────────────────────────────────────
    cmax = np.maximum(np.maximum(R, G), B)
    cmin = np.minimum(np.minimum(R, G), B)

    kovac = (
        (R > 95) & (G > 40) & (B > 20) &
        ((cmax - cmin) > 15) &
        (np.abs(R - G) > 15) &
        (R > G) & (R > B)
    )

    # ── Rule Set 2: YCbCr rules ──────────────────────────────────────────────
    # Convert RGB → YCbCr
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
    Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128

    ycbcr = (
        (Y > 80) &
        (Cb >= 77) & (Cb <= 127) &
        (Cr >= 133) & (Cr <= 173)
    )

    # Pixel is skin if it passes EITHER rule set
    skin_mask = kovac | ycbcr
    skin_ratio = float(skin_mask.sum()) / (128 * 128)

    return skin_ratio >= threshold, round(skin_ratio, 3)

@app.route("/")
def health():
    return jsonify({"status": "running", "model_ready": model_ready})

@app.route("/classify", methods=["POST"])
def classify():
    if not model_ready:
        return jsonify({
            "error": "Model is warming up. Please try again in 20 seconds.",
            "code": "NOT_READY",
            "retryable": True
        }), 503

    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided."}), 400

    try:
        image_pil = decode_image(data["image"])
    except Exception:
        return jsonify({"error": "Could not read image. Please try again."}), 400

    # ── Skin Detection ───────────────────────────────────────────────────────
    skin_ok, skin_ratio = is_skin(image_pil)

    if not skin_ok:
        return jsonify({
            "error": "No skin detected. Please upload a close-up photo of a skin lesion on your body.",
            "code": "NO_SKIN",
            "skin_ratio": skin_ratio,
        }), 422

    # ── Classification ───────────────────────────────────────────────────────
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    predictions = [
        {"label": model.config.id2label[i], "score": round(p, 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "predictions": predictions,
        "gradcam": None,
        "top_class": model.config.id2label[int(torch.argmax(probs).item())],
        "skin_ratio": skin_ratio,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))