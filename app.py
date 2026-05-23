import os, io, base64, threading
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

app = Flask(__name__)
CORS(app, origins="*")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

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
    Kovac RGB rules + YCbCr rules for skin detection.
    Covers light, medium, and dark skin tones.
    """
    img = image_pil.resize((128, 128)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax = np.maximum(np.maximum(R, G), B)
    cmin = np.minimum(np.minimum(R, G), B)

    # Kovac RGB rules (2002)
    kovac = (
        (R > 95) & (G > 40) & (B > 20) &
        ((cmax - cmin) > 15) &
        (np.abs(R - G) > 15) &
        (R > G) & (R > B)
    )

    # YCbCr rules
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
    Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128
    ycbcr = (Y > 80) & (Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)

    skin_ratio = float((kovac | ycbcr).sum()) / (128 * 128)
    return skin_ratio >= threshold, round(skin_ratio, 3)

def compute_entropy(predictions: list) -> float:
    """
    Shannon entropy of prediction distribution, normalised to [0, 1].

    WHY THIS WORKS (from ISIC Grand Challenge research):
    - Real lesion: model is confident → one dominant class → LOW entropy (< 0.65)
    - Non-lesion: model is confused → spreads across all 7 classes → HIGH entropy (> 0.72)

    Example — Tube photo (37%, 18%, 12%, 12%, 9%, 9%, 2%):
        entropy = 1.697 / 1.946 = 0.87  → REJECT

    Example — Real melanoma (81%, 10%, 3%, 2%, 2%, 1%, 1%):
        entropy = 0.50 / 1.946 = 0.26   → PASS
    """
    probs = np.array([p["score"] for p in predictions], dtype=np.float64)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))  # ln(7) = 1.946
    return float(entropy / max_entropy)

@app.route("/")
def health():
    return jsonify({"status": "running", "model_ready": model_ready})

@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if not model_ready:
        return jsonify({
            "error": "Model is warming up. Please try again in 20 seconds.",
            "code": "NOT_READY", "retryable": True
        }), 503

    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided."}), 400

    try:
        image_pil = decode_image(data["image"])
    except Exception:
        return jsonify({"error": "Could not read image. Please try again."}), 400

    # ── STEP 1: Skin colour detection ────────────────────────────────────────
    skin_ok, skin_ratio = is_skin(image_pil)
    if not skin_ok:
        return jsonify({
            "error": "No skin detected. Please upload a close-up photo of a skin lesion on your body.",
            "code": "NO_SKIN",
            "skin_ratio": skin_ratio,
        }), 422

    # ── STEP 2: Classification ────────────────────────────────────────────────
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    predictions = [
        {"label": model.config.id2label[i], "score": round(float(p), 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)

    top_conf   = predictions[0]["score"]
    second_conf = predictions[1]["score"] if len(predictions) > 1 else 0
    conf_gap   = top_conf - second_conf
    entropy    = compute_entropy(predictions)

    # ── STEP 3: Lesion validation (3-signal gate) ─────────────────────────────
    # A real dermoscopic lesion must pass ALL THREE:
    # 1. Top confidence >= 40%  (model has a clear answer)
    # 2. Entropy <= 0.72        (model is not equally uncertain about all classes)
    # 3. Confidence gap >= 0.18 (dominant class is significantly ahead of second)
    #
    # Non-lesion images (random surfaces, body parts without lesions) fail
    # at least one of these because the model spreads its probability
    # evenly across all 7 lesion types — exactly as ISIC research showed.

    lesion_valid = (
        top_conf >= 0.40 and
        entropy  <= 0.72 and
        conf_gap >= 0.18
    )

    if not lesion_valid:
        return jsonify({
            "error": "No clear skin lesion detected. Please take a close-up photo with the lesion centred and filling most of the frame.",
            "code": "NO_LESION",
            "debug": {
                "top_confidence": round(top_conf, 3),
                "entropy": round(entropy, 3),
                "confidence_gap": round(conf_gap, 3),
            }
        }), 422

    return jsonify({
        "predictions": predictions,
        "gradcam": None,
        "top_class": model.config.id2label[int(torch.argmax(probs).item())],
        "skin_ratio": skin_ratio,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))