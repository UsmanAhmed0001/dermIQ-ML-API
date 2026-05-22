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
    img = image_pil.resize((128, 128)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax = np.maximum(np.maximum(R, G), B)
    cmin = np.minimum(np.minimum(R, G), B)
    kovac = (
        (R > 95) & (G > 40) & (B > 20) &
        ((cmax - cmin) > 15) &
        (np.abs(R - G) > 15) &
        (R > G) & (R > B)
    )
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
    Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128
    ycbcr = (Y > 80) & (Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)
    skin_ratio = float((kovac | ycbcr).sum()) / (128 * 128)
    return skin_ratio >= threshold, round(skin_ratio, 3)

@app.route("/")
def health():
    return jsonify({"status": "running", "model_ready": model_ready})

@app.route("/classify", methods=["POST"])
def classify():
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

    # Skin detection
    skin_ok, skin_ratio = is_skin(image_pil)
    if not skin_ok:
        return jsonify({
            "error": "No skin detected. Please upload a close-up photo of a skin lesion on your body.",
            "code": "NO_SKIN", "skin_ratio": skin_ratio,
        }), 422

    # Classification
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    predictions = [
        {"label": model.config.id2label[i], "score": round(p, 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)

    # Confidence gate — rejects images where skin is visible but no clear lesion
    if predictions[0]["score"] < 0.35:
        return jsonify({
            "error": "No skin lesion detected. Please take a close-up photo of the specific lesion you want to analyse.",
            "code": "NO_LESION",
        }), 422

    return jsonify({
        "predictions": predictions,
        "gradcam": None,
        "top_class": model.config.id2label[int(torch.argmax(probs).item())],
        "skin_ratio": skin_ratio,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))