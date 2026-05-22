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
    print("Loading model in background...")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    model_ready = True
    print("Model ready!")

# Load model in background thread at startup
# This means Railway boots instantly AND model is ready within 30s
threading.Thread(target=load_model_background, daemon=True).start()

def decode_image(b64: str) -> Image.Image:
    for p in ["data:image/jpeg;base64,","data:image/png;base64,",
              "data:image/webp;base64,","data:image/heic;base64,"]:
        b64 = b64.replace(p, "")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

@app.route("/")
def health():
    return jsonify({"status": "running", "model_ready": model_ready})

@app.route("/classify", methods=["POST"])
def classify():
    if not model_ready:
        return jsonify({"error": "Model is warming up. Please try again in 20 seconds.", "retryable": True}), 503

    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided."}), 400

    try:
        image_pil = decode_image(data["image"])
    except Exception:
        return jsonify({"error": "Could not read image."}), 400

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
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))