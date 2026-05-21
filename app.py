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
    for p in ["data:image/jpeg;base64,","data:image/png;base64,","data:image/webp;base64,","data:image/heic;base64,"]:
        b64 = b64.replace(p, "")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

@app.route("/")
def health():
    return jsonify({"status": "running"})

@app.route("/classify", methods=["POST"])
def classify():
    proc, mdl = get_model()
    data = request.json
    image_pil = decode_image(data["image"])

    inputs = proc(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = mdl(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    predictions = [
        {"label": mdl.config.id2label[i], "score": round(p, 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "predictions": predictions,
        "gradcam": None,
        "top_class": mdl.config.id2label[int(torch.argmax(probs).item())],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))