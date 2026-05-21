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

# Lazy loading — model loads on first request, not at startup
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

def generate_attention_map(image_pil: Image.Image, proc, mdl) -> str:
    inputs = proc(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = mdl(**inputs, output_attentions=True)
    attentions = outputs.attentions
    att_mat = torch.stack([a.squeeze(0).mean(0) for a in attentions])
    result = torch.eye(att_mat.size(-1))
    for att in att_mat:
        att_r = att + torch.eye(att.size(-1))
        att_r = att_r / att_r.sum(dim=-1, keepdim=True)
        result = att_r @ result
    mask = result[0, 1:].reshape(14, 14).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    img_resized = image_pil.resize((224, 224))
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    mask_arr = np.array(mask_pil) / 255.0
    heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap[:, :, 0] = (mask_arr * 255).astype(np.uint8)
    heatmap[:, :, 1] = ((1 - np.abs(mask_arr - 0.5) * 2) * 200).astype(np.uint8)
    heatmap[:, :, 2] = ((1 - mask_arr) * 255).astype(np.uint8)
    orig_arr = np.array(img_resized)
    blended = (orig_arr * 0.5 + heatmap * 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

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
    top_class_idx = int(torch.argmax(probs).item())

    gradcam_image = None
    try:
        gradcam_image = generate_attention_map(image_pil, proc, mdl)
    except Exception as e:
        print(f"Attention map error: {e}")

    return jsonify({
        "predictions": predictions,
        "gradcam": gradcam_image,
        "top_class": mdl.config.id2label[top_class_idx],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))