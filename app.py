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
<<<<<<< HEAD
=======

# Lazy loading — model loads on first request, not at startup
>>>>>>> parent of 1d5421f (Remove gradcam, clean working version)
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

<<<<<<< HEAD
def is_skin_image(image_pil: Image.Image, threshold: float = 0.06) -> tuple:
    """
    Checks for skin-coloured pixels using broad HSV ranges.
    Threshold lowered to 0.06 (6%) to handle:
    - Close-up lesion photos (lesion fills frame, little surrounding skin)
    - Dark skin tones
    - Different lighting conditions
    """
    img = image_pil.resize((64, 64)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

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

    # Broad skin ranges — covers light to dark skin tones
    skin_mask = (
        ((h_deg <= 40) | (h_deg >= 320)) &
        (s >= 0.05) & (s <= 0.95) &
        (v >= 0.15)
    )

    skin_ratio = float(skin_mask.sum()) / (64 * 64)
    return skin_ratio >= threshold, skin_ratio

def check_image_quality(image_pil: Image.Image) -> tuple:
    w, h = image_pil.size
    if w < 50 or h < 50:
        return False, "Image too small. Please take a closer photo."

    arr = np.array(image_pil.convert("L").resize((64, 64))).astype(np.float32)
    avg_brightness = arr.mean()

    if avg_brightness < 15:
        return False, "Image too dark. Please ensure good lighting."
    if avg_brightness > 248:
        return False, "Image overexposed. Please reduce lighting."

    return True, ""
=======
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
>>>>>>> parent of 1d5421f (Remove gradcam, clean working version)

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

    # Quality check
    quality_ok, quality_msg = check_image_quality(image_pil)
    if not quality_ok:
        return jsonify({"error": quality_msg, "code": "QUALITY_FAIL"}), 422

    # Skin detection
    skin_ok, skin_ratio = is_skin_image(image_pil)
    if not skin_ok:
        return jsonify({
            "error": "No skin detected. Please upload a close-up photo of a skin lesion on your body.",
            "code": "NO_SKIN",
            "skin_ratio": round(skin_ratio, 3),
        }), 422

    # Classification
    inputs = proc(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = mdl(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    predictions = [
        {"label": mdl.config.id2label[i], "score": round(p, 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)
<<<<<<< HEAD
    top_confidence = predictions[0]["score"]

    # Confidence gate
    if top_confidence < 0.15:
        return jsonify({
            "error": "Could not identify a skin lesion. Please take a clearer, closer photo.",
            "code": "LOW_CONFIDENCE",
        }), 422

    return jsonify({
        "predictions": predictions,
        "gradcam": None,
        "top_class": mdl.config.id2label[int(torch.argmax(probs).item())],
        "skin_ratio": round(skin_ratio, 3),
=======
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
>>>>>>> parent of 1d5421f (Remove gradcam, clean working version)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))