import os, io, base64
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter
from transformers import ViTForImageClassification, ViTImageProcessor

app = Flask(__name__)
CORS(app)

MODEL_ID = "Uzzyy/dermiq-skin-classifier"

print("Loading model...")
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model ready!")

def decode_image(b64: str) -> Image.Image:
    for p in ["data:image/jpeg;base64,","data:image/png;base64,","data:image/webp;base64,","data:image/heic;base64,"]:
        b64 = b64.replace(p, "")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def generate_attention_map(image_pil: Image.Image, class_idx: int) -> str:
    """
    ViT Attention Rollout — no opencv needed.
    Uses the attention weights from all transformer heads to produce
    a heatmap showing which image patches the model focused on.
    """
    inputs = processor(images=image_pil, return_tensors="pt")

    # Collect attention maps from all layers
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # list of (1, heads, tokens, tokens) per layer

    # Average attention across all heads in each layer
    att_mat = torch.stack([a.squeeze(0).mean(0) for a in attentions])  # (layers, tokens, tokens)

    # Attention rollout — multiply attention matrices across layers
    # Add identity matrix to account for residual connections
    result = torch.eye(att_mat.size(-1))
    for att in att_mat:
        att_with_residual = att + torch.eye(att.size(-1))
        att_with_residual = att_with_residual / att_with_residual.sum(dim=-1, keepdim=True)
        result = att_with_residual @ result

    # Extract attention from [CLS] token to all patch tokens
    # CLS token is index 0, patches are 1:197
    mask = result[0, 1:]  # shape: (196,) for 14x14 patches
    mask = mask.reshape(14, 14).numpy()

    # Normalise to 0-1
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Resize mask to match image size
    img_resized = image_pil.resize((224, 224))
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    mask_arr = np.array(mask_pil) / 255.0

    # Create colourmap (blue→green→red) without opencv
    heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap[:, :, 0] = (mask_arr * 255).astype(np.uint8)          # Red channel
    heatmap[:, :, 1] = ((1 - np.abs(mask_arr - 0.5) * 2) * 200).astype(np.uint8)  # Green
    heatmap[:, :, 2] = ((1 - mask_arr) * 255).astype(np.uint8)    # Blue

    # Blend with original image
    orig_arr = np.array(img_resized)
    blended = (orig_arr * 0.5 + heatmap * 0.5).astype(np.uint8)

    # Convert to base64
    result_pil = Image.fromarray(blended)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

@app.route("/")
def health():
    return jsonify({"status": "running", "model": MODEL_ID})

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    image_pil = decode_image(data["image"])

    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    predictions = [
        {"label": model.config.id2label[i], "score": round(p, 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)
    top_class_idx = int(torch.argmax(probs).item())

    gradcam_image = None
    try:
        gradcam_image = generate_attention_map(image_pil, top_class_idx)
    except Exception as e:
        print(f"Attention map error: {e}")

    return jsonify({
        "predictions": predictions,
        "gradcam": gradcam_image,
        "top_class": model.config.id2label[top_class_idx],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))