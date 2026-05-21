import os, io, base64
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)
CORS(app)

MODEL_ID = "Uzzyy/dermiq-skin-classifier"

print("Loading model...")
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model ready!")

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layer = model.vit.encoder.layer[-1].layernorm_before
cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

def decode_image(b64_string: str) -> Image.Image:
    for prefix in ["data:image/jpeg;base64,","data:image/png;base64,","data:image/webp;base64,","data:image/heic;base64,"]:
        b64_string = b64_string.replace(prefix, "")
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")

def generate_gradcam(image_pil: Image.Image, class_idx: int) -> str:
    inputs = processor(images=image_pil, return_tensors="pt")
    input_tensor = inputs["pixel_values"]
    img_resized = image_pil.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True,
        colormap=2,
        image_weight=0.5,
    )
    pil_result = Image.fromarray(visualization)
    buffer = io.BytesIO()
    pil_result.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

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

    predictions = []
    for idx, prob in enumerate(probs.tolist()):
        predictions.append({"label": model.config.id2label[idx], "score": round(prob, 6)})
    predictions.sort(key=lambda x: x["score"], reverse=True)

    top_class_idx = int(torch.argmax(probs).item())

    gradcam_image = None
    try:
        gradcam_image = generate_gradcam(image_pil, top_class_idx)
    except Exception as e:
        print(f"Grad-CAM error: {e}")

    return jsonify({
        "predictions": predictions,
        "gradcam": gradcam_image,
        "top_class": model.config.id2label[top_class_idx],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))