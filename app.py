import os, io, base64, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import pipeline

app = Flask(__name__)
CORS(app)

print("Loading model...")
classifier = pipeline(
    "image-classification",
    model="Uzzyy/dermiq-skin-classifier",
    top_k=7
)
print("Model ready!")

@app.route("/")
def health():
    return jsonify({"status": "running"})

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    b64 = data["image"]
    for prefix in ["data:image/jpeg;base64,","data:image/png;base64,","data:image/webp;base64,"]:
        b64 = b64.replace(prefix, "")
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    results = classifier(image)
    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
