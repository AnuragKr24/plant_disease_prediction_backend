from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# -------- Load Models --------
baseline_model = load_model("models/cnn_baseline.h5")
pso_model = load_model("models/cnn_pso_tuned.h5")

# -------- Class Names --------
# Paste your class names here exactly in the same index order printed by train_data.class_indices
CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

IMG_SIZE = (128, 128)

# -------- CORS Headers --------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

# -------- Preprocess Image --------
def preprocess(image_file):
    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------- Prediction Route --------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        img = preprocess(request.files["image"])

        # Baseline model
        base_prob = baseline_model.predict(img, verbose=0)[0]
        base_idx = np.argmax(base_prob)
        base_label = CLASSES[base_idx]
        base_conf = float(base_prob[base_idx])

        # PSO model
        pso_prob = pso_model.predict(img, verbose=0)[0]
        pso_idx = np.argmax(pso_prob)
        pso_label = CLASSES[pso_idx]
        pso_conf = float(pso_prob[pso_idx])

        return jsonify({
            "baseline_prediction": base_label,
            "baseline_confidence": base_conf,
            "pso_prediction": pso_label,
            "pso_confidence": pso_conf
        })

    except Exception as e:
        print("Backend Error:", e)
        return jsonify({"error": str(e)}), 500

# -------- Health Check --------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend running"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
