# app.py
# FarmSense Backend - Plant Disease Detection using MobileViT-XS

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
from predict import predict_with_confidence

app = Flask(__name__)
CORS(app)  # allow frontend to access backend

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_leaf():

    # check if file is sent
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["file"]

    # get file extension (default .jpg)
    ext = os.path.splitext(file.filename)[1]
    if ext == "":
        ext = ".jpg"

    # save image temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    file.save(temp_file.name)
    temp_path = temp_file.name

    try:
        # run model
        class_name, confidence = predict_with_confidence(temp_path)

        # Handle case when no leaf is detected
        if class_name == "No Leaf Detected":

            response = {
                "class": class_name,
                "plant": "Unknown",
                "disease": "No Leaf Detected",
                "confidence": 100,
                "is_healthy": False
            }

            return jsonify(response)


        # Normal disease prediction
        parts = class_name.split("_")

        plant = parts[0]
        disease = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"

        is_healthy = disease.lower() == "healthy"

        response = {
            "class": class_name,
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence * 100, 2),
            "is_healthy": is_healthy
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except PermissionError:
            pass


# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend running"})


if __name__ == "__main__":
    print("🌿 FarmSense Backend Running at http://localhost:5000")
    app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)