# predict.py
# MobileViT-XS based Plant Disease Classifier with Face Rejection

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Checkpoint
# -----------------------------
checkpoint = torch.load("model/plant_disease_model.pth", map_location=device)

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

num_classes = len(class_to_idx)

# -----------------------------
# Create Model
# -----------------------------
model = timm.create_model("mobilevit_xs", num_classes=num_classes)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Model loaded successfully")
print("Total classes:", num_classes)

# -----------------------------
# Face Detector (OpenCV)
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_with_confidence(image_path):

    # ---------- Load image with OpenCV ----------
    image_cv = cv2.imread(image_path)

    if image_cv is None:
        return "Invalid Image", 0.0

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # ---------- Face Detection ----------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        return "No Leaf Detected", 1.0

    # ---------- Convert to PIL ----------
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # ---------- Transform ----------
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # ---------- Model Prediction ----------
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    confidence_value = confidence.item()
    class_name = idx_to_class[predicted.item()]

    # ---------- Reject Low Confidence ----------
    if confidence_value < 0.80:
        return "No Leaf Detected", 1.0

    return class_name, confidence_value