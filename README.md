# 🌿 FarmSense – AI Plant Disease Detection System

FarmSense is an AI-powered Plant Disease Detection System built using **Python, Flask, HTML, CSS, JavaScript, and MobileViT-XS deep learning model**. The system helps identify plant diseases from uploaded leaf images and provides prediction confidence in real-time.

This project was developed as a major project focused on improving agricultural assistance using computer vision and deep learning.

---

# 🚀 Features

* 🌱 Detects plant diseases using AI
* 📸 Upload leaf images for prediction
* 🤖 Deep Learning based classification
* ⚡ Real-time prediction results
* 📊 Confidence score display
* 🌐 Flask backend API
* 🎨 Interactive frontend UI
* 🧠 MobileViT-XS model integration
* 📂 Supports multiple disease classes

---

# 🛠️ Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript

## Backend

* Python
* Flask
* Flask-CORS

## Deep Learning

* PyTorch
* MobileViT-XS
* OpenCV
* NumPy
* PIL

---

# 📁 Project Structure

```bash
plant_disease_detection/
│
├── app.py                  # Flask backend server
├── predict.py              # Prediction logic
├── farmsense.html          # Frontend UI
├── classes.txt             # Disease class labels
├── model/                  # Trained model files
├── test_imgs/              # Sample test images
├── gradcam.py              # GradCAM visualization
├── confusion_matrix.py     # Evaluation metrics
├── plot_training.py        # Training graph plotting
├── training_results.png    # Training results
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/farmsense-plant-disease-detection.git
cd farmsense-plant-disease-detection
```

---

## 2️⃣ Install Dependencies

```bash
pip install flask flask-cors torch torchvision pillow numpy opencv-python
```

Or install using requirements file:

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run the Flask Backend

```bash
python app.py
```

Backend will start at:

```bash
http://127.0.0.1:5000
```

---

## 4️⃣ Open Frontend

Open:

```bash
farmsense.html
```

using:

* Live Server in VS Code
* OR directly in browser

---

# 🔍 How It Works

1. User uploads a plant leaf image.
2. Frontend sends image to Flask backend.
3. Backend processes image.
4. MobileViT-XS model predicts disease.
5. Prediction and confidence score are returned.
6. Result is displayed on frontend UI.

---

# 📊 Model Information

* Model Architecture: **MobileViT-XS**
* Total Classes: **29**
* Framework: **PyTorch**
* Task: Multi-class plant disease classification

---

# 🧪 API Endpoint

## Predict Disease

```http
POST /predict
```

### Request

Form-data:

```bash
file: image
```

### Response

```json
{
  "prediction": "Tomato Leaf Blight",
  "confidence": 98.45
}
```

---

# 📸 Screenshots

Add your project screenshots here.

Example:

```md
![Home Page](screenshots/home.png)
![Prediction Result](screenshots/result.png)
```

---

# 📈 Future Improvements

* 🌐 Deploy on cloud
* 📱 Mobile application support
* 🎤 Voice assistance for farmers
* 🌍 Multi-language support
* 📷 Real-time camera detection
* ☁️ Online disease database integration

---

# 🤝 Contributors

* Sumit Kumar

---

# 📄 License

This project is developed for educational and research purposes.

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
