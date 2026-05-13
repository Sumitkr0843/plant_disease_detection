# PLANT DISEASE DETECTION USING DEEP LEARNING

**Major B.Tech Project**  
**Greater Noida Institute of Technology (GNIT-IPU Campus)**

**Team Members:**  
- Sreya Bhattacharjee (00327202722)  
- Priya Kumari (09527202722)  
- Sumit Kumar (02427202722)  
- Ankit Tiwari (00927202722)  

**Project Guide:** Mr. Sujeet Kumar (Asst. Prof.)

**Date:** June 2026

---

## 📋 Project Overview

An intelligent **Plant Disease Detection System** built using **MobileViT-XS** (Hybrid CNN + Vision Transformer). The model can classify **29 disease classes** across **14 crop species** from the Plant Village Dataset with high accuracy (~99.3%).

**Key Features:**
- Real-time disease prediction via Streamlit
- Grad-CAM visualization for model explainability
- Treatment recommendations
- Lightweight & deployable model

---

## 🛠️ Software & Tools Requirements

- **Python** 3.8 or higher
- **PyTorch** + torchvision
- **Streamlit**
- **timm** (for MobileViT model)
- **Pillow**, **OpenCV**, **NumPy**, **Matplotlib**
- **scikit-learn**

**Recommended:**
- NVIDIA GPU with CUDA support (for faster inference/training)
- VS Code / PyCharm

---

## 📁 Project Structure
CLLG-MAJORPROJECT/
├── plant_disease_detection/
│   ├── app.py                      # Main Streamlit Application
│   ├── classes.txt                 # List of 29 disease classes
│   ├── predict.py                  # Prediction script
│   ├── gradcam.py                  # Grad-CAM visualization
│   ├── confusion_matrix.py
│   ├── plot_training.py
│   ├── model/                      # Trained model files
│   ├── test_imgs/                  # Sample test images
│   ├── pycache/
│   └── style.css                   # Custom styling for Streamlit
├── farmsense.html
└── ... (other supporting files)
text---

## 📥 Dataset Setup

1. The model uses **29 classes** defined in `plant_disease_detection/classes.txt`:

   **Classes (29):**
   1. Apple_Apple_scab  
   2. Apple_Black_rot  
   3. Apple_Cedar_apple_rust  
   4. Apple_Healthy  
   5. BellPepper_Bacterial_spot  
   6. BellPepper_Healthy  
   7. Cherry_Powdery_mildew  
   8. Cherry_Healthy  
   9. Corn_Cercospora_leaf_spot  
   10. Corn_Common_rust  
   11. Corn_Northern_leaf_blight  
   12. Corn_Healthy  
   13. Grape_Black_rot  
   14. Grape_Esca  
   15. Grape_Leaf_blight  
   16. Grape_Healthy  
   17. Peach_Bacterial_spot  
   18. Peach_Healthy  
   19. Potato_Early_blight  
   20. Potato_Late_blight  
   21. Potato_Healthy  
   22. Strawberry_Leaf_scorch  
   23. Strawberry_Healthy  
   24. Tomato_Bacterial_spot  
   25. Tomato_Early_blight  
   26. Tomato_Late_blight  
   27. Tomato_Septoria_leaf_spot  
   28. Tomato_Yellow_leaf_curl_virus  
   29. Tomato_Healthy  

2. For testing, place your leaf images in the `test_imgs/` folder.

---

## 🚀 Project Execution Steps

### 1. Setup Project
- Download or open the project folder `CLLG-MAJORPROJECT`.
- Navigate to the project directory:
  ```bash
  cd CLLG-MAJORPROJECT/plant_disease_detection
2. Install Dependencies
Bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm streamlit pillow opencv-python numpy matplotlib scikit-learn albumentations
3. Run the Application
Bashstreamlit run app.py

Open the local URL shown in the terminal (usually http://localhost:8501).


🎯 How to Use

Run streamlit run app.py
Upload a plant leaf image or select from test images
Click Predict
View:
Predicted Disease + Confidence Score
Grad-CAM Heatmap (shows where model is looking)
Recommended Treatment



📊 Model Details

Architecture: MobileViT-XS
Training: Transfer Learning (ImageNet-1K)
Optimizer: AdamW + Cosine Annealing LR
Input Size: 224×224
Classes: 29 (as per classes.txt)


🔧 Additional Scripts

predict.py → Standalone prediction
gradcam.py → Generate Grad-CAM visualizations
confusion_matrix.py → View model performance


⚠️ Important Notes

Make sure the model/ folder contains the trained .pth model file.
Color-distorting augmentations were avoided during training (color is a key indicator for diseases).
For best results, use clear, well-lit leaf images with plain background.


🔮 Future Scope

Mobile App Development
Disease Severity Estimation
Multi-disease detection
Real-field image support



Department of Computer Science & Engineering
Greater Noida Institute of Technology (GNIT-IPU Campus)
