# 🎯 Gender and Age Prediction System

A machine learning-based system for predicting gender and age from facial images with high accuracy. Built with TensorFlow/Keras, featuring both standalone models and a web interface.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.9+-D00000?style=flat-square&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.44+-FF6B6B?style=flat-square&logo=gradio&logoColor=white)

---

## 📋 Table of Contents

- [Features](#-features)
- [Accuracy](#-accuracy)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Troubleshooting](#-troubleshooting)
- [Developer](#-developer)
- [License](#-license)

---

## ✨ Features

### 🤖 AI-Powered Predictions
- ✅ **Gender Prediction**: Binary classification (Male/Female)
- ✅ **Age Prediction**: Regression-based age estimation
- ✅ **Combined Model**: Simultaneous gender and age prediction
- ✅ **Web Interface**: User-friendly Gradio-based web application
- ✅ **Batch Processing**: Support for multiple images
- ✅ **Real-time Inference**: Fast prediction on uploaded images

### 📊 Model Capabilities
- ✅ Pre-trained on large facial datasets
- ✅ EfficientNetB0 backbone for feature extraction
- ✅ Data augmentation for robust training
- ✅ Model checkpointing and early stopping
- ✅ Confusion matrix and classification reports

### 🖥️ User Interfaces
- ✅ Jupyter Notebook interfaces for testing
- ✅ Gradio web app for easy deployment
- ✅ Command-line prediction scripts
- ✅ Visualization of predictions with images

---

## 🎯 Accuracy

| Model Type | Accuracy |
|------------|----------|
| **Gender Only** | **99%** |
| **Gender + Age** | **91-92%** |

*Note: Accuracy may vary based on image quality, lighting, and facial orientation.*

---

## 💻 Requirements

| Component | Minimum Version |
|-----------|-----------------|
| Python | 3.8 or higher |
| TensorFlow | 2.9.1 |
| Keras | 2.9.0 |
| OpenCV | 4.11.0 |
| NumPy | 1.23.0 |
| Matplotlib | 3.9.4 |
| Scikit-learn | 1.6.1 |
| Gradio | 4.44.1 |
| Dlib | 19.24.6 |

---

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/pawit5001/gender-age-prediction.git
cd gender-age-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models
The pre-trained models are included in the `model/` directory:
- `GenderModel-Pro.keras` - Gender prediction model
- `AgeGenderModel-Pro.keras` - Combined gender and age prediction model

---

## 📖 Usage Guide

### 🧪 Testing with Jupyter Notebooks

#### 1. Gender Prediction Only
1. Open `prediction_test.ipynb`
2. Update the `img_path` variable to point to your test image
3. Run the cells to see gender prediction results

#### 2. Gender + Age Prediction
1. Open `prediction_test.ipynb`
2. Scroll to the "For gender + age" section
3. Update the `image_path` variable
4. Run the cells to get both predictions

#### 3. Training the Model
1. Open `GenderModel.ipynb`
2. Ensure dataset is in `dataset/` folder with Train/Validation/Test subfolders
3. Run all cells to train the model from scratch

### 🌐 Web Application

#### Running the Gradio App
1. Open `web_app.ipynb`
2. Run all cells in the notebook
3. The Gradio interface will launch in your browser
4. Upload an image and get instant predictions

#### Alternative: Run as Python Script
```bash
python -c "
import gradio as gr
# Copy the code from web_app.ipynb and run
"
```

### 🔧 Command Line Usage

#### Gender Prediction
```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('model/GenderModel-Pro.keras')

def predict_gender(image_path):
    img = Image.open(image_path).resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return 'Male' if prediction < 0.5 else 'Female'

print(predict_gender('test.jpg'))
```

#### Age + Gender Prediction
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/AgeGenderModel-Pro.keras')

def predict_age_gender(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100)) / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    gender = 'Male' if predictions[0][0][0] < 0.5 else 'Female'
    age = int(predictions[1][0][0])
    
    return gender, age

gender, age = predict_age_gender('test.jpg')
print(f'Gender: {gender}, Age: {age}')
```

---

## 📁 Project Structure

```
gender-age-prediction/
├── GenderModel.ipynb              # Main training notebook
├── prediction_test.ipynb          # Testing and prediction notebook
├── web_app.ipynb                  # Gradio web interface
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── dataset/                       # Training data
│   ├── Train/                     # Training images
│   ├── Validation/                # Validation images
│   └── Test/                      # Test images
│
├── model/                         # Pre-trained models
│   ├── GenderModel-Pro.keras      # Gender prediction model
│   └── AgeGenderModel-Pro.keras   # Combined model
│
├── flagged/                       # Processed/problematic images
├── test.jpg                       # Sample test image
├── test2.jpg                      # Additional test image
├── test3.jpg                      # Additional test image
└── .ipynb_checkpoints/            # Jupyter checkpoints
```

---

## 🧠 Model Details

### Gender Model
- **Architecture**: EfficientNetB0 + Custom head
- **Input Size**: 100x100 RGB images
- **Output**: Binary classification (0: Male, 1: Female)
- **Training**: Binary cross-entropy loss, Adam optimizer
- **Accuracy**: 99% on test set

### Age + Gender Model
- **Architecture**: Convolutional neural network
- **Input Size**: 100x100 RGB images
- **Outputs**: 
  - Gender: Binary classification
  - Age: Regression (continuous value)
- **Loss Functions**: 
  - Gender: Binary cross-entropy
  - Age: Mean squared error
- **Accuracy**: 91-92% combined performance

### Training Details
- **Data Augmentation**: Rotation, shift, shear, zoom, flip
- **Batch Size**: 64
- **Epochs**: Variable with early stopping
- **Callbacks**: Model checkpointing, early stopping

---

## 🔧 Troubleshooting

### ❌ Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### ❌ Model Loading Errors
- Verify model files exist in `model/` directory
- Check TensorFlow version compatibility
- Try reinstalling TensorFlow: `pip install tensorflow==2.9.1`

### ❌ Image Processing Errors
- Ensure images are in RGB format
- Check image dimensions (should be resizable to 100x100)
- Verify OpenCV installation: `python -c "import cv2"`

### ❌ Web App Not Launching
- Check if port 7860 is available
- Try running in a different environment
- Ensure Gradio is properly installed

### ❌ Low Prediction Accuracy
- Use high-quality, well-lit facial images
- Ensure face is clearly visible and centered
- Avoid extreme angles or occlusions

---

## 👨‍💻 Developer

- **Project**: Gender and Age Prediction System
- **GitHub**: [pawit5001/gender-age-prediction](https://github.com/pawit5001/gender-age-prediction)
- **Model**: Based on EfficientNet and custom CNN architectures
- **Framework**: TensorFlow/Keras with OpenCV preprocessing

---

## 📄 License

MIT License - Free to use and modify for educational and research purposes.
